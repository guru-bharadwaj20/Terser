"""The two-tier rewrite service (rows 11.3.1-11.3.6), independent of the HTTP layer.

Request flow for ``rewrite(prompt, target, mode)``:

1. **Cache** (11.3.4): LRU lookup on ``sha256(target, mode, normalised prompt)``; a hit returns the
   stored response with ``cached=True``.
2. **Rewrite**: ``mode="fast"`` runs the CPU fast path. ``mode="quality"`` runs the quality
   rewriter on a worker thread with a latency budget of ``quality_timeout_s``.
3. **Timeout fallback** (11.3.5): if the quality path is over budget, raises, or is not
   configured, the fast-path result is used and ``fallback_used=True`` with a
   ``fallback_reason`` (``timeout`` / ``error`` / ``unavailable``). If the fast path also fails the
   original prompt is returned, so a request never fails because a rewriter did.
4. **Gate**: features of original and rewrite come from :mod:`terser.features.shared` (the same
   code used offline, row 11.1.3); if ``P(degraded) > gate_threshold`` the original prompt is
   returned (``gate_rejected=True``).
5. **Accounting and observability** (11.3.6, 11.5.1, 11.5.6): net savings per request, Prometheus
   metrics, and one structured log line with hashes (raw text only with consent).

Fallback results are not cached, so a transient quality-path outage does not pin fast-path
rewrites in the cache. A timed-out worker thread cannot be killed; it finishes in the background
and its result is discarded (the HTTP client's own timeout bounds it).
"""

from __future__ import annotations

import concurrent.futures as cf
import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from typing import Any

from terser.features.shared import extract_features
from terser.serving.accounting import Pricing, net_savings
from terser.serving.cache import LRUCache, cache_key
from terser.serving.paths import ConstraintGate, FastPath, Gate, Rewriter, RewriteResult
from terser.serving.request_log import build_record, log_request, new_request_id

__all__ = ["MODES", "RewriteResponse", "RewriteService"]

MODES: tuple[str, ...] = ("fast", "quality")
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RewriteResponse:
    """What ``POST /rewrite`` returns."""

    rewrite: str
    tokens_before: int
    tokens_after: int
    gate_score: float
    fallback_used: bool
    path: str
    mode: str
    target: str
    gate_rejected: bool
    fallback_reason: str | None
    cached: bool
    net_savings_usd: float
    request_id: str
    latency_ms: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class RewriteService:
    """Cache -> rewrite (with fallback) -> gate -> accounting."""

    def __init__(
        self,
        *,
        pricing: Pricing,
        fast: Rewriter | None = None,
        quality: Rewriter | None = None,
        gate: Gate | None = None,
        gate_threshold: float = 0.5,
        quality_timeout_s: float = 0.4,
        cache_size: int = 10_000,
        metrics: Any | None = None,
        counter: Callable[[str, str], int] | None = None,
        max_workers: int = 8,
    ) -> None:
        self.pricing = pricing
        self.fast: Rewriter = fast if fast is not None else FastPath()
        self.quality = quality
        self.gate: Gate = gate if gate is not None else ConstraintGate()
        self.gate_threshold = gate_threshold
        self.quality_timeout_s = quality_timeout_s
        self.cache: LRUCache[RewriteResponse] = LRUCache(cache_size)
        self.metrics = metrics
        self._counter = counter
        self._pool = cf.ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="quality")

    def features(self, prompt: str, target: str) -> dict[str, float]:
        """Online feature path: identical code to the offline ``featurize_*`` functions."""
        counter = None
        if self._counter is not None:
            fn = self._counter
            counter = lambda s: fn(s, target)  # noqa: E731
        return extract_features(prompt, target, counter)

    def _run_quality(self, prompt: str, target: str) -> tuple[RewriteResult | None, str | None]:
        if self.quality is None:
            return None, "unavailable"
        future = self._pool.submit(self.quality, prompt, target)
        try:
            return future.result(timeout=self.quality_timeout_s), None
        except cf.TimeoutError:
            future.cancel()
            return None, "timeout"
        except Exception as exc:  # noqa: BLE001 - any rewriter failure falls back
            log.warning("quality path failed: %s", exc)
            return None, "error"

    def _run_fast(self, prompt: str, target: str) -> tuple[RewriteResult, bool]:
        try:
            return self.fast(prompt, target), True
        except Exception as exc:  # noqa: BLE001 - last resort: the original prompt
            log.warning("fast path failed: %s", exc)
            return RewriteResult(prompt), False

    def rewrite(
        self,
        prompt: str,
        target: str,
        mode: str = "fast",
        *,
        consent_raw_text: bool = False,
        request_id: str | None = None,
    ) -> RewriteResponse:
        """Serve one request (see the module docstring for the flow)."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.pricing.target_price(target)  # unknown target -> KeyError before any work
        t0 = time.perf_counter()
        rid = request_id or new_request_id()
        key = cache_key(prompt, target, mode)
        hit = self.cache.get(key)
        if hit is not None:
            # a hit costs no rewriter work, so its net savings are the gross savings
            gross = self.pricing.target_price(target) * (hit.tokens_before - hit.tokens_after)
            resp = replace(
                hit,
                cached=True,
                request_id=rid,
                net_savings_usd=gross,
                latency_ms=(time.perf_counter() - t0) * 1e3,
            )
            self._observe(resp, prompt, consent_raw_text)
            return resp

        fallback_reason: str | None = None
        result: RewriteResult | None = None
        path = "fast"
        if mode == "quality":
            result, fallback_reason = self._run_quality(prompt, target)
            if result is not None:
                path = "quality"
        if result is None:
            result, ok = self._run_fast(prompt, target)
            if not ok:
                fallback_reason = fallback_reason or "fast_error"
                path = "original"

        orig = self.features(prompt, target)
        rewrite = result.text
        rew = self.features(rewrite, target) if rewrite != prompt else orig
        score = float(self.gate.score(orig, rew)) if rewrite != prompt else 0.0
        rejected = score > self.gate_threshold
        if rejected:
            rewrite, rew = prompt, orig
        before, after = int(orig["n_tokens"]), int(rew["n_tokens"])
        savings = net_savings(
            self.pricing,
            target,
            before,
            after,
            path=path,
            rewriter_tokens_in=result.tokens_in,
            rewriter_tokens_out=result.tokens_out,
        )
        resp = RewriteResponse(
            rewrite=rewrite,
            tokens_before=before,
            tokens_after=after,
            gate_score=score,
            fallback_used=fallback_reason is not None,
            path=path,
            mode=mode,
            target=target,
            gate_rejected=rejected,
            fallback_reason=fallback_reason,
            cached=False,
            net_savings_usd=savings.net_savings,
            request_id=rid,
            latency_ms=(time.perf_counter() - t0) * 1e3,
        )
        if fallback_reason is None:
            self.cache.put(key, resp)
        self._observe(resp, prompt, consent_raw_text)
        return resp

    def _observe(self, resp: RewriteResponse, prompt: str, consent: bool) -> None:
        saved = resp.tokens_before - resp.tokens_after
        rec = build_record(
            prompt=prompt,
            rewrite=resp.rewrite,
            consent_raw_text=consent,
            request_id=resp.request_id,
            target=resp.target,
            mode=resp.mode,
            path=resp.path,
            tokens_before=resp.tokens_before,
            tokens_after=resp.tokens_after,
            gate_score=resp.gate_score,
            gate_rejected=resp.gate_rejected,
            fallback_used=resp.fallback_used,
            fallback_reason=resp.fallback_reason,
            cached=resp.cached,
            latency_ms=resp.latency_ms,
            tokens_saved=saved,
            rewriter_cost_usd=self._cost(resp),
            net_savings_usd=resp.net_savings_usd,
        )
        log_request(rec)
        if self.metrics is not None:
            self.metrics.observe(
                path=resp.path,
                mode=resp.mode,
                target=resp.target,
                latency_s=resp.latency_ms / 1e3,
                tokens_saved=saved,
                net_savings_usd=resp.net_savings_usd,
                gate_rejected=resp.gate_rejected,
                fallback_reason=resp.fallback_reason,
                cached=resp.cached,
            )

    def _cost(self, resp: RewriteResponse) -> float:
        gross = self.pricing.target_price(resp.target) * (resp.tokens_before - resp.tokens_after)
        return gross - resp.net_savings_usd

    def close(self) -> None:
        """Stop the quality worker pool."""
        self._pool.shutdown(wait=False, cancel_futures=True)
