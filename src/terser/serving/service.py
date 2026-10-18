"""The two-tier rewrite service (rows 11.3.1-11.3.6), independent of the HTTP layer.

Request flow for ``rewrite(prompt, target, mode)``:

1. **Cache** (11.3.4): LRU lookup on ``sha256(target, mode, normalised prompt)``; a hit returns the
   stored response with ``cached=True``.
2. **Rewrite**: ``mode="fast"`` runs the CPU fast path. ``mode="quality"`` runs the quality
   rewriter on a worker thread within a budget of ``quality_timeout_s`` measured from the start
   of the request (cache lookup included); when every worker is busy the request falls back at
   once instead of queueing.
3. **Timeout fallback** (11.3.5): if the quality path is over budget, raises, or is not
   configured, the fast-path result is used and ``fallback_used=True`` with a
   ``fallback_reason`` (``timeout`` / ``error`` / ``unavailable``; an empty quality rewrite is an
   ``error``). If the fast path also fails the original prompt is returned (``path="original"``;
   the reason stays the quality path's, or ``fast_error`` when there was none), so a request
   never fails because a rewriter did.
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
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from typing import Any

from terser.features.shared import extract_features
from terser.serving.accounting import Pricing, net_savings
from terser.serving.cache import LRUCache, cache_key, normalise
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
        gate_feature_target: str | None = None,
    ) -> None:
        """``gate_feature_target`` pins the tokenizer of the gate's features to the one the gate
        was trained on (``None`` = the request's target); token accounting always uses the
        request's target."""
        self.pricing = pricing
        self.gate_feature_target = gate_feature_target
        self.fast: Rewriter = fast if fast is not None else FastPath()
        self.quality = quality
        self.gate: Gate = gate if gate is not None else ConstraintGate()
        self.gate_threshold = gate_threshold
        self.quality_timeout_s = quality_timeout_s
        self.cache: LRUCache[RewriteResponse] = LRUCache(cache_size)
        self.metrics = metrics
        self._counter = counter
        self._pool = cf.ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="quality")
        # One slot per worker: a request that finds every worker busy (e.g. with timed-out calls
        # still running in the background) falls back at once instead of queueing behind them.
        self._slots = threading.BoundedSemaphore(max_workers)

    def features(self, prompt: str, target: str) -> dict[str, float]:
        """Online feature path: identical code to the offline ``featurize_*`` functions."""
        counter = None
        if self._counter is not None:
            fn = self._counter
            counter = lambda s: fn(s, target)  # noqa: E731
        return extract_features(prompt, target, counter)

    def _run_quality(
        self, prompt: str, target: str, deadline: float
    ) -> tuple[RewriteResult | None, str | None]:
        """Quality rewrite within the request's budget (``deadline`` on ``perf_counter``)."""
        if self.quality is None:
            return None, "unavailable"
        if not self._slots.acquire(blocking=False):
            return None, "timeout"  # every worker is busy: the budget cannot be met
        quality = self.quality

        def call() -> RewriteResult:
            try:
                return quality(prompt, target)
            finally:
                self._slots.release()

        try:
            future = self._pool.submit(call)
        except RuntimeError:  # pool shut down
            self._slots.release()
            return None, "unavailable"
        try:
            result = future.result(timeout=max(0.0, deadline - time.perf_counter()))
        except cf.TimeoutError:
            future.cancel()
            return None, "timeout"
        except Exception as exc:
            log.warning("quality path failed: %s", exc)
            return None, "error"
        if not result.text.strip():
            # An empty rewrite would be served, cached and counted as 100% savings.
            log.warning("quality path returned an empty rewrite")
            return None, "error"
        return result, None

    def _run_fast(self, prompt: str, target: str) -> tuple[RewriteResult, bool]:
        try:
            return self.fast(prompt, target), True
        except Exception as exc:  # last resort: the original prompt
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
            # The key normalises whitespace and line endings, so the hit may come from a variant
            # of this prompt: count this prompt's own tokens, and if the stored result was the
            # unchanged prompt, return this prompt rather than the variant.
            before = int(self.features(prompt, target)["n_tokens"])
            unchanged = normalise(hit.rewrite) == normalise(prompt)
            rewrite = prompt if unchanged else hit.rewrite
            after = before if unchanged else hit.tokens_after
            # a hit costs no rewriter work, so its net savings are the gross savings
            gross = self.pricing.target_price(target) * (before - after)
            resp = replace(
                hit,
                rewrite=rewrite,
                tokens_before=before,
                tokens_after=after,
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
            # The budget covers the whole request, not only the quality call.
            deadline = t0 + self.quality_timeout_s
            result, fallback_reason = self._run_quality(prompt, target, deadline)
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
        score = 0.0
        if rewrite != prompt:
            gt = self.gate_feature_target
            if gt is None or gt == target:
                score = float(self.gate.score(orig, rew))
            else:
                score = float(
                    self.gate.score(self.features(prompt, gt), self.features(rewrite, gt))
                )
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
