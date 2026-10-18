"""Service-level tests: cache (11.3.4), timeout fallback fault injection (11.3.5), net-savings
accounting (11.3.6) and structured logging (11.5.6). No HTTP, no tokenizer downloads."""

from __future__ import annotations

import json
import logging
import threading
import time

import pytest

from terser.serving.accounting import Pricing, net_savings
from terser.serving.cache import LRUCache, cache_key, normalise
from terser.serving.paths import RewriteResult
from terser.serving.request_log import LOG_FIELDS, LOGGER_NAME, build_record
from terser.serving.service import RewriteService

PRICING = Pricing({"t": 1_000_000.0}, rewriter_input_per_mtok=1e6, rewriter_output_per_mtok=2e6)
PROMPT = "hey so um could you maybe please list three primes only"


def words(s: str, _target: str) -> int:
    return len(s.split())


def fast(prompt: str, target: str) -> RewriteResult:
    return RewriteResult("list three primes only")


class Quality:
    """Injectable quality rewriter: ok / slow / crash."""

    def __init__(self, behaviour: str, delay: float = 0.0) -> None:
        self.behaviour, self.delay, self.calls = behaviour, delay, 0
        self.release = threading.Event()

    def __call__(self, prompt: str, target: str) -> RewriteResult:
        self.calls += 1
        if self.behaviour == "crash":
            raise RuntimeError("CUDA error: out of memory")
        if self.behaviour == "hang":
            self.release.wait(5)
        time.sleep(self.delay)
        return RewriteResult("list 3 primes only", tokens_in=10, tokens_out=4)


def svc(**kw: object) -> RewriteService:
    return RewriteService(pricing=PRICING, fast=fast, counter=words, **kw)  # type: ignore[arg-type]


# ---- cache -------------------------------------------------------------------------------------
def test_normalise_and_key() -> None:
    assert normalise("  a b  \r\nc \n\n") == "a b\nc"
    assert cache_key("a b \n", "t") == cache_key("a b", "t")
    assert cache_key("A b", "t") != cache_key("a b", "t")  # case can change tokens
    assert cache_key("a", "t1") != cache_key("a", "t2")
    assert cache_key("a", "t", "fast") != cache_key("a", "t", "quality")


def test_lru_eviction_order() -> None:
    c: LRUCache[int] = LRUCache(2)
    c.put("a", 1)
    c.put("b", 2)
    assert c.get("a") == 1  # a is now most recent
    c.put("c", 3)
    assert "b" not in c and "a" in c and "c" in c
    assert c.stats.evictions == 1 and c.stats.hits == 1
    with pytest.raises(ValueError):
        LRUCache(0)


def test_service_cache_hit() -> None:
    s = svc()
    r1 = s.rewrite(PROMPT, "t")
    r2 = s.rewrite(PROMPT + "  \n", "t")
    assert not r1.cached and r2.cached and r2.rewrite == r1.rewrite
    assert r2.request_id != r1.request_id
    assert s.cache.stats.hits == 1


# ---- fault injection: timeout fallback (11.3.5) ------------------------------------------------
def test_quality_ok_uses_quality_path() -> None:
    r = svc(quality=Quality("ok"), quality_timeout_s=1.0).rewrite(PROMPT, "t", "quality")
    assert r.path == "quality" and not r.fallback_used and r.rewrite == "list 3 primes only"


def test_quality_over_budget_falls_back_to_fast_path() -> None:
    q = Quality("hang")
    s = svc(quality=q, quality_timeout_s=0.05)
    t0 = time.perf_counter()
    r = s.rewrite(PROMPT, "t", "quality")
    elapsed = time.perf_counter() - t0
    q.release.set()
    assert r.fallback_used and r.fallback_reason == "timeout" and r.path == "fast"
    assert r.rewrite == "list three primes only"  # the fast-path result
    assert elapsed < 1.0
    assert len(s.cache) == 0  # fallbacks are not cached


@pytest.mark.parametrize(
    ("quality", "reason"), [(Quality("crash"), "error"), (None, "unavailable")]
)
def test_quality_crash_or_missing_falls_back(quality: Quality | None, reason: str) -> None:
    r = svc(quality=quality).rewrite(PROMPT, "t", "quality")
    assert r.fallback_used and r.fallback_reason == reason and r.path == "fast"


def test_both_paths_fail_returns_original() -> None:
    def broken(prompt: str, target: str) -> RewriteResult:
        raise RuntimeError("tagger crashed")

    s = RewriteService(pricing=PRICING, fast=broken, quality=Quality("crash"), counter=words)
    r = s.rewrite(PROMPT, "t", "quality")
    assert r.rewrite == PROMPT and r.path == "original" and r.fallback_used
    assert r.fallback_reason == "error"  # the quality path's reason is kept
    assert r.tokens_after == r.tokens_before
    fast_only = RewriteService(pricing=PRICING, fast=broken, counter=words).rewrite(PROMPT, "t")
    assert fast_only.fallback_reason == "fast_error" and fast_only.path == "original"


def test_empty_quality_rewrite_falls_back() -> None:
    def empty(prompt: str, target: str) -> RewriteResult:
        return RewriteResult("   ", tokens_in=5, tokens_out=0)

    r = svc(quality=empty, quality_timeout_s=1.0).rewrite(PROMPT, "t", "quality")
    assert r.fallback_reason == "error" and r.path == "fast" and r.rewrite.strip()


def test_busy_pool_falls_back_without_waiting() -> None:
    q = Quality("hang")
    s = svc(quality=q, quality_timeout_s=0.05, max_workers=1)
    s.rewrite(PROMPT, "t", "quality")  # times out; its worker is still busy
    t0 = time.perf_counter()
    r = s.rewrite(PROMPT + " again", "t", "quality")
    q.release.set()
    assert r.fallback_reason == "timeout" and time.perf_counter() - t0 < 0.05


def test_cache_hit_counts_this_prompts_tokens() -> None:
    s = svc(gate_threshold=-1.0)  # every rewrite rejected: the response is the prompt itself
    s.rewrite(PROMPT, "t")
    variant = PROMPT + "   \n"
    r = s.rewrite(variant, "t")
    assert r.cached and r.rewrite == variant and r.tokens_before == r.tokens_after


def test_repeated_faults_always_fall_back() -> None:
    s = svc(quality=Quality("crash"))
    results = [s.rewrite(f"{PROMPT} {i}", "t", "quality") for i in range(50)]
    assert all(r.fallback_used and r.rewrite for r in results)


# ---- gate --------------------------------------------------------------------------------------
def test_gate_rejection_returns_original() -> None:
    def drops_constraint(prompt: str, target: str) -> RewriteResult:
        return RewriteResult("list three primes")  # lost "only"

    s = RewriteService(pricing=PRICING, fast=drops_constraint, counter=words)
    r = s.rewrite(PROMPT, "t")
    assert r.gate_rejected and r.gate_score == 1.0 and r.rewrite == PROMPT


def test_unknown_target_and_mode() -> None:
    with pytest.raises(KeyError):
        svc().rewrite(PROMPT, "nope")
    with pytest.raises(ValueError):
        svc().rewrite(PROMPT, "t", "turbo")


# ---- accounting (11.3.6) -----------------------------------------------------------------------
def test_net_savings_formula() -> None:
    p = Pricing({"t": 2.0}, rewriter_input_per_mtok=1.0, rewriter_output_per_mtok=3.0)
    s = net_savings(p, "t", 100, 60, path="quality", rewriter_tokens_in=100, rewriter_tokens_out=60)
    assert s.tokens_saved == 40
    assert s.gross_savings == pytest.approx(2.0e-6 * 40)
    assert s.rewriter_cost == pytest.approx((100 * 1.0 + 60 * 3.0) / 1e6)
    assert s.net_savings == pytest.approx(s.gross_savings - s.rewriter_cost)
    assert net_savings(p, "t", 10, 10, path="fast").net_savings == 0.0
    with pytest.raises(KeyError):
        p.target_price("x")


def test_net_savings_logged_per_request(caplog: pytest.LogCaptureFixture) -> None:
    s = svc(quality=Quality("ok"), quality_timeout_s=1.0)
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        r_fast = s.rewrite(PROMPT, "t", "fast")
        r_q = s.rewrite(PROMPT, "t", "quality")
    recs = [json.loads(m.message) for m in caplog.records if m.name == LOGGER_NAME]
    assert len(recs) == 2
    # PRICING: 1 USD per target token, rewriter 1 USD per input token, 2 USD per output token
    assert recs[0]["tokens_saved"] == 11 - 4 and recs[0]["net_savings_usd"] == 7.0
    assert recs[1]["net_savings_usd"] == pytest.approx(11 - 4 - (10 * 1 + 4 * 2))
    assert recs[1]["rewriter_cost_usd"] == pytest.approx(18.0)
    assert r_fast.net_savings_usd == 7.0 and r_q.net_savings_usd == pytest.approx(-11.0)


# ---- structured logging (11.5.6) ---------------------------------------------------------------
def test_log_hashes_by_default_raw_only_with_consent(caplog: pytest.LogCaptureFixture) -> None:
    s = svc()
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        s.rewrite(PROMPT, "t")
        s.rewrite(PROMPT + " again", "t", consent_raw_text=True)
    a, b = (json.loads(m.message) for m in caplog.records if m.name == LOGGER_NAME)
    assert PROMPT not in json.dumps(a) and "prompt" not in a
    assert set(LOG_FIELDS) <= set(a) and len(a["prompt_sha256"]) == 64
    assert b["prompt"] == PROMPT + " again" and b["consent_raw_text"] is True


def test_build_record_rejects_unknown_fields() -> None:
    with pytest.raises(KeyError):
        build_record(prompt="a", rewrite="b", user_email="x@example.com")
