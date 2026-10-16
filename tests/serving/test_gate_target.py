"""The gate scores features on the tokenizer it was trained on (``gate_feature_target``)."""

from __future__ import annotations

from collections.abc import Mapping

from terser.serving.accounting import Pricing
from terser.serving.paths import RewriteResult
from terser.serving.service import RewriteService

PRICING = Pricing({"a": 1.0, "b": 1.0})


def counter(s: str, target: str) -> int:
    n = len(s.split())
    return n if target == "a" else 10 * n


class RecordingGate:
    def __init__(self) -> None:
        self.seen: list[float] = []

    def score(self, orig: Mapping[str, float], rew: Mapping[str, float]) -> float:
        self.seen.append(orig["n_tokens"])
        return 0.0


def fast(prompt: str, target: str) -> RewriteResult:
    return RewriteResult("list primes")


def test_gate_uses_pinned_target_but_accounting_uses_request_target() -> None:
    gate = RecordingGate()
    svc = RewriteService(
        pricing=PRICING, fast=fast, gate=gate, counter=counter, gate_feature_target="a"
    )
    r = svc.rewrite("please could you list primes", "b")
    assert gate.seen == [5.0]  # 5 words on tokenizer "a"
    assert r.tokens_before == 50  # accounting on the request target "b"


def test_default_gate_target_is_request_target() -> None:
    gate = RecordingGate()
    svc = RewriteService(pricing=PRICING, fast=fast, gate=gate, counter=counter)
    svc.rewrite("please could you list primes", "b")
    assert gate.seen == [50.0]
