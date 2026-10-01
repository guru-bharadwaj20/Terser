"""Decoding safeguards and the inference-time quality gate (plan rows **8.3.7**, **8.3.8**).

**Safeguard (8.3.7).** A rewrite is only worth sending if it is strictly cheaper on the target
tokenizer ``T``. :func:`safeguard` returns the rewrite when it is non-empty and has fewer tokens
than the original on ``T``; otherwise it falls back to the original. The invariant, checked by a
property test, is ``tokens_T(output) <= tokens_T(original)`` and ``output in {original,
rewrite}``.

**Gate (8.3.8).** :class:`GatedRewriter` holds a ladder of rewriters ordered from heaviest
(most aggressive, e.g. the SFT model or all operators) to lightest (e.g. filler removal only).
For each rung it applies the safeguard and asks the champion quality gate for
``P(degraded | original, rewrite)``; the first rung below the gate threshold wins. If every rung is
rejected the original is returned (0% savings, 100% preservation by construction). The gate is
any callable, so the champion of C5.1 plugs in unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

__all__ = ["GateDecision", "GatedRewriter", "safeguard"]

TokenCounter = Callable[[str], int]
Rewriter = Callable[[str], str]
Gate = Callable[[str, str], float]


def safeguard(original: str, rewrite: str, count: TokenCounter) -> tuple[str, bool]:
    """``(text, fell_back)``: rewrite if non-empty and strictly cheaper on ``T``, else original."""
    if not rewrite.strip() or count(rewrite) >= count(original):
        return original, True
    return rewrite, False


@dataclass(frozen=True)
class GateDecision:
    """Outcome of one gated rewrite.

    ``rung`` is the index of the accepted rewriter, or ``None`` when the original was kept.
    ``trace`` holds ``(rung, reason, p_degraded)`` for every rung tried; reason is one of
    ``accepted``, ``not_shorter`` or ``gate_rejected``.
    """

    text: str
    rung: int | None
    tokens_original: int
    tokens_output: int
    trace: tuple[tuple[int, str, float | None], ...] = field(default=())

    @property
    def savings(self) -> float:
        """Fractional token savings on the target (0 when the original is kept)."""
        if self.tokens_original == 0:
            return 0.0
        return 1.0 - self.tokens_output / self.tokens_original


@dataclass
class GatedRewriter:
    """Try rewriters heaviest-first; keep the first that is shorter and passes the gate.

    Attributes:
        rewriters: Ladder ordered from most to least aggressive.
        gate: ``(original, rewrite) -> P(degraded)``.
        count: Token counter for the target ``T``.
        threshold: Reject when ``P(degraded) >= threshold`` (set on validation for
            sensitivity 0.95, row 3.2.6).
    """

    rewriters: Sequence[Rewriter]
    gate: Gate
    count: TokenCounter
    threshold: float = 0.5

    def __call__(self, original: str) -> GateDecision:
        n_orig = self.count(original)
        trace: list[tuple[int, str, float | None]] = []
        for i, rewrite_fn in enumerate(self.rewriters):
            text, fell_back = safeguard(original, rewrite_fn(original), self.count)
            if fell_back:
                trace.append((i, "not_shorter", None))
                continue
            p = float(self.gate(original, text))
            if p >= self.threshold:
                trace.append((i, "gate_rejected", p))
                continue
            trace.append((i, "accepted", p))
            return GateDecision(text, i, n_orig, self.count(text), tuple(trace))
        return GateDecision(original, None, n_orig, n_orig, tuple(trace))
