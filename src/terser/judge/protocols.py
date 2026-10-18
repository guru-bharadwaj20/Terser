"""Judge protocols compared in ADR-0005 and the side each one assigns to a pair.

Plan rows **2.3.3 / 2.3.4**; checkpoint C2.6. Every protocol reduces a judged (original output,
rewrite output) pair to sides in the ``original_first`` and ``rewrite_first`` orders, which
:func:`terser.judge.labels.label_pair` turns into a label with the unchanged row 2.3.4 rule
(``preserved`` iff tie-or-rewrite-better in both orders):

``pairwise``
    Greedy pairwise verdict in each order (:class:`terser.judge.pairwise.PairwiseJudge`). The
    two orders are judged independently, so they can disagree (position bias).
``pairwise_avg``
    The same two pairwise calls with ``top_logprobs``; the A/B/tie probabilities of each order
    are mapped to sides and **averaged over the two orders** (:func:`order_averaged_side`). The
    argmax side is used for both orders, so the label cannot depend on presentation order. A
    call without probabilities (e.g. the provider returned no ``top_logprobs``, or the verdict
    token was not in the top list) contributes a one-hot vector of its parsed side. Retries
    request logprobs too.
``pointwise``
    Each output scored alone against the category rubric
    (:class:`terser.judge.pointwise.PointwiseJudge`); the side comes from comparing the scores
    and is the same for both orders, again order-free by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, get_args

from terser.judge.pairwise import DEFAULT_PROMPT_VERSION, PairwiseJudge, PairwiseResult

if TYPE_CHECKING:
    from terser.data.categories import Category
    from terser.judge.pairwise import JudgeCall, Side
    from terser.llm import LLMClient

__all__ = [
    "DEFAULT_TOP_LOGPROBS",
    "LOW_MARGIN",
    "PROTOCOLS",
    "AveragedResult",
    "averaged_judge",
    "judge_pair_averaged",
    "JudgeProtocol",
    "call_side_probs",
    "order_averaged_side",
]

JudgeProtocol = Literal["pairwise", "pairwise_avg", "pointwise"]
PROTOCOLS: tuple[str, ...] = get_args(JudgeProtocol)

_SIDES: tuple[Side, ...] = ("original", "rewrite", "tie")


def call_side_probs(call: JudgeCall) -> dict[Side, float] | None:
    """Side probabilities of one judge call; one-hot of its side without logprobs; None if unparsed.

    In ``original_first`` A is the original's output; in ``rewrite_first`` A is the rewrite's.
    """
    if call.side is None:
        return None
    probs = call.verdict_probs
    if probs is None:
        return {side: float(side == call.side) for side in _SIDES}
    a, b = probs.get("A", 0.0), probs.get("B", 0.0)
    if call.order == "original_first":
        return {"original": a, "rewrite": b, "tie": probs.get("tie", 0.0)}
    return {"original": b, "rewrite": a, "tie": probs.get("tie", 0.0)}


def order_averaged_side(
    original_first: JudgeCall, rewrite_first: JudgeCall
) -> tuple[Side | None, dict[Side, float] | None]:
    """Average the two orders' side probabilities; return ``(argmax side, averaged probs)``.

    Exact probability ties are broken towards ``tie``, then ``original`` (the conservative
    choice: neither breaks towards ``preserved`` by accident of ordering). ``(None, None)`` if
    either order is unparsed.
    """
    p1, p2 = call_side_probs(original_first), call_side_probs(rewrite_first)
    if p1 is None or p2 is None:
        return None, None
    avg: dict[Side, float] = {side: (p1[side] + p2[side]) / 2 for side in _SIDES}
    preference: tuple[Side, ...] = ("tie", "original", "rewrite")
    best = max(preference, key=lambda side: (avg[side], -preference.index(side)))
    return best, avg


DEFAULT_TOP_LOGPROBS = 20
LOW_MARGIN = 0.2
"""Averaged-probability margin below which a ``pairwise_avg`` verdict is flagged for review."""


@dataclass(frozen=True)
class AveragedResult:
    """A ``pairwise_avg`` judgement: both raw orders plus the order-averaged side."""

    pairwise: PairwiseResult
    side: Side | None
    probs: dict[Side, float] | None

    @property
    def margin(self) -> float | None:
        """Averaged probability of the chosen side minus the runner-up (None if unparsed)."""
        if self.probs is None:
            return None
        top, second = sorted(self.probs.values(), reverse=True)[:2]
        return top - second

    @property
    def low_confidence(self) -> bool:
        """True when the margin is below :data:`LOW_MARGIN` (or the pair is unparsed)."""
        return self.margin is None or self.margin < LOW_MARGIN

    @property
    def sides(self) -> tuple[Side | None, Side | None]:
        """The side passed to the row 2.3.4 rule for both orders (order-free by construction)."""
        return self.side, self.side

    def aggregate(self) -> dict[str, Any]:
        """The ``aggregate`` field of a label record."""
        return {
            "protocol": "pairwise_avg",
            "side": self.side,
            "probs": self.probs,
            "margin": self.margin,
            "low_confidence": self.low_confidence,
        }


def averaged_judge(
    client: LLMClient,
    *,
    provider: str,
    model: str,
    max_tokens: int = 256,
    retries: int = 2,
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    top_logprobs: int = DEFAULT_TOP_LOGPROBS,
) -> PairwiseJudge:
    """A :class:`PairwiseJudge` configured for ``pairwise_avg`` (temperature 0, logprobs on)."""
    return PairwiseJudge(
        client,
        provider=provider,
        model=model,
        max_tokens=max_tokens,
        retries=retries,
        prompt_version=prompt_version,
        top_logprobs=top_logprobs,
    )


def judge_pair_averaged(
    judge: PairwiseJudge,
    prompt: str,
    original_output: str,
    rewrite_output: str,
    category: Category | str | None = None,
) -> AveragedResult:
    """Judge both orders and average them (ADR-0005 protocol ``pairwise_avg``).

    Raises:
        ValueError: If ``judge`` does not request logprobs (``top_logprobs`` unset).
    """
    if not judge.top_logprobs:
        raise ValueError("pairwise_avg needs a PairwiseJudge with top_logprobs set")
    result = judge.judge_pair(prompt, original_output, rewrite_output, category)
    side, probs = order_averaged_side(result.original_first, result.rewrite_first)
    return AveragedResult(result, side, probs)
