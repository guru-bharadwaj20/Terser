"""Label definition for the ``label`` stage (plan row **2.3.4**).

This module is the single source of truth for turning a task-metric comparison and the two
position-swapped judge verdicts into a ``preserved`` / ``degraded`` label. The rule
(``docs/categories.md`` and ``docs/decisions/ADR-0002-problem-framing.md``) is:

    ``preserved`` iff the category's task metric is **not worse** for the rewrite's answer
    **and** the pairwise judge says *tie or rewrite better* in **both** A/B orders.

Everything else is ``degraded``, with one exception: when a judge verdict could not be parsed and
no other evidence already implies degradation, the pair is left **unlabelled** (``label=None``,
reason ``unparseable_verdict``) by default so it can be re-judged instead of silently entering
the training set as a positive. ``on_missing="degraded"`` makes such pairs degraded instead.

Inputs:

* ``task_not_worse`` - precomputed ``bool`` from :func:`terser.judge.metrics.not_worse`, or
  ``None`` when the category has no automatic task metric (``explain``, ``creative``), in which
  case the judge alone decides. :func:`task_not_worse_from_results` wraps the metrics module
  with a lazy import.
* ``verdict_ab`` / ``verdict_ba`` - the judge's verdict in the ``original_first`` and
  ``rewrite_first`` orders, already translated to *sides* (``"original"``, ``"rewrite"``,
  ``"tie"``; ``None`` = unparseable) by :func:`terser.judge.pairwise.to_side`.

Labels use the positive-class convention of :mod:`terser.eval.labels` (``degraded`` = positive).
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Literal, TypedDict, get_args

from terser.eval.labels import LABEL_NAMES, to_binary

if TYPE_CHECKING:
    from terser.judge.pairwise import PairwiseResult

__all__ = [
    "LABEL_RECORD_FIELDS",
    "LABEL_SCHEMA_VERSION",
    "REASON_JUDGE_PREFERS_ORIGINAL",
    "REASON_PRESERVED",
    "REASON_TASK_METRIC_WORSE",
    "REASON_UNPARSEABLE_VERDICT",
    "Label",
    "LabelDecision",
    "LabelRecord",
    "OnMissing",
    "Side",
    "label_pair",
    "label_pair_from_result",
    "make_label_record",
    "task_not_worse_from_results",
    "validate_label_record",
]

Label = Literal["preserved", "degraded"]
Side = Literal["original", "rewrite", "tie"]
OnMissing = Literal["abstain", "degraded"]

REASON_PRESERVED: Final = "preserved"
REASON_TASK_METRIC_WORSE: Final = "task_metric_worse"
REASON_JUDGE_PREFERS_ORIGINAL: Final = "judge_prefers_original"
REASON_UNPARSEABLE_VERDICT: Final = "unparseable_verdict"

LABEL_SCHEMA_VERSION: Final = "label_v1"

_SIDES: Final[frozenset[str]] = frozenset(get_args(Side))


@dataclass(frozen=True)
class LabelDecision:
    """Outcome of :func:`label_pair`.

    Attributes:
        label: ``"preserved"``, ``"degraded"``, or ``None`` (unlabelled: unparseable verdict and
            ``on_missing="abstain"``).
        reason: The primary reason code, in priority order ``task_metric_worse`` >
            ``judge_prefers_original`` > ``unparseable_verdict`` > ``preserved``.
        reasons: Every reason code that applies (``("preserved",)`` for a preserved pair).
        position_consistent: ``True`` if both orders favour the same side, ``False`` if they
            differ, ``None`` if either verdict is missing (position bias is undefined then).
    """

    label: Label | None
    reason: str
    reasons: tuple[str, ...]
    position_consistent: bool | None

    @property
    def label_int(self) -> int | None:
        """The label as ``PRESERVED = 0`` / ``DEGRADED = 1`` (``None`` if unlabelled)."""
        return None if self.label is None else to_binary(self.label)


def _check_side(name: str, value: object) -> None:
    if value is not None and value not in _SIDES:
        raise ValueError(f"{name} must be one of {sorted(_SIDES)} or None, got {value!r}")


def label_pair(
    task_not_worse: bool | None,
    verdict_ab: Side | None,
    verdict_ba: Side | None,
    *,
    on_missing: OnMissing = "abstain",
) -> LabelDecision:
    """Apply the row 2.3.4 label rule to one (original, rewrite) pair.

    Truth table (``T/F/-`` = task metric not worse / worse / no metric; ``o/r/t/x`` = judge side
    original / rewrite / tie / unparseable; "ok" = ``r`` or ``t``):

    ========= ============== ============== ======================================
    task      verdict_ab     verdict_ba     label (reason)
    ========= ============== ============== ======================================
    F         any            any            degraded (task_metric_worse)
    T or -    o              any            degraded (judge_prefers_original)
    T or -    any            o              degraded (judge_prefers_original)
    T or -    x              ok or x        None (unparseable_verdict) [#]_
    T or -    ok or x        x              None (unparseable_verdict) [#]_
    T or -    ok             ok             preserved
    ========= ============== ============== ======================================

    .. [#] ``degraded`` when ``on_missing="degraded"``.

    Args:
        task_not_worse: Whether the rewrite's answer is not worse on the category's task metric;
            ``None`` when the category has no task metric.
        verdict_ab: Judge side in the ``original_first`` order.
        verdict_ba: Judge side in the ``rewrite_first`` order.
        on_missing: What to do when a verdict is missing and nothing else implies degradation.

    Returns:
        A :class:`LabelDecision`.

    Raises:
        TypeError: If ``task_not_worse`` is not a bool or None.
        ValueError: If a verdict is not a valid side, or ``on_missing`` is unknown.
    """
    if task_not_worse is not None and not isinstance(task_not_worse, bool):
        raise TypeError(f"task_not_worse must be bool or None, got {type(task_not_worse).__name__}")
    _check_side("verdict_ab", verdict_ab)
    _check_side("verdict_ba", verdict_ba)
    if on_missing not in ("abstain", "degraded"):
        raise ValueError(f"on_missing must be 'abstain' or 'degraded', got {on_missing!r}")

    sides = (verdict_ab, verdict_ba)
    reasons: list[str] = []
    if task_not_worse is False:
        reasons.append(REASON_TASK_METRIC_WORSE)
    if "original" in sides:
        reasons.append(REASON_JUDGE_PREFERS_ORIGINAL)
    missing = None in sides
    if missing:
        reasons.append(REASON_UNPARSEABLE_VERDICT)

    consistent = None if missing else verdict_ab == verdict_ba

    label: Label | None
    if task_not_worse is False or "original" in sides:
        label = "degraded"
    elif missing:
        label = "degraded" if on_missing == "degraded" else None
    else:
        label = "preserved"
        reasons.append(REASON_PRESERVED)
    return LabelDecision(
        label=label, reason=reasons[0], reasons=tuple(reasons), position_consistent=consistent
    )


def label_pair_from_result(
    task_not_worse: bool | None,
    result: PairwiseResult,
    *,
    on_missing: OnMissing = "abstain",
) -> LabelDecision:
    """:func:`label_pair` on a :class:`terser.judge.pairwise.PairwiseResult`."""
    return label_pair(
        task_not_worse,
        result.original_first.side,
        result.rewrite_first.side,
        on_missing=on_missing,
    )


def task_not_worse_from_results(original_result: Any, rewrite_result: Any) -> bool:
    """Compare two task-metric results with :func:`terser.judge.metrics.not_worse` (lazy import).

    Kept lazy so this module does not depend on the metrics module (row 2.3.1) at import time.
    """
    metrics = importlib.import_module("terser.judge.metrics")
    return bool(metrics.not_worse(original_result, rewrite_result))


class LabelRecord(TypedDict):
    """One line of the ``label`` stage output (``labels.jsonl``), schema ``label_v1``."""

    schema_version: str
    id: str
    prompt_id: str
    category: str
    target_model: str
    judge_model: str
    judge_prompt_version: str
    task_not_worse: bool | None
    verdicts: dict[str, Any]
    label: Label | None
    label_int: int | None
    reason: str
    reasons: list[str]
    position_consistent: bool | None
    stand_in: bool


LABEL_RECORD_FIELDS: Final[tuple[str, ...]] = tuple(LabelRecord.__annotations__)
"""Required keys of a ``label_v1`` record, in canonical order."""


def make_label_record(
    *,
    id: str,
    prompt_id: str,
    category: str,
    target_model: str,
    judge_model: str,
    judge_prompt_version: str,
    task_not_worse: bool | None,
    verdict_ab: Side | None,
    verdict_ba: Side | None,
    stand_in: bool,
    on_missing: OnMissing = "abstain",
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a validated ``label_v1`` record.

    ``verdicts`` holds the two sides plus ``position_consistent``; ``extra`` keys (e.g. full
    judge calls, candidate text) are added after the required fields and may not overwrite them.
    """
    decision = label_pair(task_not_worse, verdict_ab, verdict_ba, on_missing=on_missing)
    record: dict[str, Any] = {
        "schema_version": LABEL_SCHEMA_VERSION,
        "id": id,
        "prompt_id": prompt_id,
        "category": category,
        "target_model": target_model,
        "judge_model": judge_model,
        "judge_prompt_version": judge_prompt_version,
        "task_not_worse": task_not_worse,
        "verdicts": {
            "original_first": verdict_ab,
            "rewrite_first": verdict_ba,
            "position_consistent": decision.position_consistent,
        },
        "label": decision.label,
        "label_int": decision.label_int,
        "reason": decision.reason,
        "reasons": list(decision.reasons),
        "position_consistent": decision.position_consistent,
        "stand_in": stand_in,
    }
    if extra:
        clash = sorted(set(extra) & set(record))
        if clash:
            raise ValueError(f"extra keys overwrite required fields: {clash}")
        record.update(extra)
    validate_label_record(record)
    return record


def validate_label_record(record: Mapping[str, Any]) -> None:
    """Check a ``label_v1`` record: fields present, types sane, label consistent with the rule.

    Raises:
        ValueError: Describing the first problem found.
    """
    missing = [k for k in LABEL_RECORD_FIELDS if k not in record]
    if missing:
        raise ValueError(f"label record missing field(s) {missing}")
    if record["schema_version"] != LABEL_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version {record['schema_version']!r}")
    for key in ("id", "prompt_id", "category"):
        if not isinstance(record[key], str) or not record[key]:
            raise ValueError(f"{key} must be a non-empty string")
    if not isinstance(record["stand_in"], bool):
        raise ValueError("stand_in must be a bool")
    verdicts = record["verdicts"]
    if not isinstance(verdicts, Mapping):
        raise ValueError("verdicts must be an object")
    ab, ba = verdicts.get("original_first"), verdicts.get("rewrite_first")
    if isinstance(ab, Mapping):  # full JudgeCall dicts are allowed
        ab = ab.get("side")
    if isinstance(ba, Mapping):
        ba = ba.get("side")
    try:
        expected = label_pair(record["task_not_worse"], ab, ba, on_missing="abstain")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid label inputs: {exc}") from exc
    label = record["label"]
    allowed = {expected.label}
    if expected.label is None:
        allowed.add("degraded")  # produced with on_missing="degraded"
    if label not in allowed:
        raise ValueError(
            f"label {label!r} contradicts the 2.3.4 rule (expected {expected.label!r})"
        )
    if record["label_int"] != (None if label is None else to_binary(label)):
        raise ValueError("label_int does not match label")
    if label is not None and label not in LABEL_NAMES.values():
        raise ValueError(f"unknown label {label!r}")
    if record["position_consistent"] != expected.position_consistent:
        raise ValueError("position_consistent does not match the verdicts")
