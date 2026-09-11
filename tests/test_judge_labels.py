"""Tests for terser.judge.labels: the row 2.3.4 label rule and the label_v1 record schema."""

from __future__ import annotations

import itertools
import json
import random
from typing import Any

import pytest

from terser.eval.labels import DEGRADED, PRESERVED
from terser.judge.labels import (
    LABEL_RECORD_FIELDS,
    LABEL_SCHEMA_VERSION,
    REASON_JUDGE_PREFERS_ORIGINAL,
    REASON_PRESERVED,
    REASON_TASK_METRIC_WORSE,
    REASON_UNPARSEABLE_VERDICT,
    label_pair,
    label_pair_from_result,
    make_label_record,
    validate_label_record,
)
from terser.judge.pairwise import JudgeCall, PairwiseResult, label_from_sides

TASK = (True, False, None)
SIDES: tuple[Any, ...] = ("original", "rewrite", "tie", None)
ALL_INPUTS = list(itertools.product(TASK, SIDES, SIDES))


def expected_label(task: bool | None, ab: Any, ba: Any) -> str | None:
    """Independent restatement of the rule, written as the plan row reads."""
    ok = {"rewrite", "tie"}
    if task is False or "original" in (ab, ba):
        return "degraded"
    if ab is None or ba is None:
        return None
    assert ab in ok and ba in ok
    return "preserved"


# Full truth table: 3 task states x 4 x 4 sides = 48 rows, written out explicitly.
O, R, T, X = "original", "rewrite", "tie", None  # noqa: E741 - compact truth-table aliases
P, D, N = "preserved", "degraded", None
TRUTH_TABLE: list[tuple[bool | None, Any, Any, str | None, str]] = []
for task in TASK:
    for ab in SIDES:
        for ba in SIDES:
            if task is False:
                row = (D, REASON_TASK_METRIC_WORSE)
            elif O in (ab, ba):
                row = (D, REASON_JUDGE_PREFERS_ORIGINAL)
            elif X in (ab, ba):
                row = (N, REASON_UNPARSEABLE_VERDICT)
            else:
                row = (P, REASON_PRESERVED)
            TRUTH_TABLE.append((task, ab, ba, *row))


def test_truth_table_is_complete() -> None:
    assert len(TRUTH_TABLE) == 48
    assert {r[:3] for r in TRUTH_TABLE} == set(ALL_INPUTS)


@pytest.mark.parametrize(("task", "ab", "ba", "label", "reason"), TRUTH_TABLE)
def test_truth_table(task: bool | None, ab: Any, ba: Any, label: str | None, reason: str) -> None:
    decision = label_pair(task, ab, ba)
    assert decision.label == label
    assert decision.reason == reason
    assert decision.reasons[0] == reason


@pytest.mark.parametrize(
    ("task", "ab", "ba", "label"),
    [
        (True, R, R, P),
        (True, T, T, P),
        (None, R, T, P),
        (None, T, R, P),
        (True, O, R, D),
        (True, R, O, D),
        (None, O, O, D),
        (False, R, R, D),
        (False, T, T, D),
        (False, X, X, D),
        (True, X, R, N),
        (None, T, X, N),
        (True, X, O, D),
    ],
)
def test_hand_picked_cases(task: bool | None, ab: Any, ba: Any, label: str | None) -> None:
    assert label_pair(task, ab, ba).label == label


@pytest.mark.parametrize(("task", "ab", "ba"), ALL_INPUTS)
def test_properties(task: bool | None, ab: Any, ba: Any) -> None:
    d = label_pair(task, ab, ba)
    # Matches an independent restatement of the rule.
    assert d.label == expected_label(task, ab, ba)
    # Preserved requires every condition; a failed task metric always wins.
    if d.label == "preserved":
        assert task is not False and {ab, ba} <= {"rewrite", "tie"}
        assert d.reasons == (REASON_PRESERVED,)
    else:
        assert REASON_PRESERVED not in d.reasons
    if task is False:
        assert d.label == "degraded"
    # Order symmetry: swapping the two verdicts never changes the label or consistency.
    swapped = label_pair(task, ba, ab)
    assert swapped.label == d.label
    assert swapped.position_consistent == d.position_consistent
    # Position consistency is only defined when both verdicts parsed.
    if ab is None or ba is None:
        assert d.position_consistent is None
    else:
        assert d.position_consistent == (ab == ba)
    # on_missing="degraded" only turns abstentions into degraded.
    strict = label_pair(task, ab, ba, on_missing="degraded")
    assert strict.label == (d.label or "degraded")
    assert strict.reasons == d.reasons
    # label_int follows the degraded = positive convention.
    assert d.label_int == {None: None, "preserved": PRESERVED, "degraded": DEGRADED}[d.label]


@pytest.mark.parametrize(("ab", "ba"), list(itertools.product(SIDES, SIDES)))
def test_judge_part_matches_pairwise_label_from_sides(ab: Any, ba: Any) -> None:
    """With no task metric, the rule reduces to the pilot judge rule of pairwise.py."""
    assert label_pair(None, ab, ba).label == label_from_sides(ab, ba)


def test_monotone_random_sequences() -> None:
    """Making any input worse never turns degraded into preserved."""
    rng = random.Random(0)
    worse_side = {"rewrite": "tie", "tie": "original", "original": "original", None: None}
    for _ in range(500):
        task, ab, ba = rng.choice(ALL_INPUTS)
        before = label_pair(task, ab, ba).label
        after = label_pair(False if task is False else task, worse_side[ab], ba).label
        if before == "degraded":
            assert after == "degraded"


def test_reasons_accumulate() -> None:
    d = label_pair(False, "original", None)
    assert d.reasons == (
        REASON_TASK_METRIC_WORSE,
        REASON_JUDGE_PREFERS_ORIGINAL,
        REASON_UNPARSEABLE_VERDICT,
    )
    assert d.label == "degraded"


@pytest.mark.parametrize(
    ("args", "exc"),
    [
        ((1, "tie", "tie"), TypeError),
        (("yes", "tie", "tie"), TypeError),
        ((True, "A", "tie"), ValueError),
        ((True, "tie", "better"), ValueError),
    ],
)
def test_invalid_inputs(args: tuple[Any, Any, Any], exc: type[Exception]) -> None:
    with pytest.raises(exc):
        label_pair(*args)


def test_invalid_on_missing() -> None:
    with pytest.raises(ValueError):
        label_pair(True, "tie", "tie", on_missing="ignore")  # type: ignore[arg-type]


def _call(order: Any, side: Any) -> JudgeCall:
    verdict = None if side is None else "tie"
    return JudgeCall(order, verdict, side, "", "", 1, False, 0)


def test_label_pair_from_result() -> None:
    result = PairwiseResult(
        _call("original_first", "rewrite"), _call("rewrite_first", "original"), "degraded"
    )
    d = label_pair_from_result(True, result)
    assert d.label == "degraded"
    assert d.position_consistent is False


def _record(**overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "id": "p1:c3",
        "prompt_id": "p1",
        "category": "code",
        "target_model": "Qwen/Qwen2.5-7B-Instruct",
        "judge_model": "Qwen/Qwen2.5-32B-Instruct-AWQ",
        "judge_prompt_version": "pairwise_v1",
        "task_not_worse": True,
        "verdict_ab": "tie",
        "verdict_ba": "rewrite",
        "stand_in": False,
    }
    kwargs.update(overrides)
    return make_label_record(**kwargs)


def test_make_label_record_schema() -> None:
    rec = _record()
    assert list(rec)[: len(LABEL_RECORD_FIELDS)] == list(LABEL_RECORD_FIELDS)
    assert rec["schema_version"] == LABEL_SCHEMA_VERSION
    assert rec["label"] == "preserved" and rec["label_int"] == PRESERVED
    assert rec["position_consistent"] is False
    assert json.loads(json.dumps(rec)) == rec
    validate_label_record(rec)


def test_make_label_record_extra_and_on_missing() -> None:
    rec = _record(verdict_ba=None, on_missing="degraded", extra={"candidate": "x"})
    assert rec["label"] == "degraded" and rec["candidate"] == "x"
    with pytest.raises(ValueError, match="overwrite"):
        _record(extra={"label": "preserved"})


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda r: r.pop("category"), "missing"),
        (lambda r: r.update(schema_version="label_v0"), "schema_version"),
        (lambda r: r.update(label="degraded", label_int=1), "contradicts"),
        (lambda r: r.update(label_int=1), "label_int"),
        (lambda r: r.update(position_consistent=True), "position_consistent"),
        (lambda r: r.update(stand_in="no"), "stand_in"),
        (lambda r: r["verdicts"].update(original_first="A"), "invalid label inputs"),
    ],
)
def test_validate_label_record_rejects(mutate: Any, match: str) -> None:
    rec = _record()
    mutate(rec)
    with pytest.raises(ValueError, match=match):
        validate_label_record(rec)


def test_validate_accepts_full_judge_call_dicts() -> None:
    rec = _record()
    rec["verdicts"]["original_first"] = _call("original_first", "tie").to_dict()
    rec["verdicts"]["rewrite_first"] = _call("rewrite_first", "rewrite").to_dict()
    validate_label_record(rec)
