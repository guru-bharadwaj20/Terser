"""Tests for the positive-class convention in terser.eval.labels (plan row 1.1.3)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from sklearn.metrics import confusion_matrix, recall_score

from terser.eval.labels import (
    DEGRADED,
    LABEL_NAMES,
    POSITIVE_LABEL,
    PRESERVED,
    confusion_counts,
    sensitivity,
    specificity,
    to_binary,
)


def test_constants() -> None:
    assert PRESERVED == 0
    assert DEGRADED == 1
    assert POSITIVE_LABEL == DEGRADED
    assert LABEL_NAMES == {0: "preserved", 1: "degraded"}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("preserved", 0),
        ("degraded", 1),
        (" Degraded\n", 1),
        ("PRESERVED", 0),
        (True, 1),
        (False, 0),
        (0, 0),
        (1, 1),
        (np.int64(1), 1),
        (np.bool_(True), 1),
    ],
)
def test_to_binary_valid(raw: str | bool | int, expected: int) -> None:
    assert to_binary(raw) == expected


@pytest.mark.parametrize("raw", ["ok", "", "positive", 2, -1])
def test_to_binary_invalid_value(raw: str | int) -> None:
    with pytest.raises(ValueError):
        to_binary(raw)


@pytest.mark.parametrize("raw", [0.5, 1.0, None])
def test_to_binary_invalid_type(raw: object) -> None:
    with pytest.raises(TypeError):
        to_binary(raw)  # type: ignore[arg-type]


def test_hand_computed_example() -> None:
    y_true = ["degraded", "degraded", "degraded", "degraded", "preserved", "preserved", "preserved"]
    y_pred = ["degraded", "degraded", "degraded", "preserved", "preserved", "degraded", "preserved"]
    assert confusion_counts(y_true, y_pred) == (2, 1, 1, 3)
    assert sensitivity(y_true, y_pred) == pytest.approx(3 / 4)
    assert specificity(y_true, y_pred) == pytest.approx(2 / 3)


def test_mixed_input_forms_agree() -> None:
    strings = ["degraded", "preserved", "degraded"]
    bools = [True, False, True]
    ints = np.array([1, 0, 1])
    assert confusion_counts(strings, bools) == confusion_counts(ints, ints)


@pytest.mark.parametrize("seed", range(20))
def test_matches_sklearn_recall_with_pos_label(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(5, 200))
    y_true = rng.integers(0, 2, size=n)
    y_pred = rng.integers(0, 2, size=n)
    # Guarantee both classes are present so neither ratio is undefined.
    y_true[0], y_true[1] = DEGRADED, PRESERVED

    sk_sens = recall_score(y_true, y_pred, pos_label=POSITIVE_LABEL)
    sk_spec = recall_score(y_true, y_pred, pos_label=PRESERVED)
    assert abs(sensitivity(y_true, y_pred) - sk_sens) < 1e-12
    assert abs(specificity(y_true, y_pred) - sk_spec) < 1e-12

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[PRESERVED, DEGRADED]).ravel()
    assert confusion_counts(y_true, y_pred) == (tn, fp, fn, tp)


def test_undefined_ratios_are_nan() -> None:
    assert math.isnan(sensitivity(["preserved", "preserved"], ["preserved", "degraded"]))
    assert math.isnan(specificity(["degraded"], ["degraded"]))


def test_empty_input() -> None:
    assert confusion_counts([], []) == (0, 0, 0, 0)
    assert math.isnan(sensitivity([], []))


def test_length_mismatch() -> None:
    with pytest.raises(ValueError, match="differ in length"):
        sensitivity([1, 0], [1])


def test_rejects_2d_array() -> None:
    with pytest.raises(ValueError, match="1-D"):
        specificity(np.zeros((2, 2), dtype=int), np.zeros((2, 2), dtype=int))


def test_always_preserved_baseline() -> None:
    """The 'always preserved' baseline (3.6.5) catches nothing and accepts everything."""
    y_true = [1, 0, 1, 0, 0]
    y_pred = [PRESERVED] * 5
    assert sensitivity(y_true, y_pred) == 0.0
    assert specificity(y_true, y_pred) == 1.0
