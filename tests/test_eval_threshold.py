"""Tests for terser.eval.threshold (row 3.2.3)."""

from __future__ import annotations

import numpy as np
import pytest

from terser.eval.labels import sensitivity, specificity
from terser.eval.threshold import MIN_SENSITIVITY, apply_threshold, choose_threshold


def test_hand_example() -> None:
    # 20 degraded: 19 score high, one scores 0.35; preserved scores spread 0.0-0.6.
    y = np.array([1] * 20 + [0] * 10)
    s = np.array([0.9] * 19 + [0.35] + [0.0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65])
    choice = choose_threshold(y, s)
    # Sensitivity 19/20 = 0.95 is reachable at 0.9 with every preserved accepted.
    assert choice.threshold == 0.9
    assert choice.sensitivity == 0.95
    assert choice.specificity == 1.0
    full = choose_threshold(y, s, min_sensitivity=1.0)
    assert full.threshold == 0.35 and full.sensitivity == 1.0
    assert full.specificity == pytest.approx(0.4)


@pytest.mark.parametrize("seed", range(15))
def test_brute_force_optimum(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(30, 300))
    y = (rng.random(n) < 0.3).astype(int)
    y[:2] = [0, 1]
    s = np.round(y * 0.7 + rng.normal(0, 0.5, n), 2)
    choice = choose_threshold(y, s)
    pred = apply_threshold(s, choice.threshold)
    assert sensitivity(y, pred) == pytest.approx(choice.sensitivity)
    assert specificity(y, pred) == pytest.approx(choice.specificity)
    assert choice.sensitivity >= MIN_SENSITIVITY
    best = max(
        specificity(y, apply_threshold(s, t))
        for t in np.unique(s)
        if sensitivity(y, apply_threshold(s, t)) >= MIN_SENSITIVITY
    )
    assert choice.specificity == pytest.approx(best)


def test_rejects_bad_floor() -> None:
    with pytest.raises(ValueError):
        choose_threshold([0, 1], [0.1, 0.9], min_sensitivity=0.0)
    with pytest.raises(ValueError):
        choose_threshold([0, 1], [0.1, 0.9], min_sensitivity=1.1)
