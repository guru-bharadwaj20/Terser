"""Tests for terser.eval.significance (row 3.2.8)."""

from __future__ import annotations

import numpy as np
import pytest

from terser.eval.curves import roc_auc
from terser.eval.significance import bootstrap_ci, mcnemar, paired_bootstrap_diff


def _acc(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(y == p))


def test_mcnemar_hand_example() -> None:
    y = np.zeros(20, dtype=int)
    a = np.zeros(20, dtype=int)  # always correct
    b = np.zeros(20, dtype=int)
    b[:8] = 1  # wrong on 8
    r = mcnemar(y, a, b)
    assert (r.b, r.c) == (8, 0)
    assert r.statistic == pytest.approx(49 / 8)
    assert r.p_value == pytest.approx(2 * 0.5**8)


def test_mcnemar_symmetry_and_ties() -> None:
    y = np.array([0, 1, 0, 1, 1, 0])
    a = np.array([0, 1, 1, 1, 0, 0])
    b = np.array([1, 1, 0, 0, 1, 0])
    assert mcnemar(y, a, b).p_value == pytest.approx(mcnemar(y, b, a).p_value)
    assert mcnemar(y, a, a).p_value == 1.0
    with pytest.raises(ValueError):
        mcnemar(y, a, b[:-1])


def test_bootstrap_covers_truth_and_groups() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.where(rng.random(500) < 0.8, y, 1 - y)
    ci = bootstrap_ci(_acc, y, p, n_resamples=500)
    assert ci.low < ci.estimate < ci.high
    assert ci.low < 0.8 < ci.high
    groups = [i // 5 for i in range(500)]
    gci = bootstrap_ci(_acc, y, p, groups=groups, n_resamples=500)
    assert gci.low < gci.estimate < gci.high
    s = y + rng.normal(0, 1, 500)
    auc = bootstrap_ci(roc_auc, y, s, n_resamples=200)
    assert 0.5 < auc.low < auc.estimate < auc.high < 1.0


def test_paired_diff_detects_better_model() -> None:
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    good = np.where(rng.random(400) < 0.9, y, 1 - y)
    bad = np.where(rng.random(400) < 0.6, y, 1 - y)
    d = paired_bootstrap_diff(_acc, y, good, bad, n_resamples=500)
    assert d.low > 0
    same = paired_bootstrap_diff(_acc, y, good, good, n_resamples=50)
    assert same.low == same.high == same.estimate == 0.0
