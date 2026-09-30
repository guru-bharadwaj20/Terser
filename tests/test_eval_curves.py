"""Tests for terser.eval.curves against scikit-learn (row 3.2.2)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn import metrics as skm

from terser.eval.curves import average_precision, plot_curves, pr_curve, roc_auc, roc_curve


def _data(seed: int, ties: bool) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(20, 400))
    y = (rng.random(n) < 0.25).astype(int)
    y[:2] = [0, 1]
    s = y * 0.8 + rng.normal(0, 0.6, n)
    if ties:
        s = np.round(s, 1)
    return y, s


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("ties", [False, True])
def test_matches_sklearn(seed: int, ties: bool) -> None:
    y, s = _data(seed, ties)
    assert roc_auc(y, s) == pytest.approx(skm.roc_auc_score(y, s), abs=1e-9)
    assert average_precision(y, s) == pytest.approx(skm.average_precision_score(y, s), abs=1e-9)
    fpr, tpr, _ = skm.roc_curve(y, s, drop_intermediate=False)
    c = roc_curve(y, s)
    np.testing.assert_allclose(c.x, fpr)
    np.testing.assert_allclose(c.y, tpr)


def test_pr_curve_shape_and_perfect_scores() -> None:
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    assert roc_auc(y, s) == 1.0
    assert average_precision(y, s) == 1.0
    c = pr_curve(y, s)
    assert c.x[0] == 0.0 and c.y[0] == 1.0 and c.x[-1] == 1.0
    assert np.all(np.diff(c.x) >= 0)


def test_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        roc_auc([0, 0], [0.1, 0.2])
    with pytest.raises(ValueError):
        roc_auc([0, 1], [0.1])
    with pytest.raises(ValueError):
        roc_auc([0, 1], [0.1, float("nan")])


def test_plot_curves_writes_png(tmp_path: Path) -> None:
    y, s = _data(0, False)
    out = tmp_path / "curves.png"
    fig = plot_curves(y, {"model": s, "noise": np.zeros_like(s)}, title="t", path=out)
    assert out.stat().st_size > 0
    assert len(fig.axes) == 2
