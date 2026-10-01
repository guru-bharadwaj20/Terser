"""Rows 3.4.2 / 3.4.3 / 3.4.5: penalised LR, calibration metrics and odds ratios."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from sklearn.calibration import calibration_curve
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import StandardScaler

from terser.eval.calibration import (
    brier_score,
    expected_calibration_error,
    plot_reliability,
    reliability_curve,
)
from terser.models.classical.scratch_penalized_lr import (
    PenalizedLogisticRegression,
    odds_ratio_table,
    soft_threshold,
)


@pytest.fixture(scope="module")
def data() -> tuple[np.ndarray, np.ndarray]:
    X, y = make_classification(
        n_samples=300, n_features=8, n_informative=4, n_redundant=0, random_state=0
    )
    return StandardScaler().fit_transform(X), y


def test_soft_threshold() -> None:
    out = soft_threshold(np.array([-3.0, -0.5, 0.0, 0.5, 2.0]), 1.0)
    assert out.tolist() == [-2.0, 0.0, 0.0, 0.0, 1.0]


@pytest.mark.parametrize(
    ("penalty", "C", "l1_ratio"),
    [
        ("l2", 1.0, None),
        ("l2", 0.05, None),
        ("l1", 1.0, None),
        ("l1", 0.05, None),
        ("elasticnet", 0.1, 0.5),
        ("elasticnet", 1.0, 0.2),
    ],
)
def test_matches_sklearn(
    data: tuple[np.ndarray, np.ndarray], penalty: str, C: float, l1_ratio: float | None
) -> None:
    X, y = data
    ours = PenalizedLogisticRegression(
        penalty=penalty,  # type: ignore[arg-type]
        C=C,
        l1_ratio=l1_ratio if l1_ratio is not None else 0.5,
    ).fit(X, y)
    ratio = {"l2": 0.0, "l1": 1.0}.get(penalty, l1_ratio)
    ref = LogisticRegression(
        C=C,
        l1_ratio=ratio,
        solver="saga",
        tol=1e-12,
        max_iter=200_000,
    ).fit(X, y)
    np.testing.assert_allclose(ours.coef_, ref.coef_[0], atol=1e-3)
    assert ours.intercept_ == pytest.approx(ref.intercept_[0], abs=1e-3)


def test_l1_gives_exact_zeros(data: tuple[np.ndarray, np.ndarray]) -> None:
    X, y = data
    sparse = PenalizedLogisticRegression(penalty="l1", C=0.02).fit(X, y)
    dense = PenalizedLogisticRegression(penalty="l2", C=0.02).fit(X, y)
    assert np.sum(sparse.coef_ == 0.0) > 0
    assert np.sum(dense.coef_ == 0.0) == 0


def test_input_validation() -> None:
    with pytest.raises(ValueError):
        PenalizedLogisticRegression(C=0).fit([[0.0], [1.0]], [0, 1])
    with pytest.raises(ValueError):
        PenalizedLogisticRegression().fit([[0.0], [1.0]], [0, 2])
    with pytest.raises(ValueError):
        PenalizedLogisticRegression(penalty="elasticnet", l1_ratio=2).fit([[0.0], [1.0]], [0, 1])


def test_odds_ratio_table() -> None:
    t = odds_ratio_table([0.1, -2.0, 0.0], ["a", "b", "c"])
    assert t["feature"].tolist() == ["b", "a", "c"]
    assert t.loc[0, "odds_ratio"] == pytest.approx(math.exp(-2.0))


def test_brier_matches_sklearn() -> None:
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    p = rng.random(200)
    assert brier_score(y, p) == pytest.approx(brier_score_loss(y, p), abs=1e-12)


def test_reliability_curve_matches_sklearn() -> None:
    rng = np.random.default_rng(2)
    p = rng.random(500)
    y = (rng.random(500) < p**2).astype(int)
    p[:5] = [0.0, 0.1, 0.5, 0.9, 1.0]  # bin edges
    ours = reliability_curve(y, p, n_bins=10)
    true, pred = calibration_curve(y, p, n_bins=10, strategy="uniform")
    np.testing.assert_allclose(ours.prob_true, true, atol=1e-12)
    np.testing.assert_allclose(ours.prob_pred, pred, atol=1e-12)
    assert ours.counts.sum() == 500


def test_ece_hand_example() -> None:
    # bin [0.0, 0.5): p = 0.2, 0.2 with outcomes 0, 1 -> |0.5 - 0.2| = 0.3, weight 2/4
    # bin [0.5, 1.0]: p = 0.8, 0.8 with outcomes 1, 1 -> |1.0 - 0.8| = 0.2, weight 2/4
    assert expected_calibration_error([0, 1, 1, 1], [0.2, 0.2, 0.8, 0.8], n_bins=2) == (
        pytest.approx(0.25)
    )
    assert expected_calibration_error([0, 1], [0.0, 1.0], n_bins=5) == 0.0


def test_probability_validation() -> None:
    with pytest.raises(ValueError):
        brier_score([0, 1], [0.5, 1.5])
    with pytest.raises(ValueError):
        brier_score([0, 1], [0.5])


def test_plot_reliability(tmp_path: Path) -> None:
    out = plot_reliability({"m": ([0, 1, 1], [0.1, 0.7, 0.9])}, tmp_path / "rel.png")
    assert out.exists() and out.stat().st_size > 0
