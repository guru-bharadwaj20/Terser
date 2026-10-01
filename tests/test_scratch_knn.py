"""Rows 3.5.1 / 3.5.2 / 3.5.4: from-scratch KNN and the skip-short-prompts rule."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import pairwise_distances as sk_pairwise
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from terser.models.classical.scratch_knn import KNNClassifier, KNNRegressor, pairwise_distances
from terser.models.classical.skip_rule import best_threshold, net_savings_curve, should_rewrite

RNG = np.random.default_rng(0)
X_TR = RNG.normal(size=(120, 4))
Y_CLS = (X_TR[:, 0] + 0.5 * X_TR[:, 1] + RNG.normal(scale=0.5, size=120) > 0).astype(int)
Y_REG = X_TR[:, 0] * 3 + RNG.normal(size=120)
X_TE = RNG.normal(size=(40, 4))


@pytest.mark.parametrize("metric", ["euclidean", "manhattan", "cosine"])
def test_distances_match_sklearn(metric: str) -> None:
    ours = pairwise_distances(X_TE, X_TR, metric)  # type: ignore[arg-type]
    np.testing.assert_allclose(ours, sk_pairwise(X_TE, X_TR, metric=metric), atol=1e-9)


def test_unknown_metric() -> None:
    with pytest.raises(ValueError):
        pairwise_distances(X_TE, X_TR, "chebyshev")  # type: ignore[arg-type]


@pytest.mark.parametrize("k", [1, 3, 5, 7, 11, 15, 25])
@pytest.mark.parametrize("metric", ["euclidean", "manhattan", "cosine"])
@pytest.mark.parametrize("weights", ["uniform", "distance"])
def test_classifier_matches_sklearn(k: int, metric: str, weights: str) -> None:
    ours = KNNClassifier(k=k, metric=metric, weights=weights).fit(X_TR, Y_CLS)  # type: ignore[arg-type]
    ref = KNeighborsClassifier(n_neighbors=k, metric=metric, weights=weights, algorithm="brute")
    ref.fit(X_TR, Y_CLS)
    np.testing.assert_allclose(ours.predict_proba(X_TE), ref.predict_proba(X_TE)[:, 1], atol=1e-9)


@pytest.mark.parametrize("k", [1, 5, 25])
@pytest.mark.parametrize("metric", ["euclidean", "manhattan", "cosine"])
@pytest.mark.parametrize("weights", ["uniform", "distance"])
def test_regressor_matches_sklearn(k: int, metric: str, weights: str) -> None:
    ours = KNNRegressor(k=k, metric=metric, weights=weights).fit(X_TR, Y_REG)  # type: ignore[arg-type]
    ref = KNeighborsRegressor(n_neighbors=k, metric=metric, weights=weights, algorithm="brute")
    ref.fit(X_TR, Y_REG)
    np.testing.assert_allclose(ours.predict(X_TE), ref.predict(X_TE), atol=1e-9)


def test_hand_example_and_zero_distance() -> None:
    X = np.array([[0.0], [1.0], [3.0], [10.0]])
    y = np.array([0, 1, 1, 0])
    uni = KNNClassifier(k=3).fit(X, y)
    assert uni.predict_proba([[1.5]]).tolist() == pytest.approx([2 / 3])
    dist = KNNClassifier(k=3, weights="distance").fit(X, y)
    # neighbours of 1.5: 1.0 (d .5, y1), 3.0 (d 1.5, y1), 0.0 (d 1.5, y0)
    w = np.array([2.0, 1 / 1.5, 1 / 1.5])
    assert dist.predict_proba([[1.5]])[0] == pytest.approx((w[0] + w[1]) / w.sum())
    # exact match: only the zero-distance neighbour votes
    assert dist.predict_proba([[10.0]])[0] == 0.0
    reg = KNNRegressor(k=2).fit(X, [0.0, 2.0, 4.0, 100.0])
    assert reg.predict([[2.0]])[0] == pytest.approx(3.0)


def test_validation() -> None:
    with pytest.raises(ValueError):
        KNNClassifier(k=10).fit([[0.0], [1.0]], [0, 1])
    with pytest.raises(ValueError):
        KNNClassifier(k=1).fit([[0.0], [1.0]], [0, 2])
    with pytest.raises(ValueError):
        KNNRegressor(k=1).fit([[0.0]], [1.0, 2.0])


def test_skip_rule_hand_example() -> None:
    predicted = [2.0, 10.0, 30.0, 5.0]
    actual = [1.0, 12.0, 25.0, 6.0]
    curve = net_savings_curve(predicted, actual, overhead=5.0)
    assert curve.thresholds.tolist() == [-np.inf, 2.0, 5.0, 10.0, 30.0, np.inf]
    # gains after overhead: -4, 7, 20, 1
    assert curve.net_savings.tolist() == pytest.approx([24 / 4, 28 / 4, 27 / 4, 20 / 4, 0, 0])
    assert curve.rewrite_rate.tolist() == pytest.approx([1, 0.75, 0.5, 0.25, 0, 0])
    assert best_threshold(curve) == 2.0
    assert should_rewrite(predicted, 2.0).tolist() == [False, True, True, True]


def test_skip_rule_length_mismatch() -> None:
    with pytest.raises(ValueError):
        net_savings_curve([1.0], [1.0, 2.0], 0.0)
