"""Tests for terser.models.classical.gmm (row 5.2.5)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.mixture import GaussianMixture

from terser.models.classical.gmm import GaussianMixtureEM


def _blobs(seed: int, d: int = 2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 1, (150, d))
    b = rng.normal(3, 0.5, (100, d))
    c = rng.normal([-3] + [2] * (d - 1), 0.8, (80, d))
    return np.vstack([a, b, c])


@pytest.mark.parametrize("seed", range(5))
def test_log_likelihood_non_decreasing(seed: int) -> None:
    X = _blobs(seed)
    m = GaussianMixtureEM(n_components=3, seed=seed, tol=0.0, max_iter=100).fit(X)
    h = np.array(m.log_likelihood_history_)
    assert np.all(np.diff(h) >= -1e-10)
    assert len(h) > 20 and h[-1] > h[0]


@pytest.mark.parametrize("d", [1, 2, 3])
def test_matches_sklearn_from_same_init(d: int) -> None:
    X = _blobs(10 + d, d)
    rng = np.random.default_rng(0)
    means0 = X[rng.choice(len(X), 3, replace=False)]
    ours = GaussianMixtureEM(n_components=3, tol=1e-10, max_iter=2000).fit(X, means_init=means0)
    sk = GaussianMixture(
        n_components=3,
        means_init=means0,
        weights_init=np.full(3, 1 / 3),
        precisions_init=np.linalg.inv(ours_init_cov(X, 3)),
        tol=1e-10,
        max_iter=2000,
        reg_covar=1e-6,
    ).fit(X)
    np.testing.assert_allclose(ours.means_, sk.means_, atol=1e-4)
    np.testing.assert_allclose(ours.weights_, sk.weights_, atol=1e-4)
    np.testing.assert_allclose(ours.covariances_, sk.covariances_, atol=1e-4)
    assert ours.score(X) == pytest.approx(sk.score(X), abs=1e-6)
    np.testing.assert_array_equal(ours.predict(X), sk.predict(X))


def ours_init_cov(X: np.ndarray, k: int) -> np.ndarray:
    d = X.shape[1]
    base = np.atleast_2d(np.cov(X, rowvar=False)) + 1e-6 * np.eye(d)
    return np.repeat(base[None], k, axis=0)


def test_one_d_input_and_recovery() -> None:
    rng = np.random.default_rng(3)
    x = np.r_[rng.normal(-2, 0.5, 400), rng.normal(2, 0.5, 600)]
    m = GaussianMixtureEM(n_components=2, seed=1).fit(x)
    order = np.argsort(m.means_[:, 0])
    np.testing.assert_allclose(m.means_[order, 0], [-2, 2], atol=0.1)
    np.testing.assert_allclose(m.weights_[order], [0.4, 0.6], atol=0.03)
    assert m.predict_proba(x).shape == (1000, 2)
    assert m.converged_
    with pytest.raises(ValueError):
        GaussianMixtureEM(n_components=3).fit(np.zeros((2, 1)))
