"""Tests for terser.models.classical.logistic (row 3.4.1: coefficients within 1e-3 of sklearn)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from terser.models.classical.logistic import GDLogisticRegression, log_loss, sigmoid


def _data(seed: int, n: int = 400, d: int = 5) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, d))
    w = rng.normal(0, 1, d)
    y = (rng.random(n) < sigmoid(X @ w - 0.5)).astype(int)
    return X, y


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("C", [None, 1.0, 0.05])
def test_matches_sklearn(seed: int, C: float | None) -> None:
    X, y = _data(seed)
    ours = GDLogisticRegression(C=C).fit(X, y)
    sk = LogisticRegression(C=C if C is not None else np.inf, tol=1e-12)
    sk.fit(X, y)
    np.testing.assert_allclose(ours.coef_, sk.coef_[0], atol=1e-3)
    assert ours.intercept_ == pytest.approx(sk.intercept_[0], abs=1e-3)
    np.testing.assert_allclose(ours.predict_proba(X), sk.predict_proba(X)[:, 1], atol=1e-3)


def test_gradient_matches_finite_differences() -> None:
    X, y = _data(9, n=50, d=3)
    m = GDLogisticRegression(C=2.0)
    w = np.array([0.3, -0.2, 0.1])
    b = 0.05

    def J(w: np.ndarray, b: float) -> float:
        return log_loss(y, sigmoid(X @ w + b)) + w @ w / (2 * 2.0 * len(y))

    gw, gb = m._gradient(X, y.astype(float), w, b)
    h = 1e-6
    num = [(J(w + h * e, b) - J(w - h * e, b)) / (2 * h) for e in np.eye(3)]
    np.testing.assert_allclose(gw, num, rtol=1e-6)
    assert gb == pytest.approx((J(w, b + h) - J(w, b - h)) / (2 * h), rel=1e-6)


def test_loss_decreases_and_validation() -> None:
    X, y = _data(1)
    m = GDLogisticRegression(C=1.0).fit(X, y)
    assert all(a >= b - 1e-12 for a, b in zip(m.loss_history_, m.loss_history_[1:], strict=False))
    assert set(m.predict(X).tolist()) <= {0, 1}
    assert sigmoid(np.array([-1000.0, 1000.0])).tolist() == [0.0, 1.0]
    with pytest.raises(ValueError):
        GDLogisticRegression().fit(X, y[:-1])
    with pytest.raises(ValueError):
        GDLogisticRegression().fit(X, y + 1)
