"""Row 4.4.4: cvxopt dual SVM vs scikit-learn (same SV set, ||w - w_sk|| / ||w_sk|| < 1e-3)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.svm import SVC

from terser.models.classical.scratch_svm import DualSVM


def _data(seed: int, n: int = 120, d: int = 4, sep: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    X = rng.normal(0, 1, (n, d)) + sep * (2 * y[:, None] - 1) * np.linspace(1, 0.2, d)
    return X, y


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("C", [0.01, 0.1, 1.0, 10.0])
def test_linear_matches_sklearn(seed: int, C: float) -> None:
    X, y = _data(seed)
    ours = DualSVM(C=C).fit(X, y)
    sk = SVC(kernel="linear", C=C, tol=1e-10).fit(X, y)
    assert set(ours.support_) == set(sk.support_)
    w_sk = sk.coef_[0]
    assert np.linalg.norm(ours.coef_ - w_sk) / np.linalg.norm(w_sk) < 1e-3
    assert ours.intercept_ == pytest.approx(sk.intercept_[0], abs=1e-3)
    np.testing.assert_array_equal(ours.predict(X), sk.predict(X))


def test_hard_margin_separable() -> None:
    X, y = _data(5, sep=4.0)
    ours = DualSVM(C=None).fit(X, y)
    sk = SVC(kernel="linear", C=1e8, tol=1e-10).fit(X, y)
    assert set(ours.support_) == set(sk.support_)
    ys = np.where(y == 1, 1.0, -1.0)
    margins = ys * ours.decision_function(X)
    assert margins.min() == pytest.approx(1.0, abs=1e-5)
    np.testing.assert_allclose(margins[ours.support_], 1.0, atol=1e-5)
    assert ours.margin_width == pytest.approx(2 / np.linalg.norm(ours.coef_))


@pytest.mark.parametrize("degree,coef0", [(2, 1.0), (3, 0.5)])
def test_poly_matches_sklearn(degree: int, coef0: float) -> None:
    X, y = _data(1, n=80, d=2, sep=0.6)
    ours = DualSVM(C=1.0, kernel="poly", degree=degree, coef0=coef0).fit(X, y)
    sk = SVC(kernel="poly", degree=degree, coef0=coef0, C=1.0, gamma="scale", tol=1e-10)
    sk.fit(X, y)
    assert set(ours.support_) == set(sk.support_)
    np.testing.assert_allclose(ours.decision_function(X), sk.decision_function(X), atol=1e-3)
    with pytest.raises(AttributeError):
        _ = ours.coef_


def test_string_labels_and_validation() -> None:
    X, y = _data(2)
    labels = np.where(y == 1, "filler", "content")
    m = DualSVM(C=1.0).fit(X, labels)
    assert set(m.predict(X)) <= {"filler", "content"}
    with pytest.raises(ValueError):
        DualSVM().fit(X, np.zeros(len(y)))
