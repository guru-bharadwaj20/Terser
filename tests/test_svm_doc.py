"""The running example in docs/derivations/svm.md (row 4.4.1)."""

from __future__ import annotations

import math

import numpy as np
import pytest
from sklearn.svm import SVC

X = np.array([[1.0, 1.0], [-1.0, -1.0]])
Y = np.array([1, -1])


def test_margin_and_primal_solution() -> None:
    svm = SVC(kernel="linear", C=1e6).fit(X, Y)
    w = svm.coef_[0]
    np.testing.assert_allclose(w, [0.5, 0.5], atol=1e-6)
    assert svm.intercept_[0] == pytest.approx(0.0, abs=1e-6)
    assert 2 / np.linalg.norm(w) == pytest.approx(2 * math.sqrt(2), abs=1e-5)
    np.testing.assert_allclose(Y * (X @ w + svm.intercept_[0]), [1.0, 1.0], atol=1e-6)


def test_soft_margin_slack_is_hinge() -> None:
    rng = np.random.default_rng(0)
    Xs = rng.normal(0, 1, (60, 2))
    ys = np.where(Xs[:, 0] + 0.5 * rng.normal(0, 1, 60) > 0, 1, -1)
    for C in (0.01, 1.0, 100.0):
        svm = SVC(kernel="linear", C=C).fit(Xs, ys)
        w, b = svm.coef_[0], svm.intercept_[0]
        slack = np.maximum(0.0, 1 - ys * (Xs @ w + b))
        primal = 0.5 * w @ w + C * slack.sum()
        # Any perturbation of the optimum must not lower the hinge-loss objective.
        for dw in rng.normal(0, 1e-2, (20, 2)):
            s2 = np.maximum(0.0, 1 - ys * (Xs @ (w + dw) + b))
            assert primal <= 0.5 * (w + dw) @ (w + dw) + C * s2.sum() + 1e-6
