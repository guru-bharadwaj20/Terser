"""The running example in docs/derivations/svm.md (rows 4.4.1-4.4.3)."""

from __future__ import annotations

import math
from pathlib import Path

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


def test_dual_solution_and_recovery() -> None:
    svm = SVC(kernel="linear", C=1e6).fit(X, Y)
    alpha_y = np.zeros(2)
    alpha_y[svm.support_] = svm.dual_coef_[0]
    np.testing.assert_allclose(alpha_y, [0.25, -0.25], atol=1e-6)
    alpha = alpha_y * Y
    assert alpha @ Y == pytest.approx(0.0, abs=1e-9)
    np.testing.assert_allclose(alpha_y @ X, svm.coef_[0], atol=1e-9)
    K = X @ X.T
    dual = alpha.sum() - 0.5 * (alpha * Y) @ K @ (alpha * Y)
    assert dual == pytest.approx(0.25, abs=1e-6)
    assert dual == pytest.approx(0.5 * svm.coef_[0] @ svm.coef_[0], abs=1e-6)


def test_kkt_groups_on_noisy_data() -> None:
    rng = np.random.default_rng(1)
    Xs = rng.normal(0, 1, (80, 2))
    ys = np.where(Xs[:, 0] - Xs[:, 1] + 0.7 * rng.normal(0, 1, 80) > 0, 1, -1)
    C = 1.0
    svm = SVC(kernel="linear", C=C, tol=1e-8).fit(Xs, ys)
    alpha = np.zeros(80)
    alpha[svm.support_] = np.abs(svm.dual_coef_[0])
    margin = ys * svm.decision_function(Xs)
    assert np.all(margin[alpha < 1e-8] >= 1 - 1e-4)
    free = (alpha > 1e-6) & (alpha < C - 1e-6)
    np.testing.assert_allclose(margin[free], 1.0, atol=1e-3)
    assert np.all(margin[alpha > C - 1e-6] <= 1 + 1e-4)


def test_linear_kernel_dual_matches_primal_decision() -> None:
    rng = np.random.default_rng(2)
    Xs = rng.normal(0, 1, (50, 3))
    ys = np.where(Xs @ [1.0, -0.5, 0.2] + 0.3 * rng.normal(0, 1, 50) > 0, 1, -1)
    lin = SVC(kernel="linear", C=1.0).fit(Xs, ys)
    pre = SVC(kernel="precomputed", C=1.0).fit(Xs @ Xs.T, ys)
    Z = rng.normal(0, 1, (20, 3))
    gram_test = Z @ Xs[pre.support_].T
    via_kernel = gram_test @ pre.dual_coef_[0] + pre.intercept_[0]
    np.testing.assert_allclose(via_kernel, lin.decision_function(Z), atol=1e-6)
    np.testing.assert_allclose(Z @ lin.coef_[0] + lin.intercept_[0], via_kernel, atol=1e-6)
    assert np.linalg.eigvalsh(Xs @ Xs.T).min() > -1e-9
    np.testing.assert_allclose(X @ X.T, [[2.0, -2.0], [-2.0, 2.0]])


def test_polynomial_feature_count_quoted_in_doc() -> None:
    from math import comb

    assert 5e8 < comb(1536 + 3, 3) < 7e8


DOC = Path(__file__).resolve().parents[1] / "docs" / "derivations" / "svm.md"


def test_numbers_quoted_in_doc() -> None:
    text = DOC.read_text(encoding="utf-8")
    assert "$w = (0.5, 0.5)$" in text
    assert r"$2\sqrt 2 \approx 2.828427$" in text
    assert f"{2 * math.sqrt(2):.6f}" == "2.828427"
    assert r"$6 \times 10^8$ features" in text


def test_project_dual_svm_reproduces_running_example() -> None:
    from terser.models.classical.scratch_svm import DualSVM

    svm = DualSVM(C=None, kernel="linear").fit(X, Y)
    np.testing.assert_allclose(svm.coef_, [0.5, 0.5], atol=1e-5)
    assert svm.intercept_ == pytest.approx(0.0, abs=1e-5)
    assert svm.margin_width == pytest.approx(2 * math.sqrt(2), abs=1e-4)
    np.testing.assert_allclose(svm.alpha_, [0.25, 0.25], atol=1e-5)
