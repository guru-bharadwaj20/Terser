"""Support vector machine solved in the dual with ``cvxopt`` (plan rows **4.4.4**-**4.4.8**).

Derivation: ``docs/derivations/svm.md`` §3-5. With labels ``y_i in {-1, +1}`` and kernel
``K``, the soft-margin dual is

    max_a  sum_i a_i - 1/2 sum_ij a_i a_j y_i y_j K(x_i, x_j)
    s.t.   0 <= a_i <= C,   sum_i a_i y_i = 0

(hard margin: no upper bound). ``cvxopt.solvers.qp`` minimises ``1/2 a^T P a + q^T a`` subject to
``G a <= h`` and ``A a = b``, so ``P = (y y^T) * K``, ``q = -1``, ``A = y^T``, ``b = 0`` and ``G``
stacks ``-I`` (and ``+I`` with ``h = C`` for the soft margin).

Support vectors are the points with ``a_i > sv_tol``. For the linear kernel the weight vector is
``w = sum_i a_i y_i x_i``. The bias is averaged over *margin* support vectors (``a_i < C``), for
which the KKT conditions give ``y_i f(x_i) = 1`` exactly: ``b = mean(y_i - sum_j a_j y_j K_ji)``.
The geometric margin width is ``2 / ||w||`` (row 4.4.1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt

__all__ = ["DualSVM", "kernel_matrix"]

FloatArray = npt.NDArray[np.float64]
Kernel = Literal["linear", "poly"]


def kernel_matrix(
    X: FloatArray, Z: FloatArray, kernel: Kernel, degree: int, gamma: float, coef0: float
) -> FloatArray:
    """Gram matrix ``K[i, j] = K(X_i, Z_j)``; poly is ``(gamma x.z + coef0) ** degree``."""
    lin = X @ Z.T
    if kernel == "linear":
        return lin
    return np.asarray((gamma * lin + coef0) ** degree)


@dataclass
class DualSVM:
    """Binary SVM fitted in the dual.

    Args:
        C: Soft-margin penalty; ``None`` for the hard margin (separable data only).
        kernel: ``"linear"`` or ``"poly"``.
        degree, gamma, coef0: Polynomial kernel parameters (scikit-learn conventions;
            ``gamma=None`` means ``1 / (n_features * X.var())`` like ``gamma="scale"``).
        sv_tol: Multipliers above this (relative to ``max(a)``) count as support vectors.
    """

    C: float | None = 1.0
    kernel: Kernel = "linear"
    degree: int = 3
    gamma: float | None = None
    coef0: float = 0.0
    sv_tol: float = 1e-6
    alpha_: FloatArray = field(init=False, repr=False)
    support_: npt.NDArray[np.int64] = field(init=False, repr=False)
    intercept_: float = field(init=False, default=0.0)

    def _y(self, y: npt.ArrayLike) -> FloatArray:
        ya = np.asarray(y)
        classes = np.unique(ya)
        if len(classes) != 2:
            raise ValueError(f"need exactly two classes, got {classes}")
        self.classes_ = classes
        return np.where(ya == classes[1], 1.0, -1.0)

    def _k(self, A: FloatArray, B: FloatArray) -> FloatArray:
        return kernel_matrix(A, B, self.kernel, self.degree, self._gamma, self.coef0)

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> DualSVM:
        """Solve the dual QP; ``y`` may use any two labels (the larger maps to +1)."""
        from cvxopt import matrix, solvers

        Xa = np.asarray(X, dtype=np.float64)
        ys = self._y(y)
        n = len(ys)
        var = float(Xa.var())
        self._gamma = self.gamma if self.gamma is not None else 1.0 / (Xa.shape[1] * (var or 1.0))
        K = self._k(Xa, Xa)
        P = np.outer(ys, ys) * K
        G = -np.eye(n)
        h = np.zeros(n)
        if self.C is not None:
            G = np.vstack([G, np.eye(n)])
            h = np.concatenate([h, np.full(n, float(self.C))])
        opts = {"show_progress": False, "abstol": 1e-12, "reltol": 1e-12, "feastol": 1e-12}
        sol = solvers.qp(
            matrix(P),
            matrix(-np.ones(n)),
            matrix(G),
            matrix(h),
            matrix(ys.reshape(1, -1)),
            matrix(0.0),
            options=opts,
        )
        alpha = np.clip(np.asarray(sol["x"]).ravel(), 0.0, self.C)
        self.alpha_ = alpha
        self.support_ = np.flatnonzero(alpha > self.sv_tol * max(alpha.max(), 1e-300))
        self._sv_X = Xa[self.support_]
        self._sv_coef = alpha[self.support_] * ys[self.support_]
        upper = self.C if self.C is not None else np.inf
        margin = self.support_[alpha[self.support_] < upper * (1 - 1e-6)]
        idx = margin if len(margin) else self.support_
        f_no_b = self._k(Xa[idx], self._sv_X) @ self._sv_coef
        self.intercept_ = float(np.mean(ys[idx] - f_no_b))
        return self

    @property
    def coef_(self) -> FloatArray:
        """Primal weights ``w = sum a_i y_i x_i`` (linear kernel only)."""
        if self.kernel != "linear":
            raise AttributeError("coef_ exists only for the linear kernel")
        return np.asarray(self._sv_coef @ self._sv_X)

    @property
    def margin_width(self) -> float:
        """Geometric margin ``2 / ||w||`` (linear kernel only)."""
        return float(2.0 / np.linalg.norm(self.coef_))

    def decision_function(self, X: npt.ArrayLike) -> FloatArray:
        """``f(x) = sum_i a_i y_i K(x_i, x) + b``; positive means ``classes_[1]``."""
        Xa = np.asarray(X, dtype=np.float64)
        return np.asarray(self._k(Xa, self._sv_X) @ self._sv_coef + self.intercept_)

    def predict(self, X: npt.ArrayLike) -> npt.NDArray[np.generic]:
        """Class labels in the original encoding."""
        return np.asarray(self.classes_[(self.decision_function(X) > 0).astype(int)])
