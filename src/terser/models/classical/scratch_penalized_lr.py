"""L1 / L2 / elastic-net logistic regression by proximal gradient (FISTA), from scratch.

Plan rows **3.4.2** (regularisation) and **3.4.5** (odds ratios). Derivation:
``docs/derivations/logistic_regression.md``.

Objective, scaled so its minimiser equals scikit-learn's ``LogisticRegression(penalty=...,
C=C, l1_ratio=r)`` (whose objective is ``C sum log-loss + (1 - r)/2 ||w||^2 + r ||w||_1``)::

    J(w, b) = mean log-loss + [(1 - r)/2 ||w||^2 + r ||w||_1] / (C n)

``penalty="l2"`` is ``r = 0``, ``"l1"`` is ``r = 1``. The smooth part (log-loss + L2) is handled by
a gradient step of size ``1/L`` with ``L = sigma_max(X)^2 / (4n) + (1 - r)/(C n)``, its Lipschitz
constant; the L1 part by its proximal operator, soft-thresholding at ``step * r / (C n)``. The
intercept is never penalised. FISTA momentum gives the ``O(1/k^2)`` rate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from terser.models.classical.logistic import sigmoid

__all__ = ["PenalizedLogisticRegression", "odds_ratio_table", "soft_threshold"]

FloatArray = npt.NDArray[np.float64]
Penalty = Literal["l1", "l2", "elasticnet"]


def soft_threshold(z: FloatArray, t: float) -> FloatArray:
    """Proximal operator of ``t ||.||_1``: ``sign(z) max(|z| - t, 0)``."""
    return np.asarray(np.sign(z) * np.maximum(np.abs(z) - t, 0.0), dtype=np.float64)


@dataclass
class PenalizedLogisticRegression:
    """Binary logistic regression with an L1, L2 or elastic-net penalty.

    Attributes:
        penalty: ``"l1"``, ``"l2"`` or ``"elasticnet"``.
        C: Inverse regularisation strength (scikit-learn convention).
        l1_ratio: Elastic-net mixing ``r`` in [0, 1]; ignored unless ``penalty="elasticnet"``.
        max_iter: Iteration cap.
        tol: Stop when the parameter change (infinity norm) falls below this.
    """

    penalty: Penalty = "l2"
    C: float = 1.0
    l1_ratio: float = 0.5
    max_iter: int = 50_000
    tol: float = 1e-10
    coef_: FloatArray = field(init=False, repr=False)
    intercept_: float = field(init=False, default=0.0)
    n_iter_: int = field(init=False, default=0)

    @property
    def _r(self) -> float:
        if self.penalty == "l1":
            return 1.0
        if self.penalty == "l2":
            return 0.0
        if not 0.0 <= self.l1_ratio <= 1.0:
            raise ValueError("l1_ratio must be in [0, 1]")
        return self.l1_ratio

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> PenalizedLogisticRegression:
        Xa = np.asarray(X, dtype=np.float64)
        ya = np.asarray(y, dtype=np.float64)
        if Xa.ndim != 2 or ya.shape != (Xa.shape[0],):
            raise ValueError(f"expected X (n, d) and y (n,), got {Xa.shape} and {ya.shape}")
        if not np.isin(ya, (0.0, 1.0)).all():
            raise ValueError("y must be binary 0/1")
        if self.C <= 0:
            raise ValueError("C must be positive")
        n, d = Xa.shape
        r = self._r
        lam2, lam1 = (1 - r) / (self.C * n), r / (self.C * n)
        # augmented design with a column of ones for the intercept
        Z = np.hstack([Xa, np.ones((n, 1))])
        lip = float(np.linalg.norm(Z, 2) ** 2) / (4 * n) + lam2
        step = 1.0 / lip
        theta = np.zeros(d + 1)
        mom, t = theta.copy(), 1.0
        for it in range(1, self.max_iter + 1):
            self.n_iter_ = it
            p = sigmoid(Z @ mom)
            grad = Z.T @ (p - ya) / n
            grad[:d] += lam2 * mom[:d]
            nxt = mom - step * grad
            nxt[:d] = soft_threshold(nxt[:d], step * lam1)
            t_next = (1 + np.sqrt(1 + 4 * t * t)) / 2
            mom = nxt + ((t - 1) / t_next) * (nxt - theta)
            done = float(np.max(np.abs(nxt - theta))) < self.tol
            theta, t = nxt, t_next
            if done:
                break
        self.coef_, self.intercept_ = theta[:d], float(theta[d])
        return self

    def decision_function(self, X: npt.ArrayLike) -> FloatArray:
        return np.asarray(X, dtype=np.float64) @ self.coef_ + self.intercept_

    def predict_proba(self, X: npt.ArrayLike) -> FloatArray:
        """Probability of the positive class, shape (n,)."""
        return sigmoid(self.decision_function(X))

    def predict(self, X: npt.ArrayLike, threshold: float = 0.5) -> npt.NDArray[np.int64]:
        return (self.predict_proba(X) >= threshold).astype(np.int64)


def odds_ratio_table(coef: npt.ArrayLike, feature_names: Sequence[str]) -> pd.DataFrame:
    """Odds ratios ``exp(w_j)`` for standardised features, sorted by effect size.

    With z-scored inputs, ``exp(w_j)`` is the multiplicative change in the odds of ``degraded``
    for a one-standard-deviation increase in feature ``j``, other features held fixed.
    """
    w = np.asarray(coef, dtype=np.float64)
    if len(w) != len(feature_names):
        raise ValueError("one name per coefficient required")
    df = pd.DataFrame({"feature": list(feature_names), "coef": w, "odds_ratio": np.exp(w)})
    order = np.argsort(-np.abs(w), kind="stable")
    return df.iloc[order].reset_index(drop=True)
