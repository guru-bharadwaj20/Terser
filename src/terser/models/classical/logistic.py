"""Logistic regression trained by batch gradient descent, from scratch (plan row **3.4.1**).

Model: ``p(degraded | x) = sigmoid(w . x + b)``. Objective (mean log-loss plus optional L2):

    J(w, b) = -1/n sum_i [y_i log p_i + (1 - y_i) log(1 - p_i)] + ||w||^2 / (2 C n)

Its gradient is ``dJ/dw = X^T (p - y) / n + w / (C n)`` and ``dJ/db = mean(p - y)``. The L2 scaling
makes the minimiser identical to scikit-learn's ``LogisticRegression(C=C)`` (whose objective is
``C * sum log-loss + ||w||^2 / 2``); ``C=None`` is unregularised. The bias is never penalised.
Coefficients match scikit-learn within 1e-3 (tested).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

__all__ = ["GDLogisticRegression", "log_loss", "sigmoid"]

FloatArray = npt.NDArray[np.float64]


def sigmoid(z: npt.ArrayLike) -> FloatArray:
    """Numerically stable logistic function."""
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def log_loss(y: npt.ArrayLike, p: npt.ArrayLike, eps: float = 1e-15) -> float:
    """Mean binary cross-entropy."""
    ya = np.asarray(y, dtype=np.float64)
    pa = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return float(-np.mean(ya * np.log(pa) + (1 - ya) * np.log(1 - pa)))


@dataclass
class GDLogisticRegression:
    """Binary logistic regression fitted by full-batch gradient descent.

    Attributes:
        C: Inverse L2 strength as in scikit-learn; ``None`` for no penalty.
        lr: Step size. Features should be standardised (row 3.1.4) for a fixed step to converge.
        max_iter: Iteration cap.
        tol: Stop when the gradient's infinity norm falls below this.
    """

    C: float | None = 1.0
    lr: float = 0.5
    max_iter: int = 100_000
    tol: float = 1e-8
    coef_: FloatArray = field(init=False, repr=False)
    intercept_: float = field(init=False, default=0.0)
    n_iter_: int = field(init=False, default=0)
    loss_history_: list[float] = field(init=False, repr=False, default_factory=list)

    def _gradient(
        self, X: FloatArray, y: FloatArray, w: FloatArray, b: float
    ) -> tuple[FloatArray, float]:
        n = len(y)
        r = sigmoid(X @ w + b) - y
        gw = X.T @ r / n
        if self.C is not None:
            gw = gw + w / (self.C * n)
        return gw, float(r.mean())

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> GDLogisticRegression:
        """Fit on features ``X`` (n, d) and binary labels ``y`` (n,)."""
        Xa = np.asarray(X, dtype=np.float64)
        ya = np.asarray(y, dtype=np.float64)
        if Xa.ndim != 2 or ya.shape != (Xa.shape[0],):
            raise ValueError(f"expected X (n, d) and y (n,), got {Xa.shape} and {ya.shape}")
        if not np.isin(ya, (0.0, 1.0)).all():
            raise ValueError("y must be binary 0/1")
        w = np.zeros(Xa.shape[1])
        b = 0.0
        self.loss_history_ = []
        for it in range(1, self.max_iter + 1):
            gw, gb = self._gradient(Xa, ya, w, b)
            w -= self.lr * gw
            b -= self.lr * gb
            if it % 100 == 1:
                self.loss_history_.append(log_loss(ya, sigmoid(Xa @ w + b)))
            if max(float(np.max(np.abs(gw), initial=0.0)), abs(gb)) < self.tol:
                break
        self.coef_, self.intercept_, self.n_iter_ = w, b, it
        return self

    def decision_function(self, X: npt.ArrayLike) -> FloatArray:
        """Logit ``w . x + b``."""
        return np.asarray(X, dtype=np.float64) @ self.coef_ + self.intercept_

    def predict_proba(self, X: npt.ArrayLike) -> FloatArray:
        """Probability of the positive class (``degraded``), shape (n,)."""
        return sigmoid(self.decision_function(X))

    def predict(self, X: npt.ArrayLike, threshold: float = 0.5) -> npt.NDArray[np.int64]:
        """Hard labels at ``threshold`` (use row 3.2.3 for the gate's operating point)."""
        return (self.predict_proba(X) >= threshold).astype(np.int64)
