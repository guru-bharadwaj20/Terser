"""Discrete AdaBoost over depth-1 decision stumps, from scratch (plan row **5.1.1**).

Labels ``y_i in {-1, +1}``; sample weights start uniform, ``D_1(i) = 1/n``. Round ``t``:

1. fit the stump ``h_t`` minimising weighted error ``eps_t = sum_i D_t(i) [h_t(x_i) != y_i]``;
2. ``alpha_t = learning_rate * 1/2 ln((1 - eps_t) / eps_t)`` (scikit-learn's two-class SAMME uses
   ``ln((1-eps)/eps)``, i.e. twice this; the factor scales every ``alpha`` equally and so leaves
   the sign of ``F`` and every ranking metric such as AUC unchanged);
3. ``D_{t+1}(i) ∝ D_t(i) exp(-alpha_t y_i h_t(x_i))``: misclassified points gain weight.

The ensemble is ``F(x) = sum_t alpha_t h_t(x)``. :attr:`AdaBoostStumps.train_error_` records
training error after each round (for the error-vs-rounds plot) and
:meth:`AdaBoostStumps.hard_examples` returns the highest-weight examples after training
(the "which pairs get the largest weights" note).
Training error is bounded by ``prod_t 2 sqrt(eps_t (1 - eps_t))``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

__all__ = ["AdaBoostStumps", "Stump", "fit_stump"]

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Stump:
    """``h(x) = polarity if x[feature] > threshold else -polarity``."""

    feature: int
    threshold: float
    polarity: int

    def predict(self, X: FloatArray) -> FloatArray:
        return np.where(X[:, self.feature] > self.threshold, self.polarity, -self.polarity).astype(
            np.float64
        )


def fit_stump(X: FloatArray, y: FloatArray, w: FloatArray) -> tuple[Stump, float]:
    """Exhaustive weighted-error stump search over midpoints between sorted unique values."""
    w = w / w.sum()
    base = float(w[y < 0].sum())  # polarity +1, threshold below every value: all predicted +1
    best = (Stump(0, -np.inf, 1), base)
    for j in range(X.shape[1]):
        order = np.argsort(X[:, j], kind="stable")
        xs, ys, ws = X[order, j], y[order], w[order]
        # Moving point k to the "<= threshold" side flips its prediction to -1: error rises by
        # its weight if it is positive and falls by it if negative.
        errs = base + np.cumsum(np.where(ys > 0, ws, -ws))
        valid = np.flatnonzero(np.diff(xs) > 0)
        if len(valid) == 0:
            continue
        cand = errs[valid]
        for pol, e in ((1, cand), (-1, 1.0 - cand)):
            k = int(np.argmin(e))
            if e[k] < best[1] - 1e-15:
                thr = float((xs[valid[k]] + xs[valid[k] + 1]) / 2)
                best = (Stump(j, thr, pol), float(e[k]))
    return best


@dataclass
class AdaBoostStumps:
    """AdaBoost with depth-1 stumps on labels ``{0, 1}`` (1 = positive class)."""

    n_estimators: int = 50
    learning_rate: float = 1.0
    stumps_: list[Stump] = field(init=False, default_factory=list)
    alphas_: list[float] = field(init=False, default_factory=list)
    train_error_: list[float] = field(init=False, default_factory=list)
    sample_weight_: FloatArray = field(init=False, repr=False)

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> AdaBoostStumps:
        Xa = np.asarray(X, dtype=np.float64)
        ya = np.asarray(y)
        if not np.isin(ya, (0, 1)).all():
            raise ValueError("labels must be 0/1")
        ys = np.where(ya == 1, 1.0, -1.0)
        w = np.full(len(ys), 1.0 / len(ys))
        F = np.zeros(len(ys))
        self.stumps_, self.alphas_, self.train_error_ = [], [], []
        for _ in range(self.n_estimators):
            stump, eps = fit_stump(Xa, ys, w)
            if eps >= 0.5:
                break
            eps = max(eps, 1e-12)
            alpha = self.learning_rate * 0.5 * np.log((1 - eps) / eps)
            h = stump.predict(Xa)
            self.stumps_.append(stump)
            self.alphas_.append(float(alpha))
            F += alpha * h
            # Same rule as predict: F == 0 is the negative class, not an error.
            self.train_error_.append(float(np.mean(np.where(F > 0, 1.0, -1.0) != ys)))
            w = w * np.exp(-alpha * ys * h)
            w /= w.sum()
            if eps <= 1e-12:
                break
        self.sample_weight_ = w
        return self

    def decision_function(self, X: npt.ArrayLike) -> FloatArray:
        Xa = np.asarray(X, dtype=np.float64)
        F = np.zeros(len(Xa))
        for a, s in zip(self.alphas_, self.stumps_, strict=True):
            F += a * s.predict(Xa)
        return F

    def predict_proba(self, X: npt.ArrayLike) -> FloatArray:
        """``P(y=1) = sigmoid(2 F(x))`` (Friedman, Hastie and Tibshirani's link)."""
        return np.asarray(1.0 / (1.0 + np.exp(-2.0 * self.decision_function(X))))

    def predict(self, X: npt.ArrayLike) -> npt.NDArray[np.int64]:
        return (self.decision_function(X) > 0).astype(np.int64)

    def hard_examples(self, k: int = 10) -> npt.NDArray[np.int64]:
        """Indices of the ``k`` training examples with the largest final boosting weight."""
        return np.argsort(-self.sample_weight_, kind="stable")[:k].astype(np.int64)
