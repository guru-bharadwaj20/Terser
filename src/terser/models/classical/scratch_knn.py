"""K-nearest neighbours classification and regression from scratch (rows **3.5.1**, **3.5.2**).

Brute-force search. Distances:

* ``euclidean``: ``||a - b||_2``;
* ``manhattan``: ``||a - b||_1``;
* ``cosine``: ``1 - a.b / (||a|| ||b||)`` (0 for a zero vector paired with anything, as distance 1).

Weights: ``uniform`` (each of the k neighbours counts once) or ``distance`` (weight ``1/d``; if
any neighbour is at distance 0, only the zero-distance neighbours vote, equally, as in
scikit-learn). The classifier's score is the weighted share of positive neighbours; the
regressor predicts the weighted mean target. Neighbour ties at equal distance are broken by
training order (stable sort). Features should be standardised (row 3.1.4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt

__all__ = ["KNNClassifier", "KNNRegressor", "pairwise_distances"]

FloatArray = npt.NDArray[np.float64]
Metric = Literal["euclidean", "manhattan", "cosine"]
Weights = Literal["uniform", "distance"]


def pairwise_distances(A: npt.ArrayLike, B: npt.ArrayLike, metric: Metric) -> FloatArray:
    """Distance matrix of shape ``(len(A), len(B))``."""
    a = np.asarray(A, dtype=np.float64)
    b = np.asarray(B, dtype=np.float64)
    if metric == "euclidean":
        norms = (a**2).sum(1)[:, None] + (b**2).sum(1)[None, :]
        sq = norms - 2 * a @ b.T
        # The expansion leaves round-off (~1e-6 after the sqrt) for identical points; snap values
        # within relative round-off of zero to exactly 0 so duplicates are at distance 0.
        sq[sq <= 1e-12 * norms] = 0.0
        return np.sqrt(np.maximum(sq, 0.0))
    if metric == "manhattan":
        return np.abs(a[:, None, :] - b[None, :, :]).sum(-1)
    if metric == "cosine":
        na = np.linalg.norm(a, axis=1)[:, None]
        nb = np.linalg.norm(b, axis=1)[None, :]
        denom = na * nb
        sim = np.divide(a @ b.T, denom, out=np.zeros((len(a), len(b))), where=denom > 0)
        return np.clip(1.0 - sim, 0.0, 2.0)
    raise ValueError(f"unknown metric {metric!r}")


@dataclass
class _KNNBase:
    k: int = 5
    metric: Metric = "euclidean"
    weights: Weights = "uniform"
    X_: FloatArray = field(init=False, repr=False)
    y_: FloatArray = field(init=False, repr=False)

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> _KNNBase:
        Xa = np.asarray(X, dtype=np.float64)
        ya = np.asarray(y, dtype=np.float64)
        if Xa.ndim != 2 or ya.shape != (len(Xa),):
            raise ValueError("expected X (n, d) and y (n,)")
        if not 1 <= self.k <= len(Xa):
            raise ValueError(f"k must be in [1, {len(Xa)}], got {self.k}")
        self.X_, self.y_ = Xa, ya
        return self

    def _weighted_mean(self, X: npt.ArrayLike) -> FloatArray:
        d = pairwise_distances(X, self.X_, self.metric)
        idx = np.argsort(d, axis=1, kind="stable")[:, : self.k]
        nd = np.take_along_axis(d, idx, axis=1)
        ny = self.y_[idx]
        if self.weights == "uniform":
            return np.asarray(ny.mean(axis=1), dtype=np.float64)
        zero = nd == 0
        with np.errstate(divide="ignore"):
            w = np.where(zero.any(axis=1, keepdims=True), zero.astype(np.float64), 1.0 / nd)
        return np.asarray((w * ny).sum(axis=1) / w.sum(axis=1), dtype=np.float64)


@dataclass
class KNNClassifier(_KNNBase):
    """Binary KNN classifier with 0/1 labels."""

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> KNNClassifier:
        if not np.isin(np.asarray(y), (0, 1)).all():
            raise ValueError("y must be binary 0/1")
        super().fit(X, y)
        return self

    def predict_proba(self, X: npt.ArrayLike) -> FloatArray:
        """Weighted share of positive neighbours, shape (n,)."""
        return self._weighted_mean(X)

    def predict(self, X: npt.ArrayLike) -> npt.NDArray[np.int64]:
        """Majority vote; an exact tie goes to the negative class (``preserved``)."""
        return (self.predict_proba(X) > 0.5).astype(np.int64)


@dataclass
class KNNRegressor(_KNNBase):
    """KNN regression: weighted mean of the neighbours' targets."""

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> KNNRegressor:
        super().fit(X, y)
        return self

    def predict(self, X: npt.ArrayLike) -> FloatArray:
        return self._weighted_mean(X)
