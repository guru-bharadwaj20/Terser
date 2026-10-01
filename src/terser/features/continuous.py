"""Continuous-value handling per model family (plan row **3.1.4**).

=========================  ======================================================================
Model                      Treatment of a continuous feature
=========================  ======================================================================
Logistic regression (3.4)  :class:`Standardizer` (z-score, fitted on the training fold only), so
                           one gradient step size works for every feature and odds ratios are per
                           standard deviation (3.4.5).
KNN (3.5)                  :class:`Standardizer` as well: without it the distance is dominated by
                           whichever feature has the largest range (``token_cost`` vs flags).
C4.5-style tree (3.3.2)    Raw values. The tree searches binary thresholds at midpoints between
                           adjacent sorted values whose class differs
                           (:func:`terser.models.classical.scratch_id3.candidate_thresholds`);
                           trees are invariant to monotone rescaling, so no scaling is needed.
ID3 (3.3.1)                :class:`EqualFrequencyBinner`: ``n_bins`` quantile bins per feature
                           (edges fitted on the training fold), turning each feature into a small
                           categorical attribute for a multiway split.
=========================  ======================================================================

Every transformer is fitted inside the cross-validation fold (``fit_predict`` in
:mod:`terser.eval.cv`), never on the full data set, so no validation statistics leak into training.
Binary flags pass through the binner unchanged (two distinct values give at most two bins).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

__all__ = ["EqualFrequencyBinner", "Standardizer"]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


@dataclass
class Standardizer:
    """Z-score scaling; constant columns are centred and left with unit scale."""

    mean_: FloatArray = field(init=False, repr=False)
    scale_: FloatArray = field(init=False, repr=False)

    def fit(self, X: npt.ArrayLike) -> Standardizer:
        Xa = np.asarray(X, dtype=np.float64)
        self.mean_ = Xa.mean(axis=0)
        std = Xa.std(axis=0)
        self.scale_ = np.where(std > 0, std, 1.0)
        return self

    def transform(self, X: npt.ArrayLike) -> FloatArray:
        return (np.asarray(X, dtype=np.float64) - self.mean_) / self.scale_

    def fit_transform(self, X: npt.ArrayLike) -> FloatArray:
        return self.fit(X).transform(X)


@dataclass
class EqualFrequencyBinner:
    """Quantile binning into at most ``n_bins`` bins per column.

    Interior edges are the ``1/n_bins, ..., (n_bins-1)/n_bins`` quantiles of the training column,
    deduplicated, so a column with few distinct values gets fewer bins. Bin ``b`` holds values in
    ``(edge[b-1], edge[b]]``; values beyond the training range fall in the outer bins.
    """

    n_bins: int = 4
    edges_: list[FloatArray] = field(init=False, repr=False, default_factory=list)

    def fit(self, X: npt.ArrayLike) -> EqualFrequencyBinner:
        if self.n_bins < 2:
            raise ValueError("n_bins must be >= 2")
        Xa = np.asarray(X, dtype=np.float64)
        qs = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        self.edges_ = []
        for j in range(Xa.shape[1]):
            col = Xa[:, j]
            edges = np.unique(np.quantile(col, qs, method="lower"))
            # an edge at the column max would create an empty last bin
            self.edges_.append(edges[edges < col.max()])
        return self

    def transform(self, X: npt.ArrayLike) -> IntArray:
        Xa = np.asarray(X, dtype=np.float64)
        out = np.empty(Xa.shape, dtype=np.int64)
        for j, edges in enumerate(self.edges_):
            out[:, j] = np.searchsorted(edges, Xa[:, j], side="left")
        return out

    def fit_transform(self, X: npt.ArrayLike) -> IntArray:
        return self.fit(X).transform(X)
