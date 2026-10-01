"""K-means as the zero-variance limit of EM (plan row **6.7**).

A GMM with equal weights ``pi_k = 1/K`` and a *shared, fixed* spherical covariance ``sigma^2 I``
has responsibilities

    r_ik = exp(-||x_i - mu_k||^2 / 2 sigma^2) / sum_j exp(-||x_i - mu_j||^2 / 2 sigma^2)

and M-step ``mu_k = sum_i r_ik x_i / sum_i r_ik``. As ``sigma -> 0`` the softmax becomes an
argmin over squared distances (hard assignment) and the M-step becomes the cluster mean: exactly
Lloyd's K-means. :class:`AnnealedSphericalEM` runs EM at a decreasing ``sigma`` schedule,
warm-starting each stage from the previous means. Full derivation:
``docs/derivations/kmeans_em.md``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp

__all__ = ["AnnealedSphericalEM", "assignment_agreement", "spherical_responsibilities"]

FloatArray = npt.NDArray[np.float64]


def spherical_responsibilities(X: FloatArray, means: FloatArray, sigma: float) -> FloatArray:
    """``r_ik`` for equal weights and shared covariance ``sigma^2 I`` (log-sum-exp stable)."""
    d2 = (X**2).sum(1)[:, None] - 2.0 * X @ means.T + (means**2).sum(1)[None, :]
    logits = -np.maximum(d2, 0.0) / (2.0 * sigma**2)
    return np.asarray(np.exp(logits - logsumexp(logits, axis=1, keepdims=True)))


@dataclass
class AnnealedSphericalEM:
    """EM for the equal-weight, shared-``sigma^2 I`` GMM over an annealing schedule.

    Attributes:
        sigmas: Decreasing ``sigma`` values; EM runs to convergence at each.
        max_iter: EM iterations per stage.
        tol: Stop a stage when no mean moves more than this (squared distance).
    """

    sigmas: Sequence[float] = (4.0, 2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.01)
    max_iter: int = 200
    tol: float = 1e-10
    means_: FloatArray = field(init=False, repr=False)
    stage_labels_: list[npt.NDArray[np.int64]] = field(init=False, default_factory=list)
    stage_max_resp_: list[float] = field(init=False, default_factory=list)

    def fit(self, X: npt.ArrayLike, means_init: npt.ArrayLike) -> AnnealedSphericalEM:
        """Anneal from ``means_init`` (K, d). Records hard labels and mean max-responsibility."""
        if any(b >= a for a, b in zip(self.sigmas[:-1], self.sigmas[1:], strict=False)):
            raise ValueError("sigmas must be strictly decreasing")
        Xa = np.asarray(X, dtype=np.float64)
        means = np.asarray(means_init, dtype=np.float64).copy()
        self.stage_labels_, self.stage_max_resp_ = [], []
        for sigma in self.sigmas:
            for _ in range(self.max_iter):
                r = spherical_responsibilities(Xa, means, sigma)
                nk = r.sum(0)
                new = np.where(nk[:, None] > 0, r.T @ Xa / np.maximum(nk, 1e-300)[:, None], means)
                shift = float(((new - means) ** 2).sum(1).max())
                means = new
                if shift <= self.tol:
                    break
            r = spherical_responsibilities(Xa, means, sigma)
            self.stage_labels_.append(r.argmax(1).astype(np.int64))
            self.stage_max_resp_.append(float(r.max(1).mean()))
        self.means_ = means
        return self


def assignment_agreement(a: npt.ArrayLike, b: npt.ArrayLike) -> float:
    """Share of points in the same cluster after the best one-to-one relabelling (Hungarian)."""
    la, lb = np.asarray(a), np.asarray(b)
    if la.shape != lb.shape:
        raise ValueError("label arrays differ in shape")
    ua, ia = np.unique(la, return_inverse=True)
    ub, ib = np.unique(lb, return_inverse=True)
    table = np.zeros((len(ua), len(ub)))
    np.add.at(table, (ia, ib), 1)
    rows, cols = linear_sum_assignment(-table)
    return float(table[rows, cols].sum() / len(la))
