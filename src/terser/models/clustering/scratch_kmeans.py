"""K-means and bisecting K-means from scratch (plan rows **6.2**, **6.3**).

**K-means** minimises the within-cluster sum of squares
``SSE = sum_i ||x_i - mu_{c(i)}||^2`` by Lloyd's alternation: assign each point to its nearest
centroid, then move each centroid to the mean of its points. Neither step can increase SSE, so
the objective is non-increasing and the algorithm stops at a local minimum. Initialisation is
**k-means++**: the first centroid is a uniform random point, every further one is drawn with
probability proportional to ``D(x)^2``, the squared distance to the nearest centroid so far.

**Bisecting K-means** is the top-down variant: start with one cluster and, until there are
``k``, split the cluster with the largest SSE by a 2-means run (best of ``n_trials``).
It returns clusters of more balanced size than flat K-means and its split history is a tree
(the divisive clustering of row 6.6 uses the same split).

Both are tested against ``sklearn.cluster.KMeans`` / ``BisectingKMeans``. Derivation of K-means
as the zero-variance limit of EM: ``docs/derivations/kmeans_em.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

__all__ = ["BisectingKMeans", "KMeans", "kmeans_plusplus", "size_balance", "sse"]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def _sq_dists(X: FloatArray, C: FloatArray) -> FloatArray:
    """Squared Euclidean distances, shape (n, k)."""
    d = (X**2).sum(1)[:, None] - 2.0 * X @ C.T + (C**2).sum(1)[None, :]
    return np.maximum(d, 0.0)


def sse(X: npt.ArrayLike, labels: npt.ArrayLike) -> float:
    """Within-cluster sum of squared distances to each cluster's mean."""
    Xa = np.asarray(X, dtype=np.float64)
    la = np.asarray(labels)
    total = 0.0
    for c in np.unique(la):
        pts = Xa[la == c]
        total += float(((pts - pts.mean(0)) ** 2).sum())
    return total


def size_balance(labels: npt.ArrayLike) -> float:
    """Normalised entropy of the cluster-size distribution: 1 = equal sizes, 0 = one cluster."""
    _, counts = np.unique(np.asarray(labels), return_counts=True)
    if len(counts) < 2:
        return 0.0
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum() / np.log(len(counts)))


def kmeans_plusplus(X: FloatArray, k: int, rng: np.random.Generator) -> FloatArray:
    """k-means++ seeding: sample centroids with probability proportional to ``D(x)^2``."""
    n = len(X)
    centers = [X[rng.integers(n)]]
    d2 = _sq_dists(X, np.asarray(centers))[:, 0]
    for _ in range(1, k):
        total = d2.sum()
        idx = int(rng.integers(n)) if total <= 0 else int(rng.choice(n, p=d2 / total))
        centers.append(X[idx])
        d2 = np.minimum(d2, _sq_dists(X, X[idx : idx + 1])[:, 0])
    return np.asarray(centers, dtype=np.float64)


@dataclass
class KMeans:
    """Lloyd's K-means with k-means++ seeding and ``n_init`` restarts (lowest SSE kept).

    Attributes:
        n_clusters: ``k``.
        n_init: Independent restarts.
        max_iter: Lloyd iterations per restart.
        tol: Stop when no centroid moves more than this (squared distance).
        seed: RNG seed.
    """

    n_clusters: int = 8
    n_init: int = 10
    max_iter: int = 300
    tol: float = 1e-10
    seed: int = 0
    cluster_centers_: FloatArray = field(init=False, repr=False)
    labels_: IntArray = field(init=False, repr=False)
    inertia_: float = field(init=False, default=float("inf"))
    sse_history_: list[float] = field(init=False, repr=False, default_factory=list)

    def _lloyd(
        self, X: FloatArray, centers: FloatArray
    ) -> tuple[FloatArray, IntArray, float, list[float]]:
        history: list[float] = []
        labels = np.zeros(len(X), dtype=np.int64)
        for _ in range(self.max_iter):
            d = _sq_dists(X, centers)
            labels = d.argmin(1).astype(np.int64)
            history.append(float(d[np.arange(len(X)), labels].sum()))
            new = centers.copy()
            for j in range(len(centers)):
                members = X[labels == j]
                if len(members):
                    new[j] = members.mean(0)
            shift = float(((new - centers) ** 2).sum(1).max())
            centers = new
            if shift <= self.tol:
                break
        d = _sq_dists(X, centers)
        labels = d.argmin(1).astype(np.int64)
        return centers, labels, float(d[np.arange(len(X)), labels].sum()), history

    def fit(self, X: npt.ArrayLike, init: npt.ArrayLike | None = None) -> KMeans:
        """Fit on ``X`` (n, d). ``init`` (k, d) fixes the starting centroids (one run)."""
        Xa = np.asarray(X, dtype=np.float64)
        if len(Xa) < self.n_clusters:
            raise ValueError(f"need at least {self.n_clusters} samples, got {len(Xa)}")
        rng = np.random.default_rng(self.seed)
        best: tuple[FloatArray, IntArray, float, list[float]] | None = None
        starts = (
            [np.asarray(init, dtype=np.float64)]
            if init is not None
            else [kmeans_plusplus(Xa, self.n_clusters, rng) for _ in range(self.n_init)]
        )
        for c0 in starts:
            run = self._lloyd(Xa, c0.copy())
            if best is None or run[2] < best[2]:
                best = run
        assert best is not None
        self.cluster_centers_, self.labels_, self.inertia_, self.sse_history_ = best
        return self

    def predict(self, X: npt.ArrayLike) -> IntArray:
        """Index of the nearest centroid for each row."""
        Xa = np.asarray(X, dtype=np.float64)
        return _sq_dists(Xa, self.cluster_centers_).argmin(1).astype(np.int64)


@dataclass
class BisectingKMeans:
    """Top-down K-means: repeatedly 2-means-split the cluster with the largest SSE.

    ``splits_`` records ``(parent_label, left_label, right_label)`` in split order; the parent
    keeps its label for the left child, the right child gets the next free label.
    """

    n_clusters: int = 8
    n_trials: int = 5
    max_iter: int = 300
    seed: int = 0
    labels_: IntArray = field(init=False, repr=False)
    cluster_centers_: FloatArray = field(init=False, repr=False)
    inertia_: float = field(init=False, default=float("inf"))
    splits_: list[tuple[int, int, int]] = field(init=False, repr=False, default_factory=list)

    def fit(self, X: npt.ArrayLike) -> BisectingKMeans:
        Xa = np.asarray(X, dtype=np.float64)
        if len(Xa) < self.n_clusters:
            raise ValueError(f"need at least {self.n_clusters} samples, got {len(Xa)}")
        labels = np.zeros(len(Xa), dtype=np.int64)
        self.splits_ = []
        for step in range(1, self.n_clusters):
            sses = {
                c: sse(Xa[labels == c], np.zeros(int((labels == c).sum()))) for c in range(step)
            }
            target = max((c for c in sses if (labels == c).sum() >= 2), key=lambda c: sses[c])
            idx = np.flatnonzero(labels == target)
            km = KMeans(2, n_init=self.n_trials, max_iter=self.max_iter, seed=self.seed + step)
            km.fit(Xa[idx])
            labels[idx[km.labels_ == 1]] = step
            self.splits_.append((target, target, step))
        self.labels_ = labels
        self.cluster_centers_ = np.vstack([Xa[labels == c].mean(0) for c in range(self.n_clusters)])
        self.inertia_ = sse(Xa, labels)
        return self
