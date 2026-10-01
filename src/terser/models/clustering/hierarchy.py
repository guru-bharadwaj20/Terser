"""Agglomerative and divisive hierarchical clustering (plan rows **6.5**, **6.6**).

* **Agglomerative** (bottom-up): every point starts as its own cluster and the two closest
  clusters merge until one remains. "Closest" is the linkage: Ward (smallest SSE increase),
  average (mean pairwise distance) or complete (largest pairwise distance). Delegated to
  ``scipy.cluster.hierarchy.linkage``; cost O(n^2) memory, so row 6.5 uses a 2,000-prompt sample.
* **Divisive** (top-down, :func:`divisive_linkage`): start with one cluster and recursively
  bisect each cluster with 2-means (the bisecting K-means split of row 6.3) until singletons.
  The split history is written as a scipy linkage matrix whose merge height is the SSE of the
  merged cluster. SSE never increases from a parent to a child, so heights are monotone and
  the matrix can be drawn as a dendrogram and compared with :func:`cophenetic_correlation`.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.cluster.hierarchy import cophenet, fcluster, linkage
from scipy.spatial.distance import pdist

from terser.models.clustering.scratch_kmeans import KMeans

__all__ = [
    "LINKAGES",
    "agglomerative_linkage",
    "cophenetic_correlation",
    "cut_tree",
    "divisive_linkage",
]

FloatArray = npt.NDArray[np.float64]
LINKAGES = ("ward", "average", "complete")


def agglomerative_linkage(X: npt.ArrayLike, method: str = "ward") -> FloatArray:
    """Scipy linkage matrix (n-1, 4) for ``method`` in :data:`LINKAGES` (Euclidean)."""
    if method not in LINKAGES:
        raise ValueError(f"method must be one of {LINKAGES}, got {method!r}")
    return np.asarray(linkage(np.asarray(X, dtype=np.float64), method=method), dtype=np.float64)


def _cluster_sse(X: FloatArray) -> float:
    return float(((X - X.mean(0)) ** 2).sum())


def _bisect(X: FloatArray, seed: int) -> npt.NDArray[np.bool_]:
    """Boolean mask of the right half of a 2-means split; falls back to a halving on ties."""
    n = len(X)
    if n == 2:
        return np.array([False, True])
    km = KMeans(2, n_init=3, seed=seed).fit(X)
    mask = km.labels_ == 1
    if mask.all() or not mask.any():  # identical points: split arbitrarily
        mask = np.zeros(n, dtype=bool)
        mask[n // 2 :] = True
    return mask


def divisive_linkage(X: npt.ArrayLike, seed: int = 0) -> FloatArray:
    """Top-down recursive bisection to singletons, as a scipy linkage matrix.

    Row ``i`` merges nodes ``a`` and ``b`` at height ``SSE(a ∪ b)`` into node ``n + i``; rows are
    sorted by height (ties broken by size) so the result is a valid, monotone linkage.
    """
    Xa = np.asarray(X, dtype=np.float64)
    n = len(Xa)
    if n < 2:
        raise ValueError("need at least 2 samples")
    # Each internal node: (height, size, left_child, right_child); children are leaf ids
    # ("L", i) or internal ids ("N", j) resolved after sorting.
    nodes: list[tuple[float, int, tuple[str, int], tuple[str, int]]] = []

    def build(idx: npt.NDArray[np.int64], depth: int) -> tuple[str, int]:
        if len(idx) == 1:
            return ("L", int(idx[0]))
        mask = _bisect(Xa[idx], seed + depth * 7919 + int(idx[0]))
        left = build(idx[~mask], depth + 1)
        right = build(idx[mask], depth + 1)
        nodes.append((_cluster_sse(Xa[idx]), len(idx), left, right))
        return ("N", len(nodes) - 1)

    build(np.arange(n, dtype=np.int64), 0)
    order = sorted(range(len(nodes)), key=lambda j: (nodes[j][0], nodes[j][1]))
    new_id = {old: n + pos for pos, old in enumerate(order)}

    def ref(child: tuple[str, int]) -> int:
        return child[1] if child[0] == "L" else new_id[child[1]]

    Z = np.empty((n - 1, 4), dtype=np.float64)
    for pos, old in enumerate(order):
        h, size, left, right = nodes[old]
        Z[pos] = (ref(left), ref(right), h, size)
    return Z


def cophenetic_correlation(Z: npt.ArrayLike, X: npt.ArrayLike) -> float:
    """Pearson correlation between tree (cophenetic) distances and Euclidean distances."""
    c, _ = cophenet(np.asarray(Z, dtype=np.float64), pdist(np.asarray(X, dtype=np.float64)))
    return float(c)


def cut_tree(Z: npt.ArrayLike, k: int) -> npt.NDArray[np.int64]:
    """Flat labels ``0..k-1`` from cutting a linkage matrix into ``k`` clusters."""
    labels = fcluster(np.asarray(Z, dtype=np.float64), t=k, criterion="maxclust")
    return np.asarray(labels, dtype=np.int64) - 1
