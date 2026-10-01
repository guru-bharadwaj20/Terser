"""Phase 6 clustering: K-means, bisecting, hierarchical, K-means-as-EM, analysis, embeddings."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.cluster import KMeans as SkKMeans

from terser.features.embeddings import EmbeddingCache
from terser.models.clustering.analysis import (
    cluster_risk,
    clusters_vs_categories,
    k_sweep,
    plot_k_sweep,
)
from terser.models.clustering.hierarchy import (
    agglomerative_linkage,
    cophenetic_correlation,
    cut_tree,
    divisive_linkage,
)
from terser.models.clustering.kmeans_em import (
    AnnealedSphericalEM,
    assignment_agreement,
    spherical_responsibilities,
)
from terser.models.clustering.scratch_kmeans import BisectingKMeans, KMeans, size_balance, sse


def _blobs(seed: int = 0, n: int = 60, k: int = 4, spread: float = 0.6) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = rng.uniform(-8, 8, (k, 3))
    return np.vstack([rng.normal(c, spread, (n, 3)) for c in centers])


def test_kmeans_matches_sklearn_from_same_init() -> None:
    X = _blobs()
    init = X[[0, 70, 130, 200]]
    ours = KMeans(4).fit(X, init=init)
    sk = SkKMeans(4, init=init, n_init=1, tol=0, algorithm="lloyd").fit(X)
    assert ours.inertia_ == pytest.approx(sk.inertia_, rel=1e-9)
    assert assignment_agreement(ours.labels_, sk.labels_) == 1.0


@pytest.mark.parametrize("seed", range(3))
def test_kmeans_sse_non_increasing(seed: int) -> None:
    X = _blobs(seed, spread=2.5)
    km = KMeans(6, n_init=1, seed=seed).fit(X)
    h = np.array(km.sse_history_)
    assert np.all(np.diff(h) <= 1e-9)
    assert km.inertia_ == pytest.approx(sse(X, km.labels_))
    np.testing.assert_array_equal(km.predict(X), km.labels_)


def test_kmeans_too_few_points() -> None:
    with pytest.raises(ValueError):
        KMeans(5).fit(np.zeros((3, 2)))


def test_bisecting_recovers_blobs_and_balance() -> None:
    X = _blobs()
    bk = BisectingKMeans(4, seed=0).fit(X)
    truth = np.repeat(np.arange(4), 60)
    assert assignment_agreement(bk.labels_, truth) == 1.0
    assert len(bk.splits_) == 3
    assert size_balance(bk.labels_) == pytest.approx(1.0)
    assert size_balance(np.zeros(5)) == 0.0


def test_agglomerative_and_divisive_trees() -> None:
    X = _blobs(n=25)
    truth = np.repeat(np.arange(4), 25)
    for method in ("ward", "average", "complete"):
        Z = agglomerative_linkage(X, method)
        assert assignment_agreement(cut_tree(Z, 4), truth) == 1.0
    with pytest.raises(ValueError):
        agglomerative_linkage(X, "single")
    Zd = divisive_linkage(X)
    assert Zd.shape == (99, 4)
    assert np.all(np.diff(Zd[:, 2]) >= 0)
    assert Zd[-1, 3] == 100
    assert assignment_agreement(cut_tree(Zd, 4), truth) == 1.0
    assert cophenetic_correlation(Zd, X) > 0.8
    assert cophenetic_correlation(agglomerative_linkage(X, "average"), X) > 0.8


def test_divisive_handles_duplicates() -> None:
    Z = divisive_linkage(np.zeros((5, 2)))
    assert Z.shape == (4, 4) and Z[-1, 3] == 5


def test_responsibilities_harden_as_sigma_shrinks() -> None:
    X = _blobs()
    means = X[[0, 70, 130, 200]]
    soft = spherical_responsibilities(X, means, 50.0)
    hard = spherical_responsibilities(X, means, 0.01)
    np.testing.assert_allclose(soft.sum(1), 1.0)
    assert soft.max(1).mean() < 0.5
    assert hard.max(1).min() > 1 - 1e-9


@pytest.mark.parametrize("seed", range(3))
def test_annealed_em_agrees_with_kmeans(seed: int) -> None:
    """Row 6.7 / C6.2 mechanism: >= 98% agreement at the smallest sigma (synthetic data)."""
    X = _blobs(seed, spread=1.5)
    init = X[np.random.default_rng(seed).choice(len(X), 4, replace=False)]
    em = AnnealedSphericalEM().fit(X, init)
    km = KMeans(4).fit(X, init=em.means_)
    assert assignment_agreement(em.stage_labels_[-1], km.labels_) >= 0.98
    assert em.stage_max_resp_[-1] > 0.999
    assert em.stage_max_resp_[0] < em.stage_max_resp_[-1]
    with pytest.raises(ValueError):
        AnnealedSphericalEM(sigmas=(1.0, 2.0)).fit(X, init)


def test_assignment_agreement_relabels() -> None:
    assert assignment_agreement([0, 0, 1, 1], [1, 1, 0, 0]) == 1.0
    assert assignment_agreement([0, 0, 1, 1], [0, 1, 0, 1]) == 0.5
    with pytest.raises(ValueError):
        assignment_agreement([0], [0, 1])


def test_k_sweep_and_plot(tmp_path: Path) -> None:
    X = _blobs(n=20)
    sweep = k_sweep(X, ks=[2, 4, 6], n_init=2)
    assert set(sweep["algorithm"]) == {"kmeans", "bisecting"}
    km = sweep[sweep.algorithm == "kmeans"].set_index("k")
    assert km.loc[4, "silhouette"] == km["silhouette"].max()
    out = plot_k_sweep(sweep, tmp_path / "fig.png")
    assert out.exists() and out.stat().st_size > 0


def test_clusters_vs_categories() -> None:
    ari, nmi, table = clusters_vs_categories([0, 0, 1, 1], ["a", "a", "b", "b"])
    assert ari == pytest.approx(1.0) and nmi == pytest.approx(1.0)
    assert table.shape == (2, 2)


def test_cluster_risk_flags_high_risk() -> None:
    clusters = [0] * 10 + [1] * 10
    degraded = [1] * 6 + [0] * 4 + [1] * 1 + [0] * 9
    risk = cluster_risk(clusters, np.linspace(0, 0.5, 20), degraded)
    assert risk.loc[0, "degradation_rate"] == pytest.approx(0.6)
    assert risk.loc[0, "risk_ratio"] == pytest.approx(0.6 / 0.35)
    assert bool(risk.loc[0, "high_risk"]) and not bool(risk.loc[1, "high_risk"])
    with pytest.raises(ValueError):
        cluster_risk([0], [0.1], [2])


def test_embedding_cache_roundtrip_and_missing(tmp_path: Path) -> None:
    def fake(texts: object) -> np.ndarray:
        return np.array([[len(t), 1.0] for t in texts], dtype=np.float32)  # type: ignore[attr-defined]

    cache = EmbeddingCache.build(["a", "b"], ["xx", "yyy"], fake)
    np.testing.assert_array_equal(cache.get(["b", "a"]), [[3, 1], [2, 1]])
    with pytest.raises(KeyError, match="missing-id"):
        cache.get(["a", "missing-id"])
    cache.save(tmp_path, "train")
    loaded = EmbeddingCache.load(tmp_path, "train")
    assert loaded.ids == ["a", "b"]
    with pytest.raises(ValueError):
        EmbeddingCache(["a", "a"], np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(ValueError):
        EmbeddingCache.build(["a"], ["x", "y"], fake)
