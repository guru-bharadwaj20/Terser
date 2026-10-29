"""Choosing k and using clusters (plan rows **6.2**, **6.3**, **6.8**, **6.9**).

* :func:`k_sweep` - SSE (elbow) and silhouette for K-means and bisecting K-means over a k range,
  plus cluster-size balance; :func:`plot_k_sweep` draws ``reports/figures/p6_kmeans_k.png``.
* :func:`clusters_vs_categories` - ARI, NMI and the contingency table against the 6
  hand-defined categories.
* :func:`cluster_risk` - oracle savings and degradation rate per cluster; a cluster is
  *high risk* when its degradation rate is at least ``ratio`` (1.15) times the overall rate
  (checkpoint C6.3 as revised by ADR-0012).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

from terser.models.clustering.scratch_kmeans import BisectingKMeans, KMeans, size_balance

__all__ = ["cluster_risk", "clusters_vs_categories", "k_sweep", "plot_k_sweep"]


def k_sweep(
    X: npt.ArrayLike, ks: Iterable[int] = range(4, 31), seed: int = 0, n_init: int = 10
) -> pd.DataFrame:
    """One row per (algorithm, k): ``sse``, ``silhouette``, ``size_balance``."""
    Xa = np.asarray(X, dtype=np.float64)
    rows = []
    for k in ks:
        for name, model in (
            ("kmeans", KMeans(k, n_init=n_init, seed=seed)),
            ("bisecting", BisectingKMeans(k, seed=seed)),
        ):
            model.fit(Xa)
            labels = model.labels_
            rows.append(
                {
                    "algorithm": name,
                    "k": k,
                    "sse": float(model.inertia_),
                    "silhouette": float(silhouette_score(Xa, labels)),
                    "size_balance": size_balance(labels),
                }
            )
    return pd.DataFrame(rows)


def plot_k_sweep(sweep: pd.DataFrame, path: Path) -> Path:
    """Elbow (SSE) and silhouette panels for each algorithm, saved to ``path``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, grp in sweep.groupby("algorithm"):
        axes[0].plot(grp["k"], grp["sse"], marker="o", label=str(name))
        axes[1].plot(grp["k"], grp["silhouette"], marker="o", label=str(name))
    axes[0].set(xlabel="k", ylabel="SSE", title="Elbow")
    axes[1].set(xlabel="k", ylabel="silhouette", title="Silhouette")
    for ax in axes:
        ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def clusters_vs_categories(
    clusters: npt.ArrayLike, categories: npt.ArrayLike
) -> tuple[float, float, pd.DataFrame]:
    """``(ARI, NMI, contingency)``; contingency rows = clusters, columns = categories."""
    c, g = np.asarray(clusters), np.asarray(categories)
    table = pd.crosstab(pd.Series(c, name="cluster"), pd.Series(g, name="category"))
    return float(adjusted_rand_score(g, c)), float(normalized_mutual_info_score(g, c)), table


def cluster_risk(
    clusters: npt.ArrayLike,
    savings: npt.ArrayLike,
    degraded: npt.ArrayLike,
    ratio: float = 1.15,
) -> pd.DataFrame:
    """Per-cluster ``n``, median oracle savings, degradation rate, rate / overall, high-risk flag.

    ``savings`` is the oracle savings fraction per prompt, ``degraded`` a 0/1 label.
    """
    df = pd.DataFrame(
        {
            "cluster": np.asarray(clusters),
            "savings": np.asarray(savings, dtype=float),
            "degraded": np.asarray(degraded, dtype=float),
        }
    )
    if not df["degraded"].isin([0.0, 1.0]).all():
        raise ValueError("degraded must be 0/1")
    overall = float(df["degraded"].mean())
    out = df.groupby("cluster").agg(
        n=("savings", "size"),
        median_savings=("savings", "median"),
        degradation_rate=("degraded", "mean"),
    )
    out["risk_ratio"] = out["degradation_rate"] / overall if overall > 0 else 0.0
    out["high_risk"] = out["risk_ratio"] >= ratio
    return out.sort_values("risk_ratio", ascending=False)
