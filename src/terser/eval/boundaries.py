"""Decision-boundary figures on a shared 2-D plane (rows **3.3.5**, **3.5.3**, C3.7).

Every panel is drawn over the same axes and grid so tree, LR and KNN boundaries are directly
comparable. A panel is ``(title, fit_predict)`` where ``fit_predict(X_train_2d, y_train, grid)``
returns a degradation score in [0, 1] for each grid point; the 0.5 contour is drawn as the
boundary. For the PCA plane (3.5.3) build the 2-D coordinates with :func:`fold_pca_2d`, which
fits the projection on the training fold only.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt

from terser.eval.cv import FitPredict
from terser.features.continuous import Standardizer

__all__ = ["fold_pca_2d", "plot_boundaries"]

FloatArray = npt.NDArray[np.float64]


def fold_pca_2d(X_train: npt.ArrayLike, X_other: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Standardise and project onto the top-2 principal axes fitted on ``X_train`` only."""
    scaler = Standardizer().fit(X_train)
    Ztr = scaler.transform(X_train)
    _, _, vt = np.linalg.svd(Ztr, full_matrices=False)
    comps = vt[:2].T
    return Ztr @ comps, scaler.transform(X_other) @ comps


def plot_boundaries(
    panels: Sequence[tuple[str, FitPredict]],
    X2: npt.ArrayLike,
    y: npt.ArrayLike,
    path: str | Path,
    xlabel: str,
    ylabel: str,
    resolution: int = 200,
) -> Path:
    """One panel per model: filled score surface, 0.5 boundary, training points. Returns path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Xa = np.asarray(X2, dtype=np.float64)
    ya = np.asarray(y, dtype=np.int64)
    if Xa.ndim != 2 or Xa.shape[1] != 2:
        raise ValueError("X2 must have exactly two columns")
    pad = 0.05 * (Xa.max(axis=0) - Xa.min(axis=0) + 1e-9)
    lo, hi = Xa.min(axis=0) - pad, Xa.max(axis=0) + pad
    gx, gy = np.meshgrid(
        np.linspace(lo[0], hi[0], resolution), np.linspace(lo[1], hi[1], resolution)
    )
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False, sharex=True, sharey=True)
    for ax, (title, fit_predict) in zip(axes[0], panels, strict=True):
        z = np.asarray(fit_predict(Xa, ya, grid), dtype=np.float64).reshape(gx.shape)
        ax.contourf(gx, gy, z, levels=np.linspace(0, 1, 11), cmap="RdBu_r", alpha=0.35)
        ax.contour(gx, gy, z, levels=[0.5], colors="k", linewidths=1)
        ax.scatter(Xa[ya == 0, 0], Xa[ya == 0, 1], s=6, c="tab:blue", label="preserved")
        ax.scatter(Xa[ya == 1, 0], Xa[ya == 1, 1], s=6, c="tab:red", label="degraded")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
    axes[0][0].set_ylabel(ylabel)
    axes[0][0].legend(loc="best", fontsize=7)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
