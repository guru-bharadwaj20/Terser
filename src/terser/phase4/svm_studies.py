"""SVM studies on span / pair features (plan rows **4.4.5**-**4.4.8**), using :class:`DualSVM`.

* :func:`hard_margin_plot` (4.4.5): fit the hard-margin SVM on a linearly separable 2-D subset and
  plot points, decision line, both margins and the support vectors.
* :func:`soft_margin_sweep` (4.4.6): for each ``C`` the margin width ``2/||w||``, number of support
  vectors and validation AUC / F1.
* :func:`kernel_grid` (4.4.7): polynomial degree x coef0 x C vs the linear kernel: val AUC, fit
  time, number of SVs.
* :func:`plot_surfaces` (4.4.8): linear vs polynomial decision surfaces on a 2-D plane (the Phase 3
  plane of rows 3.3.5 / 3.5.3).

``python -m terser.phase4.svm_studies --table <features> --out reports`` runs 4.4.6 and 4.4.7 on a
grouped train / validation split and writes ``p4_svm_soft_margin.csv`` and
``p4_svm_kernels.csv``.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from terser.eval.curves import roc_auc
from terser.eval.metrics import classification_metrics
from terser.models.classical.scratch_svm import DualSVM

__all__ = [
    "C_GRID",
    "hard_margin_plot",
    "kernel_grid",
    "main",
    "plot_surfaces",
    "soft_margin_sweep",
]

FloatArray = npt.NDArray[np.float64]
C_GRID: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0, 100.0)


def soft_margin_sweep(
    X: FloatArray,
    y: npt.NDArray[np.int64],
    Xv: FloatArray,
    yv: npt.NDArray[np.int64],
    Cs: Sequence[float] = C_GRID,
) -> pd.DataFrame:
    """Row 4.4.6 table: one row per C."""
    rows = []
    for C in Cs:
        m = DualSVM(C=C).fit(X, y)
        s = m.decision_function(Xv)
        rows.append(
            {
                "C": C,
                "margin_width": m.margin_width,
                "n_support": len(m.support_),
                "val_auc": roc_auc(yv, s),
                "val_f1": classification_metrics(yv, (s > 0).astype(int)).f1,
            }
        )
    return pd.DataFrame(rows)


def kernel_grid(
    X: FloatArray,
    y: npt.NDArray[np.int64],
    Xv: FloatArray,
    yv: npt.NDArray[np.int64],
    degrees: Sequence[int] = (2, 3),
    coef0s: Sequence[float] = (0.0, 1.0),
    Cs: Sequence[float] = (0.1, 1.0, 10.0),
) -> pd.DataFrame:
    """Row 4.4.7 table: linear baseline per C plus every polynomial configuration."""
    configs: list[dict[str, Any]] = [{"kernel": "linear", "C": C} for C in Cs]
    configs += [
        {"kernel": "poly", "degree": d, "coef0": c0, "C": C}
        for d in degrees
        for c0 in coef0s
        for C in Cs
    ]
    rows = []
    for cfg in configs:
        t0 = time.perf_counter()
        m = DualSVM(**cfg).fit(X, y)
        fit_s = time.perf_counter() - t0
        rows.append(
            {
                **cfg,
                "val_auc": roc_auc(yv, m.decision_function(Xv)),
                "fit_seconds": fit_s,
                "n_support": len(m.support_),
            }
        )
    return pd.DataFrame(rows)


def _grid(X: FloatArray, n: int = 200) -> tuple[FloatArray, FloatArray, FloatArray]:
    pad = 0.1 * (X.max(0) - X.min(0))
    xs = np.linspace(X[:, 0].min() - pad[0], X[:, 0].max() + pad[0], n)
    ys = np.linspace(X[:, 1].min() - pad[1], X[:, 1].max() + pad[1], n)
    gx, gy = np.meshgrid(xs, ys)
    return gx, gy, np.c_[gx.ravel(), gy.ravel()]


def hard_margin_plot(
    X: FloatArray, y: npt.NDArray[np.int64], path: Path, axis_labels: Sequence[str] = ("x1", "x2")
) -> DualSVM:
    """Row 4.4.5: hard-margin fit on separable 2-D data, support vectors circled."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if X.shape[1] != 2:
        raise ValueError("hard_margin_plot needs 2-D inputs")
    m = DualSVM(C=None).fit(X, y)
    gx, gy, pts = _grid(X)
    f = m.decision_function(pts).reshape(gx.shape)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.contour(gx, gy, f, levels=[-1, 0, 1], linestyles=["--", "-", "--"], colors="k")
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=12)
    sv = X[m.support_]
    ax.scatter(sv[:, 0], sv[:, 1], s=90, facecolors="none", edgecolors="k", label="support vectors")
    ax.set(xlabel=axis_labels[0], ylabel=axis_labels[1], title=f"margin = {m.margin_width:.3f}")
    ax.legend(fontsize=7)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return m


def plot_surfaces(
    X: FloatArray,
    y: npt.NDArray[np.int64],
    path: Path,
    C: float = 1.0,
    degree: int = 3,
    coef0: float = 1.0,
    axis_labels: Sequence[str] = ("self_information", "token_cost"),
) -> None:
    """Row 4.4.8: linear vs polynomial decision surfaces side by side."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if X.shape[1] != 2:
        raise ValueError("plot_surfaces needs 2-D inputs")
    models = {
        "linear": DualSVM(C=C).fit(X, y),
        f"poly (d={degree}, coef0={coef0})": DualSVM(
            C=C, kernel="poly", degree=degree, coef0=coef0
        ).fit(X, y),
    }
    gx, gy, pts = _grid(X)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (name, m) in zip(axes, models.items(), strict=True):
        f = m.decision_function(pts).reshape(gx.shape)
        ax.contourf(gx, gy, f > 0, alpha=0.25, cmap="coolwarm")
        ax.contour(gx, gy, f, levels=[0], colors="k")
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", s=8)
        ax.set(title=f"{name}: {len(m.support_)} SVs", xlabel=axis_labels[0])
    axes[0].set_ylabel(axis_labels[1])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    from terser.eval.cv import grouped_folds
    from terser.phase4.tables import load_table

    ap = argparse.ArgumentParser(prog="python -m terser.phase4.svm_studies", description=__doc__)
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("reports"))
    ap.add_argument("--max-train", type=int, default=2000, help="QP is O(n^2) memory")
    args = ap.parse_args(argv)
    t = load_table(args.table)
    tr, va = next(grouped_folds(t.y, t.categories, t.groups, seed=0))
    tr = tr[: args.max_train]
    mu, sd = t.X[tr].mean(0), t.X[tr].std(0) + 1e-12
    X, Xv = (t.X[tr] - mu) / sd, (t.X[va] - mu) / sd
    args.out.mkdir(parents=True, exist_ok=True)
    soft = soft_margin_sweep(X, t.y[tr], Xv, t.y[va])
    soft.to_csv(args.out / "p4_svm_soft_margin.csv", index=False)
    kern = kernel_grid(X, t.y[tr], Xv, t.y[va])
    kern.to_csv(args.out / "p4_svm_kernels.csv", index=False)
    print(soft.to_string(index=False))
    print(kern.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
