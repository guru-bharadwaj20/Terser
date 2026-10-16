"""SVM figures for rows **4.4.5**, **4.4.6** and **4.4.8** on the real feature tables.

* 4.4.5 - hard margin on a linearly separable DROP-vs-KEEP span subset in the
  (``self_information``, ``stopword_ratio``) plane. The full span set is not separable, so the
  subset keeps the ``k`` tokens per class that a soft-margin linear SVM (C = 1, 400-token sample)
  scores most confidently, shrinking ``k`` until a near-hard (C = 1e6) fit separates them.
* 4.4.6 - margin width, support-vector count and val AUC against C from
  ``reports/p4_svm_soft_margin.csv`` (written by :mod:`terser.phase4.svm_studies`).
* 4.4.8 - linear vs polynomial surfaces on the Phase 3 plane (``max_si_removed`` x
  ``tokens_saved``, rows 3.3.5 / 3.5.3), 600 standardised train pairs.

CLI::

    python -m terser.phase4.svm_figures
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from terser.models.classical.scratch_svm import DualSVM
from terser.phase4.svm_studies import hard_margin_plot, plot_surfaces

__all__ = ["main"]


def _std(X: np.ndarray) -> np.ndarray:
    return (X - X.mean(0)) / (X.std(0) + 1e-12)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="SVM figures (rows 4.4.5, 4.4.6, 4.4.8).")
    ap.add_argument("--spans", type=Path, default=Path("data/processed/span_features.parquet"))
    ap.add_argument("--pairs", type=Path, default=Path("data/processed/pair_features.parquet"))
    ap.add_argument("--reports", type=Path, default=Path("reports"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    figs = args.reports / "figures"
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 4.4.5
    sp = pd.read_parquet(args.spans)
    sp = sp[sp.label.isin(["KEEP", "DROP"])]
    s = sp.iloc[rng.choice(len(sp), 400, replace=False)]
    X = _std(s[["self_information", "stopword_ratio"]].to_numpy(np.float64))
    y = np.where(s.label == "DROP", 1, -1)
    conf = y * DualSVM(C=1.0).fit(X, y).decision_function(X)
    for k in (60, 40, 25, 15, 8):
        keep = np.zeros(len(y), bool)
        for c in (-1, 1):
            idx = np.flatnonzero(y == c)
            keep[idx[np.argsort(-conf[idx])[:k]]] = True
        ref = DualSVM(C=1e6).fit(X[keep], y[keep])
        if np.all(y[keep] * ref.decision_function(X[keep]) > 0):
            break
    hard = hard_margin_plot(
        X[keep], y[keep], figs / "p4_svm_hard_margin.png", ("self_information", "stopword_ratio")
    )
    separable = bool(np.all(y[keep] * hard.decision_function(X[keep]) > 0))

    # 4.4.6
    sm = pd.read_csv(args.reports / "p4_svm_soft_margin.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, col in zip(axes, ("margin_width", "n_support", "val_auc"), strict=True):
        ax.plot(sm.C, sm[col], marker="o")
        ax.set(xscale="log", xlabel="C", title=col)
    fig.tight_layout()
    fig.savefig(figs / "p4_svm_soft_margin.png", dpi=150)
    plt.close(fig)

    # 4.4.8
    pf = pd.read_parquet(args.pairs)
    p = pf.iloc[rng.choice(len(pf), 600, replace=False)]
    Xp = _std(p[["max_si_removed", "tokens_saved"]].to_numpy(np.float64))
    yp = np.where(p.label == "degraded", 1, -1)
    plot_surfaces(
        Xp,
        yp,
        figs / "p4_svm_surfaces.png",
        C=1.0,
        degree=3,
        coef0=1.0,
        axis_labels=("max_si_removed (std)", "tokens_saved (std)"),
    )
    print(
        json.dumps(
            {
                "hard_margin_points": int(keep.sum()),
                "hard_margin_svs": len(hard.support_),
                "hard_margin_width": hard.margin_width,
                "separable": separable,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
