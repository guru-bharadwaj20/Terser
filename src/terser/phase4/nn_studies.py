"""Activation (row **4.2.1**) and optimizer (row **4.3.1**) studies on the NumPy MLP.

Both studies train :class:`~terser.models.neural.numpy_mlp.NumpyMLP` on one grouped train /
validation split of a feature table (:mod:`terser.phase4.tables`) over several seeds:

* :func:`activation_run` - one activation (step with straight-through estimator, sigmoid, tanh,
  ReLU) for a fixed number of epochs; records val AUC per epoch, epochs to plateau (first epoch
  within ``plateau_tol`` of the best val AUC), per-layer gradient norm averaged over training, and
  the saturated / dead hidden-unit fraction at the end.
* :func:`optimizer_run` - one optimizer configuration at a fixed **gradient-evaluation budget**
  (number of per-example gradients, so full-batch GD takes ``budget / n`` steps and mini-batch SGD
  ``budget / 64``); records train loss and val AUC against gradient evaluations.

``python -m terser.phase4.nn_studies {activations,optimizers} --table <features> --out <dir>``
writes the per-run CSV, a markdown summary table (mean ± std over seeds) and, for optimizers, the
convergence figure. Interpretation goes into ``reports/p4_activations.md`` /
``reports/p4_optimizers.md`` by hand once the real tables exist.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from terser.eval.curves import roc_auc
from terser.eval.cv import grouped_folds
from terser.models.neural.numpy_mlp import Activation, NumpyMLP
from terser.models.neural.scratch_optim import make_optimizer

__all__ = [
    "ACTIVATIONS",
    "OPTIMIZER_GRID",
    "OptimizerConfig",
    "activation_run",
    "main",
    "optimizer_run",
    "summarise",
]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
ACTIVATIONS: tuple[Activation, ...] = ("step", "sigmoid", "tanh", "relu")


@dataclass(frozen=True)
class OptimizerConfig:
    name: str  # gd | sgd | momentum | adagrad | adam
    lr: float
    batch_size: int | None  # None = full batch
    beta: float | None = None  # momentum coefficient

    @property
    def label(self) -> str:
        extra = f", beta={self.beta}" if self.beta is not None else ""
        return f"{self.name}(lr={self.lr}{extra})"


def _grid() -> list[OptimizerConfig]:
    out = [OptimizerConfig("gd", lr, None) for lr in (0.1, 0.3, 1.0)]
    out += [OptimizerConfig("sgd", lr, 64) for lr in (0.01, 0.03, 0.1)]
    out += [OptimizerConfig("momentum", lr, 64, b) for b in (0.9, 0.99) for lr in (0.001, 0.01)]
    out += [OptimizerConfig("adagrad", lr, 64) for lr in (0.01, 0.03, 0.1)]
    out += [OptimizerConfig("adam", lr, 64) for lr in (1e-4, 3e-4, 1e-3)]
    return out


OPTIMIZER_GRID: tuple[OptimizerConfig, ...] = tuple(_grid())


def _safe_auc(y: IntArray, s: FloatArray) -> float:
    return float(roc_auc(y, s)) if len(np.unique(y)) == 2 else float("nan")


def activation_run(
    activation: Activation,
    X: FloatArray,
    y: npt.NDArray[np.int64],
    Xv: FloatArray,
    yv: npt.NDArray[np.int64],
    *,
    hidden: Sequence[int] = (64, 16),
    seed: int = 0,
    epochs: int = 30,
    lr: float = 0.05,
    batch_size: int = 64,
    plateau_tol: float = 0.005,
) -> dict[str, Any]:
    """Train one MLP with ``activation`` (mini-batch SGD); return the study metrics."""
    m = NumpyMLP((X.shape[1], *hidden, 1), activation=activation, seed=seed)
    rng = np.random.default_rng(seed)
    n_layers = len(m.weights)
    grad_norm_sum = np.zeros(n_layers)  # per-step L2 norms, summed
    steps = 0
    aucs: list[float] = []
    for _ in range(epochs):
        order = rng.permutation(len(y))
        for i in range(0, len(y), batch_size):
            b = order[i : i + batch_size]
            gws, gbs = m.backward(X[b], y[b])
            for k, (w, bias, gw, gb) in enumerate(zip(m.weights, m.biases, gws, gbs, strict=True)):
                grad_norm_sum[k] += float(np.sqrt((gw**2).sum() + (gb**2).sum()))
                np.subtract(w, lr * gw, out=w)  # in place on the model's arrays
                np.subtract(bias, lr * gb, out=bias)
            steps += 1
        aucs.append(_safe_auc(yv, m.predict_proba(Xv)))
    best = float(np.nanmax(aucs)) if not np.all(np.isnan(aucs)) else float("nan")
    plateau = next((i + 1 for i, a in enumerate(aucs) if a >= best - plateau_tol), epochs)
    stats = m.hidden_unit_stats(X)
    row: dict[str, Any] = {
        "activation": activation,
        "seed": seed,
        "val_auc": aucs[-1],
        "best_val_auc": best,
        "epochs_to_plateau": plateau,
        **stats,
    }
    for k in range(n_layers):
        row[f"grad_norm_layer{k + 1}"] = grad_norm_sum[k] / max(steps, 1)
    return row


def optimizer_run(
    cfg: OptimizerConfig,
    X: FloatArray,
    y: npt.NDArray[np.int64],
    Xv: FloatArray,
    yv: npt.NDArray[np.int64],
    *,
    budget: int,
    hidden: Sequence[int] = (64, 16),
    seed: int = 0,
    n_checkpoints: int = 20,
) -> pd.DataFrame:
    """Train under a per-example gradient budget; one row per checkpoint."""
    m = NumpyMLP((X.shape[1], *hidden, 1), activation="relu", seed=seed)
    kwargs = {"beta": cfg.beta} if cfg.beta is not None else {}
    opt = make_optimizer(cfg.name, cfg.lr, **kwargs)
    bs = cfg.batch_size or len(y)
    rng = np.random.default_rng(seed)
    params = [*m.weights, *m.biases]
    used, order, pos = 0, rng.permutation(len(y)), 0
    # Checkpoints over the budget this batch size can actually spend (whole batches only), so
    # every config gets all n_checkpoints rows and its last one at its final step.
    reach = (budget // bs) * bs
    marks = np.linspace(reach / n_checkpoints, reach, n_checkpoints).astype(int).tolist()
    next_marks = sorted(set(marks))
    rows: list[dict[str, Any]] = []
    while used + bs <= budget:
        if pos + bs > len(y):
            order, pos = rng.permutation(len(y)), 0
        b = order[pos : pos + bs]
        pos += bs
        gws, gbs = m.backward(X[b], y[b])
        opt.step(params, [*gws, *gbs])
        used += len(b)
        while next_marks and used >= next_marks[0]:
            next_marks.pop(0)
            rows.append(
                {
                    "optimizer": cfg.name,
                    "config": cfg.label,
                    "lr": cfg.lr,
                    "beta": cfg.beta,
                    "seed": seed,
                    "grad_evals": used,
                    "train_loss": m.loss(X, y),
                    "val_auc": _safe_auc(yv, m.predict_proba(Xv)),
                }
            )
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame, by: str, metrics: Sequence[str]) -> str:
    """Markdown table of ``mean ± std`` per ``by`` value."""
    lines = [f"| {by} | " + " | ".join(metrics) + " |", "|---" * (len(metrics) + 1) + "|"]
    for key, grp in df.groupby(by, sort=False):
        cells = [f"{grp[m].mean():.4f} ± {grp[m].std(ddof=1):.4f}" for m in metrics]
        lines.append(f"| {key} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _split(
    table_path: Path, features: Path | None, seed: int
) -> tuple[FloatArray, IntArray, FloatArray, IntArray]:
    from terser.phase4.tables import load_table

    t = load_table(table_path, features)
    tr, va = next(grouped_folds(t.y, t.categories, t.groups, seed=seed))
    mu, sd = t.X[tr].mean(0), t.X[tr].std(0) + 1e-12
    return (t.X[tr] - mu) / sd, t.y[tr], (t.X[va] - mu) / sd, t.y[va]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m terser.phase4.nn_studies", description=__doc__)
    ap.add_argument("study", choices=["activations", "optimizers"])
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--features", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("reports"))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args(argv)
    X, y, Xv, yv = _split(args.table, args.features, seed=0)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.study == "activations":
        df = pd.DataFrame(
            [
                activation_run(a, X, y, Xv, yv, seed=s, epochs=args.epochs)
                for a in ACTIVATIONS
                for s in range(args.seeds)
            ]
        )
        df.to_csv(args.out / "p4_activations_runs.csv", index=False)
        metrics = ["val_auc", "epochs_to_plateau", "saturated_fraction", "dead_fraction"]
        print(summarise(df, "activation", metrics))
        return 0
    budget = args.epochs * len(y)
    df = pd.concat(
        [
            optimizer_run(c, X, y, Xv, yv, budget=budget, seed=s)
            for c in OPTIMIZER_GRID
            for s in range(args.seeds)
        ],
        ignore_index=True,
    )
    df.to_csv(args.out / "p4_optimizers_runs.csv", index=False)
    final = df.sort_values("grad_evals").groupby(["config", "seed"]).tail(1)
    print(summarise(final, "config", ["train_loss", "val_auc"]))
    _plot_optimizers(df, args.out / "figures" / "p4_optimizers.png")
    return 0


def _plot_optimizers(df: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for cfg, grp in df.groupby("config", sort=False):
        mean = grp.groupby("grad_evals")[["train_loss", "val_auc"]].mean()
        axes[0].plot(mean.index, mean["train_loss"], label=cfg)
        axes[1].plot(mean.index, mean["val_auc"], label=cfg)
    axes[0].set(xlabel="gradient evaluations", ylabel="train BCE", yscale="log")
    axes[1].set(xlabel="gradient evaluations", ylabel="val ROC-AUC")
    axes[1].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
