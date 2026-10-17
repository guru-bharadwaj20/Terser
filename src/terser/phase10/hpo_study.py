"""Hyperparameter-search method study (rows **10.2**, **10.5**; checkpoints **C10.1**, **C10.2**).

* ``run --model gbm --method {grid,random,tpe,gp}``: LightGBM quality gate on the row 3.1.3 pair
  table, objective = grouped 5-fold OOF ROC-AUC (row 3.2.4). 60 trials x 5 repeats (seeds) per
  method; the grid has 4 points per dimension over 3 dimensions (64 points, first 60 evaluated
  in the sampler's order).
* ``run --model mlp --method {random,tpe}``: the Phase 4 PyTorch MLP on the same features (depth
  1-3, width 64-512, dropout, lr, weight decay).
* ``report``: best-so-far curves (mean ± sd over repeats) in
  ``reports/figures/p10_hpo_methods.png`` and ``reports/p10_hpo.md``, with the untuned
  baseline for C10.2; the same table as numbers in ``reports/p10_hpo.csv``.

Every trial and best-so-far curve goes to MLflow (``terser.hpo.search.run_search(mlflow=True)``);
results also land in ``reports/p10_hpo/<model>_<method>.json``. Train split only.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from terser.eval.curves import roc_auc
from terser.eval.cv import out_of_fold_scores
from terser.hpo.search import Float, Int, SearchSpace, run_search

__all__ = ["GBM_SPACE", "MLP_SPACE", "main"]

GBM_SPACE = SearchSpace(
    {
        "learning_rate": Float(0.01, 0.3, log=True),
        "num_leaves": Int(4, 64, log=True),
        "min_child_samples": Int(5, 200, log=True),
    }
)
MLP_SPACE = SearchSpace(
    {
        "depth": Int(1, 3),
        "width": Int(64, 512, log=True),
        "dropout": Float(0.0, 0.5),
        "lr": Float(1e-4, 1e-2, log=True),
        "weight_decay": Float(1e-6, 1e-2, log=True),
    }
)
UNTUNED_GBM = {"learning_rate": 0.1, "num_leaves": 31, "min_child_samples": 20}


def _data(table: Path) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    df = pd.read_parquet(table)
    cols = [c for c in df.columns if c not in ("id", "group", "category", "label")]
    return (
        df[cols].to_numpy(np.float64),
        df.label.to_numpy(),
        df.category.astype(str).tolist(),
        df.group.astype(str).tolist(),
    )


def gbm_objective(
    X: np.ndarray, y: np.ndarray, cats: list[str], groups: list[str], seed: int
) -> Callable[[dict[str, Any]], float]:
    import lightgbm as lgb

    def objective(p: dict[str, Any]) -> float:
        def fp(Xt: np.ndarray, yt: np.ndarray, Xv: np.ndarray) -> np.ndarray:
            m = lgb.LGBMClassifier(
                n_estimators=150,
                learning_rate=p["learning_rate"],
                num_leaves=int(p["num_leaves"]),
                min_child_samples=int(p["min_child_samples"]),
                random_state=seed,
                verbose=-1,
                n_jobs=2,
            )
            return np.asarray(m.fit(Xt, yt).predict_proba(Xv))[:, 1]

        return roc_auc(y, out_of_fold_scores(fp, X, y, cats, groups, seed=seed).scores)

    return objective


def mlp_objective(
    X: np.ndarray, y: np.ndarray, cats: list[str], groups: list[str], seed: int
) -> Callable[[dict[str, Any]], float]:
    import torch
    from torch import nn

    torch.set_num_threads(2)

    def objective(p: dict[str, Any]) -> float:
        def fp(Xt: np.ndarray, yt: np.ndarray, Xv: np.ndarray) -> np.ndarray:
            torch.manual_seed(seed)
            mu, sd = Xt.mean(0), Xt.std(0) + 1e-12
            layers: list[nn.Module] = []
            d = Xt.shape[1]
            for _ in range(int(p["depth"])):
                layers += [nn.Linear(d, int(p["width"])), nn.ReLU(), nn.Dropout(p["dropout"])]
                d = int(p["width"])
            net = nn.Sequential(*layers, nn.Linear(d, 1))
            opt = torch.optim.AdamW(net.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
            xt = torch.tensor((Xt - mu) / sd, dtype=torch.float32)
            ytt = torch.tensor(yt, dtype=torch.float32)
            loss_fn = nn.BCEWithLogitsLoss()
            g = torch.Generator().manual_seed(seed)
            for _ in range(15):
                net.train()
                for b in torch.randperm(len(xt), generator=g).split(128):
                    opt.zero_grad()
                    loss_fn(net(xt[b]).squeeze(-1), ytt[b]).backward()
                    opt.step()
            net.eval()
            with torch.no_grad():
                xv = torch.tensor((Xv - mu) / sd, dtype=torch.float32)
                return torch.sigmoid(net(xv).squeeze(-1)).numpy().astype(np.float64)

        return roc_auc(y, out_of_fold_scores(fp, X, y, cats, groups, seed=seed).scores)

    return objective


def cmd_run(args: argparse.Namespace) -> int:
    X, y, cats, groups = _data(args.table)
    make, space = (gbm_objective, GBM_SPACE) if args.model == "gbm" else (mlp_objective, MLP_SPACE)
    out = {"model": args.model, "method": args.method, "repeats": []}
    for rep in range(args.repeats):
        res = run_search(
            make(X, y, cats, groups, rep),
            space,
            args.method,
            args.trials,
            seed=rep,
            grid_points=4,
            mlflow=not args.no_mlflow,
            mlflow_tags={
                "phase": 10,
                "row": "10.2",
                "task": f"hpo_{args.model}",
                "run_name": f"10-hpo-{args.model}-{args.method}-{rep}",
            },
        )
        out["repeats"].append(
            {
                "seed": rep,
                "best_value": res.best_value,
                "best_params": res.best_params,
                "best_so_far": res.best_so_far,
                "values": res.values,
            }
        )
        print(json.dumps({"rep": rep, "best": res.best_value}), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.model}_{args.method}.json").write_text(
        json.dumps(out, default=str), encoding="utf-8"
    )
    return 0


def _cell(v: object) -> str:
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def cmd_report(args: argparse.Namespace) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    X, y, cats, groups = _data(args.table)
    untuned = [gbm_objective(X, y, cats, groups, s)(UNTUNED_GBM) for s in range(5)]
    base = float(np.mean(untuned))
    rows, fig = [], plt.figure(figsize=(11, 4))
    for k, model in enumerate(("gbm", "mlp")):
        ax = fig.add_subplot(1, 2, k + 1)
        for f in sorted(args.out.glob(f"{model}_*.json")):
            r = json.loads(f.read_text(encoding="utf-8"))
            curves = np.array([rep["best_so_far"] for rep in r["repeats"]], dtype=float)
            m, s = curves.mean(0), curves.std(0)
            x = np.arange(1, len(m) + 1)
            ax.plot(x, m, label=r["method"])
            ax.fill_between(x, m - s, m + s, alpha=0.15)
            bests = [rep["best_value"] for rep in r["repeats"]]
            at20 = curves[:, min(19, curves.shape[1] - 1)]
            rows.append(
                {
                    "model": model,
                    "method": r["method"],
                    "trials": curves.shape[1],
                    "best_auc": float(np.mean(bests)),
                    "best_auc_sd": float(np.std(bests)),
                    "best_at_20": float(at20.mean()),
                    "best_params (seed 0)": r["repeats"][0]["best_params"],
                }
            )
        if model == "gbm":
            ax.axhline(base, ls="--", c="grey", label="untuned")
        ax.set(title=f"{model} quality gate", xlabel="trial", ylabel="best-so-far grouped AUC")
        ax.legend(fontsize=8)
    fig.tight_layout()
    reports = args.out.parent  # --out reports/p10_hpo -> reports/
    figs = reports / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    fig.savefig(figs / "p10_hpo_methods.png", dpi=150)
    plt.close(fig)
    df = pd.DataFrame(rows)
    tuned = float(df.loc[df.model == "gbm", "best_auc"].max())
    df.to_csv(reports / "p10_hpo.csv", index=False)
    lines = [
        "# Hyperparameter-search methods (rows 10.2, 10.5; C10.1, C10.2)",
        "",
        "Generated by `python -m terser.phase10.hpo_study run ...` / `report`. Objective: grouped "
        "5-fold OOF ROC-AUC of the quality gate on the row 3.1.3 pair table (train split only). "
        "Each method: 60 trials x 5 repeats (seeds 0-4), every trial logged to MLflow "
        "(experiment tags phase 10, row 10.2).",
        "",
        "![methods](figures/p10_hpo_methods.png)",
        "",
        "| " + " | ".join(df.columns) + " |",
        "|---" * len(df.columns) + "|",
        *("| " + " | ".join(_cell(v) for v in r) + " |" for r in df.itertuples(index=False)),
        "",
        f"Untuned LightGBM (lr 0.1, 31 leaves, min_child 20, 150 trees): AUC **{base:.4f}**. "
        f"Best tuned: **{tuned:.4f}** (gain {tuned - base:+.4f}). C10.2 (>= +0.01 or a "
        f"documented plateau): **{'pass' if tuned - base >= 0.01 else 'plateau'}**.",
        "",
    ]
    (reports / "p10_hpo.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"untuned": base, "tuned": tuned}))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 10 HPO method study.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", choices=("gbm", "mlp"), required=True)
    r.add_argument("--method", choices=("grid", "random", "tpe", "gp"), required=True)
    r.add_argument("--trials", type=int, default=60)
    r.add_argument("--repeats", type=int, default=5)
    r.add_argument("--no-mlflow", action="store_true")
    rep = sub.add_parser("report")
    for p in (r, rep):
        p.add_argument("--table", type=Path, default=Path("data/processed/p4_pairs.parquet"))
        p.add_argument("--out", type=Path, default=Path("reports/p10_hpo"))
    args = ap.parse_args(argv)
    return cmd_run(args) if args.cmd == "run" else cmd_report(args)


if __name__ == "__main__":
    raise SystemExit(main())
