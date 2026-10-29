"""Ensemble experiments for the quality gate (plan rows **5.1.1**, **5.1.3**, **5.1.5**, **5.1.6**).

* :func:`adaboost_vs_sklearn` (5.1.1): from-scratch AdaBoost vs scikit-learn's on one grouped
  split (AUC difference), training error per round for the plot, the hardest examples' ids.
* :func:`random_forest_grid` (5.1.3): ``n_estimators x max_features x bootstrap``; for bootstrap
  forests the OOB AUC beside the grouped-CV AUC (checkpoint C5.2 asks for |gap| <= 0.10, ADR-0012).
* :func:`lightgbm_grid` (5.1.5): ``learning_rate x n_estimators x max_depth`` with early stopping on
  a *grouped* validation fold carved from each training fold, so no source prompt leaks into the
  early-stopping set.
* :func:`feature_importance` (5.1.6): permutation importance (AUC drop on held-out data) and mean
  |SHAP| per task category. SHAP values come from LightGBM's built-in TreeSHAP
  (``predict(..., pred_contrib=True)``), so no extra dependency is needed.

``python -m terser.phase5.ensembles {adaboost,rf,lgbm,importance} --table <features> --out reports``
writes the tables (and ``figures/p5_adaboost_error.png`` / ``figures/p5_shap.png``).
"""

from __future__ import annotations

import argparse
import itertools
import sys
from collections.abc import Hashable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from terser.eval.curves import roc_auc
from terser.eval.cv import grouped_folds, out_of_fold_scores
from terser.models.ensembles.scratch_adaboost import AdaBoostStumps

__all__ = [
    "adaboost_vs_sklearn",
    "feature_importance",
    "lightgbm_grid",
    "main",
    "random_forest_grid",
]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def adaboost_vs_sklearn(
    X: FloatArray,
    y: IntArray,
    Xv: FloatArray,
    yv: IntArray,
    n_estimators: int = 200,
    k_hard: int = 10,
) -> dict[str, Any]:
    """Row 5.1.1: AUC of both implementations, training-error curve and hardest examples."""
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier

    ours = AdaBoostStumps(n_estimators=n_estimators).fit(X, y)
    sk = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=1), n_estimators=n_estimators, random_state=0
    ).fit(X, y)
    auc_ours = roc_auc(yv, ours.decision_function(Xv))
    auc_sk = roc_auc(yv, sk.decision_function(Xv))
    return {
        "auc_scratch": auc_ours,
        "auc_sklearn": auc_sk,
        "auc_diff": abs(auc_ours - auc_sk),
        "train_error": ours.train_error_,
        "hard_examples": ours.hard_examples(k_hard).tolist(),
        "model": ours,
    }


def random_forest_grid(
    X: FloatArray,
    y: IntArray,
    categories: Sequence[Hashable],
    groups: Sequence[Hashable],
    n_estimators: Sequence[int] = (100, 300),
    max_features: Sequence[str | float] = ("sqrt", 0.3),
    bootstrap: Sequence[bool] = (True, False),
    seed: int = 0,
) -> pd.DataFrame:
    """Row 5.1.3: grouped-CV AUC per configuration and OOB AUC when bootstrapping."""
    from sklearn.ensemble import RandomForestClassifier

    rows = []
    for n, mf, bs in itertools.product(n_estimators, max_features, bootstrap):
        params: dict[str, Any] = {
            "n_estimators": n,
            "max_features": mf,
            "bootstrap": bs,
            "random_state": seed,
            "n_jobs": 1,
        }

        def fp(Xt: FloatArray, yt: IntArray, Xq: FloatArray, p: dict[str, Any] = params) -> Any:
            return RandomForestClassifier(**p).fit(Xt, yt).predict_proba(Xq)[:, 1]

        oof = out_of_fold_scores(fp, X, y, categories, groups, seed=seed)
        cv_auc = roc_auc(y, oof.scores)
        oob_auc = float("nan")
        if bs:
            rf = RandomForestClassifier(oob_score=True, **params).fit(X, y)
            oob_auc = roc_auc(y, rf.oob_decision_function_[:, 1])
        rows.append(
            {
                "n_estimators": n,
                "max_features": mf,
                "bootstrap": bs,
                "cv_auc": cv_auc,
                "oob_auc": oob_auc,
                "oob_minus_cv": oob_auc - cv_auc,
            }
        )
    return pd.DataFrame(rows)


def _lgbm_fit(
    X: FloatArray,
    y: IntArray,
    categories: Sequence[Hashable],
    groups: Sequence[Hashable],
    params: dict[str, Any],
    seed: int,
) -> Any:
    """Fit LightGBM with early stopping on a grouped inner validation fold."""
    import lightgbm as lgb

    tr, es = next(grouped_folds(y, categories, groups, n_splits=5, seed=seed + 1))
    model = lgb.LGBMClassifier(random_state=seed, verbose=-1, n_jobs=1, **params)
    model.fit(
        X[tr],
        y[tr],
        eval_X=(X[es],),
        eval_y=(y[es],),
        eval_metric="auc",
        callbacks=[lgb.early_stopping(20, verbose=False)],
    )
    return model


def lightgbm_grid(
    X: FloatArray,
    y: IntArray,
    categories: Sequence[Hashable],
    groups: Sequence[Hashable],
    learning_rate: Sequence[float] = (0.03, 0.1),
    n_estimators: Sequence[int] = (200, 1000),
    max_depth: Sequence[int] = (3, 6, -1),
    seed: int = 0,
) -> pd.DataFrame:
    """Row 5.1.5: grouped-CV AUC per config; ``best_iteration`` averaged over outer folds."""
    rows = []
    cats, grps = list(categories), list(groups)
    for lr, n, d in itertools.product(learning_rate, n_estimators, max_depth):
        params = {"learning_rate": lr, "n_estimators": n, "max_depth": d}
        best_iters: list[int] = []
        scores = np.full(len(y), np.nan)
        for tr, va in grouped_folds(y, cats, grps, seed=seed):
            m = _lgbm_fit(X[tr], y[tr], [cats[i] for i in tr], [grps[i] for i in tr], params, seed)
            scores[va] = m.predict_proba(X[va])[:, 1]
            best_iters.append(int(m.best_iteration_ or n))
        rows.append({**params, "cv_auc": roc_auc(y, scores), "best_iteration": np.mean(best_iters)})
    return pd.DataFrame(rows)


def feature_importance(
    model: Any,
    Xv: FloatArray,
    yv: IntArray,
    categories: Sequence[Hashable],
    feature_names: Sequence[str],
    n_repeats: int = 5,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Row 5.1.6: permutation importance (AUC drop) and mean |SHAP| per category.

    Returns ``(permutation, shap)``: the first has one row per feature (mean and std AUC drop),
    the second one row per feature with one column per category plus ``all``.
    """
    from sklearn.inspection import permutation_importance

    perm = permutation_importance(
        model, Xv, yv, scoring="roc_auc", n_repeats=n_repeats, random_state=seed
    )
    perm_df = pd.DataFrame(
        {
            "feature": list(feature_names),
            "auc_drop_mean": perm.importances_mean,
            "auc_drop_std": perm.importances_std,
        }
    ).sort_values("auc_drop_mean", ascending=False, ignore_index=True)
    contrib = np.asarray(model.predict(Xv, pred_contrib=True))[:, :-1]  # last column: bias
    cats = np.asarray([str(c) for c in categories])
    shap = {"all": np.abs(contrib).mean(0)}
    for c in sorted(set(cats.tolist())):
        shap[c] = np.abs(contrib[cats == c]).mean(0)
    shap_df = pd.DataFrame(shap, index=list(feature_names)).sort_values("all", ascending=False)
    return perm_df, shap_df


def plot_shap(shap_df: pd.DataFrame, path: Path, top: int = 10) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    top_df = shap_df.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(top_df) + 1.5))
    cols = [c for c in top_df.columns if c != "all"]
    h = 0.8 / max(len(cols), 1)
    ypos = np.arange(len(top_df))
    for i, c in enumerate(cols):
        ax.barh(ypos + i * h, top_df[c], height=h, label=c)
    ax.set_yticks(ypos + 0.4 - h / 2, top_df.index)
    ax.set_xlabel("mean |SHAP| (log-odds)")
    ax.legend(fontsize=7)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_training_error(errors: Sequence[float], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(np.arange(1, len(errors) + 1), errors)
    ax.set(xlabel="boosting round", ylabel="training error", title="AdaBoost (depth-1 stumps)")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    from terser.phase4.tables import load_table

    ap = argparse.ArgumentParser(prog="python -m terser.phase5.ensembles", description=__doc__)
    ap.add_argument("study", choices=["adaboost", "rf", "lgbm", "importance"])
    ap.add_argument("--table", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("reports"))
    args = ap.parse_args(argv)
    t = load_table(args.table)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.study == "rf":
        df = random_forest_grid(t.X, t.y, t.categories, t.groups)
        df.to_csv(args.out / "p5_rf_grid.csv", index=False)
    elif args.study == "lgbm":
        df = lightgbm_grid(t.X, t.y, t.categories, t.groups)
        df.to_csv(args.out / "p5_lgbm_grid.csv", index=False)
    else:
        tr, va = next(grouped_folds(t.y, t.categories, t.groups, seed=0))
        if args.study == "adaboost":
            res = adaboost_vs_sklearn(t.X[tr], t.y[tr], t.X[va], t.y[va])
            plot_training_error(res["train_error"], args.out / "figures" / "p5_adaboost_error.png")
            hard = [t.ids[tr[i]] for i in res["hard_examples"]]
            df = pd.DataFrame(
                [{k: res[k] for k in ("auc_scratch", "auc_sklearn", "auc_diff")}
                 | {"hard_example_ids": " ".join(hard)}]  # fmt: skip
            )
            df.to_csv(args.out / "p5_adaboost.csv", index=False)
        else:
            cats_tr = [t.categories[i] for i in tr]
            grps_tr = [t.groups[i] for i in tr]
            model = _lgbm_fit(t.X[tr], t.y[tr], cats_tr, grps_tr, {"n_estimators": 1000}, 0)
            perm, shap = feature_importance(
                model, t.X[va], t.y[va], [t.categories[i] for i in va], t.feature_names
            )
            perm.to_csv(args.out / "p5_permutation_importance.csv", index=False)
            shap.to_csv(args.out / "p5_shap_by_category.csv")
            plot_shap(shap, args.out / "figures" / "p5_shap.png")
            df = perm.head(10)
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
