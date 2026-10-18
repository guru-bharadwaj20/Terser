"""Phase 3 study runner: every ❌ experiment row of Phase 3 from one labelled feature table.

Usage::

    python -m terser.phase3.studies --data data/processed/pair_features.parquet --out reports

The table is the row-3.1.3 pair feature table built from **GPU judge labels** (Phase 2): one row
per (original, rewrite) with ``prompt_id`` (CV group), ``category``, ``label``
(``preserved``/``degraded`` or 0/1), numeric feature columns, and optionally ``savings_pct``
(realised savings of the prompt's oracle rewrite) with prompt-level ``prompt_*`` feature columns
for KNN regression (3.5.2, 3.5.4, squared-loss part of 3.6.1), and ``overhead_pct``.

Outputs (under ``--out``; figures in ``--out/figures``; best grid params under ``--configs``):

==========  =========================================  ==========================================
Row         Study                                      Artifact
==========  =========================================  ==========================================
3.2.5       naive KFold vs grouped CV (LR)             ``p3_cv_leakage.md``
3.2.7       leave-one-category-out (LR, tree, KNN)     ``p3_loco.md``
3.3.3       tree depth / min-samples-leaf sweep        ``figures/p3_tree_overfit.png``
3.3.4       pre / reduced-error / rule / CCP pruning   ``p3_tree_pruning.md``
3.3.5       tree boundary on two features              ``figures/p3_tree_boundary.png``
3.3.7       pruned tree as rules                       ``p3_tree_rules.md``
3.4.2       LR penalty x C grid                        ``p3_lr_grid.md``
3.4.3       LR reliability diagram                     ``figures/p3_lr_calibration.png``
3.4.5       odds ratios on standardised features       ``p3_lr_coefficients.md``
3.5.1       KNN k x metric x weighting grid            ``p3_knn_grid.md``
3.5.2       KNN regression of savings                  ``p3_knn_regression.md``
3.5.3       KNN k = 1, 5, 25 and LR on fold PCA plane  ``figures/p3_knn_boundaries.png``
3.5.4       skip-short-prompts net-savings curve       ``p3_skip_rule.md``
3.6.1       bootstrap bias-variance                    ``p3_bias_variance.md`` + figure
3.6.3       learning curves for tree, LR, KNN          ``figures/p3_learning_curves.png``
3.6.4       nested-CV grid search                      ``configs/model/p3_*.yaml``
3.6.5       baselines                                  a row in every table
==========  =========================================  ==========================================

Every score comes from the grouped protocol of row 3.2.4 unless the study says otherwise, and every
table carries per-category rows and the three baselines. Seeds default to 0, 1, 2 (mean ± std).
The runner refuses frozen held-out split paths (never tune on held-out data) and stand-in tables
(``stand_in`` column true) unless ``--allow-stand-in`` is given, in which case every report is
stamped ``STAND-IN``.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from terser.eval.baselines import BASELINES
from terser.eval.bias_variance import (
    bootstrap_predictions,
    decompose_squared,
    decompose_zero_one,
)
from terser.eval.boundaries import fold_pca_2d, plot_boundaries
from terser.eval.calibration import plot_reliability
from terser.eval.cv import FitPredict, grouped_folds, nested_cv_scores, out_of_fold_scores
from terser.eval.labels import _as_binary_array
from terser.eval.metrics import classification_metrics
from terser.eval.studies import (
    expand_grid,
    leakage_gap,
    learning_curve,
    leave_one_category_out,
    safe_auc,
    write_best_params,
)
from terser.features.continuous import Standardizer
from terser.models.classical.scratch_id3 import (
    DecisionTree,
    export_rules,
    rule_post_prune,
    tree_to_rules,
)
from terser.models.classical.scratch_knn import KNNClassifier, KNNRegressor
from terser.models.classical.scratch_penalized_lr import (
    PenalizedLogisticRegression,
    odds_ratio_table,
)
from terser.models.classical.skip_rule import best_threshold, net_savings_curve

__all__ = ["Grids", "StudyData", "load_table", "main", "run"]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
PROMPT_PREFIX = "prompt_"
META_COLUMNS = frozenset(
    {"id", "prompt_id", "category", "label", "savings_pct", "overhead_pct", "stand_in"}
)


# ---------------------------------------------------------------------------- data
@dataclass(frozen=True)
class StudyData:
    """Features, labels, categories, groups and provenance of the input table."""

    X: FloatArray
    y: IntArray
    categories: list[str]
    groups: list[str]
    features: list[str]
    savings: FloatArray | None
    overhead: FloatArray | None
    source: str
    sha256: str
    stand_in: bool


def load_table(path: Path, allow_stand_in: bool = False) -> StudyData:
    """Read a CSV / parquet / JSONL pair feature table.

    Raises:
        ValueError: On a test-split path, a stand-in table without ``allow_stand_in``, missing
            columns, or no numeric features.
    """
    parts = [p.lower() for p in path.resolve().parts]
    if any(p.startswith("test") for p in parts[-1:]) and "processed" in parts:
        raise ValueError("refusing to read a test split outside terser.eval.final")
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".jsonl":
        df = pd.read_json(path, lines=True)
    else:
        df = pd.read_csv(path)
    missing = {"prompt_id", "category", "label"} - set(df.columns)
    if missing:
        raise ValueError(f"table lacks columns {sorted(missing)}")
    stand_in = bool("stand_in" in df.columns and df["stand_in"].astype(bool).any())
    if stand_in and not allow_stand_in:
        raise ValueError("table contains stand-in labels; pass --allow-stand-in for smoke runs")
    feats = [
        c for c in df.columns if c not in META_COLUMNS and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not feats:
        raise ValueError("no numeric feature columns")
    savings = df["savings_pct"].to_numpy(np.float64) if "savings_pct" in df else None
    overhead = df["overhead_pct"].to_numpy(np.float64) if "overhead_pct" in df else None
    return StudyData(
        X=df[feats].to_numpy(np.float64),
        y=_as_binary_array(df["label"].tolist()),
        categories=df["category"].astype(str).tolist(),
        groups=df["prompt_id"].astype(str).tolist(),
        features=feats,
        savings=savings,
        overhead=overhead,
        source=path.as_posix(),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest()[:16],
        stand_in=stand_in,
    )


# -------------------------------------------------------------------------- models
def _proba(model: Any, X: FloatArray) -> FloatArray:
    p = model.predict_proba(X)
    if p.ndim == 1:
        return np.asarray(p, dtype=np.float64)
    if p.shape[1] == 1:
        return np.full(len(X), float(model.classes_[0]))
    return np.asarray(p[:, 1], dtype=np.float64)


def tree_fp(max_depth: int | None = None, min_samples_leaf: int = 1) -> FitPredict:
    def fp(Xtr: FloatArray, ytr: IntArray, Xva: FloatArray) -> FloatArray:
        m = DecisionTree("c45", max_depth=max_depth, min_samples_leaf=min_samples_leaf)
        return _proba(m.fit(Xtr, ytr), Xva)

    return fp


def lr_fp(penalty: str = "l2", C: float = 1.0, l1_ratio: float = 0.5) -> FitPredict:
    def fp(Xtr: FloatArray, ytr: IntArray, Xva: FloatArray) -> FloatArray:
        if len(np.unique(ytr)) < 2:
            return np.full(len(Xva), float(ytr[0]))
        s = Standardizer().fit(Xtr)
        m = PenalizedLogisticRegression(penalty=penalty, C=C, l1_ratio=l1_ratio)  # type: ignore[arg-type]
        return m.fit(s.transform(Xtr), ytr).predict_proba(s.transform(Xva))

    return fp


def knn_fp(k: int = 5, metric: str = "euclidean", weights: str = "uniform") -> FitPredict:
    def fp(Xtr: FloatArray, ytr: IntArray, Xva: FloatArray) -> FloatArray:
        s = Standardizer().fit(Xtr)
        m = KNNClassifier(k=min(k, len(ytr)), metric=metric, weights=weights)  # type: ignore[arg-type]
        return m.fit(s.transform(Xtr), ytr).predict_proba(s.transform(Xva))

    return fp


def _make(model: str) -> Callable[[dict[str, object]], FitPredict]:
    makers: dict[str, Callable[..., FitPredict]] = {"tree": tree_fp, "lr": lr_fp, "knn": knn_fp}
    return lambda params: makers[model](**params)


@dataclass(frozen=True)
class Grids:
    """Hyperparameter grids for 3.3.3, 3.4.2, 3.5.1, 3.6.1 and 3.6.4."""

    tree_depths: tuple[int | None, ...] = (1, 2, 3, 4, 6, 8, 10, 12, 15, 20, 25, 30)
    tree_min_leaf: tuple[int, ...] = (1, 5, 10, 25, 50)
    lr_penalties: tuple[str, ...] = ("l1", "l2", "elasticnet")
    lr_C: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
    knn_k: tuple[int, ...] = (1, 3, 5, 7, 11, 15, 25)
    knn_metrics: tuple[str, ...] = ("euclidean", "manhattan", "cosine")
    knn_weights: tuple[str, ...] = ("uniform", "distance")
    bv_depths: tuple[int | None, ...] = (2, 4, 8, 16, None)
    bv_k: tuple[int, ...] = (1, 3, 5, 15, 51)
    n_boot: int = 200
    grid_tree: dict[str, list[Any]] = field(
        default_factory=lambda: {"max_depth": [2, 4, 8, 16, None], "min_samples_leaf": [1, 10]}
    )

    @classmethod
    def quick(cls) -> Grids:
        """Tiny grids for CPU smoke tests."""
        return cls(
            tree_depths=(1, 3, None),
            tree_min_leaf=(1, 10),
            lr_penalties=("l2",),
            lr_C=(0.1, 1.0),
            knn_k=(1, 5),
            knn_metrics=("euclidean",),
            knn_weights=("uniform",),
            bv_depths=(2, None),
            bv_k=(1, 15),
            n_boot=5,
            grid_tree={"max_depth": [2, None], "min_samples_leaf": [1]},
        )


# --------------------------------------------------------------------- reporting
def _fmt(x: float) -> str:
    return "n/a" if x != x else f"{x:.3f}"  # NaN check


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        cells = [_fmt(v) if isinstance(v, float) else str(v) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _header(title: str, row: str, data: StudyData, seeds: Sequence[int]) -> str:
    stamp = "\n> **STAND-IN DATA**: smoke run only, not a result.\n" if data.stand_in else ""
    return (
        f"# {title}\n\nPlan row {row}. Generated by `python -m terser.phase3.studies`.{stamp}\n"
        f"- data: `{data.source}` (sha256 {data.sha256}), {len(data.y)} pairs, "
        f"{len(set(data.groups))} source prompts, degraded share {data.y.mean():.3f}\n"
        f"- seeds: {', '.join(map(str, seeds))}\n\n"
    )


def _mean_std(vals: Sequence[float]) -> str:
    a = np.asarray(vals, dtype=np.float64)
    return f"{np.nanmean(a):.3f} ± {np.nanstd(a):.3f}"


def _per_category_auc(
    fps: dict[str, FitPredict], data: StudyData, seeds: Sequence[int]
) -> pd.DataFrame:
    """Grouped-OOF AUC (mean ± std over seeds) overall and per category for each model."""
    cats = sorted(set(data.categories))
    cat_arr = np.asarray(data.categories)
    rows = []
    for name, fp in fps.items():
        per: dict[str, list[float]] = {c: [] for c in ["all", *cats]}
        for s in seeds:
            oof = out_of_fold_scores(fp, data.X, data.y, data.categories, data.groups, seed=s)
            per["all"].append(safe_auc(data.y, oof.scores))
            for c in cats:
                m = cat_arr == c
                per[c].append(safe_auc(data.y[m], oof.scores[m]))
        rows.append({"model": name, **{k: _mean_std(v) for k, v in per.items()}})
    return pd.DataFrame(rows)


def _baselines() -> dict[str, FitPredict]:
    return {f"baseline: {k}": make() for k, make in BASELINES.items()}


# ------------------------------------------------------------------------ studies
def study_leakage(data: StudyData, out: Path, seeds: Sequence[int]) -> dict[str, float]:
    gap = leakage_gap(lr_fp(), data.X, data.y, data.categories, data.groups, seeds=seeds)
    text = _header("Leakage demonstration: naive KFold vs grouped CV", "3.2.5", data, seeds)
    text += "Model: L2 logistic regression (C = 1) on standardised features. ROC-AUC.\n\n"
    text += _md_table(gap) + "\n\n"
    text += (
        f"Mean gap (naive - grouped): **{gap['gap'].mean():.3f} ± {gap['gap'].std(ddof=0):.3f}**."
        " A positive gap is optimism from paraphrases of one source prompt on both sides.\n\n"
    )
    text += "## Grouped-CV AUC per category\n\n"
    text += _md_table(_per_category_auc({"LR": lr_fp(), **_baselines()}, data, seeds)) + "\n"
    (out / "p3_cv_leakage.md").write_text(text, encoding="utf-8")
    return {"leakage_gap_auc": float(gap["gap"].mean())}


def study_loco(data: StudyData, out: Path, seeds: Sequence[int]) -> dict[str, float]:
    parts = []
    models = {"tree(depth 4)": tree_fp(4), "LR": lr_fp(), "KNN(k 15)": knn_fp(15)}
    for name, fp in {**models, **_baselines()}.items():
        t = leave_one_category_out(fp, data.X, data.y, data.categories)
        t.insert(0, "model", name)
        parts.append(t)
    table = pd.concat(parts, ignore_index=True)
    text = _header("Leave-one-category-out generalisation", "3.2.7", data, seeds)
    text += "Train on five categories, test on the held-out one (deterministic: no seeds).\n\n"
    text += _md_table(table) + "\n"
    (out / "p3_loco.md").write_text(text, encoding="utf-8")
    real = table[~table["model"].str.startswith("baseline")]
    return {"loco_min_auc": float(np.nanmin(real["auc"].to_numpy(np.float64)))}


def _holdout(data: StudyData, seed: int) -> tuple[IntArray, IntArray]:
    """First grouped fold as validation, the rest as training."""
    return next(grouped_folds(data.y, data.categories, data.groups, seed=seed))


def _f1(y: IntArray, pred: npt.ArrayLike) -> float:
    return float(classification_metrics(y, np.asarray(pred, dtype=np.int64)).f1)


def study_tree(
    data: StudyData, out: Path, figs: Path, seeds: Sequence[int], grids: Grids, xy: tuple[str, str]
) -> dict[str, float]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 3.3.3 overfitting sweep
    depth_rows, leaf_rows = [], []
    for s in seeds:
        tr, va = _holdout(data, s)
        for d in grids.tree_depths:
            m = DecisionTree("c45", max_depth=d).fit(data.X[tr], data.y[tr])
            depth_rows.append(
                (
                    s,
                    d if d is not None else 99,
                    _f1(data.y[tr], m.predict(data.X[tr])),
                    _f1(data.y[va], m.predict(data.X[va])),
                )
            )
        for leaf in grids.tree_min_leaf:
            m = DecisionTree("c45", min_samples_leaf=leaf).fit(data.X[tr], data.y[tr])
            leaf_rows.append(
                (
                    s,
                    leaf,
                    _f1(data.y[tr], m.predict(data.X[tr])),
                    _f1(data.y[va], m.predict(data.X[va])),
                )
            )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, sweep, xlabel in (
        (axes[0], depth_rows, "max depth"),
        (axes[1], leaf_rows, "min samples per leaf"),
    ):
        df = pd.DataFrame(sweep, columns=["seed", "x", "train", "val"]).groupby("x")
        mean, std = df.mean(), df.std(ddof=0)
        for col in ("train", "val"):
            ax.errorbar(mean.index, mean[col], yerr=std[col], marker="o", capsize=3, label=col)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("F1 (degraded)")
        ax.legend()
    fig.suptitle("Tree overfitting (row 3.3.3)")
    fig.tight_layout()
    fig.savefig(figs / "p3_tree_overfit.png", dpi=150)
    plt.close(fig)

    # 3.3.4 pruning comparison and 3.3.7 rules
    rows = []
    rules_text = ""
    for s in seeds:
        tr, va = _holdout(data, s)
        # pruning set: a grouped split of the training part, never the validation fold
        sub_cat = [data.categories[i] for i in tr]
        sub_grp = [data.groups[i] for i in tr]
        g_tr, g_pr = next(grouped_folds(data.y[tr], sub_cat, sub_grp, n_splits=3, seed=s))
        fit_i, prune_i = tr[g_tr], tr[g_pr]
        Xf, yf, Xp, yp = data.X[fit_i], data.y[fit_i], data.X[prune_i], data.y[prune_i]
        full = DecisionTree("c45").fit(Xf, yf)
        _alphas, trees = full.cost_complexity_path()
        ccp = max(trees, key=lambda t: _f1(yp, t.predict(Xp)))
        variants: dict[str, Any] = {
            "unpruned": full,
            "pre-pruning (depth 4, leaf 10)": DecisionTree(
                "c45", max_depth=4, min_samples_leaf=10
            ).fit(Xf, yf),
            "reduced-error": full.reduced_error_prune(Xp, yp),
            "rule post-pruning": rule_post_prune(full, Xp, yp),
            "cost-complexity": ccp,
        }
        for name, m in variants.items():
            leaves = len(m.rules) if hasattr(m, "rules") else m.n_leaves
            tr_f1, va_f1 = _f1(yf, m.predict(Xf)), _f1(data.y[va], m.predict(data.X[va]))
            rows.append(
                {
                    "seed": s,
                    "method": name,
                    "leaves/rules": leaves,
                    "train F1": tr_f1,
                    "val F1": va_f1,
                    "gap": tr_f1 - va_f1,
                }
            )
        for bname, make in BASELINES.items():
            fp = make()
            tr_f1 = _f1(yf, fp(Xf, yf, Xf) >= 0.5)
            va_f1 = _f1(data.y[va], fp(Xf, yf, data.X[va]) >= 0.5)
            rows.append(
                {
                    "seed": s,
                    "method": f"baseline: {bname}",
                    "leaves/rules": 0,
                    "train F1": tr_f1,
                    "val F1": va_f1,
                    "gap": tr_f1 - va_f1,
                }
            )
        if s == seeds[0]:
            rules = sorted(tree_to_rules(variants["reduced-error"]), key=lambda r: -r.coverage)
            rules_text = "\n".join(
                f"- `{line}`"
                for line in export_rules(rules, data.features, ["preserved", "degraded"])
            )
    table = (
        pd.DataFrame(rows).groupby("method", sort=False).agg(["mean", "std"]).drop(columns="seed")
    )
    table.columns = [f"{a} {b}" for a, b in table.columns]
    table = table.reset_index()
    text = _header("Tree overfitting solutions", "3.3.4", data, seeds)
    text += (
        "Validation = first grouped fold; pruning set = a grouped third of the remaining "
        "training data; trees are fit on the other two thirds.\n\n"
    )
    text += _md_table(table) + "\n"
    (out / "p3_tree_pruning.md").write_text(text, encoding="utf-8")
    rtext = _header("Pruned tree as rules", "3.3.7", data, seeds)
    rtext += "Reduced-error-pruned tree, seed 0, one rule per leaf, most-covered first.\n\n"
    (out / "p3_tree_rules.md").write_text(rtext + rules_text + "\n", encoding="utf-8")

    # 3.3.5 boundary on two named features
    ix, iy = data.features.index(xy[0]), data.features.index(xy[1])
    X2 = data.X[:, [ix, iy]]
    plot_boundaries(
        [("tree (depth 4)", tree_fp(4))], X2, data.y, figs / "p3_tree_boundary.png", xy[0], xy[1]
    )
    unpruned = table.loc[table["method"] == "unpruned", "gap mean"].iloc[0]
    pruned = (table["method"] != "unpruned") & ~table["method"].str.startswith("baseline")
    best_gap = table.loc[pruned, "gap mean"].min()
    return {"tree_gap_unpruned": float(unpruned), "tree_gap_best_pruned": float(best_gap)}


def study_lr(
    data: StudyData, out: Path, figs: Path, seeds: Sequence[int], grids: Grids
) -> dict[str, float]:
    fps = {f"LR {p} C={c}": lr_fp(p, c) for p in grids.lr_penalties for c in grids.lr_C}
    table = _per_category_auc({**fps, **_baselines()}, data, seeds)
    text = _header("Logistic regression regularisation grid", "3.4.2", data, seeds)
    text += "Grouped-OOF ROC-AUC, elastic net at l1_ratio 0.5.\n\n" + _md_table(table) + "\n"
    (out / "p3_lr_grid.md").write_text(text, encoding="utf-8")
    # 3.4.3 calibration on OOF probabilities of the default model
    oof = out_of_fold_scores(lr_fp(), data.X, data.y, data.categories, data.groups, seed=seeds[0])
    plot_reliability({"LR l2 C=1 (OOF)": (data.y, oof.scores)}, figs / "p3_lr_calibration.png")
    # 3.4.5 odds ratios from a full-data fit on standardised features
    s = Standardizer().fit(data.X)
    m = PenalizedLogisticRegression("l2", 1.0).fit(s.transform(data.X), data.y)
    ctext = _header("LR coefficients as odds ratios", "3.4.5", data, seeds)
    ctext += "L2, C = 1, fitted on all labelled pairs; odds ratio per +1 SD.\n\n"
    ctext += _md_table(odds_ratio_table(m.coef_, data.features)) + "\n"
    (out / "p3_lr_coefficients.md").write_text(ctext, encoding="utf-8")
    return {"lr_auc_default": safe_auc(data.y, oof.scores)}


def study_knn(
    data: StudyData, out: Path, figs: Path, seeds: Sequence[int], grids: Grids
) -> dict[str, float]:
    fps = {
        f"KNN k={k} {m} {w}": knn_fp(k, m, w)
        for k in grids.knn_k
        for m in grids.knn_metrics
        for w in grids.knn_weights
    }
    table = _per_category_auc({**fps, **_baselines()}, data, seeds)
    text = _header("KNN quality-gate grid", "3.5.1", data, seeds)
    text += "Grouped-OOF ROC-AUC on standardised features.\n\n" + _md_table(table) + "\n"
    (out / "p3_knn_grid.md").write_text(text, encoding="utf-8")

    # 3.5.3 boundaries on the fold PCA plane (fit on the training fold of seed 0)
    tr, _ = _holdout(data, seeds[0])
    Z, _ = fold_pca_2d(data.X[tr], data.X[tr])
    panels = [(f"KNN k={k}", knn_fp(k)) for k in (1, 5, 25)] + [("LR", lr_fp())]
    plot_boundaries(panels, Z, data.y[tr], figs / "p3_knn_boundaries.png", "PC1", "PC2")

    metrics: dict[str, float] = {}
    if data.savings is not None and _prompt_features(data):
        metrics.update(_study_knn_regression(data, out, seeds, grids))
    return metrics


def _prompt_features(data: StudyData) -> list[str]:
    return [f for f in data.features if f.startswith(PROMPT_PREFIX)]


def _per_prompt(data: StudyData) -> tuple[FloatArray, FloatArray, list[str]]:
    """Prompt-level features, savings target and category, one row per source prompt.

    Only ``prompt_*`` columns are used: pair features describe a particular rewrite (e.g.
    ``tokens_saved``) and would leak the savings target.
    """
    assert data.savings is not None
    cols = _prompt_features(data)
    df = pd.DataFrame(data.X[:, [data.features.index(c) for c in cols]], columns=cols)
    df["g"], df["c"], df["s"] = data.groups, data.categories, data.savings
    per = df.groupby("g", sort=True).first()
    return per[cols].to_numpy(np.float64), per["s"].to_numpy(np.float64), per["c"].tolist()


def _study_knn_regression(
    data: StudyData, out: Path, seeds: Sequence[int], grids: Grids
) -> dict[str, float]:
    X, yv, cats = _per_prompt(data)
    rows, preds_seed0 = [], np.zeros(len(yv))
    rng_cats = np.asarray(cats)
    for k in grids.knn_k:
        for s in seeds:
            pred = np.zeros(len(yv))
            folds = np.random.default_rng(s).permutation(len(yv)) % 5
            for f in range(5):
                tr, va = folds != f, folds == f
                sc = Standardizer().fit(X[tr])
                m = KNNRegressor(k=min(k, int(tr.sum())), weights="distance").fit(
                    sc.transform(X[tr]), yv[tr]
                )
                pred[va] = m.predict(sc.transform(X[va]))
            if k == grids.knn_k[0] and s == seeds[0]:
                preds_seed0 = pred.copy()
            for c in ["all", *sorted(set(cats))]:
                msk = np.ones(len(yv), bool) if c == "all" else rng_cats == c
                err = pred[msk] - yv[msk]
                ss = float(np.sum((yv[msk] - yv[msk].mean()) ** 2))
                rows.append(
                    {
                        "k": k,
                        "seed": s,
                        "category": c,
                        "MAE": float(np.mean(np.abs(err))),
                        "R2": 1 - float(np.sum(err**2)) / ss if ss > 0 else float("nan"),
                    }
                )
    t = (
        pd.DataFrame(rows)
        .groupby(["k", "category"])
        .agg(MAE=("MAE", "mean"), MAE_sd=("MAE", "std"), R2=("R2", "mean"), R2_sd=("R2", "std"))
        .reset_index()
    )
    base = np.mean(np.abs(yv - yv.mean()))
    text = _header("KNN regression of achievable savings", "3.5.2", data, seeds)
    text += (
        f"One row per source prompt ({len(yv)}), 5-fold CV by prompt, distance weighting. "
        f"Baseline (predict the mean): MAE {base:.3f}, R² 0.\n\n"
    )
    text += _md_table(t) + "\n"
    (out / "p3_knn_regression.md").write_text(text, encoding="utf-8")
    # 3.5.4 skip rule on OOF predictions
    overhead = 0.0
    if data.overhead is not None:
        overhead = float(pd.Series(data.overhead).groupby(data.groups).first().mean())
    curve = net_savings_curve(preds_seed0, yv, overhead)
    tau = best_threshold(curve)
    stext = _header("Skip-short-prompts rule", "3.5.4", data, seeds)
    stext += (
        f"Overhead per prompt: {overhead:.3f}. Best threshold on OOF predictions "
        f"(choose on validation only): {tau:.3f}.\n\n"
    )
    stext += (
        _md_table(
            pd.DataFrame(
                {
                    "threshold": curve.thresholds,
                    "net saving": curve.net_savings,
                    "rewrite rate": curve.rewrite_rate,
                }
            )
        )
        + "\n"
    )
    (out / "p3_skip_rule.md").write_text(stext, encoding="utf-8")
    best = t[t["category"] == "all"].sort_values("MAE").iloc[0]
    return {"knn_reg_mae": float(best["MAE"]), "knn_reg_r2": float(best["R2"])}


def study_bias_variance(
    data: StudyData, out: Path, figs: Path, seeds: Sequence[int], grids: Grids
) -> dict[str, float]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for s in seeds:
        tr, va = _holdout(data, s)

        def hard(fp: FitPredict) -> Callable[[FloatArray, FloatArray, FloatArray], FloatArray]:
            return lambda a, b, c: (fp(a, b.astype(np.int64), c) >= 0.5).astype(np.float64)

        for d in grids.bv_depths:
            P = bootstrap_predictions(
                hard(tree_fp(d)), data.X[tr], data.y[tr], data.X[va], grids.n_boot, s
            )
            bv = decompose_zero_one(P, data.y[va])
            rows.append(
                {"model": "tree", "param": d if d is not None else 99, "seed": s, **bv.__dict__}
            )
        for k in grids.bv_k:
            P = bootstrap_predictions(
                hard(knn_fp(k)), data.X[tr], data.y[tr], data.X[va], grids.n_boot, s
            )
            bv = decompose_zero_one(P, data.y[va])
            rows.append({"model": "knn", "param": k, "seed": s, **bv.__dict__})
        if data.savings is not None and _prompt_features(data):
            # squared loss: KNN regression of savings, one row per source prompt
            Xp, yp, _ = _per_prompt(data)
            perm = np.random.default_rng(s).permutation(len(yp))
            cut = int(0.8 * len(yp))
            ptr, pva = perm[:cut], perm[cut:]
            sc = Standardizer().fit(Xp[ptr])
            for k in grids.bv_k:

                def reg(a: FloatArray, b: FloatArray, c: FloatArray, k: int = k) -> FloatArray:
                    return KNNRegressor(k=min(k, len(b))).fit(a, b).predict(c)

                P = bootstrap_predictions(
                    reg, sc.transform(Xp[ptr]), yp[ptr], sc.transform(Xp[pva]), grids.n_boot, s
                )
                bv = decompose_squared(P, yp[pva])
                rows.append({"model": "knn_regression", "param": k, "seed": s, **bv.__dict__})
    df = pd.DataFrame(rows)
    agg = df.groupby(["model", "param"]).mean(numeric_only=True).drop(columns="seed").reset_index()
    panels = [
        ("tree", "max depth (99 = unlimited)", "0-1 loss"),
        ("knn", "k", "0-1 loss"),
        ("knn_regression", "k", "squared loss"),
    ]
    panels = [p for p in panels if (agg["model"] == p[0]).any()]
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4), squeeze=False)
    for ax, (model, xlabel, loss) in zip(axes[0], panels, strict=True):
        a = agg[agg["model"] == model]
        for col in ("loss", "bias", "variance"):
            ax.plot(a["param"], a[col], marker="o", label=col)
        ax.set_xlabel(xlabel)
        ax.set_title(f"{model}: {loss} decomposition")
        ax.legend()
    fig.tight_layout()
    fig.savefig(figs / "p3_bias_variance.png", dpi=150)
    plt.close(fig)
    text = _header("Bias-variance decomposition", "3.6.1", data, seeds)
    text += (
        f"{grids.n_boot} bootstrap resamples of the training part; evaluation = first "
        "grouped fold. 0-1 loss (Domingos): loss = bias + V_unbiased - V_biased. Squared loss "
        "(KNN regression of savings, 80/20 split of source prompts): loss = bias² + variance.\n\n"
    )
    text += _md_table(agg) + "\n\n"
    for model, *_ in panels:
        a = agg[agg["model"] == model]
        best = a.loc[a["loss"].idxmin()]
        text += f"- {model}: minimum expected loss {best['loss']:.3f} at param {best['param']}.\n"
    text += (
        "\nInterpretation (bias vs variance at the minimum) is written by the analyst from "
        "these numbers.\n"
    )
    (out / "p3_bias_variance.md").write_text(text, encoding="utf-8")
    return {}


def study_learning_curves(data: StudyData, figs: Path, seeds: Sequence[int]) -> dict[str, float]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    models = {"tree (depth 8)": tree_fp(8), "LR": lr_fp(), "KNN (k 15)": knn_fp(15)}
    for ax, (name, fp) in zip(axes, models.items(), strict=True):
        parts = [
            learning_curve(fp, data.X, data.y, data.categories, data.groups, seed=s) for s in seeds
        ]
        t = pd.concat(parts).groupby("fraction").mean().reset_index()
        ax.plot(t["n_train"], t["train_mean"], marker="o", label="train")
        ax.plot(t["n_train"], t["val_mean"], marker="o", label="validation")
        ax.set_title(name)
        ax.set_xlabel("training pairs")
    axes[0].set_ylabel("ROC-AUC")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(figs / "p3_learning_curves.png", dpi=150)
    plt.close(fig)
    return {}


def study_grid(
    data: StudyData, configs: Path, seeds: Sequence[int], grids: Grids
) -> dict[str, float]:
    spaces = {
        "tree": expand_grid(grids.grid_tree),
        "knn": expand_grid(
            {
                "k": list(grids.knn_k),
                "metric": list(grids.knn_metrics),
                "weights": list(grids.knn_weights),
            }
        ),
        "lr": expand_grid({"penalty": list(grids.lr_penalties), "C": list(grids.lr_C)}),
    }
    metrics = {}
    for model, grid in spaces.items():
        chosen: list[dict[str, object]] = []
        aucs = []
        for s in seeds:
            res = nested_cv_scores(
                _make(model), grid, safe_auc, data.X, data.y, data.categories, data.groups, seed=s
            )
            chosen.extend(res.best_params)
            aucs.append(safe_auc(data.y, res.scores))
        write_best_params(configs / f"p3_{model}.yaml", model, chosen)
        metrics[f"nested_auc_{model}"] = float(np.mean(aucs))
    return metrics


STUDIES = (
    "leakage",
    "loco",
    "tree",
    "lr",
    "knn",
    "knn_regression",
    "bias_variance",
    "learning_curves",
    "grid",
)


def run(
    data: StudyData,
    out: Path,
    configs: Path,
    studies: Sequence[str] = STUDIES,
    seeds: Sequence[int] = (0, 1, 2),
    grids: Grids | None = None,
    xy: tuple[str, str] = ("max_si_removed", "tokens_saved"),
) -> dict[str, float]:
    """Run the selected studies; return headline metrics."""
    g = grids or Grids()
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, float] = {}
    for name in studies:
        if name == "leakage":
            metrics.update(study_leakage(data, out, seeds))
        elif name == "loco":
            metrics.update(study_loco(data, out, seeds))
        elif name == "tree":
            metrics.update(study_tree(data, out, figs, seeds, g, xy))
        elif name == "lr":
            metrics.update(study_lr(data, out, figs, seeds, g))
        elif name == "knn":
            metrics.update(study_knn(data, out, figs, seeds, g))
        elif name == "knn_regression":  # 3.5.2 + 3.5.4 only, without the 3.5.1 grid
            metrics.update(_study_knn_regression(data, out, seeds, g))
        elif name == "bias_variance":
            metrics.update(study_bias_variance(data, out, figs, seeds, g))
        elif name == "learning_curves":
            metrics.update(study_learning_curves(data, figs, seeds))
        elif name == "grid":
            metrics.update(study_grid(data, configs, seeds, g))
        else:
            raise ValueError(f"unknown study {name!r}; choose from {STUDIES}")
    return metrics


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m terser.phase3.studies",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data", type=Path, required=True, help="pair feature table (row 3.1.3)")
    p.add_argument("--out", type=Path, default=Path("reports"))
    p.add_argument("--configs", type=Path, default=Path("configs/model"))
    p.add_argument("--studies", nargs="+", default=list(STUDIES), choices=STUDIES)
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument(
        "--xy",
        nargs=2,
        default=["max_si_removed", "tokens_saved"],
        help="two feature columns for the tree boundary (3.3.5)",
    )
    p.add_argument(
        "--overhead-pct",
        type=float,
        default=None,
        help="rewrite overhead per prompt in percent for the 3.5.4 skip rule (overrides the "
        "table's overhead_pct column)",
    )
    p.add_argument("--quick", action="store_true", help="tiny grids (smoke test only)")
    p.add_argument("--allow-stand-in", action="store_true")
    p.add_argument("--no-track", action="store_true", help="skip MLflow logging")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    args = _parser().parse_args(argv)
    try:
        data = load_table(args.data, allow_stand_in=args.allow_stand_in)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.overhead_pct is not None:
        data = dataclasses.replace(data, overhead=np.full(len(data.y), args.overhead_pct))
    if data.overhead is None and "knn_regression" in args.studies:
        print(
            "warning: no overhead_pct; the 3.5.4 skip rule assumes zero overhead", file=sys.stderr
        )
    grids = Grids.quick() if args.quick else Grids()

    def execute() -> dict[str, float]:
        return run(
            data, args.out, args.configs, args.studies, args.seeds, grids, (args.xy[0], args.xy[1])
        )

    if args.no_track:
        metrics = execute()
    else:  # pragma: no cover - needs an MLflow server
        import mlflow

        from terser.tracking import start_run

        variant = "standin" if data.stand_in else "labelled"
        with start_run(
            phase=3,
            row="3.2-3.6",
            task="classical",
            run_name=f"3-classical-studies-" f"{variant}-{args.seeds[0]}",
            params={
                "data": data.source,
                "data_sha256": data.sha256,
                "studies": ",".join(args.studies),
                "seeds": ",".join(map(str, args.seeds)),
                "quick": args.quick,
            },
        ):
            mlflow.set_tag("stand_in", str(data.stand_in).lower())
            metrics = execute()
            mlflow.log_metrics({k: v for k, v in metrics.items() if v == v})
            mlflow.log_artifacts(str(args.out), artifact_path="reports")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
