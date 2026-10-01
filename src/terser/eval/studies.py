"""Phase 3 evaluation studies built on the grouped-CV protocol.

* :func:`naive_kfold_scores` / :func:`leakage_gap`: row **3.2.5**, naive ``KFold`` (paraphrases of
  one source prompt can sit in train and validation) against grouped CV on the same model.
* :func:`leave_one_category_out`: row **3.2.7**, train on five categories, test on the sixth.
* :func:`learning_curve`: row **3.6.3**, train on a growing share of each fold's training
  *groups* (10% -> 100%) and score train and validation.
* :func:`expand_grid` / :func:`write_best_params`: row **3.6.4**, grid definitions for
  :func:`terser.eval.cv.nested_cv_scores` and recording the chosen parameters under ``configs/``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Hashable, Mapping, Sequence
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
import yaml
from sklearn.model_selection import KFold

from terser.eval.curves import roc_auc
from terser.eval.cv import FitPredict, grouped_folds, out_of_fold_scores
from terser.eval.labels import ArrayLike, _as_binary_array

__all__ = [
    "Score",
    "expand_grid",
    "leakage_gap",
    "learning_curve",
    "leave_one_category_out",
    "naive_kfold_scores",
    "safe_auc",
    "write_best_params",
]

FloatArray = npt.NDArray[np.float64]
Score = Callable[[npt.NDArray[np.int64], FloatArray], float]


def safe_auc(y: npt.NDArray[np.int64], scores: FloatArray) -> float:
    """ROC-AUC, or NaN when only one class is present."""
    if len(np.unique(y)) < 2:
        return float("nan")
    return roc_auc(y, scores)


def naive_kfold_scores(
    fit_predict: FitPredict, X: npt.ArrayLike, y: ArrayLike, n_splits: int = 5, seed: int = 0
) -> FloatArray:
    """OOF scores under shuffled ``KFold`` that ignores groups (the leaky protocol)."""
    Xa = np.asarray(X, dtype=np.float64)
    labels = _as_binary_array(y)
    scores = np.full(len(labels), np.nan)
    for tr, va in KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(Xa):
        scores[va] = fit_predict(Xa[tr], labels[tr], Xa[va])
    return scores


def leakage_gap(
    fit_predict: FitPredict,
    X: npt.ArrayLike,
    y: ArrayLike,
    categories: Sequence[Hashable],
    groups: Sequence[Hashable],
    seeds: Sequence[int] = (0, 1, 2),
    score: Score = safe_auc,
) -> pd.DataFrame:
    """Naive vs grouped CV score per seed; ``gap = naive - grouped`` (positive = leakage)."""
    labels = _as_binary_array(y)
    rows = []
    for s in seeds:
        naive = score(labels, naive_kfold_scores(fit_predict, X, labels, seed=s))
        grouped = score(
            labels, out_of_fold_scores(fit_predict, X, labels, categories, groups, seed=s).scores
        )
        rows.append({"seed": s, "naive": naive, "grouped": grouped, "gap": naive - grouped})
    return pd.DataFrame(rows)


def leave_one_category_out(
    fit_predict: FitPredict,
    X: npt.ArrayLike,
    y: ArrayLike,
    categories: Sequence[Hashable],
    threshold: float = 0.5,
) -> pd.DataFrame:
    """One row per held-out category: size, degraded share, AUC, accuracy at ``threshold``."""
    Xa = np.asarray(X, dtype=np.float64)
    labels = _as_binary_array(y)
    cats = np.asarray([str(c) for c in categories])
    rows = []
    for c in sorted(set(cats.tolist())):
        te = cats == c
        if te.all():
            raise ValueError("leave-one-category-out needs at least two categories")
        s = np.asarray(fit_predict(Xa[~te], labels[~te], Xa[te]), dtype=np.float64)
        rows.append(
            {
                "held_out": c,
                "n": int(te.sum()),
                "degraded_share": float(labels[te].mean()),
                "auc": safe_auc(labels[te], s),
                "accuracy": float(np.mean((s >= threshold).astype(int) == labels[te])),
            }
        )
    return pd.DataFrame(rows)


def learning_curve(
    fit_predict: FitPredict,
    X: npt.ArrayLike,
    y: ArrayLike,
    categories: Sequence[Hashable],
    groups: Sequence[Hashable],
    fractions: Sequence[float] = (0.1, 0.2, 0.4, 0.6, 0.8, 1.0),
    score: Score = safe_auc,
    seed: int = 0,
) -> pd.DataFrame:
    """Train / validation score vs share of training groups, mean and std over grouped folds.

    Subsampling is by source-prompt group, so a smaller training set never splits a group.
    """
    Xa = np.asarray(X, dtype=np.float64)
    labels = _as_binary_array(y)
    grp = np.asarray([str(g) for g in groups])
    rng = np.random.default_rng(seed)
    per: dict[float, list[tuple[float, float, int]]] = {f: [] for f in fractions}
    for tr, va in grouped_folds(labels, categories, groups, seed=seed):
        uniq = np.unique(grp[tr])
        order = rng.permutation(uniq)
        for f in fractions:
            if not 0 < f <= 1:
                raise ValueError("fractions must lie in (0, 1]")
            keep = set(order[: max(1, round(f * len(uniq)))].tolist())
            sub = tr[np.isin(grp[tr], list(keep))]
            if len(np.unique(labels[sub])) < 2:
                continue
            tr_s = fit_predict(Xa[sub], labels[sub], Xa[sub])
            va_s = fit_predict(Xa[sub], labels[sub], Xa[va])
            per[f].append((score(labels[sub], tr_s), score(labels[va], va_s), len(sub)))
    rows = []
    for f, vals in per.items():
        arr = np.asarray(vals, dtype=np.float64).reshape(-1, 3)
        rows.append(
            {
                "fraction": f,
                "n_train": float(arr[:, 2].mean()) if len(arr) else float("nan"),
                "train_mean": float(arr[:, 0].mean()) if len(arr) else float("nan"),
                "train_std": float(arr[:, 0].std()) if len(arr) else float("nan"),
                "val_mean": float(arr[:, 1].mean()) if len(arr) else float("nan"),
                "val_std": float(arr[:, 1].std()) if len(arr) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def expand_grid(grid: Mapping[str, Sequence[Any]]) -> list[dict[str, object]]:
    """Cartesian product of a ``{name: values}`` grid as a list of parameter dicts."""
    keys = list(grid)
    return [dict(zip(keys, combo, strict=True)) for combo in product(*(grid[k] for k in keys))]


def write_best_params(
    path: str | Path, model: str, chosen: Sequence[Mapping[str, object]], row: str = "3.6.4"
) -> dict[str, object]:
    """Record the most frequently chosen outer-fold parameters as a Hydra-style YAML file.

    Returns the written mapping: ``model``, ``row``, ``params`` (the modal choice) and
    ``outer_fold_choices`` (all of them, for transparency).
    """
    if not chosen:
        raise ValueError("no parameter choices given")
    keyed = Counter(yaml.safe_dump(dict(c), sort_keys=True) for c in chosen)
    modal = yaml.safe_load(keyed.most_common(1)[0][0])
    doc: dict[str, object] = {
        "model": model,
        "row": row,
        "params": modal,
        "outer_fold_choices": [dict(c) for c in chosen],
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return doc
