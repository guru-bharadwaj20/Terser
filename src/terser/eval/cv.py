"""Grouped stratified cross-validation (plan row **3.2.4**).

Every Phase 3-5 model is scored under one protocol: ``StratifiedGroupKFold(k=5)`` with
**group = source prompt**, so paraphrases and rewrites of one prompt never straddle train and
validation, and **stratum = category x label**, so each fold keeps the category mix and the
degraded share. Out-of-fold (OOF) predictions are the shared unit: every example gets exactly one
score from a model that never saw its group, and all model comparisons (row 3.2.8) and threshold
choices (row 3.2.3) use those scores.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterator, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt
from sklearn.model_selection import StratifiedGroupKFold

from terser.eval.labels import ArrayLike, _as_binary_array

__all__ = [
    "N_SPLITS",
    "OofResult",
    "grouped_folds",
    "out_of_fold_scores",
    "strata",
]

N_SPLITS: Final[int] = 5
IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]

FitPredict = Callable[[npt.NDArray[np.float64], IntArray, npt.NDArray[np.float64]], FloatArray]
"""``fit_predict(X_train, y_train, X_val) -> degradation scores for X_val``."""


def strata(y: ArrayLike, categories: Sequence[Hashable]) -> IntArray:
    """Integer stratum id per example for ``category x label``."""
    labels = _as_binary_array(y)
    if len(labels) != len(categories):
        raise ValueError(f"lengths differ: y {len(labels)}, categories {len(categories)}")
    keys = [f"{c}|{v}" for c, v in zip(categories, labels.tolist(), strict=True)]
    _, ids = np.unique(np.asarray(keys), return_inverse=True)
    return ids.astype(np.int64)


def grouped_folds(
    y: ArrayLike,
    categories: Sequence[Hashable],
    groups: Sequence[Hashable],
    n_splits: int = N_SPLITS,
    seed: int = 0,
) -> Iterator[tuple[IntArray, IntArray]]:
    """Yield ``(train_idx, val_idx)`` for ``StratifiedGroupKFold`` on ``category x label``.

    Raises:
        ValueError: If lengths differ.
    """
    s = strata(y, categories)
    if len(groups) != len(s):
        raise ValueError(f"lengths differ: groups {len(groups)}, y {len(s)}")
    _, g = np.unique(np.asarray([str(x) for x in groups]), return_inverse=True)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, va in cv.split(np.zeros((len(s), 1)), s, g):
        yield tr.astype(np.int64), va.astype(np.int64)


@dataclass(frozen=True)
class OofResult:
    """Out-of-fold scores and the fold each example was validated in."""

    scores: FloatArray
    fold: IntArray


def out_of_fold_scores(
    fit_predict: FitPredict,
    X: npt.ArrayLike,
    y: ArrayLike,
    categories: Sequence[Hashable],
    groups: Sequence[Hashable],
    n_splits: int = N_SPLITS,
    seed: int = 0,
) -> OofResult:
    """One OOF degradation score per example under the grouped protocol.

    Args:
        fit_predict: Trains a fresh model on the fold's training part and scores the held-out part.
            Preprocessing that learns from data (scaling, PCA) belongs inside it.
        X: Feature matrix, one row per example.
        y: Labels.
        categories: Task category per example.
        groups: Source-prompt id per example.
        n_splits: Number of folds.
        seed: Shuffle seed for fold assignment.

    Raises:
        ValueError: If ``fit_predict`` returns the wrong number of scores.
    """
    Xa = np.asarray(X, dtype=np.float64)
    labels = _as_binary_array(y)
    scores = np.full(len(labels), np.nan)
    fold = np.full(len(labels), -1, dtype=np.int64)
    for k, (tr, va) in enumerate(grouped_folds(labels, categories, groups, n_splits, seed)):
        pred = np.asarray(fit_predict(Xa[tr], labels[tr], Xa[va]), dtype=np.float64)
        if pred.shape != (len(va),):
            raise ValueError(f"fit_predict returned shape {pred.shape}, expected ({len(va)},)")
        scores[va] = pred
        fold[va] = k
    return OofResult(scores=scores, fold=fold)
