"""Tests for terser.eval.cv (row 3.2.4)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from terser.eval.cv import grouped_folds, out_of_fold_scores, strata


def _data(seed: int = 0) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    rng = np.random.default_rng(seed)
    cats, groups, ys = [], [], []
    for g in range(150):
        cat = ["code", "fact", "reason"][g % 3]
        for _ in range(int(rng.integers(1, 5))):
            cats.append(cat)
            groups.append(f"src-{g}")
            ys.append(int(rng.random() < 0.3))
    y = np.array(ys)
    X = np.c_[y + rng.normal(0, 1, len(y)), rng.normal(0, 1, len(y))]
    return X, y, cats, groups


def test_groups_never_straddle_and_every_example_validated_once() -> None:
    X, y, cats, groups = _data()
    seen = np.zeros(len(y), dtype=int)
    g = np.array(groups)
    for tr, va in grouped_folds(y, cats, groups):
        assert not set(g[tr]) & set(g[va])
        seen[va] += 1
    assert np.all(seen == 1)


def test_stratification_keeps_degraded_share() -> None:
    X, y, cats, groups = _data(1)
    shares = [y[va].mean() for _, va in grouped_folds(y, cats, groups)]
    assert max(shares) - min(shares) < 0.1
    assert len(np.unique(strata(y, cats))) == 6


def test_oof_scores() -> None:
    X, y, cats, groups = _data(2)

    def fit_predict(Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray) -> np.ndarray:
        return LogisticRegression().fit(Xtr, ytr).predict_proba(Xva)[:, 1]

    res = out_of_fold_scores(fit_predict, X, y, cats, groups, seed=3)
    assert not np.isnan(res.scores).any()
    assert set(res.fold.tolist()) == set(range(5))
    again = out_of_fold_scores(fit_predict, X, y, cats, groups, seed=3)
    np.testing.assert_array_equal(res.scores, again.scores)
    with pytest.raises(ValueError):
        out_of_fold_scores(lambda a, b, c: np.zeros(1), X, y, cats, groups)


def test_length_mismatch() -> None:
    with pytest.raises(ValueError):
        strata([0, 1], ["a"])
    with pytest.raises(ValueError):
        list(grouped_folds([0, 1], ["a", "b"], ["g"]))
