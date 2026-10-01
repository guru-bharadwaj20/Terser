"""Rows 3.2.5, 3.2.7, 3.3.5, 3.5.3, 3.6.1, 3.6.3, 3.6.4, 3.6.5: study utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from terser.eval.baselines import BASELINES, always_preserved, majority, stratified
from terser.eval.bias_variance import (
    bootstrap_predictions,
    decompose_squared,
    decompose_zero_one,
)
from terser.eval.boundaries import fold_pca_2d, plot_boundaries
from terser.eval.studies import (
    expand_grid,
    leakage_gap,
    learning_curve,
    leave_one_category_out,
    naive_kfold_scores,
    write_best_params,
)
from terser.models.classical.scratch_knn import KNNClassifier, KNNRegressor

CATS = ["code", "fact", "reason"]


def knn_fp(k: int = 1):  # type: ignore[no-untyped-def]
    def fp(Xtr, ytr, Xva):  # type: ignore[no-untyped-def]
        return KNNClassifier(k=k).fit(Xtr, ytr).predict_proba(Xva)

    return fp


def leaky_data(seed: int = 0) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Groups of near-duplicate rows sharing a random label: memorisable only across groups."""
    rng = np.random.default_rng(seed)
    n_groups, per = 60, 4
    centres = rng.normal(size=(n_groups, 3))
    labels = np.tile([0, 1], n_groups // 2)
    X = np.repeat(centres, per, axis=0) + rng.normal(scale=0.01, size=(n_groups * per, 3))
    y = np.repeat(labels, per)
    groups = [f"g{i // per}" for i in range(n_groups * per)]
    cats = [CATS[(i // per) % 3] for i in range(n_groups * per)]
    return X, y, cats, groups


def test_baselines() -> None:
    X = np.zeros((5, 1))
    assert majority(X, np.array([1, 1, 0]), X).tolist() == [1.0] * 5
    assert majority(X, np.array([1, 0]), X).tolist() == [0.0] * 5
    assert always_preserved(X, np.array([1, 1]), X).tolist() == [0.0] * 5
    s = stratified(0)(X, np.array([1, 0, 0, 0]), np.zeros((4000, 1)))
    assert set(s.tolist()) <= {0.0, 1.0} and abs(s.mean() - 0.25) < 0.03
    assert set(BASELINES) == {"majority", "stratified", "always_preserved"}


def test_leakage_gap_detects_group_memorisation() -> None:
    X, y, cats, groups = leaky_data()
    naive = naive_kfold_scores(knn_fp(), X, y)
    assert not np.isnan(naive).any()
    table = leakage_gap(knn_fp(), X, y, cats, groups, seeds=(0, 1))
    assert list(table.columns) == ["seed", "naive", "grouped", "gap"]
    # naive KFold sees a near-duplicate of every validation row: AUC ~1; grouped ~0.5
    assert (table["naive"] > 0.95).all()
    assert (table["gap"] > 0.3).all()


def test_leave_one_category_out() -> None:
    X, y, cats, _ = leaky_data()
    t = leave_one_category_out(knn_fp(5), X, y, cats)
    assert t["held_out"].tolist() == sorted(CATS)
    assert t["n"].sum() == len(y)
    with pytest.raises(ValueError):
        leave_one_category_out(knn_fp(), X, y, ["code"] * len(y))


def test_learning_curve_shape_and_train_fit() -> None:
    X, y, cats, groups = leaky_data()
    t = learning_curve(knn_fp(1), X, y, cats, groups, fractions=(0.2, 1.0))
    assert t["fraction"].tolist() == [0.2, 1.0]
    assert (t["train_mean"] == 1.0).all()  # 1-NN memorises its training set
    assert t.loc[0, "n_train"] < t.loc[1, "n_train"]
    with pytest.raises(ValueError):
        learning_curve(knn_fp(), X, y, cats, groups, fractions=(1.5,))


def test_expand_grid_and_write_best_params(tmp_path: Path) -> None:
    grid = expand_grid({"k": [1, 3], "metric": ["euclidean", "cosine"]})
    assert len(grid) == 4 and grid[0] == {"k": 1, "metric": "euclidean"}
    doc = write_best_params(
        tmp_path / "configs" / "knn.yaml", "knn", [{"k": 3}, {"k": 5}, {"k": 3}]
    )
    assert doc["params"] == {"k": 3}
    loaded = yaml.safe_load((tmp_path / "configs" / "knn.yaml").read_text(encoding="utf-8"))
    assert loaded["params"] == {"k": 3} and len(loaded["outer_fold_choices"]) == 3
    with pytest.raises(ValueError):
        write_best_params(tmp_path / "x.yaml", "knn", [])


def test_squared_decomposition_hand_example() -> None:
    preds = np.array([[1.0, 0.0], [3.0, 0.0]])  # point 1: mean 2, var 1; point 2: mean 0, var 0
    bv = decompose_squared(preds, [0.0, 1.0])
    assert bv.bias == pytest.approx((4 + 1) / 2)
    assert bv.variance == pytest.approx(0.5)
    assert bv.loss == pytest.approx(bv.bias + bv.variance)


def test_zero_one_decomposition_hand_example() -> None:
    # point 1: y=1, preds 1,1,0 -> main 1, unbiased, var 1/3
    # point 2: y=0, preds 1,1,0 -> main 1, biased, var 1/3
    preds = np.array([[1, 1], [1, 1], [0, 0]], dtype=float)
    bv = decompose_zero_one(preds, [1, 0])
    assert bv.bias == pytest.approx(0.5)
    assert bv.variance_unbiased == pytest.approx(1 / 6)
    assert bv.variance_biased == pytest.approx(1 / 6)
    assert bv.loss == pytest.approx(bv.bias + bv.variance_unbiased - bv.variance_biased)
    with pytest.raises(ValueError):
        decompose_zero_one([[0.5]], [1])


def test_knn_bias_variance_tradeoff_with_k() -> None:
    rng = np.random.default_rng(0)
    X = rng.uniform(-2, 2, size=(150, 1))
    y = np.sin(2 * X[:, 0]) + rng.normal(scale=0.3, size=150)
    Xe = np.linspace(-1.8, 1.8, 40)[:, None]
    ye = np.sin(2 * Xe[:, 0])
    res = {}
    for k in (1, 51):

        def fp(a, b, c, k=k):  # type: ignore[no-untyped-def]
            return KNNRegressor(k=k).fit(a, b).predict(c)

        res[k] = decompose_squared(bootstrap_predictions(fp, X, y, Xe, n_boot=30), ye)
    assert res[1].variance > res[51].variance
    assert res[1].bias < res[51].bias


def test_fold_pca_and_boundary_plot(tmp_path: Path) -> None:
    X, y, _, _ = leaky_data()
    Ztr, Zte = fold_pca_2d(X[:200], X[200:])
    assert Ztr.shape == (200, 2) and Zte.shape == (40, 2)
    assert abs(Ztr[:, 0].mean()) < 1e-9
    out = plot_boundaries(
        [("1-NN", knn_fp(1)), ("5-NN", knn_fp(5))],
        Ztr,
        y[:200],
        tmp_path / "b.png",
        "PC1",
        "PC2",
        resolution=30,
    )
    assert out.exists()
    with pytest.raises(ValueError):
        plot_boundaries([("x", knn_fp())], X, y, tmp_path / "c.png", "a", "b")
