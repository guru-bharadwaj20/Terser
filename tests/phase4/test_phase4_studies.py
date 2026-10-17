"""Rows 4.1.2, 4.1.4, 4.2.1, 4.3.1, 4.4.5-4.4.8: study code on synthetic fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from terser.features.pair_embedding import PAIR_DIM, build_cache, load_cache, pair_features
from terser.models.neural.numpy_mlp import NumpyMLP
from terser.models.neural.scratch_optim import SGD, Adagrad, Adam, Momentum, make_optimizer
from terser.phase4 import nn_studies, svm_studies
from terser.phase4.tables import from_frame, load_table


def _blobs(seed: int, n: int = 160, d: int = 4) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    X = rng.normal(0, 1, (n, d)) + 1.2 * (2 * y[:, None] - 1) * (np.arange(d) == 0)
    return X, y


def _table(n: int = 200, seed: int = 0) -> pd.DataFrame:
    X, y = _blobs(seed, n)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    df.insert(0, "id", [f"p{i}" for i in range(n)])
    df.insert(1, "group", [f"g{i // 2}" for i in range(n)])
    df.insert(2, "category", ["code", "qa"] * (n // 2))
    df.insert(3, "label", y)
    return df


# ------------------------------------------------------------------ 4.1.2 pair embedding


def test_pair_features_layout() -> None:
    a = np.array([[1.0, -2.0]])
    b = np.array([[3.0, 1.0]])
    np.testing.assert_allclose(pair_features(a, b), [[1, -2, 3, 1, 2, 3, 3, -2]])
    with pytest.raises(ValueError):
        pair_features(a, np.zeros((1, 3)))


def test_pair_cache_roundtrip(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)

    def embed(texts: object) -> np.ndarray:
        return rng.normal(size=(len(texts), 384))  # type: ignore[arg-type]

    ids = [f"pair{i}" for i in range(5)]
    path = build_cache(ids, ["o"] * 5, ["r"] * 5, embed, tmp_path, batch_size=2)
    mat, got = load_cache(tmp_path)
    assert path.name == "pair_emb.npy" and mat.shape == (5, PAIR_DIM) and got == ids
    with pytest.raises(ValueError):
        build_cache(["a", "a"], ["o"] * 2, ["r"] * 2, embed, tmp_path)


# ------------------------------------------------------------------ tables


def test_load_table(tmp_path: Path) -> None:
    df = _table(20)
    df.to_csv(tmp_path / "t.csv", index=False)
    t = load_table(tmp_path / "t.csv")
    assert t.X.shape == (20, 4) and t.feature_names[0] == "x0" and len(t.groups) == 20
    np.save(tmp_path / "f.npy", np.ones((20, 3)))
    assert load_table(tmp_path / "t.csv", tmp_path / "f.npy").X.shape == (20, 3)
    # With an ids file next to the matrix, rows are aligned by id, not by order.
    ids = df["id"].astype(str).tolist()
    np.save(tmp_path / "e.npy", np.arange(20, dtype=float)[::-1, None])
    (tmp_path / "e_ids.json").write_text(json.dumps(ids[::-1]), encoding="utf-8")
    aligned = load_table(tmp_path / "t.csv", tmp_path / "e.npy")
    assert aligned.X[:, 0].tolist() == list(range(20))
    (tmp_path / "e_ids.json").write_text(json.dumps(ids[:-1]), encoding="utf-8")
    with pytest.raises(KeyError):
        load_table(tmp_path / "t.csv", tmp_path / "e.npy")
    with pytest.raises(FileNotFoundError):
        load_table(tmp_path / "missing.csv")
    with pytest.raises(ValueError):
        from_frame(df.drop(columns=["group"]))


# ------------------------------------------------------------------ 4.3.1 optimizers


def test_optimizer_updates_match_hand_computation() -> None:
    g = np.array([1.0, -2.0])
    p = np.zeros(2)
    SGD(lr=0.1).step([p], [g])
    np.testing.assert_allclose(p, [-0.1, 0.2])

    p, opt = np.zeros(2), Momentum(lr=0.1, beta=0.9)
    opt.step([p], [g])
    opt.step([p], [g])  # v = g, then 1.9 g
    np.testing.assert_allclose(p, -0.1 * 2.9 * g)

    p, opt2 = np.zeros(2), Adagrad(lr=1.0)
    opt2.step([p], [g])
    np.testing.assert_allclose(p, -np.sign(g), atol=1e-9)

    p, opt3 = np.zeros(2), Adam(lr=0.01)
    for _ in range(3):
        opt3.step([p], [g])  # constant gradient: bias-corrected step = lr * sign(g)
    np.testing.assert_allclose(p, -0.03 * np.sign(g), atol=1e-7)

    with pytest.raises(ValueError):
        make_optimizer("lbfgs", 0.1)


@pytest.mark.parametrize("name", ["sgd", "momentum", "adagrad", "adam"])
def test_optimizers_reduce_loss(name: str) -> None:
    X, y = _blobs(1)
    m = NumpyMLP((4, 8, 1), seed=0)
    opt = make_optimizer(name, 0.05)
    before = m.loss(X, y)
    for _ in range(50):
        gws, gbs = m.backward(X, y)
        opt.step([*m.weights, *m.biases], [*gws, *gbs])
    assert m.loss(X, y) < before


def test_optimizer_run_respects_budget() -> None:
    X, y = _blobs(2)
    Xv, yv = _blobs(3)
    budget = 10 * len(y)
    for cfg in (nn_studies.OptimizerConfig("gd", 0.5, None), nn_studies.OPTIMIZER_GRID[-1]):
        df = nn_studies.optimizer_run(cfg, X, y, Xv, yv, budget=budget, hidden=(8,))
        assert df["grad_evals"].max() <= budget
        assert df["grad_evals"].max() > budget - 64
        assert df["val_auc"].iloc[-1] > 0.7


# ------------------------------------------------------------------ 4.2.1 activations


@pytest.mark.parametrize("act", nn_studies.ACTIVATIONS)
def test_activation_run(act: str) -> None:
    X, y = _blobs(4)
    Xv, yv = _blobs(5)
    row = nn_studies.activation_run(act, X, y, Xv, yv, hidden=(8, 4), epochs=8)  # type: ignore[arg-type]
    assert row["activation"] == act and 1 <= row["epochs_to_plateau"] <= 8
    assert {"grad_norm_layer1", "grad_norm_layer3", "saturated_fraction", "dead_fraction"} <= set(
        row
    )
    assert 0 <= row["saturated_fraction"] <= 1 and 0 <= row["dead_fraction"] <= 1


def test_hidden_unit_stats_detect_dead_and_saturated() -> None:
    X = np.abs(np.random.default_rng(0).normal(size=(20, 3)))
    relu = NumpyMLP((3, 4, 1), activation="relu")
    relu.weights[0][:] = -1.0  # positive inputs -> all pre-activations negative -> all dead
    assert relu.hidden_unit_stats(X)["dead_fraction"] == 1.0
    sig = NumpyMLP((3, 4, 1), activation="sigmoid")
    sig.weights[0][:] = 100.0
    assert sig.hidden_unit_stats(X)["saturated_fraction"] == 1.0


def test_step_activation_uses_ste() -> None:
    m = NumpyMLP((2, 3, 1), activation="step", seed=0)
    X, y = _blobs(6, d=2)
    gws, _ = m.backward(X, y)
    assert np.abs(gws[0]).sum() > 0  # the true gradient would be identically zero


def test_summarise_and_cli(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": ["x", "x", "y", "y"], "m": [1.0, 3.0, 2.0, 2.0]})
    md = nn_studies.summarise(df, "a", ["m"])
    assert "| x | 2.0000 ± 1.4142 |" in md
    _table(120).to_csv(tmp_path / "t.csv", index=False)
    out = tmp_path / "out"
    assert nn_studies.main(["activations", "--table", str(tmp_path / "t.csv"),
                            "--out", str(out), "--seeds", "2", "--epochs", "2"]) == 0  # fmt: skip
    assert (out / "p4_activations_runs.csv").is_file()
    assert nn_studies.main(["optimizers", "--table", str(tmp_path / "t.csv"),
                            "--out", str(out), "--seeds", "1", "--epochs", "1"]) == 0  # fmt: skip
    assert (out / "figures" / "p4_optimizers.png").is_file()


# ------------------------------------------------------------------ 4.4.5-4.4.8 SVM studies


def test_soft_margin_sweep_widens_with_small_c() -> None:
    X, y = _blobs(7, n=80, d=2)
    Xv, yv = _blobs(8, n=80, d=2)
    df = svm_studies.soft_margin_sweep(X, y, Xv, yv, Cs=(0.01, 1.0, 100.0))
    assert list(df["C"]) == [0.01, 1.0, 100.0]
    assert df["margin_width"].is_monotonic_decreasing
    assert df["n_support"].iloc[0] >= df["n_support"].iloc[-1]
    assert (df["val_auc"] > 0.7).all()


def test_kernel_grid_and_plots(tmp_path: Path) -> None:
    X, y = _blobs(9, n=60, d=2)
    Xv, yv = _blobs(10, n=60, d=2)
    df = svm_studies.kernel_grid(X, y, Xv, yv, degrees=(2,), coef0s=(1.0,), Cs=(1.0,))
    assert set(df["kernel"]) == {"linear", "poly"} and (df["fit_seconds"] > 0).all()
    sep_X = np.r_[np.random.default_rng(0).normal(-3, 0.5, (20, 2)),
                  np.random.default_rng(1).normal(3, 0.5, (20, 2))]  # fmt: skip
    sep_y = np.r_[np.zeros(20, int), np.ones(20, int)]
    m = svm_studies.hard_margin_plot(sep_X, sep_y, tmp_path / "hm.png")
    assert (tmp_path / "hm.png").is_file() and len(m.support_) >= 2
    svm_studies.plot_surfaces(X, y, tmp_path / "s.png", degree=2)
    assert (tmp_path / "s.png").is_file()
    with pytest.raises(ValueError):
        svm_studies.plot_surfaces(np.zeros((4, 3)), np.array([0, 1, 0, 1]), tmp_path / "x.png")


def test_svm_cli(tmp_path: Path) -> None:
    df = _table(80)
    df[["id", "group", "category", "label", "x0", "x1"]].to_csv(tmp_path / "t.csv", index=False)
    assert svm_studies.main(["--table", str(tmp_path / "t.csv"), "--out", str(tmp_path)]) == 0
    assert (tmp_path / "p4_svm_kernels.csv").is_file()


# ------------------------------------------------------------------ 4.1.4 torch gate


def test_torch_quality_gate_smoke() -> None:
    pytest.importorskip("torch")
    from terser.phase4.quality_gate import make_torch_fit_predict

    X, y = _blobs(11, n=200, d=6)
    fp = make_torch_fit_predict(sizes=(6, 8, 4, 1), epochs=5)
    s = fp(X[:150], y[:150], X[150:])
    assert s.shape == (50,) and ((s >= 0) & (s <= 1)).all()
