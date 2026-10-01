"""Rows 5.1.1, 5.1.3, 5.1.5, 5.1.6, 5.2.3, 5.2.4, 5.2.6, 5.3.5, 5.3.6: study code on fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from terser.models.classical.hmm import DiscreteHMM, TokenClassEncoder
from terser.phase5 import bayes_studies, ensembles, hmm_tagger


def _data(seed: int, n: int = 240) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, 4))
    y = (rng.random(n) < 1 / (1 + np.exp(-(2 * X[:, 0] - X[:, 1])))).astype(int)
    cats = ["code" if i % 2 else "qa" for i in range(n)]
    groups = [f"g{i // 3}" for i in range(n)]
    return X, y, cats, groups


# ------------------------------------------------------------------ ensembles


def test_adaboost_vs_sklearn() -> None:
    X, y, _, _ = _data(0)
    Xv, yv, _, _ = _data(1, n=1000)
    res = ensembles.adaboost_vs_sklearn(X, y, Xv, yv, n_estimators=50, k_hard=5)
    assert res["auc_diff"] < 0.01 and len(res["train_error"]) == 50
    assert len(res["hard_examples"]) == 5


def test_random_forest_grid_oob_close_to_cv() -> None:
    X, y, c, g = _data(2)
    df = ensembles.random_forest_grid(X, y, c, g, n_estimators=(100,), max_features=("sqrt",))
    assert len(df) == 2 and df.loc[df.bootstrap, "oob_auc"].notna().all()
    assert df.loc[~df.bootstrap, "oob_auc"].isna().all()
    assert abs(df.loc[df.bootstrap, "oob_minus_cv"].iloc[0]) < 0.1
    assert (df["cv_auc"] > 0.7).all()


def test_lightgbm_grid_and_importance(tmp_path: Path) -> None:
    X, y, c, g = _data(3, n=300)
    df = ensembles.lightgbm_grid(
        X, y, c, g, learning_rate=(0.1,), n_estimators=(50,), max_depth=(3,)
    )
    assert len(df) == 1 and df["cv_auc"].iloc[0] > 0.7 and df["best_iteration"].iloc[0] <= 50
    model = ensembles._lgbm_fit(X[:200], y[:200], c[:200], g[:200], {"n_estimators": 100}, 0)
    names = ["signal", "anti", "noise1", "noise2"]
    perm, shap = ensembles.feature_importance(model, X[200:], y[200:], c[200:], names)
    assert perm["feature"].iloc[0] == "signal"
    assert shap.index[0] == "signal" and {"all", "code", "qa"} <= set(shap.columns)
    ensembles.plot_shap(shap, tmp_path / "shap.png")
    ensembles.plot_training_error([0.3, 0.2], tmp_path / "err.png")
    assert (tmp_path / "shap.png").is_file() and (tmp_path / "err.png").is_file()


def test_ensembles_cli(tmp_path: Path) -> None:
    X, y, c, g = _data(4, n=150)
    df = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    df.insert(0, "id", [f"p{i}" for i in range(150)])
    df.insert(1, "group", g)
    df.insert(2, "category", c)
    df.insert(3, "label", y)
    df.to_csv(tmp_path / "t.csv", index=False)
    for study in ("adaboost", "importance"):
        assert (
            ensembles.main([study, "--table", str(tmp_path / "t.csv"), "--out", str(tmp_path)]) == 0
        )
    assert (tmp_path / "figures" / "p5_shap.png").is_file()
    assert (tmp_path / "p5_adaboost.csv").is_file()


# ------------------------------------------------------------------ Bayes


def test_nb_vs_lr() -> None:
    rng = np.random.default_rng(0)
    filler = ["um so like", "basically just", "honestly um", "so yeah like", "kinda maybe"]
    content = ["sort the list", "return json", "parse the date", "write a test", "add types"]
    texts, y = [], []
    for _ in range(60):
        lab = int(rng.integers(0, 2))
        texts.append(str(rng.choice(filler if lab else content)))
        y.append(lab)
    cats = ["x"] * 60
    groups = [f"g{i}" for i in range(60)]
    df, curves = bayes_studies.nb_vs_lr(texts, np.array(y), cats, groups)
    assert set(df["model"]) == {"naive_bayes", "logistic_regression"}
    assert (df["roc_auc"] > 0.95).all() and set(curves) == set(df["model"])
    assert bayes_studies.precision_at_recall(np.array([0, 1, 1]), np.array([0.1, 0.9, 0.8])) == 1.0


def test_bayes_optimal_vs_map() -> None:
    rng = np.random.default_rng(1)
    y_tr, y_va = rng.integers(0, 2, 40), rng.integers(0, 2, 40)

    # Three hypotheses, each 75% accurate with independent errors.
    def flip(y: np.ndarray, s: int) -> np.ndarray:
        return np.where(np.random.default_rng(s).random(len(y)) < 0.25, 1 - y, y)

    p_tr = np.stack([flip(y_tr, s) for s in range(3)])
    p_va = np.stack([flip(y_va, s + 10) for s in range(3)])
    res = bayes_studies.bayes_optimal_vs_map(p_tr, y_tr, p_va, y_va, noise=0.3)
    assert 0 <= res["acc_map"] <= 1 and 0 <= res["acc_bayes_optimal"] <= 1
    assert 0 < res["map_posterior"] <= 1


def test_gmm_on_deltas_separates_modes() -> None:
    rng = np.random.default_rng(2)
    deltas = np.r_[rng.normal(-2, 0.4, 60), rng.normal(0.1, 0.3, 140)]
    labels = np.r_[np.ones(60, int), np.zeros(140, int)]
    res = bayes_studies.gmm_on_deltas(deltas, labels)
    assert res["ari"] > 0.9 and res["mean_degraded"] < res["mean_preserved"]  # type: ignore[operator]


# ------------------------------------------------------------------ HMM tagger


def _tagged(seed: int, n: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    true = DiscreteHMM(
        start=np.array([0.6, 0.3, 0.1]),
        trans=np.array([[0.8, 0.15, 0.05], [0.4, 0.55, 0.05], [0.7, 0.1, 0.2]]),
        emit=np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.1, 0.7]]),
    )
    rng = np.random.default_rng(seed)
    obs, tags = [], []
    for _ in range(n):
        s = [rng.choice(3, p=true.start)]
        for _ in range(11):
            s.append(rng.choice(3, p=true.trans[s[-1]]))
        tags.append(np.array(s))
        obs.append(np.array([rng.choice(3, p=true.emit[k]) for k in s]))
    return obs, tags


def test_token_macro_f1() -> None:
    assert hmm_tagger.token_macro_f1([np.array([0, 1, 2])], [np.array([0, 1, 2])]) == 1.0
    assert hmm_tagger.token_macro_f1([np.array([0, 0])], [np.array([1, 1])]) == 0.0


def test_baum_welch_f1() -> None:
    tr_o, tr_t = _tagged(0, 15)
    un_o, _ = _tagged(1, 80)
    va_o, va_t = _tagged(2, 40)
    res = hmm_tagger.baum_welch_f1(tr_o, tr_t, un_o, va_o, va_t, n_obs=3)
    assert 0 < res["f1_supervised"] <= 1 and 0 < res["f1_baum_welch"] <= 1  # type: ignore[operator]
    hist = res["log_likelihood"]
    assert np.all(np.diff(hist) >= -1e-8)  # type: ignore[arg-type]


def test_compress_drops_lexicon_words() -> None:
    enc = TokenClassEncoder()
    n = enc.n_obs
    emit = np.full((3, n), 1.0)
    lex_ids = {enc.encode_word(w) for w in ("um", "so", "basically", "um,")}
    for o in range(n):
        if o in lex_ids:
            emit[1, o] = 50.0
        else:
            emit[0, o] = 50.0
    emit /= emit.sum(1, keepdims=True)
    hmm = DiscreteHMM(np.array([0.5, 0.4, 0.1]), np.full((3, 3), 1 / 3), emit)
    out = hmm_tagger.compress(hmm, ["um, so basically sort the list", "", "um"])
    assert out[0] == "sort the list" and out[1] == "" and out[2] == "um"
