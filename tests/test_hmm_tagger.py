"""Rows 5.3.1 (model definition), 5.3.2 (supervised MLE), 5.3.5 (Baum-Welch), 5.3.6 (rewrite)."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from terser.models.classical.hmm import (
    POS_CLASSES,
    TAGS,
    DiscreteHMM,
    TokenClassEncoder,
    apply_tags,
    baum_welch,
    coarse_pos,
    fit_supervised,
)


def _random_hmm(seed: int, s: int = 3, v: int = 4) -> DiscreteHMM:
    rng = np.random.default_rng(seed)
    return DiscreteHMM(
        start=rng.dirichlet(np.ones(s)),
        trans=rng.dirichlet(np.ones(s), size=s),
        emit=rng.dirichlet(np.ones(v), size=s),
    )


def _sample(hmm: DiscreteHMM, n: int, length: int, seed: int) -> tuple[list, list]:
    rng = np.random.default_rng(seed)
    obs, tags = [], []
    for _ in range(n):
        s = [rng.choice(hmm.n_states, p=hmm.start)]
        for _ in range(length - 1):
            s.append(rng.choice(hmm.n_states, p=hmm.trans[s[-1]]))
        obs.append(np.array([rng.choice(hmm.emit.shape[1], p=hmm.emit[x]) for x in s]))
        tags.append(np.array(s))
    return obs, tags


def test_model_definition() -> None:
    assert TAGS == ("KEEP", "DROP", "REPLACE")
    enc = TokenClassEncoder(cost=len)
    assert enc.n_obs == (len(enc.lexicon) + 1) * len(POS_CLASSES) * 3
    ids = enc.encode("um so please write the code 42 !".split())
    assert ids.min() >= 0 and ids.max() < enc.n_obs
    # "write" and "code": both non-lexicon, OTHER, cost >= 3 -> same class; all else distinct.
    assert ids[3] == ids[5]
    assert len(set(ids.tolist())) == len(ids) - 1
    # Same lexicon id and POS but different cost bucket -> different observation.
    assert enc.encode_word("a") != TokenClassEncoder(cost=lambda _w: 5).encode_word("a")
    assert coarse_pos("The") == "DET" and coarse_pos("3.5") == "NUM" and coarse_pos("?!") == "PUNCT"
    assert enc.encode_word("Um") == enc.encode_word("um")


@pytest.mark.parametrize("smoothing", [0.1, 1.0])
def test_supervised_rows_sum_to_one(smoothing: float) -> None:
    obs, tags = _sample(_random_hmm(0, v=6), 30, 8, seed=1)
    hmm = fit_supervised(obs, tags, n_states=3, n_obs=6, smoothing=smoothing)
    np.testing.assert_allclose(hmm.start.sum(), 1.0)
    np.testing.assert_allclose(hmm.trans.sum(axis=1), 1.0)
    np.testing.assert_allclose(hmm.emit.sum(axis=1), 1.0)
    assert (hmm.trans > 0).all() and (hmm.emit > 0).all()


def test_supervised_matches_hand_counts() -> None:
    obs = [np.array([0, 1, 1]), np.array([1, 0])]
    tags = [np.array([0, 1, 1]), np.array([1, 0])]
    hmm = fit_supervised(obs, tags, n_states=2, n_obs=2, smoothing=1.0)
    # start counts (1, 1) + 1 -> (2, 2)/4; transitions 0->1:1, 1->1:1, 1->0:1.
    np.testing.assert_allclose(hmm.start, [0.5, 0.5])
    np.testing.assert_allclose(hmm.trans, [[1 / 3, 2 / 3], [2 / 4, 2 / 4]])
    np.testing.assert_allclose(hmm.emit, [[3 / 4, 1 / 4], [1 / 5, 4 / 5]])


def test_supervised_recovers_parameters() -> None:
    true = _random_hmm(3, v=5)
    obs, tags = _sample(true, 400, 20, seed=4)
    est = fit_supervised(obs, tags, 3, 5, smoothing=1e-3)
    np.testing.assert_allclose(est.trans, true.trans, atol=0.03)
    np.testing.assert_allclose(est.emit, true.emit, atol=0.03)


def test_supervised_validation() -> None:
    with pytest.raises(ValueError):
        fit_supervised([], [], 2, 2)
    with pytest.raises(ValueError):
        fit_supervised([np.array([0])], [np.array([0])], 2, 2, smoothing=0)
    with pytest.raises(ValueError):
        fit_supervised([np.array([0, 1])], [np.array([0])], 2, 2)


@pytest.mark.parametrize("seed", range(3))
def test_backward_consistent_with_forward(seed: int) -> None:
    hmm = _random_hmm(seed)
    o = np.array([0, 3, 1, 2, 2])
    ll = hmm.log_likelihood(o)
    a, b = hmm.forward(o), hmm.backward(o)
    for t in range(len(o)):
        assert np.logaddexp.reduce(a[t] + b[t]) == pytest.approx(ll)
    brute = sum(
        np.prod(
            [hmm.start[s[0]] * hmm.emit[s[0], o[0]]]
            + [hmm.trans[s[t - 1], s[t]] * hmm.emit[s[t], o[t]] for t in range(1, 5)]
        )
        for s in itertools.product(range(3), repeat=5)
    )
    assert ll == pytest.approx(np.log(brute))


@pytest.mark.parametrize("seed", range(3))
def test_baum_welch_monotone_and_improves(seed: int) -> None:
    true = _random_hmm(seed + 10, v=5)
    obs, _ = _sample(true, 60, 15, seed=seed)
    init = _random_hmm(seed + 100, v=5)
    fitted, hist = baum_welch(init, obs, n_iter=40)
    assert len(hist) >= 2
    assert np.all(np.diff(hist) >= -1e-8)
    assert sum(fitted.log_likelihood(o) for o in obs) >= hist[0]
    np.testing.assert_allclose(fitted.trans.sum(1), 1.0)


def test_apply_tags() -> None:
    words = "hey can you please utilize the list".split()
    tags = [1, 0, 0, 1, 2, 0, 0]
    assert apply_tags(words, tags, {"utilize": "use"}) == "can you use the list"
    assert apply_tags(["utilize"], [2]) == "utilize"
    assert apply_tags(["x"], [2], {"x": ""}) == ""
