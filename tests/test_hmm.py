"""Tests for terser.models.classical.hmm against brute-force enumeration."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from terser.models.classical.hmm import DiscreteHMM


def _random_hmm(seed: int, s: int = 3, v: int = 4) -> DiscreteHMM:
    rng = np.random.default_rng(seed)
    return DiscreteHMM(
        start=rng.dirichlet(np.ones(s)),
        trans=rng.dirichlet(np.ones(s), size=s),
        emit=rng.dirichlet(np.ones(v), size=s),
    )


def _joint(hmm: DiscreteHMM, states: tuple[int, ...], obs: np.ndarray) -> float:
    p = hmm.start[states[0]] * hmm.emit[states[0], obs[0]]
    for t in range(1, len(obs)):
        p *= hmm.trans[states[t - 1], states[t]] * hmm.emit[states[t], obs[t]]
    return float(p)


@pytest.mark.parametrize("seed", range(10))
def test_forward_matches_brute_force(seed: int) -> None:
    hmm = _random_hmm(seed)
    obs = np.random.default_rng(seed + 100).integers(0, 4, 5)
    total = sum(_joint(hmm, st, obs) for st in itertools.product(range(3), repeat=5))
    assert hmm.log_likelihood(obs) == pytest.approx(np.log(total), abs=1e-12)
    alpha = hmm.forward(obs)
    assert alpha.shape == (5, 3)


def test_forward_handles_zeros_and_long_sequences() -> None:
    hmm = DiscreteHMM(
        start=np.array([1.0, 0.0]),
        trans=np.array([[0.9, 0.1], [0.0, 1.0]]),
        emit=np.array([[0.5, 0.5], [1.0, 0.0]]),
    )
    assert hmm.log_likelihood([0]) == pytest.approx(np.log(0.5))
    long = np.zeros(5000, dtype=int)
    assert np.isfinite(hmm.log_likelihood(long))


def test_validation() -> None:
    with pytest.raises(ValueError):
        DiscreteHMM(np.array([0.5, 0.5]), np.eye(3), np.ones((2, 2)) / 2)
    with pytest.raises(ValueError):
        DiscreteHMM(np.array([0.6, 0.6]), np.eye(2), np.ones((2, 2)) / 2)
    hmm = _random_hmm(0)
    with pytest.raises(ValueError):
        hmm.log_likelihood([])
    with pytest.raises(ValueError):
        hmm.log_likelihood([4])
