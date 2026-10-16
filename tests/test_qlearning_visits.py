"""Visit-count step size (alpha = 1 / N(s, a)) makes Q the running mean of its targets."""

from __future__ import annotations

import pytest

from terser.models.rl.scratch_qlearning import QLearner


def test_visits_step_size_is_one_over_n() -> None:
    ql = QLearner(alpha="visits")
    assert [ql.step_size("s", "a") for _ in range(4)] == [1.0, 0.5, 1 / 3, 0.25]
    assert ql.step_size("s", "b") == 1.0


def test_running_mean_of_noisy_targets() -> None:
    ql = QLearner(alpha="visits")
    targets = [10.0, -100.0, 40.0, 10.0]
    for t in targets:
        ql.q[("s", "a")] += ql.step_size("s", "a") * (t - ql.q[("s", "a")])
    assert ql.q[("s", "a")] == pytest.approx(sum(targets) / len(targets))


def test_constant_alpha_unchanged_and_bad_alpha_rejected() -> None:
    assert QLearner(alpha=0.1).step_size("s", "a") == 0.1
    with pytest.raises(ValueError):
        QLearner(alpha="sqrt")
