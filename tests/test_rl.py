"""Rows 9.1.2 (environment), 9.1.4 (rewards), 9.2.1 (Q-learning mechanics)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from terser.models.rl.env import ACTIONS, STEP_PENALTY, STOP, RewriteEnv, stop_reward
from terser.models.rl.scratch_qlearning import QLearner
from terser.operators.registry import apply_ops

FIXTURES = Path(__file__).resolve().parents[1] / "data" / "samples" / "casual_prompts.jsonl"


def _words(t: str) -> int:
    return len(t.split())


def _env(gate_value: float = 0.0) -> RewriteEnv:
    return RewriteEnv(count=_words, gate=lambda o, r: gate_value)


PROMPT = "hey so um could you maybe please just write a python function that sorts a list thanks"


def test_stop_reward_formula() -> None:
    assert stop_reward(100, 60, False) == pytest.approx(40.0)
    assert stop_reward(100, 60, True) == pytest.approx(-60.0)
    assert stop_reward(10, 10, False) == 0.0
    assert stop_reward(100, 50, True, lam=10) == pytest.approx(40.0)
    with pytest.raises(ValueError):
        stop_reward(0, 0, False)


def test_step_penalty_and_stop() -> None:
    env = _env()
    s0 = env.reset(PROMPT, "code", ["python", "sorts"])
    assert s0[3] == "code" and s0[4] == "" and s0[0] >= 2 and s0[1] == 1
    assert env.legal_actions() == list(ACTIONS)
    s1, r, done = env.step("F")
    assert r == STEP_PENALTY and not done and s1[4] == "F"
    assert env.draft == apply_ops(PROMPT, "F")
    assert "F" not in env.legal_actions()
    with pytest.raises(ValueError):
        env.step("F")
    _, r, done = env.step(STOP)
    expected = 100 * (1 - _words(apply_ops(PROMPT, "F")) / _words(PROMPT))
    assert done and r == pytest.approx(expected)
    with pytest.raises(RuntimeError):
        env.step(STOP)


def test_degraded_stop_is_penalised() -> None:
    env = _env(gate_value=0.9)
    env.reset(PROMPT, "code")
    env.step("F")
    _, r, _ = env.step(STOP)
    assert r == pytest.approx(100 * (1 - _words(apply_ops(PROMPT, "F")) / _words(PROMPT)) - 100)


def test_draft_depends_on_set_not_order() -> None:
    a, b = _env(), _env()
    a.reset(PROMPT, "code")
    b.reset(PROMPT, "code")
    for x in "FH":
        a.step(x)
    for x in "HF":
        b.step(x)
    assert a.draft == b.draft and a.state() == b.state()


def test_all_ops_implicit_stop() -> None:
    env = _env()
    env.reset(PROMPT, "code")
    for i, a in enumerate("FHDILC"):
        _, r, done = env.step(a)
        assert not done and r == STEP_PENALTY, i
    _, r, done = env.step("S")
    assert done and r != STEP_PENALTY


def test_q_learning_prefers_safe_savings() -> None:
    """Gate penalises dropping 'python'; the learnt greedy policy should keep it and still save."""
    rows = [json.loads(x) for x in FIXTURES.read_text(encoding="utf-8").splitlines()][:5]
    eps = [(r["prompt"], r["category"], r["constraints"]) for r in rows]

    def gate(o: str, r: str) -> float:
        return 1.0 if len(r) < 0.3 * len(o) else 0.0

    env = RewriteEnv(count=_words, gate=gate)
    q = QLearner(alpha=0.3, gamma=0.9, decay=0.998, seed=0).fit(env, eps, 3000)
    assert len(q.returns_) == 3000
    explore = sum(q.returns_[:50]) / 50  # near-random behaviour policy
    greedy_returns = []
    for ep in eps:
        seq, draft = q.policy_rollout(env, ep)
        assert seq.endswith(STOP) or len(seq) == 7
        assert gate(ep[0], draft) == 0.0
        saved = 100 * (1 - _words(draft) / _words(ep[0]))
        greedy_returns.append(saved + STEP_PENALTY * (len(seq) - 1))
    assert sum(greedy_returns) / len(eps) > explore
    assert len(q.max_delta_) == 3000
