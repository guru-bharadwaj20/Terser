"""Rewrite MDP over operators (plan rows **9.1.2**, **9.1.4**).

State: ``(filler_bucket, constraint_bucket, length_bucket, category, ops_applied)`` where
``ops_applied`` is the sorted string of operator codes used so far. Actions: the operator codes
``F H D I L C S`` (each usable once) or ``STOP``.

A step with an operator re-applies the chosen operator *set* to the original prompt (operators
run in canonical order, row 1.2.8), so the draft is a function of the set. Rewards (9.1.4):

* every operator step: ``-0.01``;
* ``STOP`` (or the step after which no operator is left): ``savings_pct - lambda * 1[degraded]``,
  with ``savings_pct = 100 * (1 - tokens_T(draft) / tokens_T(original))`` and ``lambda = 100``.

``degraded`` comes from the quality gate (``gate(original, draft) >= threshold``), the cheap
reward model; no LLM is called during learning. Derivation: ``docs/derivations/rl.md``.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from terser.operators.registry import apply_ops

__all__ = ["ACTIONS", "OPERATOR_ACTIONS", "STEP_PENALTY", "STOP", "RewriteEnv", "stop_reward"]

STOP = "STOP"
OPERATOR_ACTIONS = ("F", "H", "D", "I", "L", "C", "S")
ACTIONS = (*OPERATOR_ACTIONS, STOP)
STEP_PENALTY = -0.01
LAMBDA = 100.0
FILLER = re.compile(
    r"\b(um+|uh+|so|basically|just|really|actually|like|maybe|please|kind of|sort of|"
    r"i was wondering|could you|thanks?)\b",
    re.IGNORECASE,
)
State = tuple[int, int, int, str, str]


def stop_reward(
    tokens_original: int, tokens_draft: int, degraded: bool, lam: float = LAMBDA
) -> float:
    """``savings_pct - lam * 1[degraded]`` on STOP."""
    if tokens_original <= 0:
        raise ValueError("original must have at least one token")
    savings = 100.0 * (1.0 - tokens_draft / tokens_original)
    return savings - lam * float(degraded)


def _bucket(x: float, edges: Sequence[float]) -> int:
    return sum(x >= e for e in edges)


@dataclass
class RewriteEnv:
    """One episode per prompt; ``reset(prompt, category, constraints)`` then ``step(action)``."""

    count: Callable[[str], int]
    gate: Callable[[str, str], float]
    threshold: float = 0.5
    lam: float = LAMBDA
    target: str | None = None
    # Operator application; injectable so callers can cache it (operators are deterministic).
    apply: Callable[[str, str, str | None], str] = apply_ops
    _original: str = field(init=False, default="")
    _category: str = field(init=False, default="")
    _constraints: tuple[str, ...] = field(init=False, default=())
    _applied: str = field(init=False, default="")
    _draft: str = field(init=False, default="")
    done: bool = field(init=False, default=True)

    def reset(self, prompt: str, category: str, constraints: Sequence[str] = ()) -> State:
        self._original, self._draft, self._applied = prompt, prompt, ""
        self._category, self._constraints = category, tuple(constraints)
        self.done = False
        return self.state()

    @property
    def draft(self) -> str:
        return self._draft

    def state(self) -> State:
        text = self._draft
        fillers = len(FILLER.findall(text))
        kept = sum(c.lower() in text.lower() for c in self._constraints)
        return (
            _bucket(fillers, (1, 3, 6)),
            _bucket(kept, (1, 3, 6)),
            _bucket(self.count(text), (32, 64, 128, 256)),
            self._category,
            self._applied,
        )

    def legal_actions(self) -> list[str]:
        return [a for a in OPERATOR_ACTIONS if a not in self._applied] + [STOP]

    def _final(self) -> float:
        degraded = self.gate(self._original, self._draft) >= self.threshold
        self.done = True
        return stop_reward(self.count(self._original), self.count(self._draft), degraded, self.lam)

    def step(self, action: str) -> tuple[State, float, bool]:
        """Apply ``action``; returns ``(state, reward, done)``."""
        if self.done:
            raise RuntimeError("episode finished; call reset()")
        if action not in self.legal_actions():
            raise ValueError(f"illegal action {action!r}; legal: {self.legal_actions()}")
        if action == STOP:
            return self.state(), self._final(), True
        self._applied = "".join(sorted(self._applied + action))
        self._draft = self.apply(self._original, self._applied, self.target)
        reward = STEP_PENALTY
        if len(self._applied) == len(OPERATOR_ACTIONS):  # nothing left: implicit STOP
            reward += self._final()
        return self.state(), reward, self.done
