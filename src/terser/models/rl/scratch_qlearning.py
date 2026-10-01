"""Tabular Q-learning from scratch (plan rows **9.2.1**, **9.2.2**).

Update (derived from the Bellman optimality equation in ``docs/derivations/rl.md``):

    Q(s, a) <- Q(s, a) + alpha * (r + gamma * max_a' Q(s', a') - Q(s, a))

with ``max`` over legal actions in ``s'`` and target ``r`` at terminal steps. Behaviour policy:
epsilon-greedy with ``epsilon_t = max(eps_min, eps0 * decay^t)`` per episode. ``max_delta_``
records the largest |Q change| per episode, for the C9.1 convergence check.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass, field
from typing import Any

from terser.models.rl.env import RewriteEnv

__all__ = ["QLearner"]

Episode = tuple[str, str, Sequence[str]]  # prompt, category, constraints


@dataclass
class QLearner:
    alpha: float = 0.1
    gamma: float = 0.99
    eps0: float = 1.0
    eps_min: float = 0.05
    decay: float = 0.995
    seed: int = 0
    q: dict[tuple[Hashable, str], float] = field(default_factory=lambda: defaultdict(float))
    returns_: list[float] = field(default_factory=list)
    max_delta_: list[float] = field(default_factory=list)

    def greedy(self, state: Hashable, legal: Sequence[str]) -> str:
        return max(legal, key=lambda a: (self.q[(state, a)], -legal.index(a)))

    def fit(
        self,
        env: RewriteEnv,
        episodes: Sequence[Episode],
        n_episodes: int,
        callback: Callable[[int, float], Any] | None = None,
    ) -> QLearner:
        rng = random.Random(self.seed)
        for t in range(n_episodes):
            eps = max(self.eps_min, self.eps0 * self.decay**t)
            prompt, category, constraints = episodes[rng.randrange(len(episodes))]
            s = env.reset(prompt, category, constraints)
            total, delta, done = 0.0, 0.0, False
            while not done:
                legal = env.legal_actions()
                a = rng.choice(legal) if rng.random() < eps else self.greedy(s, legal)
                s2, r, done = env.step(a)
                target = (
                    r
                    if done
                    else r + self.gamma * max(self.q[(s2, b)] for b in env.legal_actions())
                )
                change = self.alpha * (target - self.q[(s, a)])
                self.q[(s, a)] += change
                delta = max(delta, abs(change))
                total += r
                s = s2
            self.returns_.append(total)
            self.max_delta_.append(delta)
            if callback:
                callback(t, total)
        return self

    def policy_rollout(self, env: RewriteEnv, episode: Episode) -> tuple[str, str]:
        """Greedy operator sequence and final draft for one prompt."""
        s = env.reset(*episode)
        seq, done = [], False
        while not done:
            a = self.greedy(s, env.legal_actions())
            seq.append(a)
            s, _, done = env.step(a)
        return "".join(seq), env.draft
