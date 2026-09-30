"""Discrete hidden Markov model inference (plan rows **5.3.3**, **5.3.4**).

Hidden states ``s_t`` in ``{0..S-1}`` (the Phase 5 tagger uses KEEP / DROP / REPLACE, row 5.3.1),
observations ``o_t`` in ``{0..V-1}``. Parameters:

* ``start[s]      = P(s_1 = s)``
* ``trans[s, s']  = P(s_{t+1} = s' | s_t = s)``
* ``emit[s, o]    = P(o_t = o | s_t = s)``

All recursions run in log space so long prompts do not underflow.

**Forward algorithm (5.3.3).** ``alpha_t(s) = log P(o_1..o_t, s_t = s)``:
``alpha_1(s) = log start[s] + log emit[s, o_1]`` and
``alpha_t(s') = logsumexp_s(alpha_{t-1}(s) + log trans[s, s']) + log emit[s', o_t]``;
``log P(o) = logsumexp_s alpha_T(s)``. Cost ``O(T S^2)`` instead of ``O(T S^T)`` enumeration.

**Viterbi decoding (5.3.4).** ``delta_t(s) = max over s_1..s_{t-1} of log P(s_1..s_{t-1}, s_t = s,
o_1..o_t)``: the forward recursion with ``max`` in place of ``logsumexp``, plus back-pointers
``psi_t(s') = argmax_s(delta_{t-1}(s) + log trans[s, s'])``. Backtracking from
``argmax_s delta_T(s)`` yields the single most likely state sequence (ties go to the lowest
state id).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.special import logsumexp

__all__ = ["DiscreteHMM"]

FloatArray = npt.NDArray[np.float64]


def _log(p: FloatArray) -> FloatArray:
    with np.errstate(divide="ignore"):
        return np.log(p)


@dataclass(frozen=True)
class DiscreteHMM:
    """HMM with categorical emissions. Rows of every matrix must sum to 1."""

    start: FloatArray
    trans: FloatArray
    emit: FloatArray

    def __post_init__(self) -> None:
        s = len(self.start)
        if self.trans.shape != (s, s) or self.emit.ndim != 2 or self.emit.shape[0] != s:
            raise ValueError(
                f"shape mismatch: start {self.start.shape}, trans {self.trans.shape}, "
                f"emit {self.emit.shape}"
            )
        for name, m in (("start", self.start), ("trans", self.trans), ("emit", self.emit)):
            if np.any(m < 0) or not np.allclose(m.sum(axis=-1), 1.0):
                raise ValueError(f"{name} must be non-negative with rows summing to 1")

    @property
    def n_states(self) -> int:
        return len(self.start)

    def _check(self, obs: npt.ArrayLike) -> npt.NDArray[np.int64]:
        o = np.asarray(obs, dtype=np.int64)
        if o.ndim != 1 or len(o) == 0:
            raise ValueError("observation sequence must be 1-D and non-empty")
        if o.min() < 0 or o.max() >= self.emit.shape[1]:
            raise ValueError(f"observation ids must be in [0, {self.emit.shape[1]})")
        return o

    def forward(self, obs: npt.ArrayLike) -> FloatArray:
        """Log forward table ``alpha``, shape (T, S)."""
        o = self._check(obs)
        log_t, log_e = _log(self.trans), _log(self.emit)
        alpha = np.empty((len(o), self.n_states))
        alpha[0] = _log(self.start) + log_e[:, o[0]]
        for t in range(1, len(o)):
            alpha[t] = logsumexp(alpha[t - 1][:, None] + log_t, axis=0) + log_e[:, o[t]]
        return alpha

    def log_likelihood(self, obs: npt.ArrayLike) -> float:
        """``log P(o_1..o_T)`` by the forward algorithm."""
        return float(logsumexp(self.forward(obs)[-1]))

    def viterbi(self, obs: npt.ArrayLike) -> tuple[npt.NDArray[np.int64], float]:
        """Most likely state path and its joint log-probability ``log P(path, o)``."""
        o = self._check(obs)
        log_t, log_e = _log(self.trans), _log(self.emit)
        n = len(o)
        delta = _log(self.start) + log_e[:, o[0]]
        back = np.zeros((n, self.n_states), dtype=np.int64)
        for t in range(1, n):
            scores = delta[:, None] + log_t
            back[t] = np.argmax(scores, axis=0)
            delta = scores[back[t], np.arange(self.n_states)] + log_e[:, o[t]]
        path = np.empty(n, dtype=np.int64)
        path[-1] = int(np.argmax(delta))
        for t in range(n - 1, 0, -1):
            path[t - 1] = back[t, path[t]]
        return path, float(delta[path[-1]])
