"""Discrete hidden Markov model span tagger (plan rows **5.3.1**-**5.3.6**).

Row 5.3.1 model definition: hidden states :data:`TAGS` = KEEP / DROP / REPLACE; observations are
:class:`TokenClassEncoder` ids (filler-lexicon id x coarse POS x token-cost bucket). Rows 5.3.2
(:func:`fit_supervised`), 5.3.5 (:func:`baum_welch`) and 5.3.6 (:func:`apply_tags`) follow the
inference code below.

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

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.special import logsumexp

__all__ = [
    "FILLER_LEXICON",
    "POS_CLASSES",
    "TAGS",
    "DiscreteHMM",
    "TokenClassEncoder",
    "apply_tags",
    "baum_welch",
    "coarse_pos",
    "fit_supervised",
]

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

    def backward(self, obs: npt.ArrayLike) -> FloatArray:
        """Log backward table ``beta_t(s) = log P(o_{t+1}..o_T | s_t = s)``, shape (T, S)."""
        o = self._check(obs)
        log_t, log_e = _log(self.trans), _log(self.emit)
        beta = np.zeros((len(o), self.n_states))
        for t in range(len(o) - 2, -1, -1):
            beta[t] = logsumexp(log_t + (log_e[:, o[t + 1]] + beta[t + 1])[None, :], axis=1)
        return beta


# --------------------------------------------------------------------------------------------
# Row 5.3.1: the span-tagger model definition.
# --------------------------------------------------------------------------------------------

TAGS: tuple[str, ...] = ("KEEP", "DROP", "REPLACE")
"""Hidden states of the span tagger, in state-id order."""

FILLER_LEXICON: tuple[str, ...] = (
    "um", "uh", "erm", "hmm", "like", "so", "well", "basically", "literally", "honestly",
    "totally", "actually", "really", "just", "maybe", "perhaps", "please", "kindly", "hey", "hi",
    "hello", "thanks", "thank", "kind", "sort", "kinda", "sorta", "know", "mean", "wondering",
    "could", "would", "possibly", "lol", "haha",
)  # fmt: skip
"""Filler-lexicon entries; lexicon id ``i + 1`` for entry ``i`` and ``0`` for any other word."""

POS_CLASSES: tuple[str, ...] = ("PRON", "DET", "ADP", "CONJ", "AUX", "NUM", "PUNCT", "OTHER")
_CLOSED_CLASS: dict[str, str] = {
    **dict.fromkeys("i me my you your we our it its they them he she this that".split(), "PRON"),
    **dict.fromkeys("a an the some any every each all".split(), "DET"),
    **dict.fromkeys("of in on at to for with from by about into over as".split(), "ADP"),
    **dict.fromkeys("and or but if because while although than".split(), "CONJ"),
    **dict.fromkeys(
        (
            "is are was were be been am do does did can could would will should may might must "
            "have has had"
        ).split(),
        "AUX",
    ),
}
COST_BUCKETS: tuple[int, ...] = (1, 2, 3)
"""Upper edges of the token-cost buckets: cost 1, cost 2, cost >= 3 tokens on the target."""


def coarse_pos(word: str) -> str:
    """Closed-class lookup POS (no tagger dependency); open-class words are ``OTHER``."""
    w = word.lower()
    if w in _CLOSED_CLASS:
        return _CLOSED_CLASS[w]
    if w.replace(".", "", 1).replace(",", "").isdigit():
        return "NUM"
    if w and not any(ch.isalnum() for ch in w):
        return "PUNCT"
    return "OTHER"


def _unit_cost(_word: str) -> int:
    return 1


@dataclass(frozen=True)
class TokenClassEncoder:
    """Discretise a word into one observation id: filler-lexicon id x POS x cost bucket.

    ``obs = (lexicon_id * n_pos + pos_id) * n_cost + cost_bucket``, so
    ``n_obs = (len(lexicon) + 1) * len(POS_CLASSES) * len(COST_BUCKETS)``.

    Args:
        lexicon: Filler lexicon (lower-case words).
        pos: Word -> POS class in :data:`POS_CLASSES` (default :func:`coarse_pos`; a spaCy tagger
            can be passed in after mapping to these classes).
        cost: Word -> token count on the target tokenizer (default: 1 per word).
    """

    lexicon: tuple[str, ...] = FILLER_LEXICON
    pos: Callable[[str], str] = coarse_pos
    cost: Callable[[str], int] = _unit_cost

    @property
    def n_obs(self) -> int:
        return (len(self.lexicon) + 1) * len(POS_CLASSES) * len(COST_BUCKETS)

    def encode_word(self, word: str) -> int:
        """Observation id of one word."""
        core = word.strip(".,!?;:\"'") or word  # "?!" stays punctuation
        w = core.lower()  # "um," and "so," hit the lexicon
        lex = self.lexicon.index(w) + 1 if w in self.lexicon else 0
        pos = POS_CLASSES.index(self.pos(core))  # "and," is CONJ, like "and"
        c = self.cost(word)
        bucket = next((i for i, e in enumerate(COST_BUCKETS) if c <= e), len(COST_BUCKETS) - 1)
        return (lex * len(POS_CLASSES) + pos) * len(COST_BUCKETS) + bucket

    def encode(self, words: Sequence[str]) -> npt.NDArray[np.int64]:
        """Observation ids of a word sequence."""
        return np.array([self.encode_word(w) for w in words], dtype=np.int64)


# --------------------------------------------------------------------------------------------
# Row 5.3.2: supervised maximum likelihood with additive (Laplace) smoothing.
# --------------------------------------------------------------------------------------------


def _normalise(counts: FloatArray) -> FloatArray:
    return np.asarray(counts / counts.sum(axis=-1, keepdims=True))


def fit_supervised(
    obs_seqs: Sequence[npt.ArrayLike],
    tag_seqs: Sequence[npt.ArrayLike],
    n_states: int,
    n_obs: int,
    smoothing: float = 1.0,
) -> DiscreteHMM:
    """MLE of start / transition / emission from tagged sequences, with additive smoothing.

    Each parameter is a normalised count plus ``smoothing`` pseudo-counts per cell, i.e. the MAP
    estimate under a symmetric Dirichlet(1 + smoothing) prior, so unseen transitions and
    emissions keep non-zero probability and every row sums to 1.
    """
    if len(obs_seqs) != len(tag_seqs) or not obs_seqs:
        raise ValueError("need equally many (non-zero) observation and tag sequences")
    if smoothing <= 0:
        raise ValueError("smoothing must be > 0 so that every row is a proper distribution")
    start = np.full(n_states, smoothing)
    trans = np.full((n_states, n_states), smoothing)
    emit = np.full((n_states, n_obs), smoothing)
    for o_raw, s_raw in zip(obs_seqs, tag_seqs, strict=True):
        o, s = np.asarray(o_raw, dtype=np.int64), np.asarray(s_raw, dtype=np.int64)
        if o.shape != s.shape or o.ndim != 1 or len(o) == 0:
            raise ValueError("each observation / tag pair must be 1-D, non-empty, same length")
        start[s[0]] += 1
        np.add.at(trans, (s[:-1], s[1:]), 1)
        np.add.at(emit, (s, o), 1)
    return DiscreteHMM(_normalise(start), _normalise(trans), _normalise(emit))


# --------------------------------------------------------------------------------------------
# Row 5.3.5: Baum-Welch (EM) on unlabelled sequences.
# --------------------------------------------------------------------------------------------


def baum_welch(
    hmm: DiscreteHMM,
    obs_seqs: Sequence[npt.ArrayLike],
    n_iter: int = 50,
    tol: float = 1e-8,
    smoothing: float = 0.0,
) -> tuple[DiscreteHMM, list[float]]:
    """Re-estimate ``hmm`` by EM; returns the new model and total log-likelihood per iteration.

    E-step: state posteriors ``gamma_t(s) ∝ alpha_t(s) beta_t(s)`` and pair posteriors
    ``xi_t(s, s') ∝ alpha_t(s) trans[s, s'] emit[s', o_{t+1}] beta_{t+1}(s')``. M-step: expected
    counts, normalised (plus optional ``smoothing`` pseudo-counts). With ``smoothing=0`` the
    log-likelihood is non-decreasing; ``history[i]`` is the likelihood *before* update ``i``.
    """
    seqs = [hmm._check(o) for o in obs_seqs]
    s_n, v_n = hmm.n_states, hmm.emit.shape[1]
    history: list[float] = []
    for _ in range(n_iter):
        log_t, log_e = _log(hmm.trans), _log(hmm.emit)
        start = np.full(s_n, smoothing)
        trans = np.full((s_n, s_n), smoothing)
        emit = np.full((s_n, v_n), smoothing)
        total = 0.0
        for o in seqs:
            alpha, beta = hmm.forward(o), hmm.backward(o)
            ll = float(logsumexp(alpha[-1]))
            total += ll
            gamma = np.exp(alpha + beta - ll)
            start += gamma[0]
            np.add.at(emit.T, o, gamma)
            if len(o) > 1:
                emit_next = (log_e[:, o[1:]].T + beta[1:])[:, None, :]
                xi = alpha[:-1, :, None] + log_t[None] + emit_next - ll
                trans += np.exp(xi).sum(axis=0)
        history.append(total)
        if len(history) > 1 and history[-1] - history[-2] < tol * max(1.0, abs(history[-2])):
            break
        # Rows that received no expected mass keep their previous distribution.
        trans = np.where(trans.sum(1, keepdims=True) > 0, trans, hmm.trans)
        emit = np.where(emit.sum(1, keepdims=True) > 0, emit, hmm.emit)
        hmm = DiscreteHMM(_normalise(start), _normalise(trans), _normalise(emit))
    return hmm, history


# --------------------------------------------------------------------------------------------
# Row 5.3.6: Viterbi tags -> compressed prompt.
# --------------------------------------------------------------------------------------------


def apply_tags(
    words: Sequence[str], tags: Sequence[int], replacements: Mapping[str, str] | None = None
) -> str:
    """Rebuild a prompt from tagged words: KEEP keeps, DROP removes, REPLACE substitutes.

    A REPLACE word without an entry in ``replacements`` (lower-cased lookup) is kept, the
    conservative choice; an empty replacement deletes it.
    """
    reps = replacements or {}
    out: list[str] = []
    for w, t in zip(words, tags, strict=True):
        tag = TAGS[int(t)]
        if tag == "KEEP":
            out.append(w)
        elif tag == "REPLACE":
            r = reps.get(w.lower(), w)
            if r:
                out.append(r)
    return " ".join(out)
