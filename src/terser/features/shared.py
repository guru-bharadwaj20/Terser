"""Prompt features computed by the *same* code offline (training) and online (serving).

Plan row **11.1.3**. Training/serving skew is one of the production risks named in ADR-0010: if
the gate is trained on features computed one way and served on features computed another, its
scores drift silently. Both paths therefore call :func:`extract_features`:

* offline, :func:`featurize_records` / :func:`featurize_frame` build the training matrix;
* online, :class:`terser.serving.service.RewriteService` calls :func:`extract_features` per
  request.

The parity test (``tests/features/test_parity.py``) checks the two paths give identical vectors
on 100 fixture prompts. Feature order is fixed by :data:`FEATURE_NAMES`.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from terser.tokenizers import count_tokens

__all__ = [
    "FEATURE_NAMES",
    "GATE_FEATURE_NAMES",
    "extract_features",
    "featurize_frame",
    "featurize_records",
    "gate_vector",
]

Counter = Callable[[str], int]

_WORD_RE = re.compile(r"[A-Za-z0-9']+")
_FILLER = frozenset(
    "um uh like basically actually literally just really so ok okay well kinda sorta".split()
)
_HEDGE_RE = re.compile(
    r"\b(could you|can you|would you|i was wondering|maybe|perhaps|if possible|please|"
    r"if you can|i think|sort of|kind of|thanks|thank you)\b",
    re.IGNORECASE,
)
_CONSTRAINT_RE = re.compile(
    r"\b(must|should|only|exactly|at least|at most|no more than|never|always|don't|do not|"
    r"without|include|exclude)\b",
    re.IGNORECASE,
)

FEATURE_NAMES: tuple[str, ...] = (
    "n_chars",
    "n_words",
    "n_tokens",
    "n_lines",
    "n_sentences",
    "chars_per_token",
    "filler_ratio",
    "n_hedges",
    "n_constraints",
    "n_questions",
    "has_code",
    "digit_ratio",
    "upper_ratio",
    "unique_word_ratio",
)


def extract_features(
    prompt: str, target: str = "gpt-4o", counter: Counter | None = None
) -> dict[str, float]:
    """Feature vector of one prompt (keys in :data:`FEATURE_NAMES` order).

    ``counter`` overrides the token counter (default: the ground-truth tokenizer of ``target``).
    """
    count = counter if counter is not None else (lambda s: count_tokens(s, target))
    words = [w.lower() for w in _WORD_RE.findall(prompt)]
    n_words = len(words)
    n_chars = len(prompt)
    n_tokens = count(prompt) if prompt else 0
    letters = [c for c in prompt if c.isalpha()]
    feats = {
        "n_chars": float(n_chars),
        "n_words": float(n_words),
        "n_tokens": float(n_tokens),
        "n_lines": float(prompt.count("\n") + 1 if prompt else 0),
        "n_sentences": float(len([s for s in re.split(r"[.!?]+(?:\s|$)", prompt) if s.strip()])),
        "chars_per_token": n_chars / n_tokens if n_tokens else 0.0,
        "filler_ratio": sum(w in _FILLER for w in words) / n_words if n_words else 0.0,
        "n_hedges": float(len(_HEDGE_RE.findall(prompt))),
        "n_constraints": float(len(_CONSTRAINT_RE.findall(prompt))),
        "n_questions": float(prompt.count("?")),
        "has_code": float("```" in prompt or bool(re.search(r"^\s{4}\S", prompt, re.M))),
        "digit_ratio": sum(c.isdigit() for c in prompt) / n_chars if n_chars else 0.0,
        "upper_ratio": sum(c.isupper() for c in letters) / len(letters) if letters else 0.0,
        "unique_word_ratio": len(set(words)) / n_words if n_words else 0.0,
    }
    return {k: feats[k] for k in FEATURE_NAMES}


def featurize_records(
    records: Iterable[Mapping[str, Any]],
    target: str = "gpt-4o",
    text_key: str = "prompt",
    counter: Counter | None = None,
) -> list[dict[str, float]]:
    """Offline batch path: one feature dict per record."""
    return [extract_features(str(r[text_key]), target, counter) for r in records]


def featurize_frame(
    frame: Any, target: str = "gpt-4o", text_key: str = "prompt", counter: Counter | None = None
) -> Any:
    """Offline batch path over a pandas DataFrame; returns a DataFrame with the same index."""
    import pandas as pd

    rows = [extract_features(str(t), target, counter) for t in frame[text_key]]
    return pd.DataFrame(rows, index=frame.index, columns=list(FEATURE_NAMES))


GATE_FEATURE_NAMES: tuple[str, ...] = (
    *(f"orig_{n}" for n in FEATURE_NAMES),
    *(f"rew_{n}" for n in FEATURE_NAMES),
    "token_ratio",
    "constraint_ratio",
)


def gate_vector(orig: Mapping[str, float], rew: Mapping[str, float]) -> list[float]:
    """Pair features for the quality gate: both prompts' features plus two ratios.

    Used to build the gate's training matrix and, per request, its serving input.
    """
    tok = orig["n_tokens"]
    con = orig["n_constraints"]
    return [
        *(float(orig[n]) for n in FEATURE_NAMES),
        *(float(rew[n]) for n in FEATURE_NAMES),
        rew["n_tokens"] / tok if tok else 1.0,
        rew["n_constraints"] / con if con else 1.0,
    ]
