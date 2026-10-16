"""Pair features for the quality gate on (original, rewrite) (plan row **3.1.3**).

=============================  ==============================================================
``token_ratio``                cost(rewrite) / cost(original) on target T
``tokens_saved``               cost(original) - cost(rewrite)
``max_si_removed``             largest self-information (bits) among tokens not kept
``sum_si_removed``             total self-information of tokens not kept
``si_removed_share``           ``sum_si_removed`` / total self-information of the original
                               (row 5.1.7: ``sum_si_removed`` tops the importance ranking, so
                               the share of the prompt's information removed is made explicit)
``negations_removed``          negations in the original minus those in the rewrite (>= 0)
``numbers_removed``            likewise for numbers
``constraints_removed``        likewise for constraint keywords
``drop_ratio``                 share of original tokens labelled DROP or REPLACE
``embedding_cosine``           cosine of sentence embeddings of the two texts
``cat_<category>``             category one-hot over :class:`terser.data.categories.Category`
=============================  ==============================================================

As in :mod:`terser.features.span`, the tokenizer, LM and embedder are injected callables.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any, Final

import numpy as np
import pandas as pd

from terser.data.categories import Category
from terser.features.alignment import KEEP, align
from terser.features.lexicons import is_constraint, is_negation, is_number
from terser.features.span import SurprisalFn

__all__ = ["PAIR_FEATURES", "cosine", "pair_feature_table", "pair_features"]

CATEGORIES: Final = tuple(c.value for c in Category)
PAIR_FEATURES: Final = (
    "token_ratio",
    "tokens_saved",
    "max_si_removed",
    "sum_si_removed",
    "si_removed_share",
    "negations_removed",
    "numbers_removed",
    "constraints_removed",
    "drop_ratio",
    "embedding_cosine",
    *(f"cat_{c}" for c in CATEGORIES),
)

EmbedFn = Callable[[Sequence[str]], np.ndarray]


def cosine(u: np.ndarray, v: np.ndarray) -> float:
    """Cosine similarity; 0 when either vector is zero."""
    nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return float(np.dot(u, v) / (nu * nv))


def _removed(pred: Callable[[str], bool], a: Sequence[str], b: Sequence[str]) -> float:
    return float(max(0, sum(map(pred, a)) - sum(map(pred, b))))


def pair_features(
    original: str,
    rewrite: str,
    category: str,
    cost_fn: Callable[[str], int],
    surprisal_fn: SurprisalFn,
    embed_fn: EmbedFn,
) -> dict[str, float]:
    """Feature dict for one pair, keys in :data:`PAIR_FEATURES` order.

    Raises:
        ValueError: If ``category`` is unknown.
    """
    if category not in CATEGORIES:
        raise ValueError(f"unknown category {category!r}")
    al = align(original, rewrite)
    c_orig, c_rew = cost_fn(original), cost_fn(rewrite)
    bits = list(surprisal_fn(al.original))
    gone = [b for b, lab in zip(bits, al.labels, strict=True) if lab != KEEP]
    emb = np.asarray(embed_fn([original, rewrite]), dtype=np.float64)
    out: dict[str, float] = {
        "token_ratio": c_rew / c_orig if c_orig else 1.0,
        "tokens_saved": float(c_orig - c_rew),
        "max_si_removed": max(gone, default=0.0),
        "sum_si_removed": float(sum(gone)),
        "si_removed_share": float(sum(gone)) / sum(bits) if sum(bits) > 0 else 0.0,
        "negations_removed": _removed(is_negation, al.original, al.rewrite),
        "numbers_removed": _removed(is_number, al.original, al.rewrite),
        "constraints_removed": _removed(is_constraint, al.original, al.rewrite),
        "drop_ratio": al.drop_ratio(),
        "embedding_cosine": cosine(emb[0], emb[1]),
    }
    out.update({f"cat_{c}": float(c == category) for c in CATEGORIES})
    return out


def pair_feature_table(
    pairs: Iterable[dict[str, Any]],
    cost_fn: Callable[[str], int],
    surprisal_fn: SurprisalFn,
    embed_fn: EmbedFn,
) -> pd.DataFrame:
    """One row per labelled pair: ``id``, ``prompt_id``, ``category``, ``label`` + features.

    Each pair needs ``id``, ``prompt_id``, ``category``, ``original``, ``rewrite`` and ``label``.
    """
    rows = [
        {
            "id": p["id"],
            "prompt_id": p["prompt_id"],
            "category": p["category"],
            "label": p["label"],
            **pair_features(
                str(p["original"]),
                str(p["rewrite"]),
                str(p["category"]),
                cost_fn,
                surprisal_fn,
                embed_fn,
            ),
        }
        for p in pairs
    ]
    return pd.DataFrame(rows, columns=["id", "prompt_id", "category", "label", *PAIR_FEATURES])
