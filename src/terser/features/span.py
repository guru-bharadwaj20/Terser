"""Per-token span features for the KEEP / DROP / REPLACE tagger (plan row **3.1.2**).

For each token of an original prompt (tokens from :func:`terser.features.alignment.tokenize`):

=====================  ==================================================================
``token_cost``         tokens the word costs on target T (with its leading space)
``self_information``   ``-log2 p(token | prefix)`` under a causal LM (GPT-2 small by default)
``stopword_ratio``     share of stopwords in a 5-token window centred on the token
``rel_position``       ``i / (n - 1)``, 0 for the first token and 1 for the last
``pos_<TAG>``          one-hot universal POS tag (NOUN, VERB, ADJ, ADV, PRON, DET, ADP, ...)
``is_filler``          hit in the filler lexicon (mirrors operators ``F`` / ``H``)
``is_negation``        negation keyword
``is_number``          digit string or number word
``is_constraint``      constraint keyword ("must", "only", "json", "at least", ...)
``is_punct``           single punctuation mark
=====================  ==================================================================

Expensive resources (the target tokenizer, the LM, the POS tagger) are passed in as callables so
the feature code is testable without downloads: ``cost_fn(text) -> int``,
``surprisal_fn(tokens) -> bits per token`` and ``pos_fn(tokens) -> tags``. :func:`gpt2_surprisal`
and :func:`spacy_pos` build the production callables (they need the ``gpu`` / ``pii`` extras).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from typing import Any, Final

import pandas as pd

from terser.features.alignment import align, tokenize
from terser.features.lexicons import (
    is_constraint,
    is_filler,
    is_negation,
    is_number,
    is_stopword,
)

__all__ = [
    "POS_TAGS",
    "SPAN_FEATURES",
    "gpt2_surprisal",
    "span_feature_table",
    "span_features",
    "spacy_pos",
    "unigram_surprisal",
]

POS_TAGS: Final = (
    "NOUN",
    "VERB",
    "ADJ",
    "ADV",
    "PRON",
    "DET",
    "ADP",
    "AUX",
    "CCONJ",
    "NUM",
    "PART",
    "PUNCT",
    "X",
)
WINDOW: Final = 2

CostFn = Callable[[str], int]
SurprisalFn = Callable[[Sequence[str]], Sequence[float]]
PosFn = Callable[[Sequence[str]], Sequence[str]]

SPAN_FEATURES: Final = (
    "token_cost",
    "self_information",
    "stopword_ratio",
    "rel_position",
    *(f"pos_{t}" for t in POS_TAGS),
    "is_filler",
    "is_negation",
    "is_number",
    "is_constraint",
    "is_punct",
)


def _is_punct(tok: str) -> bool:
    return len(tok) == 1 and not tok.isalnum()


def span_features(
    tokens: Sequence[str],
    cost_fn: CostFn,
    surprisal_fn: SurprisalFn,
    pos_fn: PosFn | None = None,
) -> list[dict[str, float]]:
    """One feature dict per token, keys in :data:`SPAN_FEATURES` order.

    Raises:
        ValueError: If ``surprisal_fn`` or ``pos_fn`` returns the wrong number of values.
    """
    n = len(tokens)
    bits = list(surprisal_fn(tokens))
    tags = list(pos_fn(tokens)) if pos_fn is not None else ["X"] * n
    if len(bits) != n or len(tags) != n:
        raise ValueError(f"expected {n} surprisals/tags, got {len(bits)}/{len(tags)}")
    stop = [is_stopword(t) for t in tokens]
    rows: list[dict[str, float]] = []
    for i, tok in enumerate(tokens):
        lo, hi = max(0, i - WINDOW), min(n, i + WINDOW + 1)
        tag = tags[i] if tags[i] in POS_TAGS else "X"
        row: dict[str, float] = {
            "token_cost": float(cost_fn(tok if _is_punct(tok) else " " + tok)),
            "self_information": float(bits[i]),
            "stopword_ratio": sum(stop[lo:hi]) / (hi - lo),
            "rel_position": i / (n - 1) if n > 1 else 0.0,
        }
        row.update({f"pos_{t}": float(t == tag) for t in POS_TAGS})
        row.update(
            {
                "is_filler": float(is_filler(tok)),
                "is_negation": float(is_negation(tok)),
                "is_number": float(is_number(tok)),
                "is_constraint": float(is_constraint(tok)),
                "is_punct": float(_is_punct(tok)),
            }
        )
        rows.append(row)
    return rows


def span_feature_table(
    pairs: Iterable[dict[str, Any]],
    cost_fn: CostFn,
    surprisal_fn: SurprisalFn,
    pos_fn: PosFn | None = None,
) -> pd.DataFrame:
    """Token-level table for (original, oracle rewrite) pairs, with alignment labels.

    Each pair needs ``prompt_id``, ``category``, ``original`` and ``rewrite``. Columns are
    ``prompt_id``, ``category``, ``position``, ``token``, ``label`` and :data:`SPAN_FEATURES`.
    """
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        al = align(str(pair["original"]), str(pair["rewrite"]))
        feats = span_features(al.original, cost_fn, surprisal_fn, pos_fn)
        for i, (tok, lab, f) in enumerate(zip(al.original, al.labels, feats, strict=True)):
            rows.append(
                {
                    "prompt_id": pair["prompt_id"],
                    "category": pair["category"],
                    "position": i,
                    "token": tok,
                    "label": lab,
                    **f,
                }
            )
    cols = ["prompt_id", "category", "position", "token", "label", *SPAN_FEATURES]
    return pd.DataFrame(rows, columns=cols)


def unigram_surprisal(corpus: Iterable[str], alpha: float = 1.0) -> SurprisalFn:
    """Context-free ``-log2 p(token)`` with add-``alpha`` smoothing, fitted on ``corpus``.

    A cheap stand-in for :func:`gpt2_surprisal` in tests and CPU smoke runs; never use it for
    reported numbers.
    """
    counts = Counter(t.lower() for text in corpus for t in tokenize(text))
    total = sum(counts.values())
    vocab = len(counts) + 1

    def fn(tokens: Sequence[str]) -> list[float]:
        return [-math.log2((counts[t.lower()] + alpha) / (total + alpha * vocab)) for t in tokens]

    return fn


def gpt2_surprisal(model_name: str = "gpt2") -> SurprisalFn:  # pragma: no cover - needs torch
    """``-log2 p(word | prefix)`` under a Hugging Face causal LM, summed over the word's sub-tokens.

    The first word is scored after the BOS token. Requires ``torch`` and ``transformers``.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).eval()

    def fn(tokens: Sequence[str]) -> list[float]:
        ids: list[int] = [tok.bos_token_id]
        owner: list[int] = [-1]
        for w, word in enumerate(tokens):
            piece = tok.encode(word if w == 0 or _is_punct(word) else " " + word)
            ids.extend(piece)
            owner.extend([w] * len(piece))
        with torch.no_grad():
            logits = model(torch.tensor([ids])).logits[0]
        logp = torch.log_softmax(logits[:-1], dim=-1)
        bits = [0.0] * len(tokens)
        for pos in range(1, len(ids)):
            bits[owner[pos]] -= float(logp[pos - 1, ids[pos]]) / math.log(2)
        return bits

    return fn


def spacy_pos(model: str = "en_core_web_sm") -> PosFn:  # pragma: no cover - needs spacy model
    """Universal POS tags from spaCy on pre-tokenised words."""
    import spacy
    from spacy.tokens import Doc

    nlp = spacy.load(model)

    def fn(tokens: Sequence[str]) -> list[str]:
        doc = nlp(Doc(nlp.vocab, words=list(tokens)))
        return [t.pos_ for t in doc]

    return fn
