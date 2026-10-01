"""Rows 3.1.2 / 3.1.3: span and pair features on hand-checkable inputs."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pytest

from terser.features.lexicons import is_constraint, is_filler, is_negation, is_number
from terser.features.pair import PAIR_FEATURES, cosine, pair_feature_table, pair_features
from terser.features.span import (
    SPAN_FEATURES,
    span_feature_table,
    span_features,
    unigram_surprisal,
)


def words(text: str) -> int:
    """Toy tokenizer: one token per whitespace word."""
    return len(text.split())


def flat_bits(tokens: Sequence[str]) -> list[float]:
    return [float(len(t)) for t in tokens]


def bag_embed(texts: Sequence[str]) -> np.ndarray:
    vocab = sorted({w for t in texts for w in t.lower().split()})
    return np.array([[t.lower().split().count(w) for w in vocab] for t in texts], dtype=float)


def test_lexicon_flags() -> None:
    assert is_negation("Don't") and is_negation("never") and not is_negation("now")
    assert is_number("42") and is_number("2.5") and is_number("five") and not is_number("fiver")
    assert is_constraint("must") and not is_constraint("banana")
    assert is_filler("um") and not is_filler("sort")


def test_span_features_hand_values() -> None:
    toks = ["um", "do", "not", "use", "5", "loops", "."]
    rows = span_features(toks, words, flat_bits, pos_fn=lambda t: ["X", "AUX", "PART"] + ["X"] * 4)
    assert len(rows) == 7 and list(rows[0]) == list(SPAN_FEATURES)
    assert rows[0]["is_filler"] == 1.0 and rows[0]["rel_position"] == 0.0
    assert rows[6]["rel_position"] == 1.0 and rows[6]["is_punct"] == 1.0
    assert rows[2]["is_negation"] == 1.0 and rows[2]["pos_PART"] == 1.0
    assert rows[4]["is_number"] == 1.0 and rows[3]["is_constraint"] == 1.0
    assert rows[5]["self_information"] == 5.0 and rows[5]["token_cost"] == 1.0
    # window of "um" is [um, do, not]: one stopword ("do")
    assert rows[0]["stopword_ratio"] == pytest.approx(1 / 3)


def test_span_features_length_mismatch() -> None:
    with pytest.raises(ValueError):
        span_features(["a", "b"], words, lambda t: [1.0])


def test_unigram_surprisal() -> None:
    fn = unigram_surprisal(["a a b"], alpha=1.0)
    # counts a=2, b=1, total 3, vocab 3 -> p(a)=3/6, p(c)=1/6
    assert fn(["a", "c"]) == pytest.approx([1.0, math.log2(6)])


def test_span_feature_table_labels() -> None:
    table = span_feature_table(
        [{"prompt_id": "p1", "category": "code", "original": "um sort it", "rewrite": "sort it"}],
        words,
        flat_bits,
    )
    assert table["label"].tolist() == ["DROP", "KEEP", "KEEP"]
    assert table["position"].tolist() == [0, 1, 2]


def test_pair_features_hand_values() -> None:
    f = pair_features(
        "please do not use more than 5 loops",
        "use 5 loops",
        "code",
        words,
        flat_bits,
        bag_embed,
    )
    assert list(f) == list(PAIR_FEATURES)
    assert f["token_ratio"] == pytest.approx(3 / 8)
    assert f["tokens_saved"] == 5.0
    assert f["negations_removed"] == 1.0
    assert f["numbers_removed"] == 0.0
    assert f["max_si_removed"] == 6.0  # "please"
    assert f["sum_si_removed"] == 6 + 2 + 3 + 4 + 4
    assert f["drop_ratio"] == pytest.approx(5 / 8)
    assert f["embedding_cosine"] == pytest.approx(3 / math.sqrt(8 * 3))
    assert f["cat_code"] == 1.0 and f["cat_fact"] == 0.0


def test_pair_features_rejects_unknown_category() -> None:
    with pytest.raises(ValueError):
        pair_features("a", "a", "poetry", words, flat_bits, bag_embed)


def test_cosine_zero_vector() -> None:
    assert cosine(np.zeros(2), np.ones(2)) == 0.0


def test_pair_feature_table() -> None:
    t = pair_feature_table(
        [
            {
                "id": "x:F",
                "prompt_id": "x",
                "category": "fact",
                "original": "hi what is 2+2",
                "rewrite": "what is 2+2",
                "label": "preserved",
            }
        ],
        words,
        flat_bits,
        bag_embed,
    )
    assert t.shape == (1, 4 + len(PAIR_FEATURES))
    assert t.loc[0, "cat_fact"] == 1.0
