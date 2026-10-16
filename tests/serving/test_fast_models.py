"""Fast-path span features (row 11.3.2): serving tokenises exactly like the training alignment."""

from __future__ import annotations

import numpy as np
import pytest

from terser.features.alignment import TOKEN_RE, tokenize
from terser.serving.fast_models import SPAN_FAST_FEATURES, span_matrix

TEXTS = [
    "hey so um, could you maybe rename my_var? It's fine",
    "don’t touch the café menu — just list 3 items",
    "naïve résumé: snake_case_name() returns 42!",
]


@pytest.mark.parametrize("text", TEXTS)
def test_serving_spans_match_training_tokens(text: str) -> None:
    assert [m.group() for m in TOKEN_RE.finditer(text)] == tokenize(text)


def test_span_matrix_shape_and_flags() -> None:
    words = tokenize("so um please do not list 3 items")
    m = span_matrix(words, lambda w: 1)
    assert m.shape == (len(words), len(SPAN_FAST_FEATURES))
    neg = SPAN_FAST_FEATURES.index("is_negation")
    assert m[words.index("not"), neg] == 1.0
    assert np.all(m[:, SPAN_FAST_FEATURES.index("token_cost")] == 1.0)
