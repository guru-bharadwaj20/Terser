"""Cheap span-tagger features and the ONNX span rewriter for the CPU fast path (row 11.3.2).

The fast path must run with no GPU and no language model, so its span tagger uses only features
that cost microseconds per word. :func:`span_matrix` is the **one** implementation used both to
build the training matrix (``terser.phase11.train_fast_path``) and at serving time
(:class:`OnnxSpanRewriter`), so training and serving cannot skew (row 11.1.3).

======================  ===============================================================
``token_cost``          target-tokenizer tokens of the word (with a leading space)
``stopword_ratio``      share of stopwords in a 5-word window
``rel_position``        ``i / (n - 1)``
``is_filler`` ...       filler / negation / number / constraint lexicon hits, punctuation
``word_len``            characters in the word
``prev_filler``         previous word is a filler-lexicon hit
======================  ===============================================================

:class:`OnnxSpanRewriter` deletes the words the tagger scores as droppable (P >= threshold),
never touching negations, numbers, constraint keywords or anything inside a fenced code block
or quotation marks, and keeps the rest of the text byte-for-byte.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from terser.features.lexicons import (
    is_constraint,
    is_filler,
    is_negation,
    is_number,
    is_stopword,
)
from terser.serving.paths import RewriteResult

__all__ = ["SPAN_FAST_FEATURES", "OnnxSpanRewriter", "span_matrix"]

SPAN_FAST_FEATURES: tuple[str, ...] = (
    "token_cost",
    "stopword_ratio",
    "rel_position",
    "is_filler",
    "is_negation",
    "is_number",
    "is_constraint",
    "is_punct",
    "word_len",
    "prev_filler",
)
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?|[^\sA-Za-z0-9]")
_PROTECTED_RE = re.compile(r"```.*?```|`[^`]*`|\"[^\"]*\"", re.S)
WINDOW = 2


def _punct(w: str) -> bool:
    return len(w) == 1 and not w.isalnum()


def span_matrix(words: Sequence[str], cost: Callable[[str], int]) -> np.ndarray:
    """``(n, len(SPAN_FAST_FEATURES))`` float32 feature matrix for one word sequence."""
    n = len(words)
    stop = [is_stopword(w) for w in words]
    fill = [is_filler(w) for w in words]
    rows = []
    for i, w in enumerate(words):
        lo, hi = max(0, i - WINDOW), min(n, i + WINDOW + 1)
        rows.append(
            [
                float(cost(w if _punct(w) else " " + w)),
                sum(stop[lo:hi]) / (hi - lo),
                i / (n - 1) if n > 1 else 0.0,
                float(fill[i]),
                float(is_negation(w)),
                float(is_number(w)),
                float(is_constraint(w)),
                float(_punct(w)),
                float(len(w)),
                float(fill[i - 1]) if i else 0.0,
            ]
        )
    return np.asarray(rows, dtype=np.float32).reshape(n, len(SPAN_FAST_FEATURES))


class OnnxSpanRewriter:
    """Fast-path rewriter: ONNX span tagger, deletes droppable words outside payload spans."""

    def __init__(
        self, path: str | Path, cost: Callable[[str], int], threshold: float = 0.5
    ) -> None:
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.cost = cost
        self.threshold = threshold

    def drop_probs(self, words: Sequence[str]) -> np.ndarray:
        if not words:
            return np.zeros(0)
        out = self.session.run(None, {self.input_name: span_matrix(words, self.cost)})
        return np.asarray(out[1])[:, 1]

    def __call__(self, prompt: str, target: str) -> RewriteResult:
        spans = list(_TOKEN_RE.finditer(prompt))
        protected = [m.span() for m in _PROTECTED_RE.finditer(prompt)]
        words = [m.group() for m in spans]
        probs = self.drop_probs(words)
        keep_mask = np.ones(len(words), bool)
        for i, (m, w) in enumerate(zip(spans, words, strict=True)):
            inside = any(a <= m.start() < b for a, b in protected)
            hard_keep = is_negation(w) or is_number(w) or is_constraint(w)
            if probs[i] >= self.threshold and not inside and not hard_keep:
                keep_mask[i] = False
        out, last = [], 0
        for m, keep in zip(spans, keep_mask, strict=True):
            if keep:
                continue
            out.append(prompt[last : m.start()])
            last = m.end()
            # swallow one following space so "so  the" does not leave a double space
            if last < len(prompt) and prompt[last] == " ":
                last += 1
        out.append(prompt[last:])
        text = re.sub(r"[ \t]{2,}", " ", "".join(out)).strip()
        return RewriteResult(text or prompt)
