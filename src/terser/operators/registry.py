"""Operator registry and canonical application order (row 1.2.8).

``apply_ops`` treats a set of operator codes as a *set*: whatever order or repetition the
caller uses, the operators run once each in ``CANONICAL_ORDER`` = ``C H F I D S L``. This makes
a rewrite a function of the chosen subset alone, which is what the Phase 1 hypothesis space
and the Phase 9 action space need.

Why this order:

* ``C`` (meta-talk) and ``H`` (hedging/politeness) run first because they remove whole
  framing clauses ("I'm going to give you some text and I want you to", "could you maybe
  please"). Those clauses contain the very words ``F`` treats as filler and the request
  shapes ``I`` rewrites; stripping them first stops ``F`` leaving a half-deleted clause and
  lets ``I`` see the real request at the start of the sentence.
* ``F`` (filler) runs before ``I`` (imperative) so interjections such as "so um, basically"
  no longer sit between the sentence start and "I was wondering if you could ...".
* ``D`` (dedupe) runs on the cleaned sentences, where restated requirements look alike.
* ``S`` (substitution) prices phrases in their final sentence context.
* ``L`` (list-ify) runs last: it changes the layout from sentences to bullets, which the
  sentence-level operators above are not written to handle, and its not-longer check must
  see the text every other operator has already shortened.
"""

from __future__ import annotations

from collections.abc import Iterable

from terser.operators.base import Operator
from terser.operators.dedupe import OPERATOR as DEDUPE
from terser.operators.filler import OPERATOR as FILLER
from terser.operators.hedging import OPERATOR as HEDGING
from terser.operators.imperative import OPERATOR as IMPERATIVE
from terser.operators.listify import OPERATOR as LISTIFY
from terser.operators.meta_talk import OPERATOR as META_TALK
from terser.operators.substitute import OPERATOR as SUBSTITUTE

CANONICAL_ORDER = "CHFIDSL"

OPERATORS: dict[str, Operator] = {
    op.code: op for op in (FILLER, HEDGING, DEDUPE, IMPERATIVE, LISTIFY, META_TALK, SUBSTITUTE)
}


def apply_ops(text: str, codes: Iterable[str], target: str | None = None) -> str:
    """Apply the operators named by ``codes`` once each, in ``CANONICAL_ORDER``.

    ``codes`` may be a string (``"FH"``) or any iterable of single-letter codes; order and
    repeats are ignored. Raises ``KeyError`` for an unknown code.
    """
    chosen = set(codes)
    unknown = sorted(chosen - OPERATORS.keys())
    if unknown:
        raise KeyError(
            f"unknown operator code(s) {unknown}; valid codes are {list(CANONICAL_ORDER)}"
        )
    for code in CANONICAL_ORDER:
        if code in chosen:
            text = OPERATORS[code](text, target)
    return text
