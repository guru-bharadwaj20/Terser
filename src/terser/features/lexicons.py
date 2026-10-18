"""Small closed word lists shared by span (3.1.2) and pair (3.1.3) features.

The lists are deliberately short and explicit: each flag answers "could deleting this token change
what the target model is asked to do?" (negations, numbers, constraint keywords) or "is this token
conversational filler?" (filler lexicon, which mirrors the ``F`` / ``H`` operators).
"""

from __future__ import annotations

import re
from typing import Final

__all__ = [
    "CONSTRAINT_WORDS",
    "FILLER_WORDS",
    "NEGATIONS",
    "STOPWORDS",
    "is_constraint",
    "is_filler",
    "is_negation",
    "is_number",
    "is_stopword",
]

NEGATIONS: Final = frozenset(
    {
        "not",
        "no",
        "never",
        "none",
        "nor",
        "without",
        "don't",
        "dont",
        "doesn't",
        "didn't",
        "can't",
        "cannot",
        "won't",
        "shouldn't",
        "mustn't",
        "isn't",
        "aren't",
        "avoid",
        "except",
    }
)

CONSTRAINT_WORDS: Final = frozenset(
    {
        "must",
        "only",
        "exactly",
        "maximum",
        "minimum",
        "max",
        "min",
        "under",
        "within",
        "limit",
        "always",
        "should",
        "require",
        "required",
        "requires",
        "ensure",
        "format",
        "json",
        "csv",
        "words",
        "sentences",
        "lines",
        "bullet",
        "bullets",
        "language",
        "return",
        "include",
        "exclude",
        "use",
        "using",
    }
)

FILLER_WORDS: Final = frozenset(
    {
        "hey",
        "hi",
        "hello",
        "um",
        "uh",
        "erm",
        "so",
        "like",
        "basically",
        "literally",
        "honestly",
        "totally",
        "just",
        "really",
        "actually",
        "please",
        "thanks",
        "thank",
        "lol",
        "haha",
        "kinda",
        "sorta",
        "maybe",
        "perhaps",
        "possibly",
        "think",
        "guess",
        "wondering",
        "ok",
        "okay",
        "well",
    }
)

STOPWORDS: Final = frozenset(
    {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "of",
        "to",
        "in",
        "on",
        "for",
        "with",
        "as",
        "by",
        "at",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "it",
        "this",
        "that",
        "these",
        "those",
        "i",
        "you",
        "me",
        "my",
        "your",
        "we",
        "can",
        "could",
        "would",
        "will",
        "do",
        "does",
        "so",
        "some",
        "any",
        "about",
        "into",
        "than",
        "then",
        "there",
        "what",
        "which",
    }
)

_NUMBER_RE: Final = re.compile(r"^\d+(?:[.,]\d+)*%?$")
_NUMBER_WORDS: Final = frozenset(
    "zero one two three four five six seven eight nine ten eleven twelve twenty thirty forty fifty "
    "hundred thousand million first second third half".split()
)


def _norm(token: str) -> str:
    """Lower case with a curly apostrophe folded to ``'`` ("Don’t" -> "don't")."""
    return token.lower().replace("’", "'")


def is_negation(token: str) -> bool:
    """True for negation words."""
    return _norm(token) in NEGATIONS


def is_number(token: str) -> bool:
    """True for digits (``3``, ``2.5``, ``40%``) and common number words."""
    t = token.lower()
    return bool(_NUMBER_RE.match(t)) or t in _NUMBER_WORDS


def is_constraint(token: str) -> bool:
    """True for words that usually state a constraint on the answer."""
    return _norm(token) in CONSTRAINT_WORDS


def is_filler(token: str) -> bool:
    """True for conversational filler / hedging words."""
    return _norm(token) in FILLER_WORDS


def is_stopword(token: str) -> bool:
    """True for English function words."""
    return _norm(token) in STOPWORDS
