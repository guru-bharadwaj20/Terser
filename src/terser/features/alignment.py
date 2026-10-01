"""Token-level KEEP / DROP / REPLACE alignment labels (plan row **3.1.1**).

Each casual prompt is diff-aligned with its oracle rewrite (the shortest candidate still judged
``preserved``). Every token of the *original* gets one label:

* ``KEEP``: the token survives unchanged in the rewrite;
* ``DROP``: the token is deleted with nothing in its place;
* ``REPLACE``: the token belongs to a span the rewrite substitutes with different text.

Tokens are words (internal apostrophes kept) and single punctuation marks, compared
case-insensitively, so a rewrite that only re-capitalises a sentence start ("so can you" ->
"Can you") keeps the word. Alignment is :class:`difflib.SequenceMatcher` with ``autojunk``
disabled: a deterministic longest-matching-block diff, adequate for prompts of a few hundred
words. The labels are what the span tagger learns to predict.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Final, Literal

__all__ = ["DROP", "KEEP", "REPLACE", "Alignment", "Label", "align", "tokenize"]

Label = Literal["KEEP", "DROP", "REPLACE"]
KEEP: Final[Label] = "KEEP"
DROP: Final[Label] = "DROP"
REPLACE: Final[Label] = "REPLACE"

_TOKEN_RE: Final = re.compile(r"\w+(?:['’]\w+)*|[^\w\s]")


def tokenize(text: str) -> list[str]:
    """Split ``text`` into words (with internal apostrophes) and single punctuation marks."""
    return _TOKEN_RE.findall(text)


@dataclass(frozen=True)
class Alignment:
    """Original tokens, rewrite tokens, and one label per original token.

    ``replacements`` maps the index of the first token of each replaced span to the rewrite
    tokens that took its place; ``inserted`` lists rewrite tokens with no original counterpart.
    """

    original: list[str]
    rewrite: list[str]
    labels: list[Label]
    replacements: dict[int, list[str]] = field(default_factory=dict)
    inserted: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len(self.labels) != len(self.original):
            raise ValueError("one label per original token required")

    def drop_ratio(self) -> float:
        """Share of original tokens that are not kept."""
        if not self.original:
            return 0.0
        return sum(lab != KEEP for lab in self.labels) / len(self.original)


def align(original: str, rewrite: str) -> Alignment:
    """Label every original token KEEP, DROP or REPLACE against ``rewrite``."""
    a, b = tokenize(original), tokenize(rewrite)
    matcher = SequenceMatcher(a=[t.lower() for t in a], b=[t.lower() for t in b], autojunk=False)
    labels: list[Label] = [KEEP] * len(a)
    replacements: dict[int, list[str]] = {}
    inserted: list[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "delete":
            labels[i1:i2] = [DROP] * (i2 - i1)
        elif tag == "replace":
            labels[i1:i2] = [REPLACE] * (i2 - i1)
            replacements[i1] = b[j1:j2]
        elif tag == "insert":
            inserted.extend(b[j1:j2])
    return Alignment(a, b, labels, replacements, inserted)
