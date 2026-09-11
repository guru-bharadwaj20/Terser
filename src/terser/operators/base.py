"""Rewrite operator interface shared by all operators.

An operator is a deterministic, interpretable text transformation. Operators are the
hypothesis-space building blocks for Phase 1 concept learning and the action set of the
Phase 9 RL environment, so each must be a pure function of ``(text, target)``.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Operator:
    """A named rewrite. ``fn`` receives the text and an optional target model name."""

    code: str
    name: str
    fn: Callable[[str, str | None], str]

    def __call__(self, text: str, target: str | None = None) -> str:
        out = self.fn(text, target)
        # A rewrite that changes nothing must return the input untouched, not a re-tidied copy.
        if out == text:
            return text
        # Tidy with payload spans (code, quotes) hidden so they come back byte for byte.
        from terser.operators.masking import with_masked  # masking imports this module

        return with_masked(out, tidy)


def tidy(text: str) -> str:
    """Repair spacing and punctuation left behind by deletions, without changing words."""
    lines = []
    for line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", line)
        line = re.sub(r"\s+([,.;:!?])", r"\1", line)
        line = re.sub(r"([,;:])(?:\s*[,;:])+", r"\1", line)
        line = re.sub(r",\s*([.!?])", r"\1", line)
        line = re.sub(r"^[\s,;:]+(?=\S)", "", line)
        line = line.strip()
        if line[:1].isalpha() and line[:1].islower() and not line.startswith(("http", "e.g")):
            line = line[0].upper() + line[1:]
        lines.append(line)
    out = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
    # Capitalise sentence starts exposed by deletions.
    out = re.sub(r"([.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), out)
    return out.strip()
