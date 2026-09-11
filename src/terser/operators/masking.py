"""Protect payload spans from rewrite operators.

Fenced code blocks, ``inline code`` and double-quoted spans (ASCII ``"..."`` and curly
``“...”``) are the user's payload: an operator must never edit them. :func:`with_masked`
swaps each such span for a placeholder built only from Unicode private-use characters, runs
the rewrite on the masked text, and restores the original spans.

Private-use characters are neither word characters, whitespace, digits nor punctuation, so no
operator regex (``\\w``, ``\\b``, ``[a-z]``, ``[.!?]``, ``\\s``) can match or split a
placeholder.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from terser.operators.base import Operator

PROTECTED = re.compile(r"```.*?```|`[^`\n]*`|\"[^\"\n]*\"|“[^”\n]*”", re.DOTALL)
"""Spans never edited by an operator: fenced code, inline code, double-quoted text."""

_OPEN = ""
_CLOSE = ""
_INDEX_BASE = 0xF0000  # Supplementary Private Use Area-A: one character per span index
_MAX_SPANS = 0xFFFD
_PLACEHOLDER = re.compile(f"{_OPEN}(.){_CLOSE}")


def mask(text: str) -> tuple[str, list[str]]:
    """Return ``text`` with protected spans replaced by placeholders, and the spans in order."""
    spans: list[str] = []

    def hide(m: re.Match[str]) -> str:
        if len(spans) >= _MAX_SPANS:
            return m.group(0)
        spans.append(m.group(0))
        return f"{_OPEN}{chr(_INDEX_BASE + len(spans) - 1)}{_CLOSE}"

    return PROTECTED.sub(hide, text), spans


def unmask(text: str, spans: list[str]) -> str:
    """Put the spans hidden by :func:`mask` back in place of their placeholders."""

    def show(m: re.Match[str]) -> str:
        index = ord(m.group(1)) - _INDEX_BASE
        return spans[index] if 0 <= index < len(spans) else m.group(0)

    return _PLACEHOLDER.sub(show, text)


def with_masked(text: str, fn: Callable[[str], str]) -> str:
    """Apply ``fn`` to ``text`` with protected spans hidden, then restore them.

    If ``fn`` drops a placeholder (which would silently delete payload), the input is
    returned unchanged.
    """
    masked, spans = mask(text)
    out = fn(masked)
    if len(_PLACEHOLDER.findall(out)) != len(spans):
        return text
    return unmask(out, spans)


MaskedOperator = Operator
"""Backward-compatible alias: every :class:`Operator` now tidies with payload spans masked."""
