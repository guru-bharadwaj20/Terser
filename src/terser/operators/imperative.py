"""``I`` imperative conversion.

Turns a request phrased as a question or a wish into a direct instruction at the start of a
sentence:

* "can / could / would / will you (please) X ..." (X a known request verb)
* "I want / need / 'd like / would like you to X ..."
* "I was wondering if you could X ..."
* "would you mind Xing ..." (known -ing forms only)
* "is it possible (for you) to X ..." (X a known text-task verb without "for you")

become "X ...". The sentence-final "?" of a rewritten sentence becomes "."; every other
sentence, including genuine questions ("What is a monad?", "Can you run Python 2 code on 3?"),
is untouched. Code, inline code and quoted spans are never edited.
"""

from __future__ import annotations

import re

from terser.operators.base import Operator
from terser.operators.masking import with_masked

_APOS = "['’]"

REQUEST_VERBS = frozenset("""
    add adapt analyse analyze answer brainstorm break build calculate change check clarify
    classify clean come compare compose compute condense convert correct create critique debug
    define describe design draft edit elaborate evaluate expand explain extract find fix format
    generate give go help identify implement improve list make modify name optimise optimize
    organise organize outline paraphrase plan polish proofread provide recommend refactor remove
    rename rephrase reply respond restructure review revise rewrite scan shorten show simplify
    solve sort spell suggest summarise summarize teach tell test tidy translate turn update walk
    write
    """.split())
"""Base-form verbs that make "can you X" a request rather than a question about ability."""

TEXT_TASK_VERBS = frozenset("""
    explain summarize summarise rewrite rephrase paraphrase translate describe draft outline
    proofread shorten simplify review critique condense
    """.split())
"""Verbs for which "is it possible to X" is almost always a request to the assistant."""

ING_TO_BASE = {
    "adding": "add",
    "analysing": "analyse",
    "analyzing": "analyze",
    "checking": "check",
    "clarifying": "clarify",
    "comparing": "compare",
    "converting": "convert",
    "correcting": "correct",
    "creating": "create",
    "debugging": "debug",
    "describing": "describe",
    "drafting": "draft",
    "editing": "edit",
    "explaining": "explain",
    "fixing": "fix",
    "formatting": "format",
    "generating": "generate",
    "giving": "give",
    "going": "go",
    "helping": "help",
    "improving": "improve",
    "listing": "list",
    "looking": "look",
    "making": "make",
    "outlining": "outline",
    "proofreading": "proofread",
    "providing": "provide",
    "refactoring": "refactor",
    "rephrasing": "rephrase",
    "reviewing": "review",
    "rewriting": "rewrite",
    "showing": "show",
    "simplifying": "simplify",
    "suggesting": "suggest",
    "summarising": "summarise",
    "summarizing": "summarize",
    "telling": "tell",
    "translating": "translate",
    "walking": "walk",
    "writing": "write",
}
"""Known -ing forms for "would you mind Xing"; unknown ones are left alone."""

_FRAME = (
    r"(?P<mind>would[ \t]+(?:you|u)[ \t]+mind(?:[ \t]+please)?)"
    r"|(?P<ask>(?:can|could|would|will)[ \t]+(?:you|u)(?:[ \t]+please)?(?:[ \t]+kindly)?)"
    r"|(?P<wish>(?:i|we)(?:[ \t]+(?:really|just))?"
    rf"(?:[ \t]+(?:want|need|would[ \t]+like)|{_APOS}d[ \t]+like)"
    r"[ \t]+(?:you|u)[ \t]+to(?:[ \t]+please)?)"
    rf"|(?P<wonder>i(?:[ \t]+was|[ \t]+am|{_APOS}m)[ \t]+(?:just[ \t]+)?wondering[ \t]+if"
    r"[ \t]+(?:you|u)[ \t]+(?:could|can|would)(?:[ \t]+please)?)"
    r"|(?P<possible>is[ \t]+it[ \t]+possible(?P<foryou>[ \t]+for[ \t]+you)?[ \t]+to)"
)

# The sentence ends at "." / "!" before whitespace, at end of line, or at a "?" followed by the
# end of line, a closing quote or a next sentence (capitalised or not: ``tidy`` capitalises
# "? word" anyway, so treating it differently would make the full stack non-idempotent). A "?"
# anywhere else (e.g. "a?b") stops the match.
_REQUEST = re.compile(
    rf"(?P<pre>^[ \t]*|(?<=[.!?])[ \t]+)(?:{_FRAME})[ \t]+(?P<verb>[A-Za-z]+)\b(?![-'’])"
    r"(?P<rest>[^\n?]*?)"
    r"(?P<end>\?(?=[ \t]*(?:$|[\"'”’)\]]|(?-i:[A-Z])|\ue000)|[ \t]+(?-i:[a-z]))|[.!](?=\s|$)|$)",
    re.IGNORECASE | re.MULTILINE,
)


def _base_verb(m: re.Match[str]) -> str | None:
    """Return the imperative verb for a matched framing, or ``None`` to leave it alone."""
    verb = m.group("verb")
    low = verb.lower()
    if m.group("mind"):
        return ING_TO_BASE.get(low)
    if low.endswith("ing"):
        return None
    if m.group("ask"):
        return verb if low in REQUEST_VERBS else None
    if m.group("possible") and not m.group("foryou"):
        return verb if low in TEXT_TASK_VERBS else None
    return verb


def _convert(m: re.Match[str]) -> str:
    verb = _base_verb(m)
    if verb is None:
        return m.group(0)
    end = "." if m.group("end") == "?" else m.group("end")
    return m.group("pre") + verb[0].upper() + verb[1:] + m.group("rest") + end


def _rewrite(text: str) -> str:
    return _REQUEST.sub(_convert, text)


def imperative(text: str, target: str | None = None) -> str:
    """Rewrite request framings as imperatives outside code and quoted spans."""
    return with_masked(text, _rewrite)


OPERATOR = Operator("I", "imperative conversion", imperative)
