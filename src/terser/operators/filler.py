"""``F`` drop filler.

Removes conversational filler that carries no instruction: leading interjections (hey, hi,
hello, ok, so, um, well), hesitation sounds (um, uh, erm), comma-bracketed "like", "you know,"
and "I mean,", intensifiers (basically, literally, honestly, totally), "kind of / sort of"
before a request verb, and laughter (lol, haha).

The rules are conservative: a missed deletion costs a few tokens, a wrong one changes meaning.
"like" as a comparison ("talk like a pirate"), "just in time", "so many", "hello world" and
intensifiers before numbers are kept. Code, inline code and quoted spans are never edited.
"""

from __future__ import annotations

import re

from terser.operators.masking import MaskedOperator, with_masked

_F = re.IGNORECASE | re.MULTILINE

# Start of a line or of a sentence; the captured lead is re-emitted.
_START = r"(?P<pre>^[ \t]*|(?<=[.!?])[ \t]+)"

_HESITATION = r"(?:u+m+|u+h+|e+r+m+|hm+)"
_SOFT_OPENERS = rf"(?:so|ok(?:ay)?|well|anyway|{_HESITATION})"
_WORD_END = r"\b(?![-'’])"

# "hey", "hi", "hello", "ok", "okay", "well": need punctuation, "there", or another opener after
# them, so "Hello world in C" and "Well documented code" are kept.
_GREETING = re.compile(
    rf"{_START}(?:hey|hi|hello|yo|ok(?:ay)?|well|anyway)(?:\s+there)?{_WORD_END}"
    rf"(?:\s*[,!.]+[ \t]*|[ \t]+(?={_SOFT_OPENERS}{_WORD_END}))(?=\S)",
    _F,
)

# "so" only before a comma, another opener, a pronoun, a wh-word or an auxiliary, so
# "So many rows fail" is kept.
_SO = re.compile(
    rf"{_START}so{_WORD_END}(?:\s*,[ \t]*|[ \t]+(?=(?:{_SOFT_OPENERS}|i|i['’]m|i['’]ve|we|my|"
    r"what|how|why|when|where|who|which|can|could|would|will|is|are|do|does|basically|"
    r"literally|honestly|please)\b(?![-'’]\w)))(?=\S)",
    _F,
)

_LAUGHTER = r"(?:lol|lmao|rofl|ha(?:ha)+h?|he(?:he)+)"
_LEADING_HESITATION = re.compile(rf"{_START}(?:{_HESITATION}|{_LAUGHTER})\b[,.!]*[ \t]*(?=\S)", _F)
# Not after a number: "10 um wide" is a micrometre unit, not a hesitation.
_INNER_HESITATION = re.compile(
    rf"(?:,[ \t]*)?(?<!\d)(?<!\d )\b{_HESITATION}\b(?:[ \t]*,)?", re.IGNORECASE
)
_COMMA_LIKE = re.compile(r",[ \t]*like[ \t]*,", re.IGNORECASE)
_YOU_KNOW_LEAD = re.compile(rf"{_START}(?:you know|i mean)[ \t]*,[ \t]*", _F)
_YOU_KNOW_BRACKETED = re.compile(r",[ \t]*(?:you know|i mean)[ \t]*,", re.IGNORECASE)
_YOU_KNOW_TRAILING = re.compile(r",[ \t]*(?:you know|i mean)[ \t]*(?=[.!?]|$)", _F)
_INTENSIFIER = re.compile(
    r"(?<!\btranslate )(?<!\btranslated )(?<!\binterpret )(?<!\bmean )(?<!\bmeant )"
    r"(?<!\bdoes )(?<!\bdo )(?<!\bword )"
    r"\b(?:basically|literally|honestly|totally)\b(?![-'’])[ \t]*,?[ \t]+"
    # the word itself is the topic ("what does literally mean") or part of a term
    r"(?!(?:mean|means|meant|ordered|bounded)\b)(?=[^\W\d_])",
    re.IGNORECASE,
)
_REQUEST_VERBS = (
    r"(?:explain|tell|show|describe|give|help|write|summari[sz]e|list|outline|walk|rewrite|"
    r"clarify|go)"
)
_KIND_OF = re.compile(
    rf"\b(?:kind of|sort of|kinda|sorta)[ \t]+(?={_REQUEST_VERBS}\b)", re.IGNORECASE
)
_LAUGH = re.compile(
    rf"(?:[ \t]*,)?[ \t]*\b{_LAUGHTER}\b(?:[ \t]*!+(?=\s|$))?", re.IGNORECASE | re.MULTILINE
)


def _drop_keep_lead(m: re.Match[str]) -> str:
    return m.group("pre")


def _rewrite(text: str) -> str:
    prev = None
    while prev != text:  # stacked filler: "hey so um, basically ..."
        prev = text
        text = _GREETING.sub(_drop_keep_lead, text)
        text = _SO.sub(_drop_keep_lead, text)
        text = _LEADING_HESITATION.sub(_drop_keep_lead, text)
        text = _YOU_KNOW_LEAD.sub(_drop_keep_lead, text)
        text = _INNER_HESITATION.sub(" ", text)
        text = _COMMA_LIKE.sub(" ", text)
        text = _YOU_KNOW_BRACKETED.sub(" ", text)
        text = _YOU_KNOW_TRAILING.sub("", text)
        text = _INTENSIFIER.sub("", text)
        text = _KIND_OF.sub("", text)
        text = _LAUGH.sub("", text)
    return text


def filler(text: str, target: str | None = None) -> str:
    """Drop conversational filler outside code and quoted spans."""
    out = with_masked(text, _rewrite)
    return out if out.strip() else text


OPERATOR = MaskedOperator("F", "drop filler", filler)
