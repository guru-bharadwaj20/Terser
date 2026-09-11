"""``H`` drop hedging and politeness.

Removes courtesy that does not change the task: "please", "thanks (so much / in advance / for
your help)", "if you don't mind", "if possible", "sorry if this is a dumb question", "any help
would be appreciated", and "maybe / possibly / perhaps" when they only soften a request
("could you maybe explain").

Hedges that carry meaning are kept: "maybe" in a statement ("maybe the cache is stale"),
"thanks" inside content ("write a thank-you note"), "to please the reviewer". Code, inline
code and quoted spans are never edited.
"""

from __future__ import annotations

import re

from terser.operators.base import Operator
from terser.operators.masking import with_masked

_I = re.IGNORECASE
_APOS = "['’]"

_PLEASE = re.compile(
    r"(?:,[ \t]*)?(?<!\bto )(?<!\bpretty )\bplease\b(?![-'’])(?:[ \t]*,)?(?![ \t]*[-'’])", _I
)

# "thanks ..." only as its own clause: bounded by start/sentence end/comma on both sides.
_THANKS = re.compile(
    r"(?:^|(?<=[.!?,;:]))[ \t]*"
    r"(?:thanks|thank you|thx|many thanks|thanks again|thank you again)"
    r"(?:[ \t]+(?:so|very)[ \t]+much|[ \t]+a[ \t]+(?:lot|ton|bunch))?"
    r"(?:[ \t]+in[ \t]+advance)?"
    r"(?:[ \t]+for[ \t]+(?:your|the|any)[ \t]+(?:help|time|assistance))?"
    r"(?:[ \t]+in[ \t]+advance)?"
    r"[ \t]*[!.]*(?:[ \t]*(?:[:;]-?\)|🙏|😊))?"
    r"(?=[ \t]*(?:$|[,;]|(?-i:[A-Z])|\ue000))",
    _I | re.MULTILINE,
)

_CONDITIONAL = re.compile(
    rf"(?:,[ \t]*)?\b(?:if you don{_APOS}?t mind|if possible|if that{_APOS}?s (?:ok(?:ay)?|alright)"
    r"|when you get a chance|if you have (?:a moment|time))\b(?:[ \t]*,)?",
    _I,
)

_DUMB = (
    r"(?:a |an )?(?:really |very |kind of )?(?:dumb|stupid|silly|basic|noob|newbie|naive|obvious)"
)
_APOLOGY = re.compile(
    rf"\b(?:(?:i{_APOS}?m )?sorry (?:if this is|for (?:the|a|this)|if it{_APOS}?s) {_DUMB} question"
    rf"|sorry to bother you|sorry for bothering you|this (?:might|may|could) be {_DUMB} question)"
    r"(?:,? but)?\b[ \t]*[,.!:;]*[ \t]*",
    _I,
)

_APPRECIATED = re.compile(
    r"\b(?:any|all) (?:help|advice|input|suggestions|pointers) (?:would be|will be|is) "
    r"(?:much |greatly |really |very much |very |highly )?appreciated\b[ \t]*[!.]*",
    _I,
)

# "maybe/possibly/perhaps" only right after a request framing.
_REQUEST_HEDGE = re.compile(
    r"(?P<frame>\b(?:(?:can|could|would|will) (?:you|u)|if you could|you could|you can|"
    rf"i was wondering if you could|i{_APOS}d like you to|is it possible to)"
    r"(?: please)?)[ \t]+(?:maybe|possibly|perhaps|kindly)\b,?(?=[ \t]+\S)",
    _I,
)
_MAYBE_YOU_COULD = re.compile(
    r"(?P<pre>^[ \t]*|(?<=[.!?])[ \t]+)(?:maybe|perhaps)[ \t]+(?=you (?:could|can)\b)",
    _I | re.MULTILINE,
)


def _rewrite(text: str) -> str:
    text = _APOLOGY.sub("", text)
    text = _APPRECIATED.sub("", text)
    text = _THANKS.sub("", text)
    text = _CONDITIONAL.sub("", text)
    text = _REQUEST_HEDGE.sub(lambda m: m.group("frame"), text)
    text = _MAYBE_YOU_COULD.sub(lambda m: m.group("pre"), text)
    text = _PLEASE.sub(" ", text)
    return text


def hedging(text: str, target: str | None = None) -> str:
    """Drop hedging and politeness outside code and quoted spans."""
    out = with_masked(text, _rewrite)
    return out if out.strip() else text


OPERATOR = Operator("H", "drop hedging/politeness", hedging)
