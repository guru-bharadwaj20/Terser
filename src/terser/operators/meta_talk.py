"""``C`` compress meta-talk.

Removes talk about the prompt rather than the task: "I'm going to give you some text and I
want you to", "I have a (quick) question:", "my question is", "so the thing is", "here's what
I need:". What follows is kept verbatim, so "I'm going to give you some text and I want you to
summarize it" becomes "Summarize it".

Framings are recognised at a sentence start or right after leading interjections that ``F``
removes ("Hey, quick question: ..."); the interjection itself is left for ``F``, so the result
does not depend on whether ``F`` ran first. A conjunction left dangling by the removal is
dropped with it: "quick question, and I forgot ..." becomes "I forgot ...".

Only framings that are clearly about the conversation are removed: "I have a question about
taxes" (no colon), "my question is simple" and "the thing is broken" are kept, as is "and"
anywhere but directly after a removed intro. Code, inline code and quoted spans are never
edited.
"""

from __future__ import annotations

import re

from terser.operators.base import Operator
from terser.operators.masking import with_masked

_F = re.IGNORECASE | re.MULTILINE
_APOS = "['’]"
_START = r"(?P<pre>^[ \t]*|(?<=[.!?])[ \t]+)"
# Leading interjections that ``F`` strips from a sentence start; matched past and kept.
_LEAD = (
    r"(?P<lead>(?:(?:(?:hey|hi|hello|yo)(?:[ \t]+there)?|ok(?:ay)?|well|anyway|so|"
    r"u+m+|u+h+|e+r+m+|hm+)\b(?![-'’])[ \t]*[,!.]*[ \t]+)*)"
)
_HEAD = _START + _LEAD
# "and" left opening the clause after a removed intro: "quick question, and I forgot ...".
_DANGLING_AND = r"(?:[ \t]*and(?=[ \t]+(?:i|honestly|basically|literally|totally)\b))?"

_GIVE_YOU_TEXT = re.compile(
    rf"\b(?:i{_APOS}m|i am|i{_APOS}ll|i will|i{_APOS}m about to|i am about to|i am going to|"
    rf"i{_APOS}m going to|i{_APOS}m gonna)[ \t]+"
    r"(?:(?:give|send|show|provide)[ \t]+you|(?:paste|share)(?:[ \t]+with[ \t]+you)?)[ \t]+"
    r"(?:some|a|an|the|my|this|a few|several)\b[^.?!\n]{0,60}?[ \t]*,?[ \t]+and[ \t]+"
    r"(?:(?:i|then i)[ \t]+(?:want|need|would[ \t]+like)|"
    rf"i{_APOS}d[ \t]+like|then[ \t]+i[ \t]+want)"
    r"[ \t]+you[ \t]+to[ \t]+",
    re.IGNORECASE,
)

_QUESTION_INTRO = re.compile(
    rf"{_HEAD}(?:(?:i[ \t]+have|i{_APOS}ve[ \t]+got|i[ \t]+got|got)[ \t]+(?:a|one)[ \t]+"
    r"(?:(?:quick|small|little|simple|short)[ \t]+)?question(?:[ \t]+for[ \t]+you)?"
    rf"|(?:quick|random|simple)[ \t]+question)[ \t]*(?:[:!.\-–—]+|,{_DANGLING_AND})[ \t]*(?=\S)",
    _F,
)

_MY_QUESTION = re.compile(
    rf"{_HEAD}(?:so[ \t]+)?my[ \t]+(?:actual[ \t]+|main[ \t]+)?question[ \t]+is"
    r"(?:[ \t]*:[ \t]*|[ \t]*,?[ \t]+(?=(?:how|what|why|when|where|who|which|whether|can|could|"
    r"should|would|is|are|do|does|did|will)\b))(?=\S)",
    _F,
)

_THING_IS = re.compile(rf"{_HEAD}so[ \t]+the[ \t]+thing[ \t]+is\b[ \t]*[,:]?[ \t]*(?=\S)", _F)

_HERE_WHAT_I_NEED = re.compile(
    rf"{_HEAD}(?:here{_APOS}s|here[ \t]+is)[ \t]+what[ \t]+"
    r"(?:i|we)[ \t]+(?:need|want)(?:[ \t]+(?:you[ \t]+to[ \t]+do|from[ \t]+you))?"
    r"[ \t]*[:.\-–—]+[ \t]*(?=\S)",
    _F,
)


def _keep_lead(m: re.Match[str]) -> str:
    return m.group("pre") + m.group("lead")


def _rewrite(text: str) -> str:
    text = _GIVE_YOU_TEXT.sub("", text)
    text = _QUESTION_INTRO.sub(_keep_lead, text)
    text = _MY_QUESTION.sub(_keep_lead, text)
    text = _THING_IS.sub(_keep_lead, text)
    text = _HERE_WHAT_I_NEED.sub(_keep_lead, text)
    return text


def meta_talk(text: str, target: str | None = None) -> str:
    """Drop meta-talk about the prompt outside code and quoted spans."""
    out = with_masked(text, _rewrite)
    return out if out.strip() else text


OPERATOR = Operator("C", "compress meta-talk", meta_talk)
