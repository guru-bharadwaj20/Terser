"""``S`` tokenizer-aware phrase substitution.

A table maps wordy phrases to shorter, meaning-preserving alternatives. For every match the
operator prices the original and each alternative **inside its sentence** on the target's
tokenizer (``terser.tokenizers.count_tokens``) and keeps the cheapest; ties keep the original,
then prefer the alternative listed first. With ``target=None`` the character length of the
sentence is used as a tokenizer-agnostic proxy.

Code blocks, ``inline code`` and "quoted" spans are never edited
(:func:`terser.operators.masking.with_masked`).

Tokenizer-dependence finding (row 1.2.7): an exhaustive search over every table entry x
:data:`SEARCH_CONTEXTS` (lower case in all 7 contexts, sentence-initial case in the 2 contexts
with no left text) x {gpt-4o, qwen2.5, llama-3.1} (486 choices, 1458 choice evaluations)
found exactly one divergent entry: ``approximately``. Sentence-initial
``Approximately`` (and ``approximately`` after an opening parenthesis) is a single o200k token,
so on gpt-4o no alternative is cheaper and the original is kept, while qwen2.5 and llama-3.1
rewrite it to ``About``. Every other choice agrees across the three tokenizers. The fixture
:data:`TOKENIZER_DIVERGENT_EXAMPLE` pins this in ``tests/operators/test_substitute.py``.
"""

from __future__ import annotations

import re

from terser.operators.base import Operator
from terser.operators.masking import mask, unmask, with_masked

PHRASE_SUBSTITUTIONS: dict[str, tuple[str, ...]] = {
    "in order to": ("to",),
    "so as to": ("to",),
    "due to the fact that": ("because",),
    "owing to the fact that": ("because",),
    "in light of the fact that": ("because",),
    "in spite of the fact that": ("although", "though"),
    "despite the fact that": ("although", "though"),
    "at this point in time": ("now",),
    "at the present time": ("now", "currently"),
    "for the purpose of": ("for",),
    "in the event that": ("if",),
    "make use of": ("use",),
    "utilize": ("use",),
    "a large number of": ("many",),
    "the vast majority of": ("most",),
    "the majority of": ("most",),
    "a majority of": ("most",),
    "a small number of": ("a few",),
    "a sufficient number of": ("enough",),
    "in close proximity to": ("near",),
    "in the vicinity of": ("near",),
    "with regard to": ("about", "regarding"),
    "in regard to": ("about", "regarding"),
    "is able to": ("can",),
    "are able to": ("can",),
    "has the ability to": ("can",),
    "have the ability to": ("can",),
    "prior to": ("before",),
    "subsequent to": ("after",),
    "each and every": ("every",),
    "first and foremost": ("first",),
    "until such time as": ("until",),
    "on a daily basis": ("daily",),
    "on a regular basis": ("regularly",),
    "in the near future": ("soon",),
    "give an explanation of": ("explain",),
    "provide an explanation of": ("explain",),
    "provide a description of": ("describe",),
    "make a decision": ("decide",),
    "take into consideration": ("consider",),
    "with the exception of": ("except", "except for"),
    "in excess of": ("more than", "over"),
    "approximately": ("about", "roughly", "around"),
    "numerous": ("many",),
    "commence": ("start", "begin"),
    "demonstrate": ("show",),
    "by means of": ("using", "via"),
    "in accordance with": ("per",),
    "at all times": ("always",),
    "in a timely manner": ("promptly", "quickly"),
    "in the absence of": ("without",),
    "has a tendency to": ("tends to",),
}
"""Phrase (lower case) -> meaning-preserving alternatives, in order of preference.

"is/are required to" -> "must" is deliberately absent: it breaks questions such as "What is
required to install Python?" where no subject precedes the phrase.
"""

_DETERMINERS = "the these those this that their our your my his her its them us all".split()
for _det in _DETERMINERS:
    # "the majority of the votes" -> "most of the votes", not "most the votes".
    for _lead in ("the vast majority of", "the majority of", "a majority of"):
        PHRASE_SUBSTITUTIONS[f"{_lead} {_det}"] = (f"most of {_det}",)

KEEP_PHRASES: tuple[str, ...] = ("make a decision tree", "make a decision trees")
"""Phrases that contain a table entry but must stay as written (a longer match wins)."""

TOKENIZER_DIVERGENT_EXAMPLE = "Approximately how many people live in Tokyo?"
"""Kept on gpt-4o; becomes "About how many people live in Tokyo?" on qwen2.5 and llama-3.1."""

SEARCH_CONTEXTS: tuple[tuple[str, str], ...] = (
    ("", "."),
    ("Please ", " the task."),
    ("We need ", " finish.\n"),
    ("I think ", " it."),
    ("(", ")"),
    ("", " x"),
    ("Note: ", ", then stop."),
)
"""Sentence contexts ``(left, right)`` used by the exhaustive tokenizer-dependence search."""

_PHRASES = sorted([*PHRASE_SUBSTITUTIONS, *KEEP_PHRASES], key=len, reverse=True)  # longest wins
_SCAN = re.compile(rf"\b(?:{'|'.join(map(re.escape, _PHRASES))})\b", re.IGNORECASE)
_BOUNDARY = re.compile(r"[.!?\n]")


def _match_case(original: str, replacement: str) -> str:
    """Carry a leading capital from the original phrase over to the replacement."""
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def _cost(sentence: str, target: str | None) -> int:
    """Price a sentence in tokens on ``target``, or in characters without a target."""
    if target is None:
        return len(sentence)
    from terser.tokenizers import count_tokens  # lazy: tokenizer backends are heavy

    return count_tokens(sentence, target)


def choose(left: str, original: str, right: str, target: str | None) -> str:
    """Return the cheapest of ``original`` and its alternatives in the context ``left _ right``.

    Ties keep the original (no gain, no risk), then prefer the earlier-listed alternative.
    """
    alternatives = PHRASE_SUBSTITUTIONS[original.lower()]
    options = [original] + [_match_case(original, alt) for alt in alternatives]
    costs = [_cost(left + option + right, target) for option in options]
    return options[costs.index(min(costs))]


def substitute(text: str, target: str | None = None) -> str:
    """Replace table phrases with their cheapest alternative on ``target``, priced in context.

    The context is the enclosing sentence (earlier substitutions already applied), with any
    payload placeholders restored so the tokenizer sees the real text.
    """
    _, spans = mask(text)

    def rewrite(masked: str) -> str:
        out: list[str] = []
        pos = 0
        for m in _SCAN.finditer(masked):
            out.append(masked[pos : m.start()])
            pos = m.end()
            if m.group(0).lower() in KEEP_PHRASES:
                out.append(m.group(0))
                continue
            done = "".join(out)
            starts = [b.end() for b in _BOUNDARY.finditer(done)]
            left = done[starts[-1] :] if starts else done
            nxt = _BOUNDARY.search(masked, m.end())
            right = masked[m.end() : nxt.end() if nxt else len(masked)]
            out.append(choose(unmask(left, spans), m.group(0), unmask(right, spans), target))
        out.append(masked[pos:])
        return "".join(out)

    return with_masked(text, rewrite)


OPERATOR = Operator("S", "tokenizer-aware phrase substitution", substitute)
