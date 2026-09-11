"""``L`` list-ify constraints.

A single-paragraph prompt made of a main request followed by three or more constraint sentences
("Keep it short. Use type hints. Don't use recursion.") is rewritten as the request followed by
a bullet list with one constraint per line. Connectives and lead-ins that only serve to chain
sentences ("also", "and", "oh and", "please", "make sure to", "it should") are dropped from the
bullets; the constraint itself is kept word for word.

Conservative by design:

* prompts with fenced code, existing lists or more than one line are left alone;
* the first sentence must be a non-constraint (the main request), and every sentence after the
  first constraint must be one too, so prose is never reordered; questions are not constraints;
* bullet markup is not free (``"\\n- "`` costs about a token more than ``". "``), so the list is
  only produced when it is **not longer** than the input on ``target``'s tokenizer (characters
  when ``target`` is ``None``). An operator in Terser must never make a prompt longer.

Inline code and quoted spans are never edited (:func:`terser.operators.masking.with_masked`).
"""

from __future__ import annotations

import re

from terser.operators.masking import MaskedOperator, mask, with_masked

MIN_CONSTRAINTS = 3
"""Fewest constraint sentences that trigger a list."""

_CONNECTIVE = r"(?:(?:oh,? )?and,? |also,? |plus,? |oh,? |please,? |finally,? )*"
_CONSTRAINT = re.compile(
    rf"^{_CONNECTIVE}"
    r"(?:make sure|be sure|ensure|remember to|keep|don'?t|do not|never|avoid|use|include|add|"
    r"no more than|no |at least|at most|limit|only|always|format|return|output|write it|"
    r"make it|it should|it must|it needs to|it has to|you should|you must|should|must|"
    r"(?:the )?(?:answer|response|reply|output|code|function|result)s? (?:should|must|needs? to))"
    r"\b",
    re.IGNORECASE,
)
_LEAD_IN = re.compile(
    rf"^{_CONNECTIVE}(?:(?:make sure|be sure|remember) to |it (?:should|needs to|has to) )?"
    r"(?:also )?",
    re.IGNORECASE,
)
_SENTENCE = re.compile(r"[^\s.!?](?:[^.!?\n]|[.!?]+(?=[^\s.!?]))*(?:[.!?]+|$)")
_EXISTING_LIST = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s", re.MULTILINE)


def _cost(text: str, target: str | None) -> int:
    """Price text in tokens on ``target``, or in characters without a target."""
    if target is None:
        return len(text)
    from terser.tokenizers import count_tokens  # lazy: tokenizer backends are heavy

    return count_tokens(text, target)


def _bullet(sentence: str) -> str:
    """Turn a constraint sentence into a bullet: drop lead-ins and the final full stop."""
    body = _LEAD_IN.sub("", sentence).rstrip(".!") or sentence
    return "- " + body[0].upper() + body[1:]


def listify(text: str, target: str | None = None) -> str:
    """Rewrite a request plus >= 3 constraint sentences as the request and a bullet list."""
    _, spans = mask(text)
    has_fence = any(s.startswith("```") for s in spans)
    if "\n" in text.strip() or _EXISTING_LIST.search(text) or has_fence:
        return text

    def rewrite(masked: str) -> str:
        sentences = _SENTENCE.findall(masked.strip())
        flags = [bool(_CONSTRAINT.match(s)) and not s.endswith("?") for s in sentences]
        if not sentences or flags[0] or sum(flags) < MIN_CONSTRAINTS:
            return masked
        first = flags.index(True)
        if not all(flags[first:]):  # prose after the constraints: moving it would reorder meaning
            return masked
        head = " ".join(sentences[:first])
        return head + "\n" + "\n".join(_bullet(s) for s in sentences[first:])

    out = with_masked(text, rewrite)
    return out if out != text and _cost(out, target) <= _cost(text, target) else text


OPERATOR = MaskedOperator("L", "list-ify constraints", listify)
