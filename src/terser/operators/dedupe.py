"""``D`` deduplicate restated requirements.

Casual prompts often repeat a requirement ("keep it short ... oh and keep it brief"). The
operator drops a sentence whose content words are contained in another sentence of the
prompt, ignoring emphasis words ("important", "remember", "definitely", ...): "mostly
contained" is made precise as "contained up to emphasis", because a single differing content
word ("into French" / "into German") is usually the whole point of the sentence. Common
constraint restatements are normalised first, so "keep it short", "keep it brief", "be
concise" and "short answer please" count as the same requirement.

Safety rules (a missed deletion costs a few tokens; a wrong one changes the request):

* a sentence is only dropped in favour of one that carries **every** number it carries, so
  "Use 3 examples. Use 5 examples." are both kept;
* a negated sentence is only covered by a negated one, and an affirmative sentence only by
  words outside the negation's scope, so "Use recursion. Do not use recursion." and "Use numpy.
  Do not use numpy for plotting." are kept, while "Use Python." is dropped in favour of "Use
  Python and do not use numpy.";
* when the later sentence is the more specific one (it contains the earlier one plus extra
  words, numbers or a negation), the **earlier** sentence is dropped and the specific one kept;
* a sentence needs at least two content words (or a normalised constraint) to be dropped, so
  "Here they are:" is never mistaken for a restatement;
* questions are only compared with questions and statements with statements;
* a sentence containing a payload (code, inline code, quoted text) is never dropped, and
  payloads are never edited (:func:`terser.operators.masking.with_masked`).
"""

from __future__ import annotations

import re

from terser.operators.base import Operator
from terser.operators.masking import mask, unmask, with_masked

_EMPHASIS = set(
    "important importantly crucial critical essential key matter matters remember forget "
    "definitely absolutely really seriously again note mention thing part especially".split()
)
"""Words that add emphasis but no requirement; they may differ between restatements."""

BREVITY = "<brevity>"
_BREVITY = re.compile(
    r"\b(?:keep (?:it|this|that|the (?:answer|response|reply)|your (?:answer|response|reply))"
    r"(?: (?:very|really|nice and|super))? (?:short|brief|concise|succinct|tight)"
    r"|(?:be|stay) (?:very |really )?(?:short|brief|concise|succinct)"
    r"|(?:short|brief|concise|succinct) and (?:short|brief|concise|succinct|sweet|to the point)"
    r"|(?:short|brief|concise|succinct) (?:answer|response|reply)"
    r"|(?:don'?t|do not) (?:ramble|be (?:too )?(?:long|verbose|wordy))"
    r"|(?:keep it |in )?(?:under|at most|no more than|fewer than|less than|max(?:imum)?)"
    r" \S+ (?:words|sentences|lines|paragraphs|bullets)"
    r"|to the point|not too long)\b",
    re.IGNORECASE,
)
"""Brevity restatements, collapsed to one canonical token before comparison."""

_NEGATION = re.compile(
    r"\b(?:not|no|never|none|nothing|nor|without|avoid|cannot|can't|won't|shouldn't|mustn't|"
    r"don't|doesn't|didn't|isn't|aren't|wasn't|weren't|dont|doesnt|isnt)\b",
    re.IGNORECASE,
)
_NUMBER_WORDS = (
    "zero one two three four five six seven eight nine ten eleven twelve twenty thirty forty "
    "fifty hundred thousand million billion half once twice"
).split()
_NUMBER = re.compile(rf"\d+(?:[.,]\d+)*|\b(?:{'|'.join(_NUMBER_WORDS)})\b", re.IGNORECASE)
_STOP = set(
    "a an the and or but so to of in on for with it its it's is are be been am was were this "
    "that these those i me my we our you your can could would should will shall please just "
    "really also like as at by from about oh ok okay very make sure ensure want need do does "
    "any some there here if then".split()
)
_SENTENCE = re.compile(r"[^\s.!?](?:[^.!?\n]|[.!?]+(?=[^\s.!?]))*(?:[.!?]+|$)", re.MULTILINE)
"""A sentence: ends at ``.``/``!``/``?`` followed by whitespace, or at the end of a line."""


def _stem(word: str) -> str:
    """Very light stemming so "keeps"/"keep" and "examples"/"example" compare equal."""
    if len(word) > 4 and word.endswith("es") and word[-3] in "sxz":
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


class _Sentence:
    """A sentence span with the features used for comparison."""

    def __init__(self, match: re.Match[str], spans: list[str]) -> None:
        self.start, self.end = match.span()
        raw = match.group(0)
        self.has_payload = unmask(raw, spans) != raw
        self.question = raw.rstrip().endswith("?")
        self.numbers = {n.lower() for n in _NUMBER.findall(raw)}
        text = _NUMBER.sub(" ", _BREVITY.sub(f" {BREVITY} ", raw.lower()))
        negation = _NEGATION.search(text)
        self.negated = negation is not None
        self.words = _words(_NEGATION.sub(" ", text))
        self.affirmed = _words(text[: negation.start()]) if negation else self.words
        """Content words stated before any negation, i.e. outside its scope."""
        self.negated_words = (
            _words(_NEGATION.sub(" ", text[negation.start() :])) if negation else set()
        )
        """Content words inside the negation's scope (from the negation to the end)."""


def _words(text: str) -> set[str]:
    """Stemmed content words of already-normalised text."""
    return {_stem(w) for w in re.findall(r"<brevity>|[a-z0-9']+", text) if w not in _STOP}


def _covered_by(a: _Sentence, b: _Sentence) -> bool:
    """True if ``a`` restates nothing beyond ``b``: dropping ``a`` while keeping ``b`` is safe."""
    if a.has_payload or a.question != b.question or not a.words:
        return False
    if not a.numbers <= b.numbers:
        return False
    core = a.words - _EMPHASIS
    if len(core) < 2 and BREVITY not in core:
        return False
    if a.negated:
        # Each word must sit on the same side of the negation in both sentences: "do not use
        # numpy for plotting" is not covered by "use numpy ... and do not use matplotlib".
        if not b.negated:
            return False
        neg = a.negated_words - _EMPHASIS
        return not neg - b.negated_words and not (a.affirmed - _EMPHASIS) - b.affirmed
    # "Use X." is covered by "Use X and do not use Y." but not by "Do not use X for Y.":
    # the shared words must sit outside the negation's scope.
    return not core - b.affirmed


def dedupe(text: str, target: str | None = None) -> str:
    """Drop sentences that restate a requirement already stated elsewhere in the prompt."""
    del target  # tokenizer-agnostic
    _, spans = mask(text)

    def rewrite(masked: str) -> str:
        sentences = [_Sentence(m, spans) for m in _SENTENCE.finditer(masked)]
        dropped: set[int] = set()
        for j, later in enumerate(sentences):
            for i in range(j):
                if i in dropped:
                    continue
                earlier = sentences[i]
                if _covered_by(later, earlier):
                    dropped.add(j)
                    break
                if _covered_by(earlier, later):
                    dropped.add(i)
        if not dropped:
            return masked
        out, pos = [], 0
        for k in sorted(dropped):
            out.append(masked[pos : sentences[k].start])
            pos = sentences[k].end
            while pos < len(masked) and masked[pos] in " \t":
                pos += 1
        out.append(masked[pos:])
        return "".join(out)

    return with_masked(text, rewrite)


OPERATOR = Operator("D", "deduplicate restated requirements", dedupe)
