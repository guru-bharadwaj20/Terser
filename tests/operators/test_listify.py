"""``L`` list-ify constraints."""

import re

import pytest

from terser.operators.listify import OPERATOR

POSITIVE = [
    (
        "Write a sort function. Make sure to use type hints. Also don't use recursion. "
        "And it should also handle empty lists.",
        "Write a sort function.\n- Use type hints\n- Don't use recursion\n- Handle empty lists",
    ),
    (
        "Summarize this article. Please keep it under 100 words. Also avoid jargon. "
        "And make sure to include the main argument.",
        "Summarize this article.\n- Keep it under 100 words\n- Avoid jargon\n"
        "- Include the main argument",
    ),
    (
        "Write a poem about rain. Make sure to keep it short. Also make sure to use rhyme. "
        "And make sure to avoid clichés.",
        "Write a poem about rain.\n- Keep it short\n- Use rhyme\n- Avoid clichés",
    ),
    (
        "Explain DNS. Please use an analogy. Also please keep it brief. "
        "Oh and please avoid acronyms.",
        "Explain DNS.\n- Use an analogy\n- Keep it brief\n- Avoid acronyms",
    ),
    (
        "Refactor `parse()` for speed. Make sure to keep the signature. "
        "Also don't change `parse()` output. And remember to add tests.",
        "Refactor `parse()` for speed.\n- Keep the signature\n- Don't change `parse()` output\n"
        "- Add tests",
    ),
    (
        'Write a tagline. Make sure to include "fast". Also make sure to avoid "cheap". '
        "Also remember to keep it short.",
        'Write a tagline.\n- Include "fast"\n- Avoid "cheap"\n- Keep it short',
    ),
    (
        "Build a REST API. It should use Flask. It should also return JSON. "
        "It needs to validate input. It has to log errors.",
        "Build a REST API.\n- Use Flask\n- Return JSON\n- Validate input\n- Log errors",
    ),
    (
        "Translate this email. Also keep the tone formal! And please don't translate names. "
        "Finally, make sure to preserve line breaks.",
        "Translate this email.\n- Keep the tone formal\n- Don't translate names\n"
        "- Preserve line breaks",
    ),
    (
        "Write a SQL query for top customers. Also use a CTE. Plus limit it to 10 rows. "
        "And make sure to order by revenue.",
        "Write a SQL query for top customers.\n- Use a CTE\n- Limit it to 10 rows\n"
        "- Order by revenue",
    ),
    (
        "Describe photosynthesis. Make sure to use simple words. "
        "Also include a diagram description. And also make sure to cite a source.",
        "Describe photosynthesis.\n- Use simple words\n- Include a diagram description\n"
        "- Cite a source",
    ),
    (
        "Draft a cover letter. Make sure to keep it to one page. Make sure to mention Python. "
        "Make sure to avoid clichés.",
        "Draft a cover letter.\n- Keep it to one page\n- Mention Python\n- Avoid clichés",
    ),
    (
        "Create a workout plan. Please include rest days. Please limit sessions to 45 minutes. "
        "Please avoid heavy weights.",
        "Create a workout plan.\n- Include rest days\n- Limit sessions to 45 minutes\n"
        "- Avoid heavy weights",
    ),
    (
        "Plan a 3-day trip to Rome. Also make sure to include museums. "
        "Also make sure to keep costs low. Also make sure to add food tips.",
        "Plan a 3-day trip to Rome.\n- Include museums\n- Keep costs low\n- Add food tips",
    ),
    (
        "Write a haiku. Make sure to use 5-7-5 syllables. Also never mention the moon. "
        "And remember to output only the poem.",
        "Write a haiku.\n- Use 5-7-5 syllables\n- Never mention the moon\n- Output only the poem",
    ),
    (
        "Generate test data. It should include 3.5% nulls. And make sure to use ISO dates. "
        "Also remember to return CSV.",
        "Generate test data.\n- Include 3.5% nulls\n- Use ISO dates\n- Return CSV",
    ),
    (
        "Explain monads. Also do not use Haskell. And also make sure to keep it short. "
        "Oh and make sure to give one example.",
        "Explain monads.\n- Do not use Haskell\n- Keep it short\n- Give one example",
    ),
    (
        "Write a parser. It is for a CLI tool. Make sure to use argparse. "
        "Also make sure to add help text. And make sure to validate paths.",
        "Write a parser. It is for a CLI tool.\n- Use argparse\n- Add help text\n"
        "- Validate paths",
    ),
]

NOOP = [
    # already efficient / no constraints
    "Explain recursion.",
    "Write a function that reverses a string.",
    # bullets would cost more than the sentences they replace
    "Write a README. Keep it short. Use headers. Avoid emojis.",
    # fewer than three constraints
    "Write a function. Make sure to use type hints. Also add docs.",
    # prose after the constraints would be reordered
    "Write a function. Make sure to use type hints. Also make sure to add docs. "
    "And make sure to avoid globals. Thanks for the help.",
    # no main request first
    "Make sure to use type hints. Write a function. Also make sure to add docs. "
    "And make sure to avoid globals.",
    # questions are not constraints
    "Write code. Should I use recursion? Should I use loops? Should I use Python?",
    # code blocks
    "Fix this:\n```\nx = 1\n```\nMake sure to keep it short. Also make sure to use hints. "
    "And make sure to avoid globals.",
    "Explain ```x = 1``` please. Make sure to keep it short. Also make sure to use hints. "
    "And make sure to avoid globals.",
    # existing lists and multi-line prompts
    "Write a function:\n- use type hints\n- avoid globals\n- add docs",
    "Write a function. 1. Make sure to use type hints 2. Also make sure to avoid globals",
    "Write a function.\nMake sure to use type hints.\nAlso make sure to avoid globals.\n"
    "And make sure to add docs.",
    # constraints inside a quoted payload
    'Print "Make sure to keep it. Also make sure to use it. And make sure to avoid it." verbatim.',
    "Echo “Please keep it. Please use it. Please avoid it. Please add it.” exactly.",
    "",
]

_LEAD_IN_WORDS = {"make", "sure", "be", "remember", "to", "it", "should", "needs", "has"}
_LEAD_IN_WORDS |= {"also", "and", "oh", "please", "plus", "finally"}


@pytest.mark.parametrize(("text", "expected"), POSITIVE)
def test_listifies_constraints(text: str, expected: str) -> None:
    assert OPERATOR(text) == expected


@pytest.mark.parametrize("text", NOOP)
def test_no_op(text: str) -> None:
    assert OPERATOR(text) == text


@pytest.mark.parametrize("text", [t for t, _ in POSITIVE] + NOOP)
def test_idempotent_and_never_longer(text: str) -> None:
    once = OPERATOR(text)
    assert OPERATOR(once) == once
    assert len(once) <= len(text)


def _words(text: str) -> list[str]:
    return re.findall(r"[\w'’%-]+", text.lower())


@pytest.mark.parametrize(("text", "expected"), POSITIVE)
def test_no_constraint_text_lost(text: str, expected: str) -> None:
    out_words = _words(OPERATOR(text))
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        words = _words(sentence)
        # Only chaining lead-ins may be dropped; every other word survives in order.
        while words and words[0] in _LEAD_IN_WORDS:
            words.pop(0)
        start = 0
        for word in words:
            assert word in out_words[start:], (word, sentence)
            start = out_words.index(word, start) + 1


def test_metadata() -> None:
    assert (OPERATOR.code, OPERATOR.name) == ("L", "list-ify constraints")
