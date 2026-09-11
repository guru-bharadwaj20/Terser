"""``I`` imperative conversion."""

import pytest

from terser.operators.imperative import ING_TO_BASE, OPERATOR, REQUEST_VERBS

POSITIVE = [
    ("Can you explain recursion?", "Explain recursion."),
    ("could you please write a haiku about rain?", "Write a haiku about rain."),
    ("I was wondering if you could explain monads?", "Explain monads."),
    ("Would you mind rewriting this paragraph?", "Rewrite this paragraph."),
    ("Is it possible to summarize this article?", "Summarize this article."),
    ("Is it possible for you to shorten this?", "Shorten this."),
    ("I want you to act as a pirate.", "Act as a pirate."),
    ("I'd like you to review my code.", "Review my code."),
    ("Can you explain X? What about Y?", "Explain X. What about Y?"),
    ("Hi. Can you fix `foo()`? Thanks", "Hi. Fix `foo()`. Thanks"),
    ('Can you explain "why?" here?', 'Explain "why?" here.'),
    ("Line one\ncould you list three fruits?\nok", "Line one\nList three fruits.\nOk"),
    ("I'm wondering if you could explain X?", "Explain X."),
    ("I’d like you to explain X.", "Explain X."),
    ("We need you to translate this.", "Translate this."),
    ("Would you please give me three examples?", "Give me three examples."),
    ("WILL YOU LIST THE PLANETS?", "LIST THE PLANETS."),
    (
        "I need you to be concise. Can you compare Go and Rust?",
        "Be concise. Compare Go and Rust.",
    ),
    ("Would you mind explaining TCP?", "Explain TCP."),
]

NOOP = [
    "What is a monad?",
    "Can you run Python 2 code on 3?",
    "Is it possible to remove items while iterating?",
    "Would you mind jumping?",
    "Will you be there?",
    "Explain recursion.",
    "Can you explain what a monad is? what about Y?",
    'Say "can you explain this?" to the user.',
    "```\ncan you explain this?\n```",
    "Why can you explain it but not me?",
    "Do you think Rust is faster?",
    "Would you recommend-ish this?",
    "I want to explain this myself.",
    "I think the bug is in line 3.",
]


@pytest.mark.parametrize(("text", "expected"), POSITIVE)
def test_converts_requests(text: str, expected: str) -> None:
    assert OPERATOR(text) == expected


@pytest.mark.parametrize("text", NOOP)
def test_no_op(text: str) -> None:
    assert OPERATOR(text) == text


@pytest.mark.parametrize("text", [t for t, _ in POSITIVE] + NOOP)
def test_idempotent_and_never_longer(text: str) -> None:
    once = OPERATOR(text)
    assert OPERATOR(once) == once
    assert len(once) <= len(text)


def test_verb_tables() -> None:
    for ing, base in ING_TO_BASE.items():
        assert ing.endswith("ing") and not base.endswith("ing")
    assert "explain" in REQUEST_VERBS and "run" not in REQUEST_VERBS


def test_metadata() -> None:
    assert (OPERATOR.code, OPERATOR.name) == ("I", "imperative conversion")
