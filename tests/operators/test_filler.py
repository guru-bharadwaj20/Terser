"""``F`` drop filler."""

import pytest

from terser.operators.filler import OPERATOR

POSITIVE = [
    ("hey so um, basically can you explain recursion?", "Can you explain recursion?"),
    ("Hi there! Explain closures.", "Explain closures."),
    ("ok so I need, like, a list of primes lol", "I need a list of primes"),
    (
        "Well, what is a monad? Um, also why does it matter?",
        "What is a monad? Also why does it matter?",
    ),
    ("I mean, the code is, you know, broken.", "The code is broken."),
    ("So, honestly, write a poem about rain haha", "Write a poem about rain"),
    ("Can you kind of explain this error?", "Can you explain this error?"),
    ("Hello, can you sort of summarize this article?", "Can you summarize this article?"),
    ("Uh, what does this regex do?", "What does this regex do?"),
    ("Okay, um, so what is Big O?", "What is Big O?"),
    ("The build, uh, fails on Windows.", "The build fails on Windows."),
    ("Hey! Um so basically I need a SQL query.", "I need a SQL query."),
    ("It literally crashes every time.", "It crashes every time."),
    (
        "First line.\nso, um, second line totally needs fixing.",
        "First line.\nSecond line needs fixing.",
    ),
    ("Explain `um basically` now, um.", "Explain `um basically` now."),
    ("HEY SO UM, EXPLAIN DNS.", "EXPLAIN DNS."),
    ("It is totally broken, lol.", "It is broken."),
    ("Hmm, list three sorting algorithms.", "List three sorting algorithms."),
    ("Haha. Explain X, haha, then Y.", "Explain X, then Y."),
    ("um, fix this:\n```\nx = 1 # um\n```", "Fix this:\n```\nx = 1 # um\n```"),
    ('So basically print "hey so um" please.', 'Print "hey so um" please.'),
]

NOOP = [
    "Explain recursion.",
    "Talk like a pirate.",
    "It is literally 3 lines.",
    "Hello world in C?",
    "So many rows fail.",
    "Well documented code is nice.",
    "Just in time compilation: explain it.",
    "Translate literally word by word.",
    "I think the bug is in line 3.",
    'Print "um, basically, lol" twice.',
    "```\n# hey so um basically\nx  =  1\n```",
    "It is kind of slow.",
    "Do you know, roughly, how big it is?",
    "Hmm",
    "OK",
    "So-called experts disagree.",
    "Things like apples and pears.",
    "explain this lowercase prompt",
]


@pytest.mark.parametrize(("text", "expected"), POSITIVE)
def test_drops_filler(text: str, expected: str) -> None:
    assert OPERATOR(text) == expected


@pytest.mark.parametrize("text", NOOP)
def test_no_op(text: str) -> None:
    assert OPERATOR(text) == text


@pytest.mark.parametrize("text", [t for t, _ in POSITIVE] + NOOP)
def test_idempotent_and_never_longer(text: str) -> None:
    once = OPERATOR(text)
    assert OPERATOR(once) == once
    assert len(once) <= len(text)


def test_metadata() -> None:
    assert (OPERATOR.code, OPERATOR.name) == ("F", "drop filler")
