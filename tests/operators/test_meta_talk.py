"""``C`` compress meta-talk."""

import pytest

from terser.operators.meta_talk import OPERATOR

POSITIVE = [
    ("I'm going to give you some text and I want you to summarize it.", "Summarize it."),
    ("I have a quick question: what is a monad?", "What is a monad?"),
    ("My question is: how do I sort a dict?", "How do I sort a dict?"),
    ("my question is how to reverse a list", "How to reverse a list"),
    ("So the thing is, my build fails.", "My build fails."),
    ("Here's what I need: a list of 5 fruits.", "A list of 5 fruits."),
    ("Here's what I need you to do - rewrite this.", "Rewrite this."),
    ("Quick question, what does `git rebase` do?", "What does `git rebase` do?"),
    (
        "Context first. I am going to paste my essay below and I'd like you to proofread it.",
        "Context first. Proofread it.",
    ),
    ("I've got a question for you: why is the sky blue?", "Why is the sky blue?"),
    ("I HAVE A QUESTION: what is DNS?", "What is DNS?"),
    ("Here is what we want: short answers.", "Short answers."),
    (
        "I'll give you a paragraph and I want you to translate it into French.",
        "Translate it into French.",
    ),
    ("Hi. My main question is why does this leak memory?", "Hi. Why does this leak memory?"),
    ("so the thing is my tests are flaky", "My tests are flaky"),
    ("Line one.\nI have a small question - is Python typed?", "Line one.\nIs Python typed?"),
    ("I’m going to send you a log and I want you to find the error.", "Find the error."),
]

NOOP = [
    "I have a question about taxes.",
    "My question is simple.",
    "The thing is broken.",
    "I have a quick question:",
    "Explain X.",
    'Say "I have a quick question: hi"',
    "Here's the code.",
    "I'm going to the store and I want you to wait.",
    "`my question is: x`",
    "Here's what I need.",
    "The question is: what is X?",
    "I think the bug is in line 3.",
]


@pytest.mark.parametrize(("text", "expected"), POSITIVE)
def test_compresses_meta_talk(text: str, expected: str) -> None:
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
    assert (OPERATOR.code, OPERATOR.name) == ("C", "compress meta-talk")
