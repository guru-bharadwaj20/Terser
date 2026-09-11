"""``H`` drop hedging and politeness."""

import pytest

from terser.operators.hedging import OPERATOR

POSITIVE = [
    (
        "Could you maybe please explain recursion? Thanks in advance!",
        "Could you explain recursion?",
    ),
    ("Thanks so much. Explain X.", "Explain X."),
    ("Sorry if this is a dumb question, but what is a closure?", "What is a closure?"),
    ("Summarize this if possible, thanks!", "Summarize this"),
    ("Please, explain X.", "Explain X."),
    ("Explain X, please.", "Explain X."),
    ("If you don't mind, list the steps.", "List the steps."),
    ("If you don’t mind, list the steps.", "List the steps."),
    ("Maybe you could explain it?", "You could explain it?"),
    ("Is it possible to possibly merge these?", "Is it possible to merge these?"),
    ("Sorry to bother you, but how do I sort a dict?", "How do I sort a dict?"),
    ("This might be a dumb question but why is the sky blue?", "Why is the sky blue?"),
    ("Explain X.\nThanks in advance", "Explain X."),
    ("Please write a haiku. Any help would be greatly appreciated!", "Write a haiku."),
    ("Can you please review my PR? Thank you for your help!", "Can you review my PR?"),
    ("PLEASE fix this bug.", "Fix this bug."),
    (
        "I was wondering if you could perhaps explain DNS?",
        "I was wondering if you could explain DNS?",
    ),
    ("Write the test, if you don't mind. Thanks a lot :)", "Write the test."),
    ('Please print "please" twice.', 'Print "please" twice.'),
    # Leading interjections (removed later by F) do not hide a sentence-start hedge.
    ("hmm, maybe you can help?", "Hmm, you can help?"),
    ("Ok so perhaps you could explain X?", "Ok so you could explain X?"),
    # Trailing laughter (left for F) does not make "thanks" part of the content.
    ("Make it 8 lines. thanks lol", "Make it 8 lines. Lol"),
]

NOOP = [
    "Maybe the cache is stale.",
    "Write a thank you note to my aunt.",
    "Thanks!",
    "Hmm, maybe the cache is stale.",
    "Tom maybe you could help.",
    "thanks lol is my catchphrase",
    "I want to please my boss.",
    'Print "please" in `please()`',
    "Thank you for your help with the code",
    "Explain X.",
    "Possibly the worst code ever: review it.",
    "I think the bug is in line 3.",
    "Any help would be greatly appreciated.",
    "“Thanks in advance” is a cliche; explain why.",
    "Write a thank-you email.",
    "Pretty please with sugar on top.",
    "```\n# please, thanks in advance\n```",
]


@pytest.mark.parametrize(("text", "expected"), POSITIVE)
def test_drops_hedging(text: str, expected: str) -> None:
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
    assert (OPERATOR.code, OPERATOR.name) == ("H", "drop hedging/politeness")
