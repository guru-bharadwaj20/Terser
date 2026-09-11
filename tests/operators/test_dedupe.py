"""``D`` deduplicate restated requirements (row 1.2.3)."""

import pytest

from terser.operators.dedupe import OPERATOR, dedupe


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "Write a poem about cats. Keep it short. Oh and keep it brief, please.",
            "Write a poem about cats. Keep it short.",
        ),
        (
            "Summarize this article. Be concise. Keep it short.",
            "Summarize this article. Be concise.",
        ),
        ("Explain TCP. Short answer please. Don't ramble.", "Explain TCP. Short answer please."),
        (
            "Write a palindrome checker. It should ignore case and spaces. "
            "Make sure it ignores case, that is important. Add type hints.",
            "Write a palindrome checker. It should ignore case and spaces. Add type hints.",
        ),
        ("Do not use recursion. Never use recursion, seriously.", "Do not use recursion."),
        ("Keep it short.\n\nKeep it brief.", "Keep it short."),
        (
            "Keep it under 100 words. Be brief.\n\nUse 2 sentences.",
            "Keep it under 100 words.\n\nUse 2 sentences.",
        ),
    ],
)
def test_drops_restatements(raw: str, expected: str) -> None:
    assert OPERATOR(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Keep it short. Keep it under 100 words.", "Keep it under 100 words."),
        (
            "Explain recursion. Explain recursion with an example.",
            "Explain recursion with an example.",
        ),
        ("Use Python. Use Python and do not use numpy.", "Use Python and do not use numpy."),
    ],
)
def test_keeps_the_more_specific_sentence(raw: str, expected: str) -> None:
    assert OPERATOR(raw) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Use 3 examples. Use 5 examples.",
        "Keep it under 100 words. Keep it under 50 words.",
        "Use recursion. Do not use recursion.",
        "Use numpy. Do not use numpy for plotting.",
        "Translate the email into formal French. Translate the email into formal German.",
        "Why is my loop slow? My loop is slow.",
        "Here they are:\n\nThey are late. Again, keep the dates as they are.",
        "Write a short story. Keep it brief.",
        "Python 3.11 is fine. Use v2.0 of the API. Cost is $1.10 in total.",
    ],
)
def test_never_drops_new_information(text: str) -> None:
    assert OPERATOR(text) == text


def test_more_specific_number_is_kept() -> None:
    assert "100" in OPERATOR("Keep it brief. Keep it under 100 words. Be concise.")


@pytest.mark.parametrize(
    "text",
    [
        'Translate "keep it short". Keep it short.',
        "Explain `keep it short`. Keep it short.",
        "Fix this:\n```python\n# keep it short\nx = 1\n```\nKeep it short.",
    ],
)
def test_payload_sentences_never_dropped_or_edited(text: str) -> None:
    assert OPERATOR(text) == text


def test_payload_restatement_outside_payload_still_deduped() -> None:
    text = 'Translate "hola". Keep it short. Keep it brief.'
    assert OPERATOR(text) == 'Translate "hola". Keep it short.'


def test_idempotent_and_target_agnostic() -> None:
    raw = "Explain DNS. Keep it short. Be concise. Keep it under 50 words."
    once = OPERATOR(raw)
    assert once == "Explain DNS. Keep it under 50 words."
    assert OPERATOR(once) == once
    assert dedupe(raw, "gpt-4o") == dedupe(raw, None)


def test_efficient_prompt_is_noop() -> None:
    text = "Write a Python function that merges two sorted lists. Return a new list."
    assert OPERATOR(text) == text
