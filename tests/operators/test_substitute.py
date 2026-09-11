"""``S`` tokenizer-aware phrase substitution (row 1.2.7)."""

import pytest

from terser.operators.substitute import (
    OPERATOR,
    PHRASE_SUBSTITUTIONS,
    SEARCH_CONTEXTS,
    TOKENIZER_DIVERGENT_EXAMPLE,
    _match_case,
    choose,
    substitute,
)
from terser.tokenizers import count_tokens

TOKENIZERS = ("gpt-4o", "qwen2.5", "llama-3.1")


def test_table_has_at_least_30_entries_all_shorter() -> None:
    assert len(PHRASE_SUBSTITUTIONS) >= 30
    for phrase, alternatives in PHRASE_SUBSTITUTIONS.items():
        assert phrase == phrase.lower()
        assert alternatives
        assert all(len(alt) < len(phrase) for alt in alternatives)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Sort the list in order to find the median.", "Sort the list to find the median."),
        ("Due to the fact that it is slow, cache it.", "Because it is slow, cache it."),
        ("Utilize a dict prior to sorting.", "Use a dict before sorting."),
        ("The vast majority of users log in daily.", "Most users log in daily."),
    ],
)
def test_substitutes_without_target(raw: str, expected: str) -> None:
    assert OPERATOR(raw) == expected


@pytest.mark.parametrize("target", TOKENIZERS)
def test_in_order_to_becomes_to_on_every_tokenizer(target: str) -> None:
    assert substitute("Sort it in order to find the median.", target) == (
        "Sort it to find the median."
    )


def test_longest_phrase_wins() -> None:
    assert substitute("The vast majority of tests pass.") == "Most tests pass."


def test_no_partial_word_matches() -> None:
    text = "Numerousness is not a word we utilized."
    assert substitute(text) == text


@pytest.mark.parametrize(
    "text",
    [
        "Run:\n```python\n# in order to utilize numpy\nimport numpy\n```",
        "Explain `in_order to` and `utilize`.",
        'Translate "in order to" into French.',
        "Translate “prior to” into German.",
    ],
)
def test_payloads_untouched(text: str) -> None:
    assert OPERATOR(text) == text
    assert OPERATOR(text, "gpt-4o") == text


def test_payload_next_to_phrase_keeps_payload() -> None:
    text = 'Use "x" in order to test.'
    assert OPERATOR(text, "gpt-4o") == 'Use "x" to test.'


def test_ties_keep_original() -> None:
    # "approximately" is one gpt-4o token at a sentence start: no alternative is cheaper.
    assert choose("", "Approximately", " how many?", "gpt-4o") == "Approximately"


def test_efficient_text_is_noop() -> None:
    text = "Write a Python function that returns the nth Fibonacci number."
    for target in (None, *TOKENIZERS):
        assert OPERATOR(text, target) == text


def test_tokenizer_divergent_fixture() -> None:
    """DoD 1.2.7: S chooses differently for two tokenizers on a fixture."""
    kept = substitute(TOKENIZER_DIVERGENT_EXAMPLE, "gpt-4o")
    rewritten = substitute(TOKENIZER_DIVERGENT_EXAMPLE, "qwen2.5")
    assert kept == TOKENIZER_DIVERGENT_EXAMPLE
    assert rewritten == "About how many people live in Tokyo?"
    assert substitute(TOKENIZER_DIVERGENT_EXAMPLE, "llama-3.1") == rewritten
    assert count_tokens(kept, "gpt-4o") <= count_tokens(rewritten, "gpt-4o")
    assert count_tokens(rewritten, "qwen2.5") < count_tokens(kept, "qwen2.5")


def test_exhaustive_search_tokenizer_dependence() -> None:
    """Record the module-docstring finding: only ``approximately`` diverges across tokenizers."""
    divergent: set[str] = set()
    evaluations = 0
    for phrase in PHRASE_SUBSTITUTIONS:
        for cased in (phrase, _match_case("X", phrase)):
            for left, right in SEARCH_CONTEXTS:
                if cased != phrase and left:
                    continue  # sentence-initial case only where nothing precedes the phrase
                choices = {choose(left, cased, right, t) for t in TOKENIZERS}
                evaluations += len(TOKENIZERS)
                if len(choices) > 1:
                    divergent.add(phrase)
    assert evaluations == len(PHRASE_SUBSTITUTIONS) * 9 * len(TOKENIZERS)
    assert divergent == {"approximately"}
