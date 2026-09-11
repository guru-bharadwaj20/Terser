"""Instance representation for concept learning (row 1.3.1)."""

from __future__ import annotations

import pytest

from terser.concept import attributes as at
from terser.tokenizers import count_tokens


def _words(n: int) -> str:
    text = " ".join(["word"] * n)
    assert count_tokens(text, "gpt-4o") == n
    return text


def identity(text: str) -> str:
    return text


def test_attribute_order_and_domains() -> None:
    assert list(at.ATTRIBUTES) == [
        "length",
        "has_constraints",
        "has_payload",
        "tone_cue",
        "filler_density",
        "op_F",
        "op_H",
        "op_D",
        "op_I",
        "op_L",
        "op_C",
        "op_S",
    ]
    assert at.ATTRIBUTES["length"] == ("short", "medium", "long")
    assert at.ATTRIBUTES["filler_density"] == ("low", "high")
    assert all(at.ATTRIBUTES[f"op_{c}"] == ("yes", "no") for c in at.OP_CODES)
    assert at.DOMAINS == tuple(at.ATTRIBUTES.values())


def test_length_thresholds() -> None:
    assert (at.SHORT_BELOW, at.MEDIUM_MAX) == (25, 80)
    assert [at.length_bucket(n) for n in (0, 24, 25, 80, 81)] == [
        "short",
        "short",
        "medium",
        "medium",
        "long",
    ]
    for n, bucket in [(24, "short"), (25, "medium"), (80, "medium"), (81, "long")]:
        assert at.describe_prompt(_words(n), remover=identity)["length"] == bucket


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Summarise this in at most 3 bullets.", True),
        ("Don't use jargon.", True),
        ("Write 200 words about tides.", True),
        ("You must cite sources", True),
        ("Explain how tides work.", False),
    ],
)
def test_has_constraints(text: str, expected: bool) -> None:
    assert at.has_constraints(text) is expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Fix this:\n```python\nprint(1)\n```", True),
        ("Why does `x += 1` fail?", True),
        ('Translate "the quick brown fox jumps over the lazy dog again and again"', True),
        ("Review:\ndef add(a, b):\n    return a + b", True),
        ("Optimise SELECT name FROM users WHERE id = 1", True),
        ("import numpy as np\nwhat does this do", True),
        ("function add(a, b) { return a + b; }", True),
        ("#include <stdio.h> why the error", True),
        ("What is a class in sociology? I want to import ideas from economics.", False),
        ('Translate "hello there"', False),
        ("Explain recursion simply.", False),
    ],
)
def test_has_payload(text: str, expected: bool) -> None:
    assert at.has_payload(text) is expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Write it in a friendly tone.", True),
        ("Explain gravity for a 5 year old.", True),
        ("Write a poem about rain.", True),
        ("List the prime numbers below 20.", False),
    ],
)
def test_tone_cue(text: str, expected: bool) -> None:
    assert at.has_tone_cue(text) is expected


def test_filler_density_uses_share_of_removed_tokens() -> None:
    text = _words(20)
    assert at.filler_ratio(text, identity) == 0.0
    # remove exactly 3 of 20 tokens -> 0.15, which is the "high" threshold (inclusive)
    three_fewer = _words(17)
    assert at.filler_ratio(text, lambda t: three_fewer) == pytest.approx(0.15)
    assert at.describe_prompt(text, lambda t: three_fewer)["filler_density"] == "high"
    two_fewer = _words(18)
    assert at.describe_prompt(text, lambda t: two_fewer)["filler_density"] == "low"
    assert at.filler_ratio(text, lambda t: text + " extra words") == 0.0  # clipped
    assert at.filler_ratio("", identity) == 0.0


def test_describe_prompt_keys_and_values() -> None:
    attrs = at.describe_prompt("Write a funny poem, at most 4 lines.", remover=identity)
    assert attrs == {
        "length": "short",
        "has_constraints": "yes",
        "has_payload": "no",
        "tone_cue": "yes",
        "filler_density": "low",
    }
    assert list(attrs) == list(at.PROMPT_ATTRIBUTES)


def test_instance_builds_ordered_tuple() -> None:
    attrs = at.describe_prompt("Explain recursion.", remover=identity)
    x = at.instance(attrs, "CHF")
    assert x == ("short", "no", "no", "no", "low", "yes", "yes", "no", "no", "no", "yes", "no")
    assert at.instance(attrs, ["F", "H", "C", "F"]) == x
    assert at.instance(attrs, "")[5:] == ("no",) * 7
    assert all(v in dom for v, dom in zip(x, at.DOMAINS, strict=True))


def test_instance_rejects_bad_input() -> None:
    attrs = at.describe_prompt("Explain recursion.", remover=identity)
    with pytest.raises(ValueError, match="unknown operator"):
        at.instance(attrs, "FX")
    with pytest.raises(ValueError, match="exactly"):
        at.instance({"length": "short"}, "F")
    with pytest.raises(ValueError, match="not in"):
        at.instance({**attrs, "length": "huge"}, "F")


def test_default_remover_integration() -> None:
    try:
        from terser.operators import apply_ops  # noqa: F401
    except ImportError:
        pytest.skip("terser.operators.apply_ops not available yet")
    casual = "hey so um, basically could you maybe please explain recursion? thanks in advance"
    attrs = at.describe_prompt(casual)
    assert attrs["filler_density"] == "high"
    assert at.describe_prompt("Explain recursion.")["filler_density"] == "low"
