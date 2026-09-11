"""Payload masking shared by the rewrite operators."""

import re

import pytest

from terser.operators.masking import MaskedOperator, mask, unmask, with_masked

PAYLOADS = [
    "```python\n# um, basically please\nx  =  1\n```",
    "`please basically um`",
    '"hey so um, thanks in advance"',
    "“could you please explain?”",
]


@pytest.mark.parametrize("payload", PAYLOADS)
def test_payload_hidden_from_regexes(payload: str) -> None:
    masked, spans = mask(f"um {payload} please.")
    assert spans == [payload]
    assert not re.search(r"[A-Za-z0-9]", masked.replace("um", "").replace("please", ""))
    placeholder = masked[3:-8]
    for pattern in (r"\w", r"\s", r"[.!?,:;]", r"\d", r"[a-z]"):
        assert not re.search(pattern, placeholder, re.IGNORECASE)


@pytest.mark.parametrize("payload", PAYLOADS)
def test_round_trip(payload: str) -> None:
    text = f"Say {payload} and {payload} again."
    masked, spans = mask(text)
    assert len(spans) == 2
    assert unmask(masked, spans) == text


def test_with_masked_edits_only_outside_payload() -> None:
    text = 'please print "please" and `please`'
    assert with_masked(text, lambda t: t.replace("please", "")) == ' print "please" and `please`'


def test_multiline_fence_is_one_span() -> None:
    text = "Fix this:\n```\nplease\n\nplease\n```\nplease"
    masked, spans = mask(text)
    assert spans == ["```\nplease\n\nplease\n```"]
    assert masked.count("please") == 1


def test_dropped_placeholder_returns_input() -> None:
    text = "keep `code` here"
    assert with_masked(text, lambda t: "keep here") == text


def test_no_protected_spans_is_identity_mask() -> None:
    assert mask("plain text") == ("plain text", [])


def test_masked_operator_tidies_outside_payload_only() -> None:
    op = MaskedOperator("X", "drop um", lambda text, target: text.replace("um ", ""))
    text = 'um fix this ,  "a  ,b":\n```\nx  =  1 ,\n```'
    assert op(text) == 'Fix this, "a  ,b":\n```\nx  =  1 ,\n```'
    assert op("keep   as is") == "keep   as is"
