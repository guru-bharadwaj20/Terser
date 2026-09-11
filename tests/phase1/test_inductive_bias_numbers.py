"""Row 1.3.8: the numbers and counter-examples in ``docs/derivations/inductive_bias.md``.

The counts are recomputed from :data:`terser.concept.attributes.ATTRIBUTES` and compared with
the values parsed from the doc's tables, so a change to the instance language (or an edit to
the doc) that makes them disagree fails here.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

from terser.concept.attributes import ATTRIBUTES, DOMAINS, OP_ATTRIBUTES, PROMPT_ATTRIBUTES
from terser.concept.candidate_elimination import candidate_elimination
from terser.concept.hypothesis import Example, hypothesis_space

DOC = Path(__file__).resolve().parents[2] / "docs" / "derivations" / "inductive_bias.md"
_ROW = re.compile(r"^\|\s*`(?P<key>[a-z0-9_]+)`\s*\|.*\|\s*(?P<value>[^|]+?)\s*\|\s*$")


def _doc_values() -> dict[str, str]:
    values: dict[str, str] = {}
    for line in DOC.read_text(encoding="utf-8").splitlines():
        m = _ROW.match(line)
        if m:
            values[m["key"]] = m["value"]
    return values


def _int(text: str) -> int:
    return int(text.replace(",", ""))


def _expected_counts() -> dict[str, int]:
    sizes = [len(d) for d in ATTRIBUTES.values()]
    n_x = math.prod(sizes)
    return {
        "prompt_values": math.prod(len(d) for d in PROMPT_ATTRIBUTES.values()),
        "op_flag_values": math.prod(len(d) for d in OP_ATTRIBUTES.values()),
        "instance_space": n_x,
        "syntactic_hypotheses": math.prod(s + 2 for s in sizes),
        "semantic_hypotheses": 1 + math.prod(s + 1 for s in sizes),
        "power_set_log2": n_x,
        "power_set_digits": len(str(2**n_x)),
        "unbiased_vs_log2_m300": n_x - 300,
    }


@pytest.mark.parametrize("key", sorted(_expected_counts()))
def test_counts_match_doc(key: str) -> None:
    values = _doc_values()
    assert key in values, f"{key} missing from the table in {DOC.name}"
    assert _int(values[key]) == _expected_counts()[key]


def test_semantic_count_matches_enumerator() -> None:
    # hypothesis_space yields every semantically distinct conjunction; count without storing.
    small = DOMAINS[:6]
    assert sum(1 for _ in hypothesis_space(small)) == 1 + math.prod(len(d) + 1 for d in small)


def test_prose_states_same_numbers() -> None:
    text = DOC.read_text(encoding="utf-8")
    c = _expected_counts()
    assert f"|X| = 48 · 128 = {c['instance_space']:,}" in text
    assert f"= {c['syntactic_hypotheses']:,}`" in text
    assert f"= {c['semantic_hypotheses']:,}`" in text
    assert f"2^{c['power_set_log2']}" in text
    mantissa = 2 ** c["power_set_log2"] / 10 ** (c["power_set_digits"] - 1)
    assert f"≈ {mantissa:.3f} × 10^{c['power_set_digits'] - 1}" in text


@pytest.mark.parametrize(
    ("key", "eps", "m"),
    [
        ("survival_eps01_m100", 0.01, 100),
        ("survival_eps01_m300", 0.01, 300),
        ("survival_eps02_m300", 0.02, 300),
        ("survival_eps05_m100", 0.05, 100),
        ("survival_eps05_m300", 0.05, 300),
    ],
)
def test_noise_survival_matches_doc(key: str, eps: float, m: int) -> None:
    stated = _doc_values()[key]
    expected = (1 - eps) ** m
    assert float(stated) == pytest.approx(expected, rel=5e-3)


# Section 1 counter-examples, attribute order of ATTRIBUTES.
def _x(prompt: str, ops: str) -> tuple[str, ...]:
    length, constraints, tone = prompt.split(",")
    flags = tuple("yes" if c in ops else "no" for c in "FHDILCS")
    return (length, constraints, "no", tone, "low", *flags)


COUNTER_EXAMPLES: dict[str, list[Example]] = {
    "exception": [
        (_x("short,no,no", "H"), True),
        (_x("short,no,yes", "F"), True),
        (_x("short,no,yes", "H"), False),
    ],
    "disjunction": [
        (_x("short,yes,no", "I"), True),
        (_x("long,no,no", "I"), True),
        (_x("long,yes,no", "I"), False),
    ],
    "threshold": [
        (_x("medium,no,no", "L"), True),
        (_x("medium,no,no", "L"), False),
    ],
    "interaction": [
        (_x("medium,no,no", "H"), True),
        (_x("medium,no,no", "I"), True),
        (_x("medium,no,no", "HI"), False),
    ],
    "count": [
        (_x("short,no,no", "F"), True),
        (_x("short,no,no", "H"), True),
        (_x("short,no,no", "FH"), False),
    ],
}


def test_counter_examples_match_doc_instances() -> None:
    text = DOC.read_text(encoding="utf-8")
    for name in ("exception", "disjunction", "threshold", "interaction"):
        for x, _ in COUNTER_EXAMPLES[name]:
            assert f"`<{', '.join(x)}>`" in text, (name, x)


@pytest.mark.parametrize("name", sorted(COUNTER_EXAMPLES))
def test_counter_examples_collapse(name: str) -> None:
    examples = COUNTER_EXAMPLES[name]
    for positives_first in (False, True):
        res = candidate_elimination(examples, DOMAINS, positives_first=positives_first)
        assert res.collapsed
        assert res.collapse is not None
        assert res.collapse.examples == tuple(range(len(examples)))


@pytest.mark.parametrize("name", ["exception", "interaction"])
def test_complement_is_conjunctive(name: str) -> None:
    flipped = [(x, not y) for x, y in COUNTER_EXAMPLES[name]]
    assert not candidate_elimination(flipped, DOMAINS).collapsed
