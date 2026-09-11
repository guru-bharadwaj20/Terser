"""Conjunctive hypotheses and the derivation toy example (row 1.3.2)."""

from __future__ import annotations

import pytest

from terser.concept import hypothesis as hy
from terser.concept.hypothesis import ANY, NONE

# Toy example from docs/derivations/concept_learning.md (0-based here; x1..x5 in the doc).
# Attributes: length, has_payload, filler_density, op_H.
TOY_DOMAINS = [("short", "medium", "long"), ("yes", "no"), ("low", "high"), ("yes", "no")]
TOY = [
    (("long", "no", "high", "yes"), True),
    (("long", "no", "low", "yes"), True),
    (("short", "yes", "high", "yes"), False),
    (("medium", "no", "high", "no"), False),
]
TOY_X5 = (("short", "yes", "low", "yes"), True)
TOY_VERSION_SPACE = {
    ("long", "no", ANY, "yes"),
    ("long", ANY, ANY, "yes"),
    ("long", "no", ANY, ANY),
    ("long", ANY, ANY, ANY),
    (ANY, "no", ANY, "yes"),
}


def test_symbols() -> None:
    assert (ANY, NONE) == ("?", "∅")


def test_covers() -> None:
    x = ("long", "no", "high", "yes")
    assert hy.covers(hy.top(4), x)
    assert not hy.covers(hy.bottom(4), x)
    assert hy.covers(("long", ANY, ANY, "yes"), x)
    assert not hy.covers(("short", ANY, ANY, ANY), x)
    assert not hy.covers(("long", NONE, ANY, ANY), x)
    with pytest.raises(ValueError):
        hy.covers(("long",), x)


def test_general_to_specific_ordering() -> None:
    h1, h2 = ("long", ANY, ANY, ANY), ("long", "no", ANY, ANY)
    assert hy.more_general_or_equal(h1, h2) and not hy.more_general_or_equal(h2, h1)
    assert hy.more_general(h1, h2) and not hy.more_general(h1, h1)
    assert hy.more_general_or_equal(h1, h1)
    assert hy.more_general_or_equal(hy.top(4), h1)
    assert hy.more_general_or_equal(h2, hy.bottom(4))
    assert hy.more_general_or_equal(h2, ("short", NONE, ANY, ANY))  # any ∅ is bottom
    assert not hy.more_general_or_equal(hy.bottom(4), h2)
    # incomparable
    h3 = (ANY, "yes", ANY, ANY)
    assert not hy.more_general_or_equal(h2, h3) and not hy.more_general_or_equal(h3, h2)


def test_ordering_agrees_with_coverage_sets() -> None:
    instances = [x for x in hy.hypothesis_space(TOY_DOMAINS) if ANY not in x and NONE not in x]
    space = list(hy.hypothesis_space(TOY_DOMAINS))
    cover = {h: frozenset(x for x in instances if hy.covers(h, x)) for h in space}
    for h1 in space:
        for h2 in space:
            assert hy.more_general_or_equal(h1, h2) == (cover[h1] >= cover[h2])


def test_min_generalise() -> None:
    x = ("long", "no", "high", "yes")
    assert hy.min_generalise(hy.bottom(4), x) == x
    assert hy.min_generalise(("long", "no", "low", "yes"), x) == ("long", "no", ANY, "yes")
    assert hy.min_generalise(("short", ANY, "high", "no"), x) == (ANY, ANY, "high", ANY)


def test_min_specialisations() -> None:
    x = ("short", "yes", "high", "yes")
    specs = hy.min_specialisations(hy.top(4), x, TOY_DOMAINS)
    assert set(specs) == {
        ("medium", ANY, ANY, ANY),
        ("long", ANY, ANY, ANY),
        (ANY, "no", ANY, ANY),
        (ANY, ANY, "low", ANY),
        (ANY, ANY, ANY, "no"),
    }
    assert all(not hy.covers(h, x) for h in specs)
    assert hy.min_specialisations(("long", ANY, ANY, ANY), x, TOY_DOMAINS) == [
        ("long", ANY, ANY, ANY)
    ]
    assert hy.min_specialisations(x, x, TOY_DOMAINS) == [hy.bottom(4)]


def test_hypothesis_space_size() -> None:
    enjoy = [("a", "b", "c"), *[("a", "b")] * 5]
    assert len(list(hy.hypothesis_space(enjoy))) == 973  # Mitchell's semantically distinct count
    assert len(set(hy.hypothesis_space(TOY_DOMAINS))) == 1 + 4 * 3 * 3 * 3


def test_boundaries_helpers() -> None:
    hs = [("long", ANY, ANY, ANY), ("long", "no", ANY, ANY), (ANY, "no", ANY, "yes")]
    assert hy.maximally_general(hs) == [("long", ANY, ANY, ANY), (ANY, "no", ANY, "yes")]
    assert hy.maximally_specific(hs) == [("long", "no", ANY, ANY), (ANY, "no", ANY, "yes")]


def test_format_and_validation() -> None:
    h = ("long", ANY, NONE, "yes")
    assert hy.format_hypothesis(h) == "<long, ?, ∅, yes>"
    names = ["length", "has_payload", "filler_density", "op_H"]
    assert hy.format_hypothesis(h, names) == "<length=long, ?, ∅, op_H=yes>"
    with pytest.raises(ValueError):
        hy.format_hypothesis(h, names[:2])
    assert hy.canonical(("long", NONE, ANY, ANY)) == hy.bottom(4)
    assert hy.canonical(h[:2]) == h[:2]
    hy.validate_domains(TOY_DOMAINS)
    for bad in ([()], [("a", "a")], [("a", ANY)]):
        with pytest.raises(ValueError):
            hy.validate_domains(bad)
    hy.validate_instance(TOY[0][0], TOY_DOMAINS)
    with pytest.raises(ValueError):
        hy.validate_instance(("huge", "no", "high", "yes"), TOY_DOMAINS)
    with pytest.raises(ValueError):
        hy.validate_instance(("long",), TOY_DOMAINS)


def test_toy_example_from_derivation() -> None:
    """Reproduce the hand-computed version space, S, G and collapse witness in the doc."""
    consistent = {h for h in hy.hypothesis_space(TOY_DOMAINS) if hy.is_consistent(h, TOY)}
    assert consistent == TOY_VERSION_SPACE
    assert hy.maximally_specific(consistent) == [("long", "no", ANY, "yes")]
    assert set(hy.maximally_general(consistent)) == {
        ("long", ANY, ANY, ANY),
        (ANY, "no", ANY, "yes"),
    }
    # Find-S trace by hand: x1 then x2
    s = hy.min_generalise(hy.bottom(4), TOY[0][0])
    assert s == ("long", "no", "high", "yes")
    assert hy.min_generalise(s, TOY[1][0]) == ("long", "no", ANY, "yes")
    # Adding x5 empties the version space; x1 and x5 alone force <?, ?, ?, yes>, covering x3.
    with_x5 = [*TOY, TOY_X5]
    assert not any(hy.is_consistent(h, with_x5) for h in hy.hypothesis_space(TOY_DOMAINS))
    witness = hy.min_generalise(TOY[0][0], TOY_X5[0])
    assert witness == (ANY, ANY, ANY, "yes") and hy.covers(witness, TOY[2][0])
    assert any(hy.is_consistent(h, [TOY[0], TOY[2]]) for h in hy.hypothesis_space(TOY_DOMAINS))
    assert any(hy.is_consistent(h, [TOY[2], TOY_X5]) for h in hy.hypothesis_space(TOY_DOMAINS))
