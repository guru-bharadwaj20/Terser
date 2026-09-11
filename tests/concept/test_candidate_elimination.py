"""Candidate-Elimination with collapse diagnosis (row 1.3.4)."""

from __future__ import annotations

import itertools
import random

import pytest

from terser.concept.candidate_elimination import (
    BoundaryTooLargeError,
    candidate_elimination,
    minimal_conflict,
)
from terser.concept.hypothesis import (
    ANY,
    Example,
    bottom,
    hypothesis_space,
    is_consistent,
    maximally_general,
    maximally_specific,
    top,
)

ENJOY_DOMAINS = [
    ("Sunny", "Cloudy", "Rainy"),
    ("Warm", "Cold"),
    ("Normal", "High"),
    ("Strong", "Weak"),
    ("Warm", "Cool"),
    ("Same", "Change"),
]
ENJOY_SPORT = [
    (("Sunny", "Warm", "Normal", "Strong", "Warm", "Same"), True),
    (("Sunny", "Warm", "High", "Strong", "Warm", "Same"), True),
    (("Rainy", "Cold", "High", "Strong", "Warm", "Change"), False),
    (("Sunny", "Warm", "High", "Strong", "Cool", "Change"), True),
]
ENJOY_S = {("Sunny", "Warm", ANY, "Strong", ANY, ANY)}
ENJOY_G = {("Sunny", ANY, ANY, ANY, ANY, ANY), (ANY, "Warm", ANY, ANY, ANY, ANY)}

# Toy example from docs/derivations/concept_learning.md.
TOY_DOMAINS = [("short", "medium", "long"), ("yes", "no"), ("low", "high"), ("yes", "no")]
TOY = [
    (("long", "no", "high", "yes"), True),
    (("long", "no", "low", "yes"), True),
    (("short", "yes", "high", "yes"), False),
    (("medium", "no", "high", "no"), False),
    (("short", "yes", "low", "yes"), True),
]


def _brute_boundaries(examples: list[Example], domains: list[tuple[str, ...]]) -> tuple[set, set]:
    vs = [h for h in hypothesis_space(domains) if is_consistent(h, examples)]
    return set(maximally_specific(vs)), set(maximally_general(vs))


def test_enjoy_sport_boundaries() -> None:
    r = candidate_elimination(ENJOY_SPORT, ENJOY_DOMAINS, trace=True)
    assert not r.collapsed
    assert set(r.S) == ENJOY_S and set(r.G) == ENJOY_G
    # Mitchell's Figure 2.4 / 2.5 intermediate boundaries
    s1, g1 = r.trace[0]
    assert s1 == (ENJOY_SPORT[0][0],) and g1 == (top(6),)
    s3, g3 = r.trace[2]
    assert s3 == (("Sunny", "Warm", ANY, "Strong", "Warm", "Same"),)
    assert set(g3) == {
        ("Sunny", ANY, ANY, ANY, ANY, ANY),
        (ANY, "Warm", ANY, ANY, ANY, ANY),
        (ANY, ANY, ANY, ANY, ANY, "Same"),
    }


def test_toy_trace_matches_derivation() -> None:
    r = candidate_elimination(TOY[:4], TOY_DOMAINS, trace=True)
    assert [(set(s), set(g)) for s, g in r.trace] == [
        ({("long", "no", "high", "yes")}, {(ANY,) * 4}),
        ({("long", "no", ANY, "yes")}, {(ANY,) * 4}),
        ({("long", "no", ANY, "yes")}, {("long", ANY, ANY, ANY), (ANY, "no", ANY, ANY)}),
        ({("long", "no", ANY, "yes")}, {("long", ANY, ANY, ANY), (ANY, "no", ANY, "yes")}),
    ]


def test_toy_collapse_reports_offender_and_minimal_conflict() -> None:
    r = candidate_elimination(TOY, TOY_DOMAINS)
    assert r.collapsed and r.collapse is not None
    c = r.collapse
    assert c.index == 4 and c.conflicting == (0, 2) and c.negative == 2
    assert c.witness == (ANY, ANY, ANY, "yes")
    assert c.examples == (0, 2, 4)
    assert "collapsed at example 4" in c.describe()
    assert "op_H=yes" in c.describe(["length", "has_payload", "filler_density", "op_H"])
    assert not r.S and not r.G


def test_collapse_on_negative_offender() -> None:
    data = [TOY[0], TOY[4], TOY[2]]  # positives first, then the negative they force in
    r = candidate_elimination(data, TOY_DOMAINS)
    assert r.collapse is not None
    assert r.collapse.index == 2 and r.collapse.conflicting == (0, 1)


def test_contradictory_duplicate_labels() -> None:
    x = TOY[0][0]
    r = candidate_elimination([TOY[1], (x, False), TOY[3], (x, True)], TOY_DOMAINS)
    assert r.collapse is not None
    assert r.collapse.index == 3 and r.collapse.conflicting == (1,)


def test_positives_first_reports_original_indices() -> None:
    r = candidate_elimination(TOY, TOY_DOMAINS, positives_first=True)
    assert r.order == (0, 1, 4, 2, 3)
    assert r.collapse is not None
    assert r.collapse.index == 2 and r.collapse.conflicting == (0, 4)


def _random_data(rng: random.Random, n_attrs: int, n: int) -> tuple[list, list[Example]]:
    domains = [tuple(f"v{j}" for j in range(rng.choice([2, 3]))) for _ in range(n_attrs)]
    target = tuple(rng.choice([ANY, ANY, *d]) for d in domains)
    data: list[Example] = []
    for _ in range(n):
        x = tuple(rng.choice(d) for d in domains)
        data.append((x, all(t == ANY or t == v for t, v in zip(target, x, strict=True))))
    return domains, data


def _assert_minimal_conflict(
    data: list[Example], domains: list, index: int, members: tuple[int, ...]
) -> None:
    subset = [data[j] for j in members]
    space = list(hypothesis_space(domains))
    assert index in members
    assert not any(is_consistent(h, subset) for h in space)
    for drop in members:
        rest = [data[j] for j in members if j != drop]
        assert any(is_consistent(h, rest) for h in space)
    assert any(is_consistent(h, data[:index]) for h in space)


@pytest.mark.parametrize("seed", range(12))
def test_matches_brute_force_and_order_independent(seed: int) -> None:
    rng = random.Random(seed)
    domains, data = _random_data(rng, 4, 8)
    expected_s, expected_g = _brute_boundaries(data, domains)
    r = candidate_elimination(data, domains)
    assert not r.collapsed  # labels come from a conjunctive target
    assert (set(r.S), set(r.G)) == (expected_s, expected_g)
    for _ in range(5):
        shuffled = data[:]
        rng.shuffle(shuffled)
        r2 = candidate_elimination(shuffled, domains)
        assert (set(r2.S), set(r2.G)) == (expected_s, expected_g)
    r3 = candidate_elimination(data, domains, positives_first=True)
    assert (set(r3.S), set(r3.G)) == (expected_s, expected_g)


@pytest.mark.parametrize("seed", range(20))
def test_noisy_collapse_is_detected_and_minimal(seed: int) -> None:
    rng = random.Random(1000 + seed)
    domains, data = _random_data(rng, 4, 10)
    data = [(x, (not y) if rng.random() < 0.3 else y) for x, y in data]
    space = list(hypothesis_space(domains))
    r = candidate_elimination(data, domains)
    if r.collapse is None:
        s, g = _brute_boundaries(data, domains)
        assert (set(r.S), set(r.G)) == (s, g) and s
        return
    i = r.collapse.index
    assert not any(is_consistent(h, data[: i + 1]) for h in space)
    _assert_minimal_conflict(data, domains, i, r.collapse.examples)


def test_all_negative_data_keeps_bottom() -> None:
    x = TOY[0][0]
    others = [(tuple(v), False) for v in itertools.product(*TOY_DOMAINS) if v != x]
    r = candidate_elimination(others, TOY_DOMAINS)
    assert not r.collapsed
    assert r.S == (bottom(4),) and r.G == (x,)
    everything = [*others, (x, False)]
    r = candidate_elimination(everything, TOY_DOMAINS, max_boundary_size=None)
    assert not r.collapsed and r.S == (bottom(4),) and r.G == (bottom(4),)


def test_boundary_guard_and_positives_first() -> None:
    rng = random.Random(7)
    n_attrs = 12
    domains = [("a", "b", "c")] * n_attrs
    negatives = [(tuple(rng.choice("abc") for _ in range(n_attrs)), False) for _ in range(6)]
    positive = (("a",) * n_attrs, True)
    negatives = [(x, False) for x, _ in negatives if x != positive[0]]
    data = [*negatives, positive]
    with pytest.raises(BoundaryTooLargeError, match="positives_first=True"):
        candidate_elimination(data, domains, max_boundary_size=500)
    r = candidate_elimination(data, domains, positives_first=True, max_boundary_size=500)
    assert r.S == (positive[0],)
    assert all(is_consistent(g, data) for g in r.G)


def test_minimal_conflict_rejects_consistent_input() -> None:
    with pytest.raises(ValueError, match="consistent"):
        minimal_conflict(TOY, 3)


def test_input_validation() -> None:
    with pytest.raises(ValueError):
        candidate_elimination([(("huge", "no", "low", "yes"), True)], TOY_DOMAINS)
