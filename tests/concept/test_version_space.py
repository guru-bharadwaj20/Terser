"""Version space enumeration (row 1.3.5)."""

from __future__ import annotations

import itertools
import random

import pytest

import terser.concept as concept
from terser.concept.candidate_elimination import candidate_elimination
from terser.concept.hypothesis import ANY, Example, bottom
from terser.concept.version_space import (
    brute_force_version_space,
    iter_version_space,
    version_space,
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

# 5-attribute toy space shaped like Terser prompt attributes.
TOY5_DOMAINS = [
    ("short", "medium", "long"),
    ("yes", "no"),
    ("yes", "no"),
    ("yes", "no"),
    ("low", "high"),
]
TOY5 = [
    (("long", "no", "no", "no", "high"), True),
    (("medium", "no", "yes", "no", "high"), True),
    (("short", "yes", "no", "no", "high"), False),
    (("long", "no", "no", "yes", "low"), False),
    (("medium", "no", "no", "no", "low"), False),
]


def _check(examples: list[Example], domains: list[tuple[str, ...]]) -> list:
    r = candidate_elimination(examples, domains)
    enumerated = version_space(r, domains)
    assert len(enumerated) == len(set(enumerated)), "duplicates"
    assert set(enumerated) == set(brute_force_version_space(examples, domains))
    return enumerated


def test_enjoy_sport_size_six() -> None:
    vs = _check(ENJOY_SPORT, ENJOY_DOMAINS)
    assert len(vs) == 6
    assert set(vs) == {
        ("Sunny", "Warm", ANY, "Strong", ANY, ANY),
        ("Sunny", ANY, ANY, "Strong", ANY, ANY),
        ("Sunny", "Warm", ANY, ANY, ANY, ANY),
        (ANY, "Warm", ANY, "Strong", ANY, ANY),
        ("Sunny", ANY, ANY, ANY, ANY, ANY),
        (ANY, "Warm", ANY, ANY, ANY, ANY),
    }


def test_five_attribute_toy_matches_brute_force() -> None:
    vs = _check(TOY5, TOY5_DOMAINS)
    assert vs  # non-empty, so the comparison is meaningful
    for k in range(len(TOY5) + 1):
        _check(TOY5[:k], TOY5_DOMAINS)
    for perm in itertools.permutations(TOY5):
        assert set(
            version_space(candidate_elimination(list(perm), TOY5_DOMAINS), TOY5_DOMAINS)
        ) == set(vs)


def test_no_examples_is_whole_space() -> None:
    assert len(_check([], TOY5_DOMAINS)) == 1 + 4 * 3 * 3 * 3 * 3


def test_only_negatives_includes_bottom() -> None:
    vs = _check(TOY5[2:], TOY5_DOMAINS)
    assert bottom(5) in vs


@pytest.mark.parametrize("seed", range(15))
def test_random_datasets_match_brute_force(seed: int) -> None:
    rng = random.Random(seed)
    domains = [tuple(f"v{j}" for j in range(rng.choice([2, 3]))) for _ in range(5)]
    target = tuple(rng.choice([ANY, ANY, *d]) for d in domains)
    noisy = seed % 3 == 0
    data: list[Example] = []
    for _ in range(rng.randint(1, 9)):
        x = tuple(rng.choice(d) for d in domains)
        label = all(t == ANY or t == v for t, v in zip(target, x, strict=True))
        data.append((x, (not label) if noisy and rng.random() < 0.2 else label))
    _check(data, domains)


def test_collapsed_is_empty() -> None:
    x = TOY5[0][0]
    data = [(x, True), (x, False)]
    r = candidate_elimination(data, TOY5_DOMAINS)
    assert r.collapsed and version_space(r, TOY5_DOMAINS) == []
    assert brute_force_version_space(data, TOY5_DOMAINS) == []


def test_iter_version_space_on_explicit_boundaries() -> None:
    s = ("long", "no", ANY, "yes")
    g = [("long", ANY, ANY, ANY), (ANY, "no", ANY, "yes")]
    toy_domains = [("short", "medium", "long"), ("yes", "no"), ("low", "high"), ("yes", "no")]
    assert len(list(iter_version_space([s], g, toy_domains))) == 5


def test_package_reexports() -> None:
    for name in (
        "ANY",
        "NONE",
        "ATTRIBUTES",
        "describe_prompt",
        "instance",
        "covers",
        "more_general_or_equal",
        "find_s",
        "candidate_elimination",
        "Collapse",
        "BoundaryTooLargeError",
        "version_space",
        "brute_force_version_space",
    ):
        assert name in concept.__all__ and hasattr(concept, name)
