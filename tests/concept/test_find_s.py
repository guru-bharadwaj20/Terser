"""Find-S (row 1.3.3)."""

from __future__ import annotations

import itertools

import pytest

from terser.concept.find_s import covered_negatives, find_s
from terser.concept.hypothesis import bottom, format_hypothesis

# Mitchell (1997), Table 2.1: EnjoySport.
ENJOY_SPORT = [
    (("Sunny", "Warm", "Normal", "Strong", "Warm", "Same"), True),
    (("Sunny", "Warm", "High", "Strong", "Warm", "Same"), True),
    (("Rainy", "Cold", "High", "Strong", "Warm", "Change"), False),
    (("Sunny", "Warm", "High", "Strong", "Cool", "Change"), True),
]


def test_enjoy_sport() -> None:
    h = find_s(ENJOY_SPORT)
    assert h == ("Sunny", "Warm", "?", "Strong", "?", "?")
    assert format_hypothesis(h) == "<Sunny, Warm, ?, Strong, ?, ?>"
    assert covered_negatives(h, ENJOY_SPORT) == []


def test_order_independent() -> None:
    for perm in itertools.permutations(ENJOY_SPORT):
        assert find_s(list(perm)) == ("Sunny", "Warm", "?", "Strong", "?", "?")


def test_no_positives_gives_bottom() -> None:
    assert find_s([ENJOY_SPORT[2]]) == bottom(6)
    assert find_s([], n_attributes=3) == bottom(3)
    with pytest.raises(ValueError):
        find_s([])


def test_single_positive_is_itself() -> None:
    assert find_s(ENJOY_SPORT[:1]) == ENJOY_SPORT[0][0]


def test_covered_negatives_reports_inconsistency() -> None:
    noisy = [*ENJOY_SPORT, (("Sunny", "Warm", "Normal", "Strong", "Cool", "Same"), False)]
    h = find_s(noisy)
    assert h == ("Sunny", "Warm", "?", "Strong", "?", "?")
    assert covered_negatives(h, noisy) == [4]


def test_length_mismatch() -> None:
    with pytest.raises(ValueError):
        find_s([(("a", "b"), True), (("a",), True)])
