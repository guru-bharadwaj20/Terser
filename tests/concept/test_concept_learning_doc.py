"""The worked example in docs/derivations/concept_learning.md (row 1.3.2).

Every S, G and version space written in section 8 of the doc is asserted here, so the doc
and the code cannot drift apart. If this test fails, fix the doc or the code, not the test
data alone.
"""

from __future__ import annotations

import pytest

from terser.concept.attributes import ATTRIBUTES
from terser.concept.candidate_elimination import BoundaryTooLargeError, candidate_elimination
from terser.concept.find_s import covered_negatives, find_s
from terser.concept.hypothesis import ANY, NONE
from terser.concept.version_space import (
    brute_force_version_space,
    iter_version_space,
    version_space,
)

NAMES = ("length", "has_constraints", "tone_cue", "op_H")
DOMAINS = [ATTRIBUTES[name] for name in NAMES]

# Section 8.1 of the doc: x0 .. x5 (label True = quality preserved).
TOY = [
    (("long", "no", "no", "yes"), True),  # x0
    (("medium", "no", "no", "yes"), True),  # x1
    (("long", "yes", "yes", "yes"), False),  # x2
    (("short", "no", "yes", "yes"), False),  # x3
    (("long", "no", "no", "no"), True),  # x4
    (("medium", "yes", "no", "yes"), False),  # x5
]
# Section 8.4: an extra negative that collapses the version space.
X6 = (("short", "no", "no", "no"), False)

# Section 8.2: (S, G) after each example.
EXPECTED_TRACE = [
    ({("long", "no", "no", "yes")}, {("?", "?", "?", "?")}),
    ({("?", "no", "no", "yes")}, {("?", "?", "?", "?")}),
    ({("?", "no", "no", "yes")}, {("?", "no", "?", "?"), ("?", "?", "no", "?")}),
    ({("?", "no", "no", "yes")}, {("?", "?", "no", "?")}),
    ({("?", "no", "no", "?")}, {("?", "?", "no", "?")}),
    ({("?", "no", "no", "?")}, {("?", "no", "no", "?")}),
]
# Section 8.3: version space after each example.
EXPECTED_VS_SIZES = [16, 8, 6, 4, 2, 1]
VS_AFTER_X3 = {
    ("?", "no", "no", "yes"),
    ("?", "no", "no", "?"),
    ("?", "?", "no", "yes"),
    ("?", "?", "no", "?"),
}


def test_domains_match_doc() -> None:
    assert DOMAINS == [("short", "medium", "long"), ("yes", "no"), ("yes", "no"), ("yes", "no")]
    assert ANY == "?" and NONE == "∅"


def test_trace_matches_doc() -> None:
    result = candidate_elimination(TOY, DOMAINS, trace=True)
    assert not result.collapsed
    assert [(set(s), set(g)) for s, g in result.trace] == EXPECTED_TRACE


def test_version_space_after_each_example() -> None:
    result = candidate_elimination(TOY, DOMAINS, trace=True)
    sizes = []
    for k, (s, g) in enumerate(result.trace):
        vs = set(iter_version_space(s, g, DOMAINS))
        assert vs == set(brute_force_version_space(TOY[: k + 1], DOMAINS))
        sizes.append(len(vs))
    assert sizes == EXPECTED_VS_SIZES
    s3, g3 = result.trace[3]
    assert set(iter_version_space(s3, g3, DOMAINS)) == VS_AFTER_X3


def test_final_version_space() -> None:
    result = candidate_elimination(TOY, DOMAINS)
    assert result.S == (("?", "no", "no", "?"),)
    assert result.G == (("?", "no", "no", "?"),)
    assert version_space(result, DOMAINS) == [("?", "no", "no", "?")]


def test_find_s_matches_doc() -> None:
    assert find_s(TOY) == ("?", "no", "no", "?")
    assert covered_negatives(find_s(TOY), TOY) == []
    assert find_s(TOY[:2]) == ("?", "no", "no", "yes")


def test_positives_first_gives_same_boundaries() -> None:
    plain = candidate_elimination(TOY, DOMAINS)
    reordered = candidate_elimination(TOY, DOMAINS, positives_first=True)
    assert reordered.order == (0, 1, 4, 2, 3, 5)
    assert (set(reordered.S), set(reordered.G)) == (set(plain.S), set(plain.G))


def test_collapse_matches_doc() -> None:
    result = candidate_elimination([*TOY, X6], DOMAINS)
    assert result.collapsed
    assert result.S == ()
    assert result.G == ()
    c = result.collapse
    assert c is not None
    assert c.index == 6
    assert c.conflicting == (1, 4)
    assert c.examples == (1, 4, 6)
    assert c.negative == 6
    assert c.witness == ("?", "no", "no", "?")
    assert covered_negatives(find_s([*TOY, X6]), [*TOY, X6]) == [6]
    assert c.describe(NAMES) == (
        "version space collapsed at example 6; minimal conflicting set [1, 4, 6]: "
        "its positives force <?, has_constraints=no, tone_cue=no, ?>, "
        "which covers negative example 6"
    )


def test_guard_matches_doc() -> None:
    negatives_first = [TOY[2], TOY[3], TOY[5], TOY[0], TOY[1], TOY[4]]
    with pytest.raises(BoundaryTooLargeError):
        candidate_elimination(negatives_first, DOMAINS, max_boundary_size=4)
    ok = candidate_elimination(negatives_first, DOMAINS, positives_first=True, max_boundary_size=4)
    assert ok.S == ok.G == (("?", "no", "no", "?"),)
