"""Hypothesis enumeration and coverage for the row 5.2.4 study."""

from __future__ import annotations

import numpy as np

from terser.concept.hypothesis import ANY, bottom, covers, hypothesis_space
from terser.phase5.bayes_optimal_study import coverage_matrix, hypothesis_codes

DOMAINS = (("a", "b", "c"), ("y", "n"))


def _decode(row: np.ndarray) -> tuple[str, ...]:
    if row[0] == -2:
        return bottom(len(DOMAINS))
    return tuple(ANY if v == -1 else DOMAINS[i][v] for i, v in enumerate(row))


def test_same_space_as_concept_module() -> None:
    H = hypothesis_codes(DOMAINS)
    assert len(H) == 1 + 4 * 3
    assert {_decode(r) for r in H} == set(hypothesis_space(DOMAINS))


def test_coverage_matches_covers() -> None:
    H = hypothesis_codes(DOMAINS)
    X = np.array([[0, 0], [2, 1], [1, 0]], dtype=np.int8)
    P = coverage_matrix(H, X)
    for h_row, p_row in zip(H, P, strict=True):
        h = _decode(h_row)
        for x_row, p in zip(X, p_row, strict=True):
            x = tuple(DOMAINS[i][v] for i, v in enumerate(x_row))
            assert bool(p) == covers(h, x)
