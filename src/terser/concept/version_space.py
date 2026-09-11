"""Enumerate the version space between the ``S`` and ``G`` boundaries.

By the version space representation theorem, a conjunctive hypothesis ``h`` is consistent
with the data iff ``s <=g h <=g g`` for some ``s`` in ``S`` and ``g`` in ``G``. For a fixed
pair the members factor per attribute:

* ``s`` concrete (not bottom): attribute ``a`` may be ``s[a]`` or, if ``g[a]`` is ``?``,
  also ``?``; so the pair contributes ``2^k`` hypotheses, ``k`` = attributes where ``s`` is
  concrete and ``g`` is ``?``.
* ``s`` bottom (no positive seen yet): bottom itself, plus every non-bottom ``h <=g g``,
  where each ``?`` of ``g`` may stay ``?`` or take any domain value.

Pairs are enumerated in boundary order and a hypothesis is yielded only for the first ``g``
that contains it, so the output has no duplicates. :func:`brute_force_version_space` filters
the whole hypothesis space instead and exists to check the enumerator.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from itertools import product

from terser.concept.candidate_elimination import CandidateEliminationResult
from terser.concept.hypothesis import (
    ANY,
    Domains,
    Example,
    Hypothesis,
    bottom,
    hypothesis_space,
    is_bottom,
    is_consistent,
    more_general_or_equal,
)


def _between(s: Hypothesis, g: Hypothesis, domains: Domains) -> Iterator[Hypothesis]:
    """Non-bottom hypotheses ``h`` with ``s <=g h <=g g`` (``g`` assumed ``>=g s``)."""
    choices: list[Sequence[str]] = []
    for a, gc in enumerate(g):
        if gc != ANY:
            choices.append((gc,))
        elif is_bottom(s):
            choices.append((ANY, *domains[a]))
        elif s[a] == ANY:
            choices.append((ANY,))
        else:
            choices.append((s[a], ANY))
    yield from product(*choices)


def iter_version_space(
    S: Sequence[Hypothesis], G: Sequence[Hypothesis], domains: Domains
) -> Iterator[Hypothesis]:
    """Yield every hypothesis lying between some member of ``S`` and some member of ``G``."""
    seen_bottom = False
    for s in S:
        if is_bottom(s) and any(more_general_or_equal(g, s) for g in G) and not seen_bottom:
            seen_bottom = True
            yield bottom(len(domains))
        pairs = [g for g in G if more_general_or_equal(g, s)]
        for j, g in enumerate(pairs):
            for h in _between(s, g, domains):
                if not any(more_general_or_equal(e, h) for e in pairs[:j]):
                    yield h


def version_space(result: CandidateEliminationResult, domains: Domains) -> list[Hypothesis]:
    """All hypotheses consistent with the examples behind ``result`` (empty if it collapsed)."""
    if result.collapsed:
        return []
    return list(iter_version_space(result.S, result.G, domains))


def brute_force_version_space(examples: Sequence[Example], domains: Domains) -> list[Hypothesis]:
    """Filter the full conjunctive hypothesis space by consistency (exponential; for checks)."""
    return [h for h in hypothesis_space(domains) if is_consistent(h, examples)]
