"""Conjunctive hypotheses over discrete attributes (Mitchell, *Machine Learning*, ch. 2).

A hypothesis is a tuple with one constraint per attribute. Each constraint is a concrete
value, :data:`ANY` (``"?"``, every value is acceptable) or :data:`NONE` (``"∅"``, no value is
acceptable). A hypothesis *covers* an instance when every constraint accepts the instance's
value, so any hypothesis containing ``∅`` covers nothing. All such hypotheses denote the same
(empty) concept; this module treats them as one element, the *bottom* hypothesis, and
:func:`canonical` rewrites them to the all-``∅`` tuple.

In Terser an instance is a prompt's discrete attributes plus the operator flags that were
applied (see :mod:`terser.concept.attributes`), and the target concept is "the rewrite
preserved output quality". Example: ``<long, ?, no, no, high, yes, ?, ?, ?, ?, ?, ?>``
reads "long, payload-free, tone-free, filler-heavy prompts rewritten with ``F`` are preserved".

The general-to-specific ordering used here is syntactic: ``h1 >=g h2`` iff every constraint of
``h1`` is ``?`` or equal to ``h2``'s, or ``h2`` is bottom. It coincides with the semantic
ordering (coverage-set inclusion) whenever every attribute domain has at least two values.
The derivations live in ``docs/derivations/concept_learning.md``.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from itertools import product

ANY = "?"
"""Constraint accepting every value."""
NONE = "∅"
"""Constraint accepting no value."""

Hypothesis = tuple[str, ...]
"""One constraint per attribute: a value, :data:`ANY` or :data:`NONE`."""
Instance = tuple[str, ...]
"""One concrete value per attribute."""
Example = tuple[Instance, bool]
"""A labelled instance; ``True`` means positive (in Terser: quality preserved)."""
Domains = Sequence[Sequence[str]]
"""Allowed values for each attribute, in attribute order."""


def bottom(n_attributes: int) -> Hypothesis:
    """The most specific hypothesis ``<∅, …, ∅>``, which covers no instance."""
    return (NONE,) * n_attributes


def top(n_attributes: int) -> Hypothesis:
    """The most general hypothesis ``<?, …, ?>``, which covers every instance."""
    return (ANY,) * n_attributes


def is_bottom(h: Hypothesis) -> bool:
    """Whether ``h`` covers nothing (contains at least one ``∅``)."""
    return NONE in h


def canonical(h: Hypothesis) -> Hypothesis:
    """Rewrite any hypothesis containing ``∅`` to the all-``∅`` bottom; others unchanged."""
    return bottom(len(h)) if is_bottom(h) else h


def validate_domains(domains: Domains) -> None:
    """Raise ``ValueError`` if a domain is empty, repeats a value or uses ``?``/``∅`` as a value."""
    for a, values in enumerate(domains):
        if not values:
            raise ValueError(f"attribute {a} has an empty domain")
        if len(set(values)) != len(values):
            raise ValueError(f"attribute {a} has duplicate values: {list(values)}")
        if ANY in values or NONE in values:
            raise ValueError(f"attribute {a} uses a reserved symbol ({ANY!r} or {NONE!r})")


def validate_instance(x: Instance, domains: Domains) -> None:
    """Raise ``ValueError`` if ``x`` has the wrong length or a value outside its domain."""
    if len(x) != len(domains):
        raise ValueError(f"instance has {len(x)} values, expected {len(domains)}: {x}")
    for a, (v, values) in enumerate(zip(x, domains, strict=True)):
        if v not in values:
            raise ValueError(f"attribute {a}: value {v!r} not in domain {list(values)}")


def covers(h: Hypothesis, x: Instance) -> bool:
    """Whether hypothesis ``h`` classifies instance ``x`` as positive."""
    return all(c == ANY or c == v for c, v in zip(h, x, strict=True))


def more_general_or_equal(h1: Hypothesis, h2: Hypothesis) -> bool:
    """``h1 >=g h2``: every instance covered by ``h2`` is also covered by ``h1``."""
    if is_bottom(h2):
        return True
    if is_bottom(h1):
        return False
    return all(a == ANY or a == b for a, b in zip(h1, h2, strict=True))


def more_general(h1: Hypothesis, h2: Hypothesis) -> bool:
    """``h1 >g h2``: ``h1 >=g h2`` and not ``h2 >=g h1``."""
    return more_general_or_equal(h1, h2) and not more_general_or_equal(h2, h1)


def min_generalise(h: Hypothesis, x: Instance) -> Hypothesis:
    """The unique minimal generalisation of ``h`` that covers ``x``.

    From bottom this is ``x`` itself; otherwise each constraint disagreeing with ``x``
    becomes ``?``.
    """
    if is_bottom(h):
        return tuple(x)
    return tuple(c if c == v else ANY for c, v in zip(h, x, strict=True))


def min_specialisations(g: Hypothesis, x: Instance, domains: Domains) -> list[Hypothesis]:
    """All minimal specialisations of ``g`` that do not cover ``x``.

    If ``g`` already excludes ``x`` the result is ``[g]``. Otherwise each ``?`` at attribute
    ``a`` may be replaced by any value other than ``x[a]``. When no such replacement exists
    (``g`` has no ``?`` with an alternative value), the only specialisation is bottom.
    """
    if not covers(g, x):
        return [g]
    out: list[Hypothesis] = [
        g[:a] + (v,) + g[a + 1 :]
        for a, values in enumerate(domains)
        if g[a] == ANY
        for v in values
        if v != x[a]
    ]
    return out or [bottom(len(g))]


def hypothesis_space(domains: Domains) -> Iterator[Hypothesis]:
    """Every semantically distinct conjunctive hypothesis: bottom, then ``∏(|D_a| + 1)`` others.

    For EnjoySport (domains of sizes 3, 2, 2, 2, 2, 2) this yields 973 hypotheses.
    """
    yield bottom(len(domains))
    yield from product(*[(ANY, *values) for values in domains])


def is_consistent(h: Hypothesis, examples: Iterable[Example]) -> bool:
    """Whether ``h`` classifies every example with its label."""
    return all(covers(h, x) == label for x, label in examples)


def maximally_general(hs: Iterable[Hypothesis]) -> list[Hypothesis]:
    """Members of ``hs`` with no strictly more general member (duplicates removed, order kept)."""
    unique = list(dict.fromkeys(hs))
    return [h for h in unique if not any(more_general(o, h) for o in unique)]


def maximally_specific(hs: Iterable[Hypothesis]) -> list[Hypothesis]:
    """Members of ``hs`` with no strictly more specific member (duplicates removed, order kept)."""
    unique = list(dict.fromkeys(hs))
    return [h for h in unique if not any(more_general(h, o) for o in unique)]


def format_hypothesis(h: Hypothesis, names: Sequence[str] | None = None) -> str:
    """Render ``h`` as ``<Sunny, Warm, ?, …>``; ``names`` renders values as ``name=value``."""
    if names is None:
        return "<" + ", ".join(h) + ">"
    if len(names) != len(h):
        raise ValueError(f"{len(names)} names for a hypothesis of length {len(h)}")
    parts = [c if c in (ANY, NONE) else f"{n}={c}" for n, c in zip(names, h, strict=True)]
    return "<" + ", ".join(parts) + ">"
