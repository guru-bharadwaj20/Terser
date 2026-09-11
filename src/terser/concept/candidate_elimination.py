"""Candidate-Elimination with version-space collapse diagnosis.

The version space ``VS(H, D)`` is the set of conjunctive hypotheses consistent with the
examples ``D``. Candidate-Elimination represents it by two boundaries: ``S`` (maximally
specific members) and ``G`` (maximally general members). For each example:

* **positive** ``x``: drop every ``g`` in ``G`` that does not cover ``x``; replace every ``s``
  in ``S`` that does not cover ``x`` by its minimal generalisation, kept only if some ``g``
  in ``G`` is more general or equal; drop members of ``S`` more general than another member.
* **negative** ``x``: drop every ``s`` in ``S`` that covers ``x``; replace every ``g`` in ``G``
  that covers ``x`` by its minimal specialisations excluding ``x``, kept only if more general
  or equal to some ``s`` in ``S``; drop members of ``G`` more specific than another member.

For conjunctions ``S`` always has at most one member. On consistent data the final ``S`` and
``G`` do not depend on example order (they are the boundaries of a set that does not).

**Collapse.** If ``S`` or ``G`` becomes empty at example ``i``, no conjunction fits the
examples seen so far, but one did before ``i``. :func:`minimal_conflict` then names an
inclusion-minimal inconsistent subset. It relies on a characterisation: a set of examples
with at least one positive is consistent iff ``find_s`` of its positives covers none of its
negatives (``find_s`` is the most specific covering hypothesis, so any consistent hypothesis
is at least as general and would cover that negative too). With no positives, bottom is
consistent. So an inconsistent set contains a negative ``n`` covered by ``find_s(P)``.
Starting from ``P`` = all positives seen, a deletion filter removes each positive in turn
whenever ``find_s`` of the rest still covers ``n``. Coverage is monotone in ``P`` (more
positives only generalise ``find_s``), so the survivors ``P'`` make ``P' ∪ {n}`` inconsistent
while every proper subset is consistent. Example ``i`` is always in the set, because the
examples before it were consistent. When ``i`` is positive, every earlier negative covered by
``find_s(P)`` is tried as ``n`` and the smallest set is reported (ties: earliest negative).
The set is minimal under inclusion, not necessarily of minimum size.

**Boundary growth.** When many negatives arrive before any positive, ``S`` is bottom and
constrains nothing, so each negative can multiply ``|G|`` by up to ``Σ_a (|D_a| − 1)``.
``max_boundary_size`` turns that into a clear :class:`BoundaryTooLargeError`. Passing
``positives_first=True`` processes positives first (stable within each label); on consistent
data the result is identical, and each specialisation must then agree with the concrete ``S``,
which keeps ``G`` small. Reported indices always refer to the caller's original order.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from terser.concept.find_s import find_s
from terser.concept.hypothesis import (
    Domains,
    Example,
    Hypothesis,
    bottom,
    covers,
    format_hypothesis,
    maximally_general,
    maximally_specific,
    min_generalise,
    min_specialisations,
    more_general_or_equal,
    top,
    validate_domains,
    validate_instance,
)

DEFAULT_MAX_BOUNDARY_SIZE = 5000
"""Default cap on candidate ``G`` (or ``S``) members before pruning."""


class BoundaryTooLargeError(RuntimeError):
    """Raised when a boundary set grows past ``max_boundary_size``."""


@dataclass(frozen=True)
class Collapse:
    """Where and why the version space became empty.

    ``index`` is the offending example; ``conflicting`` lists the other examples of an
    inclusion-minimal inconsistent subset containing it (sorted, original indices);
    ``negative`` is the negative example in that subset and ``witness`` is ``find_s`` of the
    subset's positives, the most specific hypothesis they force, which covers ``negative``.
    """

    index: int
    conflicting: tuple[int, ...]
    negative: int
    witness: Hypothesis

    @property
    def examples(self) -> tuple[int, ...]:
        """The whole minimal inconsistent subset, sorted."""
        return tuple(sorted((self.index, *self.conflicting)))

    def describe(self, names: Sequence[str] | None = None) -> str:
        """One-line human-readable diagnosis."""
        return (
            f"version space collapsed at example {self.index}; minimal conflicting set "
            f"{list(self.examples)}: its positives force {format_hypothesis(self.witness, names)}, "
            f"which covers negative example {self.negative}"
        )


@dataclass(frozen=True)
class CandidateEliminationResult:
    """Boundaries after the run (or at the collapse step) plus diagnostics.

    ``trace[k]`` holds ``(S, G)`` after the ``k``-th processed example (only if requested);
    ``order`` gives the original index of each processed example.
    """

    S: tuple[Hypothesis, ...]
    G: tuple[Hypothesis, ...]
    collapse: Collapse | None = None
    order: tuple[int, ...] = ()
    trace: tuple[tuple[tuple[Hypothesis, ...], tuple[Hypothesis, ...]], ...] = ()

    @property
    def collapsed(self) -> bool:
        """Whether no conjunctive hypothesis is consistent with the examples."""
        return self.collapse is not None


def _check_size(size: int, limit: int | None, which: str, index: int) -> None:
    if limit is not None and size > limit:
        raise BoundaryTooLargeError(
            f"{which} boundary reached {size} candidates (limit {limit}) at example {index}. "
            "This happens when many negatives precede the positives; retry with "
            "positives_first=True (same result on consistent data) or raise max_boundary_size."
        )


def minimal_conflict(
    examples: Sequence[Example], index: int, candidates: Sequence[int] | None = None
) -> Collapse:
    """Name an inclusion-minimal inconsistent subset containing ``examples[index]``.

    ``candidates`` are the indices considered (default: ``0..index``); the examples among
    them other than ``index`` must be consistent, and adding ``index`` must make them
    inconsistent. See the module docstring for the algorithm.
    """
    pool = list(range(index + 1)) if candidates is None else list(candidates)
    if index not in pool:
        pool.append(index)
    n_attrs = len(examples[index][0])
    positives = [j for j in pool if examples[j][1]]
    witness_all = find_s([examples[j] for j in positives], n_attrs)
    if examples[index][1]:
        negatives = [j for j in pool if not examples[j][1] and covers(witness_all, examples[j][0])]
    else:
        negatives = [index] if covers(witness_all, examples[index][0]) else []
    if not negatives:
        raise ValueError(f"examples {sorted(pool)} are consistent; nothing conflicts")

    best: Collapse | None = None
    for n in negatives:
        kept = list(positives)
        for p in positives:
            rest = [q for q in kept if q != p]
            if covers(find_s([examples[q] for q in rest], n_attrs), examples[n][0]):
                kept = rest
        members = sorted({*kept, n} - {index})
        witness = find_s([examples[q] for q in kept], n_attrs)
        found = Collapse(index=index, conflicting=tuple(members), negative=n, witness=witness)
        if best is None or len(found.conflicting) < len(best.conflicting):
            best = found
    assert best is not None
    return best


def candidate_elimination(
    examples: Sequence[Example],
    domains: Domains,
    *,
    positives_first: bool = False,
    max_boundary_size: int | None = DEFAULT_MAX_BOUNDARY_SIZE,
    trace: bool = False,
) -> CandidateEliminationResult:
    """Run Candidate-Elimination over ``examples`` and return the ``S``/``G`` boundaries.

    Processing stops at the first example that empties the version space; the result then
    carries a :class:`Collapse`. Raises :class:`BoundaryTooLargeError` if a boundary exceeds
    ``max_boundary_size`` candidates (``None`` disables the guard).
    """
    validate_domains(domains)
    for x, _ in examples:
        validate_instance(x, domains)
    n = len(domains)
    order = list(range(len(examples)))
    if positives_first:
        order.sort(key=lambda j: not examples[j][1])

    S: list[Hypothesis] = [bottom(n)]
    G: list[Hypothesis] = [top(n)]
    steps: list[tuple[tuple[Hypothesis, ...], tuple[Hypothesis, ...]]] = []
    for k, i in enumerate(order):
        x, label = examples[i]
        if label:
            G = [g for g in G if covers(g, x)]
            new_s: list[Hypothesis] = []
            for s in S:
                h = s if covers(s, x) else min_generalise(s, x)
                if any(more_general_or_equal(g, h) for g in G):
                    new_s.append(h)
            _check_size(len(new_s), max_boundary_size, "S", i)
            S = maximally_specific(new_s)
        else:
            S = [s for s in S if not covers(s, x)]
            new_g: list[Hypothesis] = []
            for g in G:
                for h in min_specialisations(g, x, domains):
                    if any(more_general_or_equal(h, s) for s in S):
                        new_g.append(h)
            new_g = list(dict.fromkeys(new_g))
            _check_size(len(new_g), max_boundary_size, "G", i)
            G = maximally_general(new_g)
        if trace:
            steps.append((tuple(S), tuple(G)))
        if not S or not G:
            collapse = minimal_conflict(examples, i, candidates=order[: k + 1])
            return CandidateEliminationResult(
                S=tuple(S), G=tuple(G), collapse=collapse, order=tuple(order), trace=tuple(steps)
            )
    return CandidateEliminationResult(
        S=tuple(S), G=tuple(G), order=tuple(order), trace=tuple(steps)
    )
