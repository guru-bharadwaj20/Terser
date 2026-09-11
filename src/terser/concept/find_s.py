"""Find-S: the most specific conjunctive hypothesis covering every positive example.

Find-S starts from bottom ``<∅, …, ∅>`` and, for each positive example it does not yet
cover, replaces the hypothesis with its unique minimal generalisation. Negative examples are
ignored. If the target concept is a conjunction and the labels are noise-free, the result is
consistent with the negatives too; :func:`covered_negatives` checks that assumption and names
the negatives that break it.
"""

from __future__ import annotations

from collections.abc import Sequence

from terser.concept.hypothesis import Example, Hypothesis, bottom, covers, min_generalise


def find_s(examples: Sequence[Example], n_attributes: int | None = None) -> Hypothesis:
    """Return the most specific hypothesis covering all positives in ``examples``.

    ``n_attributes`` is inferred from the first example when omitted; it is required only
    when ``examples`` is empty. With no positives the result is bottom. The result does not
    depend on the order of the examples.
    """
    if n_attributes is None:
        if not examples:
            raise ValueError("n_attributes is required when there are no examples")
        n_attributes = len(examples[0][0])
    h = bottom(n_attributes)
    for x, label in examples:
        if len(x) != n_attributes:
            raise ValueError(f"instance has {len(x)} values, expected {n_attributes}: {x}")
        if label and not covers(h, x):
            h = min_generalise(h, x)
    return h


def covered_negatives(h: Hypothesis, examples: Sequence[Example]) -> list[int]:
    """Indices of negative examples wrongly covered by ``h`` (empty iff ``h`` rejects them all)."""
    return [i for i, (x, label) in enumerate(examples) if not label and covers(h, x)]
