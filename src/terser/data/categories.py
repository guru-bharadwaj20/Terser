"""The six Terser task categories and the quality measure attached to each.

Plan row **1.1.2**. The human-readable specification (definitions, inclusion/exclusion rules,
borderline examples, rubric dimensions, what counts as a degradation, and the tie-break rule for
prompts that fit two categories) lives in ``docs/categories.md``; this module is the code-side
source of truth for the category names and their metrics.

Every category's ``preserved``/``degraded`` label (row 2.3.4) also requires the position-swapped
pairwise judge to say tie-or-rewrite-better in both orders. ``CATEGORY_METRICS`` lists the
category-specific measure applied *in addition to* (or, for ``creative``, *as*) that judge.
"""

from __future__ import annotations

import difflib
from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType

__all__ = [
    "CATEGORY_METRICS",
    "Category",
    "parse_category",
]


class Category(StrEnum):
    """A task category. Members compare equal to, and format as, their lowercase string value."""

    CODE = "code"
    FACT = "fact"
    REASON = "reason"
    EXPLAIN = "explain"
    CREATIVE = "creative"
    TRANSFORM = "transform"


CATEGORY_METRICS: Mapping[Category, tuple[str, ...]] = MappingProxyType(
    {
        Category.CODE: ("unit_test_pass_rate",),
        Category.FACT: ("exact_match", "token_f1"),
        Category.REASON: ("final_answer_accuracy",),
        Category.EXPLAIN: ("rubric_judge",),
        Category.CREATIVE: ("pairwise_judge",),
        Category.TRANSFORM: ("rubric_judge", "constraint_checklist"),
    }
)
"""Quality measure(s) per category, in priority order (see ``docs/categories.md`` section 2)."""


def parse_category(value: str | Category) -> Category:
    """Parse a category name, tolerating surrounding whitespace and case.

    Args:
        value: A :class:`Category` or a string such as ``"code"`` or ``" Fact "``.

    Returns:
        The matching :class:`Category`.

    Raises:
        TypeError: If ``value`` is not a string.
        ValueError: If ``value`` names no category. The message lists the valid names and, when
            one is close, suggests it.
    """
    if isinstance(value, Category):
        return value
    if not isinstance(value, str):
        raise TypeError(f"category must be a str, got {type(value).__name__}")
    key = value.strip().lower()
    try:
        return Category(key)
    except ValueError:
        valid = [c.value for c in Category]
        close = difflib.get_close_matches(key, valid, n=1)
        hint = f" Did you mean {close[0]!r}?" if close else ""
        raise ValueError(
            f"unknown category {value!r}; expected one of {', '.join(valid)}.{hint}"
        ) from None
