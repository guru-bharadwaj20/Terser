"""Per-category judging rubrics (plan row **2.3.2**).

One YAML file per category lives in ``src/terser/judge/rubrics/<category>.yaml``; the annotator
version is ``docs/rubrics.md`` and the category specification is ``docs/categories.md``. Each file
has:

* ``category`` (must match the file name), ``version`` (int >= 1), ``role`` (``tie_break`` when an
  automatic task metric decides first, ``primary`` when the judge is the whole decision) and a
  one-paragraph ``summary``;
* ``dimensions``: a list of ``{name, description, weight, anchors}``; ``anchors`` has exactly the
  keys 1-5, each a non-empty description; weights are positive and sum to 1;
* ``degradation_rules``: ``{id, degradation, description}``; every rubric must decide the shared
  rules in :data:`REQUIRED_RULES` (e.g. ``dropped_tone_cue`` is a degradation for ``creative`` and
  not for ``fact``);
* ``hard_fails``: ``{id, description}``; every rubric includes :data:`HARD_FAIL_CONSTRAINT`
  ("violates an explicit constraint").

:func:`render_rubric` turns a rubric into the plain-text block inserted into the pairwise judge
prompt (:mod:`terser.judge.pairwise`).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final, Literal

import yaml

from terser.data.categories import Category, parse_category

__all__ = [
    "HARD_FAIL_CONSTRAINT",
    "REQUIRED_RULES",
    "RUBRICS_DIR",
    "DegradationRule",
    "Dimension",
    "HardFail",
    "Rubric",
    "RubricError",
    "load_rubric",
    "load_rubrics",
    "render_rubric",
    "validate_rubric",
]

RUBRICS_DIR: Final = Path(__file__).resolve().parent / "rubrics"
"""Directory holding ``<category>.yaml`` rubric files."""

ANCHOR_SCORES: Final[tuple[int, ...]] = (1, 2, 3, 4, 5)

REQUIRED_RULES: Final[tuple[str, ...]] = (
    "dropped_tone_cue",
    "dropped_politeness",
    "dropped_filler",
)
"""Degradation rules every rubric must decide explicitly."""

HARD_FAIL_CONSTRAINT: Final = "violates_explicit_constraint"
"""Hard-fail id every rubric must include."""

Role = Literal["tie_break", "primary"]
_ROLES: Final = ("tie_break", "primary")
_WEIGHT_TOLERANCE: Final = 1e-6


class RubricError(ValueError):
    """A rubric file is missing or malformed."""


@dataclass(frozen=True)
class Dimension:
    """One scored rubric dimension with 1-5 anchors."""

    name: str
    description: str
    weight: float
    anchors: Mapping[int, str]


@dataclass(frozen=True)
class DegradationRule:
    """Whether dropping a kind of prompt content counts as a degradation in this category."""

    id: str
    degradation: bool
    description: str


@dataclass(frozen=True)
class HardFail:
    """A condition that makes an answer lose the comparison regardless of dimension scores."""

    id: str
    description: str


@dataclass(frozen=True)
class Rubric:
    """A validated rubric for one category."""

    category: Category
    version: int
    role: Role
    summary: str
    dimensions: tuple[Dimension, ...]
    degradation_rules: tuple[DegradationRule, ...]
    hard_fails: tuple[HardFail, ...]

    @property
    def dimension_names(self) -> tuple[str, ...]:
        """Dimension names in file order."""
        return tuple(d.name for d in self.dimensions)

    def is_degradation(self, rule_id: str) -> bool:
        """Whether the rule ``rule_id`` counts as a degradation for this category.

        Raises:
            KeyError: If the rubric has no such rule.
        """
        for rule in self.degradation_rules:
            if rule.id == rule_id:
                return rule.degradation
        raise KeyError(f"{self.category.value} rubric has no degradation rule {rule_id!r}")

    def weighted_score(self, scores: Mapping[str, int | float]) -> float:
        """Weighted mean of 1-5 dimension scores (all dimensions required).

        Raises:
            ValueError: If a dimension is missing or unknown, or a score is outside [1, 5].
        """
        names = set(self.dimension_names)
        missing = sorted(names - set(scores))
        unknown = sorted(set(scores) - names)
        if missing or unknown:
            raise ValueError(f"scores missing {missing} / unknown {unknown}")
        total = 0.0
        for dim in self.dimensions:
            value = float(scores[dim.name])
            if not 1.0 <= value <= 5.0:
                raise ValueError(f"score for {dim.name!r} must be in [1, 5], got {value}")
            total += dim.weight * value
        return total


def _require(cond: bool, where: str, message: str) -> None:
    if not cond:
        raise RubricError(f"{where}: {message}")


def _text(obj: Mapping[str, Any], key: str, where: str) -> str:
    value = obj.get(key)
    _require(
        isinstance(value, str) and bool(value.strip()), where, f"{key!r} must be a non-empty string"
    )
    assert isinstance(value, str)
    return " ".join(value.split())


def _dimension(raw: Any, where: str) -> Dimension:
    _require(isinstance(raw, Mapping), where, "dimension must be a mapping")
    name = _text(raw, "name", where)
    where = f"{where} {name!r}"
    description = _text(raw, "description", where)
    weight = raw.get("weight")
    _require(
        isinstance(weight, int | float)
        and not isinstance(weight, bool)
        and math.isfinite(weight)
        and weight > 0,
        where,
        "'weight' must be a positive number",
    )
    anchors = raw.get("anchors")
    _require(isinstance(anchors, Mapping), where, "'anchors' must be a mapping of 1-5")
    assert isinstance(anchors, Mapping)
    keys = set(anchors)
    _require(
        all(isinstance(k, int) and not isinstance(k, bool) for k in keys)
        and keys == set(ANCHOR_SCORES),
        where,
        f"'anchors' must have exactly the keys 1-5, got {sorted(map(str, keys))}",
    )
    parsed: dict[int, str] = {}
    for score in ANCHOR_SCORES:
        text = anchors[score]
        _require(
            isinstance(text, str) and bool(text.strip()), where, f"anchor {score} must be non-empty"
        )
        parsed[score] = " ".join(text.split())
    return Dimension(name, description, float(weight), MappingProxyType(parsed))


def validate_rubric(data: Any, category: Category | str | None = None) -> Rubric:
    """Validate a parsed rubric document and build a :class:`Rubric`.

    Args:
        data: The YAML document (a mapping).
        category: If given, the document's ``category`` must equal it.

    Raises:
        RubricError: Describing the first problem found.
    """
    _require(isinstance(data, Mapping), "rubric", "document must be a mapping")
    assert isinstance(data, Mapping)
    try:
        cat = parse_category(data.get("category"))  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise RubricError(f"rubric: invalid 'category': {exc}") from None
    where = f"{cat.value} rubric"
    if category is not None:
        _require(cat is parse_category(category), where, f"expected category {category!s}")
    version = data.get("version")
    _require(
        isinstance(version, int) and not isinstance(version, bool) and version >= 1,
        where,
        "'version' must be an integer >= 1",
    )
    role = data.get("role")
    _require(role in _ROLES, where, f"'role' must be one of {_ROLES}")
    summary = _text(data, "summary", where)

    raw_dims = data.get("dimensions")
    _require(
        isinstance(raw_dims, list) and bool(raw_dims),
        where,
        "'dimensions' must be a non-empty list",
    )
    assert isinstance(raw_dims, list)
    dims = tuple(_dimension(d, f"{where} dimension #{i}") for i, d in enumerate(raw_dims))
    names = [d.name for d in dims]
    _require(len(set(names)) == len(names), where, f"duplicate dimension names in {names}")
    total = sum(d.weight for d in dims)
    _require(
        abs(total - 1.0) <= _WEIGHT_TOLERANCE, where, f"dimension weights sum to {total}, not 1"
    )

    raw_rules = data.get("degradation_rules")
    _require(isinstance(raw_rules, list), where, "'degradation_rules' must be a list")
    assert isinstance(raw_rules, list)
    rules: list[DegradationRule] = []
    for i, raw in enumerate(raw_rules):
        rw = f"{where} degradation rule #{i}"
        _require(isinstance(raw, Mapping), rw, "must be a mapping")
        flag = raw.get("degradation")
        _require(isinstance(flag, bool), rw, "'degradation' must be true or false")
        rules.append(
            DegradationRule(_text(raw, "id", rw), bool(flag), _text(raw, "description", rw))
        )
    rule_ids = [r.id for r in rules]
    _require(len(set(rule_ids)) == len(rule_ids), where, "duplicate degradation rule ids")
    missing_rules = [r for r in REQUIRED_RULES if r not in rule_ids]
    _require(not missing_rules, where, f"missing required degradation rules {missing_rules}")

    raw_fails = data.get("hard_fails")
    _require(isinstance(raw_fails, list), where, "'hard_fails' must be a list")
    assert isinstance(raw_fails, list)
    fails: list[HardFail] = []
    for i, raw in enumerate(raw_fails):
        fw = f"{where} hard fail #{i}"
        _require(isinstance(raw, Mapping), fw, "must be a mapping")
        fails.append(HardFail(_text(raw, "id", fw), _text(raw, "description", fw)))
    fail_ids = [f.id for f in fails]
    _require(len(set(fail_ids)) == len(fail_ids), where, "duplicate hard-fail ids")
    _require(HARD_FAIL_CONSTRAINT in fail_ids, where, f"missing hard fail {HARD_FAIL_CONSTRAINT!r}")

    return Rubric(
        category=cat,
        version=version,  # type: ignore[arg-type]
        role=role,  # type: ignore[arg-type]
        summary=summary,
        dimensions=dims,
        degradation_rules=tuple(rules),
        hard_fails=tuple(fails),
    )


def _load(category: Category, directory: Path) -> Rubric:
    path = directory / f"{category.value}.yaml"
    if not path.is_file():
        raise RubricError(f"rubric file not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise RubricError(f"{path}: invalid YAML: {exc}") from exc
    return validate_rubric(data, category)


@cache
def _load_default(category: Category) -> Rubric:
    return _load(category, RUBRICS_DIR)


def load_rubric(category: Category | str, directory: Path | None = None) -> Rubric:
    """Load and validate the rubric for ``category`` (cached for the packaged directory)."""
    cat = parse_category(category)
    if directory is None:
        return _load_default(cat)
    return _load(cat, Path(directory))


def load_rubrics(directory: Path | None = None) -> dict[Category, Rubric]:
    """Load every category's rubric, in :class:`Category` order."""
    return {cat: load_rubric(cat, directory) for cat in Category}


def render_rubric(category: Category | str, directory: Path | None = None) -> str:
    """Render a rubric as the plain-text block used in the pairwise judge prompt.

    The block lists the dimensions with their 1-5 anchors and weights, what does and does not
    count as a degradation, and the hard-fail conditions. It contains no ``{prompt}``-style
    placeholders, so it is safe to insert into a judge template.
    """
    rubric = load_rubric(category, directory)
    role = (
        "Use this rubric to break ties; the task's automatic metric is checked separately."
        if rubric.role == "tie_break"
        else "This rubric is the main basis for your verdict."
    )
    lines = [
        f"Category: {rubric.category.value} (rubric v{rubric.version})",
        rubric.summary,
        role,
        "",
        "Dimensions (score each response 1-5; weight in brackets):",
    ]
    for dim in rubric.dimensions:
        lines.append(f"- {dim.name} [{dim.weight:g}]: {dim.description}")
        lines.extend(f"    {score} = {dim.anchors[score]}" for score in ANCHOR_SCORES)
    lines += ["", "Counts as a degradation in this category:"]
    lines += [f"- {r.description}" for r in rubric.degradation_rules if r.degradation]
    lines += ["", "Does NOT count as a degradation in this category:"]
    lines += [f"- {r.description}" for r in rubric.degradation_rules if not r.degradation]
    lines += ["", "Hard fails (a response with a hard fail loses unless both have one):"]
    lines += [f"- {f.description}" for f in rubric.hard_fails]
    return "\n".join(lines)
