"""Tests for terser.judge.rubrics (plan row 2.3.2) and the rubric in the pairwise judge prompt."""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from terser.data.categories import Category
from terser.judge.pairwise import load_prompt_template, render_prompt, rubric_section
from terser.judge.rubrics import (
    HARD_FAIL_CONSTRAINT,
    REQUIRED_RULES,
    RUBRICS_DIR,
    RubricError,
    load_rubric,
    load_rubrics,
    render_rubric,
    validate_rubric,
)


def raw(category: str) -> dict[str, Any]:
    data = yaml.safe_load((RUBRICS_DIR / f"{category}.yaml").read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def test_one_rubric_file_per_category() -> None:
    files = sorted(p.stem for p in RUBRICS_DIR.glob("*.yaml"))
    assert files == sorted(c.value for c in Category)
    rubrics = load_rubrics()
    assert list(rubrics) == list(Category)


@pytest.mark.parametrize("category", list(Category))
def test_rubric_is_valid(category: Category) -> None:
    rubric = load_rubric(category)
    assert rubric.category is category and rubric.version >= 1
    assert sum(d.weight for d in rubric.dimensions) == pytest.approx(1.0)
    for dim in rubric.dimensions:
        assert sorted(dim.anchors) == [1, 2, 3, 4, 5]
        assert all(text for text in dim.anchors.values())
    assert HARD_FAIL_CONSTRAINT in {f.id for f in rubric.hard_fails}
    for rule in REQUIRED_RULES:
        rubric.is_degradation(rule)
    assert any(r.degradation for r in rubric.degradation_rules)
    # Automatic metric decides first for code/fact/reason; the judge decides the others.
    expected_role = "tie_break" if category in ("code", "fact", "reason") else "primary"
    assert rubric.role == expected_role


def test_tone_cue_is_a_degradation_only_for_creative() -> None:
    assert load_rubric("creative").is_degradation("dropped_tone_cue") is True
    for cat in Category:
        if cat is not Category.CREATIVE:
            assert load_rubric(cat).is_degradation("dropped_tone_cue") is False, cat
        assert load_rubric(cat).is_degradation("dropped_politeness") is False
        assert load_rubric(cat).is_degradation("dropped_filler") is False
    with pytest.raises(KeyError):
        load_rubric("fact").is_degradation("no_such_rule")


def test_category_specific_degradations() -> None:
    assert load_rubric("explain").is_degradation("dropped_audience_cue")
    assert load_rubric("transform").is_degradation("altered_payload")
    assert load_rubric("fact").is_degradation("dropped_scope_qualifier")
    assert load_rubric("reason").is_degradation("changed_quantity")
    assert load_rubric("code").is_degradation("changed_signature_or_language")


def test_dimensions_match_category_spec() -> None:
    assert load_rubric("explain").dimension_names == (
        "coverage",
        "correctness",
        "depth",
        "constraint_adherence",
        "audience_fit",
    )
    assert "tone_adherence" in load_rubric("creative").dimension_names
    assert "constraint_adherence" in load_rubric("transform").dimension_names


def test_weighted_score() -> None:
    rubric = load_rubric("fact")
    assert rubric.weighted_score({n: 5 for n in rubric.dimension_names}) == pytest.approx(5.0)
    assert rubric.weighted_score(
        {"correctness": 1, "completeness": 5, "concision": 5}
    ) == pytest.approx(0.55 + 0.45 * 5)
    with pytest.raises(ValueError):
        rubric.weighted_score({"correctness": 5})
    with pytest.raises(ValueError):
        rubric.weighted_score({"correctness": 6, "completeness": 5, "concision": 5})


def _broken(mutate: Any) -> dict[str, Any]:
    data = copy.deepcopy(raw("creative"))
    mutate(data)
    return data


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d.update(category="poem"), "invalid 'category'"),
        (lambda d: d.update(version=0), "version"),
        (lambda d: d.update(role="boss"), "role"),
        (lambda d: d.update(dimensions=[]), "dimensions"),
        (lambda d: d["dimensions"][0].update(weight=0.9), "weights sum"),
        (lambda d: d["dimensions"][0].update(weight=-1), "weight"),
        (lambda d: d["dimensions"][0]["anchors"].pop(3), "exactly the keys 1-5"),
        (lambda d: d["dimensions"][0]["anchors"].update({6: "extra"}), "exactly the keys 1-5"),
        (lambda d: d["dimensions"][0]["anchors"].update({2: "  "}), "anchor 2"),
        (lambda d: d["dimensions"][1].update(name=d["dimensions"][0]["name"]), "duplicate"),
        (lambda d: d["dimensions"][0].pop("description"), "description"),
        (lambda d: d["degradation_rules"][0].update(degradation="yes"), "true or false"),
        (
            lambda d: d.update(
                degradation_rules=[r for r in d["degradation_rules"] if r["id"] != "dropped_filler"]
            ),
            "dropped_filler",
        ),
        (
            lambda d: d.update(
                hard_fails=[f for f in d["hard_fails"] if f["id"] != HARD_FAIL_CONSTRAINT]
            ),
            HARD_FAIL_CONSTRAINT,
        ),
    ],
)
def test_validation_rejects_malformed(mutate: Any, message: str) -> None:
    with pytest.raises(RubricError, match=re.escape(message)):
        validate_rubric(_broken(mutate))


def test_validation_checks_expected_category_and_files(tmp_path: Path) -> None:
    with pytest.raises(RubricError, match="expected category"):
        validate_rubric(raw("fact"), "code")
    with pytest.raises(RubricError, match="not found"):
        load_rubric("fact", tmp_path)
    (tmp_path / "fact.yaml").write_text("category: [unclosed", encoding="utf-8")
    with pytest.raises(RubricError, match="invalid YAML"):
        load_rubric("fact", tmp_path)
    (tmp_path / "fact.yaml").write_text(yaml.safe_dump(raw("fact")), encoding="utf-8")
    assert load_rubric("fact", tmp_path) == load_rubric("fact")


@pytest.mark.parametrize("category", list(Category))
def test_render_rubric(category: Category) -> None:
    text = render_rubric(category)
    rubric = load_rubric(category)
    assert text.startswith(f"Category: {category.value} (rubric v{rubric.version})")
    for dim in rubric.dimensions:
        assert f"- {dim.name} [{dim.weight:g}]" in text
        assert f"    5 = {dim.anchors[5]}" in text
    assert "Counts as a degradation" in text and "Does NOT count" in text
    assert "violates an explicit constraint" in text
    assert "{" not in text.replace("{}", "")  # no template placeholders


def test_render_rubric_tone_rule_placement() -> None:
    creative = render_rubric("creative")
    degrading, not_degrading = creative.split("Does NOT count as a degradation")
    assert "tone, mood or style cue" in degrading
    fact = render_rubric("fact")
    degrading, not_degrading = fact.split("Does NOT count as a degradation")
    assert "Tone or mood wording" in not_degrading and "Tone or mood" not in degrading


def test_judge_prompt_includes_category_rubric() -> None:
    template = load_prompt_template()
    plain = render_prompt(template, "P", "A", "B")
    assert "{rubric}" not in plain and "[Rubric" not in plain
    assert "\n\n[User request]\nP\n" in plain  # unchanged layout without a category
    with_rubric = render_prompt(template, "P", "A", "B", rubric_section("transform"))
    body = re.search(r"\[Rubric for this request\]\n(.*?)\n\[End of rubric\]", with_rubric, re.S)
    assert body is not None and render_rubric("transform") in body.group(1)
    assert with_rubric.index("[End of rubric]") < with_rubric.index("[User request]")
    assert rubric_section(None) == ""
    with pytest.raises(ValueError, match="placeholder"):
        render_prompt("no placeholder {prompt}", "P", "A", "B", rubric_section("fact"))
