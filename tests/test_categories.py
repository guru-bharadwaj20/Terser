"""Tests for terser.data.categories (plan row 1.1.2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from terser.data.categories import CATEGORY_METRICS, Category, parse_category

DOC = Path(__file__).resolve().parents[1] / "docs" / "categories.md"


def test_exactly_six_categories_in_plan_order() -> None:
    assert [c.value for c in Category] == [
        "code",
        "fact",
        "reason",
        "explain",
        "creative",
        "transform",
    ]


def test_category_behaves_like_str() -> None:
    assert Category.CODE == "code"
    assert str(Category.FACT) == "fact"
    assert f"{Category.REASON}" == "reason"
    assert isinstance(Category.EXPLAIN, str)


def test_every_category_has_metrics() -> None:
    assert set(CATEGORY_METRICS) == set(Category)
    for metrics in CATEGORY_METRICS.values():
        assert metrics and all(isinstance(m, str) and m for m in metrics)


def test_metrics_match_plan() -> None:
    assert CATEGORY_METRICS[Category.CODE] == ("unit_test_pass_rate",)
    assert CATEGORY_METRICS[Category.FACT] == ("exact_match", "token_f1")
    assert CATEGORY_METRICS[Category.REASON] == ("final_answer_accuracy",)
    assert CATEGORY_METRICS[Category.EXPLAIN] == ("rubric_judge",)
    assert CATEGORY_METRICS[Category.CREATIVE] == ("pairwise_judge",)
    assert CATEGORY_METRICS[Category.TRANSFORM] == ("rubric_judge", "constraint_checklist")


def test_metrics_mapping_is_read_only() -> None:
    with pytest.raises(TypeError):
        CATEGORY_METRICS[Category.CODE] = ("x",)  # type: ignore[index]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("code", Category.CODE),
        ("  Fact ", Category.FACT),
        ("REASON", Category.REASON),
        ("explain\n", Category.EXPLAIN),
        (Category.CREATIVE, Category.CREATIVE),
        ("transform", Category.TRANSFORM),
    ],
)
def test_parse_category_valid(raw: str, expected: Category) -> None:
    assert parse_category(raw) is expected


def test_parse_category_suggests_close_match() -> None:
    with pytest.raises(ValueError, match=r"Did you mean 'creative'\?") as exc:
        parse_category("creatve")
    assert "code, fact, reason, explain, creative, transform" in str(exc.value)


def test_parse_category_unknown_without_suggestion() -> None:
    with pytest.raises(ValueError, match="unknown category 'zzz'") as exc:
        parse_category("zzz")
    assert "Did you mean" not in str(exc.value)


@pytest.mark.parametrize("raw", ["", "   "])
def test_parse_category_empty(raw: str) -> None:
    with pytest.raises(ValueError, match="unknown category"):
        parse_category(raw)


def test_parse_category_wrong_type() -> None:
    with pytest.raises(TypeError, match="must be a str"):
        parse_category(3)  # type: ignore[arg-type]


def test_spec_document_covers_every_category() -> None:
    text = DOC.read_text(encoding="utf-8")
    for category in Category:
        assert f"`{category.value}`" in text
