"""Tests for the labelled-prompt subset (rows 2.2.1-2.4.2)."""

from __future__ import annotations

import pytest

from terser.data.label_subset import label_subset


def _recs(split: str, cat: str, groups: int, variants: int) -> list[dict[str, str]]:
    return [
        {"id": f"{split}:{cat}:{g}:{v}", "source_id": f"{split}:{cat}:{g}", "category": cat}
        for g in range(groups)
        for v in range(variants)
    ]


SPLITS = {
    "train": _recs("train", "code", 7, 3) + _recs("train", "fact", 7, 3),
    "val": _recs("val", "code", 2, 3) + _recs("val", "fact", 2, 3),
    "test": _recs("test", "code", 1, 3) + _recs("test", "fact", 1, 3),
}


def test_per_category_and_split_tag() -> None:
    rows = label_subset(SPLITS, 5)
    assert [r["category"] for r in rows].count("code") == 5
    assert [r["category"] for r in rows].count("fact") == 5
    assert all(r["split"] == r["id"].split(":")[0] for r in rows)


def test_one_variant_per_group_before_seconds() -> None:
    rows = [r for r in label_subset(SPLITS, 10) if r["category"] == "code"]
    assert len({r["source_id"] for r in rows}) == 10  # 10 groups exist, so all distinct
    rows = [r for r in label_subset(SPLITS, 12) if r["category"] == "code"]
    assert len({r["source_id"] for r in rows}) == 10


def test_prefix_property_and_determinism() -> None:
    small = [r["id"] for r in label_subset(SPLITS, 4, seed=1) if r["category"] == "fact"]
    big = [r["id"] for r in label_subset(SPLITS, 15, seed=1) if r["category"] == "fact"]
    assert big[:4] == small
    assert label_subset(SPLITS, 4, seed=1) == label_subset(SPLITS, 4, seed=1)
    assert label_subset(SPLITS, 4, seed=1) != label_subset(SPLITS, 4, seed=2)


def test_rejects_non_positive() -> None:
    with pytest.raises(ValueError):
        label_subset(SPLITS, 0)
