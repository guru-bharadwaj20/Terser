"""Tests for the pandera pair-table schema (row 11.1.4): it must fail on a corrupted fixture."""

from __future__ import annotations

from pathlib import Path

import pytest

pa = pytest.importorskip("pandera")
pd = pytest.importorskip("pandas")

from pandera.errors import SchemaErrors  # noqa: E402

from terser.data.validation import main, validate_pairs  # noqa: E402

FIX = Path(__file__).parent / "fixtures"


def _load(name: str) -> pd.DataFrame:
    return pd.read_json(FIX / name, lines=True)


def test_valid_fixture_passes() -> None:
    df = _load("pairs_valid.jsonl")
    assert len(validate_pairs(df, require_all=True)) == 12
    assert main([str(FIX / "pairs_valid.jsonl"), "--require-all"]) == 0


def test_corrupted_fixture_fails_with_every_fault() -> None:
    with pytest.raises(SchemaErrors) as exc:
        validate_pairs(_load("pairs_corrupted.jsonl"))
    failed = set(exc.value.failure_cases["check"].astype(str))
    assert any("not_nullable" in c for c in failed)  # null prompt
    assert any("in_range" in c for c in failed)  # ratio 3.0
    assert any("isin" in c for c in failed)  # bad category / label
    assert "token_ratio_matches_counts" in failed
    assert "category_balance" in failed
    assert main([str(FIX / "pairs_corrupted.jsonl")]) == 1


@pytest.mark.parametrize(
    ("column", "value"),
    [("tokens_before", 0), ("tokens_after", -1), ("category", "poetry"), ("label", "maybe")],
)
def test_single_corruption_detected(column: str, value: object) -> None:
    df = _load("pairs_valid.jsonl")
    df.loc[0, column] = value
    if column.startswith("tokens"):
        df["token_ratio"] = df["tokens_after"] / df["tokens_before"].clip(lower=1)
    with pytest.raises(SchemaErrors):
        validate_pairs(df)


def test_missing_category_with_require_all() -> None:
    df = _load("pairs_valid.jsonl")
    df = df[df["category"] != "creative"]
    validate_pairs(df)  # balance among present categories is fine
    with pytest.raises(SchemaErrors):
        validate_pairs(df, require_all=True)
