"""Tests for row 1.4.3: operator applicability shares."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from terser.phase1.applicability import (
    OVERALL,
    applicability_markdown,
    applicability_table,
    plot_applicability,
)


def _rows(pid: str, category: str, style: str, changed: dict[str, dict[str, bool]]) -> list[dict]:
    return [
        {
            "id": pid,
            "category": category,
            "style": style,
            "op_set": op,
            "target": t,
            "savings": 1.0 if c else 0.0,
            "changed": c,
        }
        for op, per_target in changed.items()
        for t, c in per_target.items()
    ]


@pytest.fixture
def results() -> pd.DataFrame:
    rows = (
        _rows("c1", "code", "texting", {"F": {"a": True, "b": True}, "S": {"a": True, "b": False}})
        + _rows(
            "c2", "code", "hedging", {"F": {"a": False, "b": False}, "S": {"a": False, "b": False}}
        )
        + _rows(
            "f1", "fact", "texting", {"F": {"a": True, "b": True}, "S": {"a": False, "b": False}}
        )
        + _rows("c1", "code", "texting", {"FHC": {"a": True, "b": True}})
    )
    return pd.DataFrame(rows)


def test_shares_by_category(results: pd.DataFrame) -> None:
    table = applicability_table(results, "category", op_codes=("F", "S"))
    assert list(table.index) == ["code", "fact", OVERALL]
    assert table.loc["code", "F"] == pytest.approx(0.5)
    assert table.loc["code", "S"] == pytest.approx(0.5)  # changed on target "a" counts
    assert table.loc["fact", "F"] == pytest.approx(1.0)
    assert table.loc[OVERALL, "F"] == pytest.approx(2 / 3)
    assert table["n"].tolist() == [2, 1, 3]
    assert "FHC" not in table.columns


def test_single_target_restriction(results: pd.DataFrame) -> None:
    table = applicability_table(results, "category", target="b", op_codes=("F", "S"))
    assert table.loc["code", "S"] == 0.0


def test_shares_by_style_and_missing_operator(results: pd.DataFrame) -> None:
    table = applicability_table(results, "style", op_codes=("F", "S", "L"))
    assert table.loc["texting", "F"] == pytest.approx(1.0)
    assert table.loc["hedging", "F"] == 0.0
    assert (table["L"] == 0.0).all()


def test_markdown_and_heatmap(results: pd.DataFrame, tmp_path: Path) -> None:
    by_cat = applicability_table(results, "category", op_codes=("F", "S"))
    by_style = applicability_table(results, "style", op_codes=("F", "S"))
    md = applicability_markdown(by_cat, by_style)
    assert "## 1.4.3" in md and "By style" in md and "| 50 |" in md.replace("  ", " ")
    out = plot_applicability(by_cat, tmp_path / "fig" / "heat.png")
    assert out.stat().st_size > 0
