"""Recompute the worked example in docs/derivations/bayes.md (row 5.2.1) from the pilot corpus."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from terser.phase5.bayes_counts import bayes_table, count_occurrences, main

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "data" / "pilot" / "prompts.jsonl"


def _texts() -> list[str]:
    lines = PROMPTS.read_text(encoding="utf-8").splitlines()
    return [json.loads(x)["prompt"] for x in lines if x.strip()]


@pytest.fixture(scope="module")
def table() -> dict[tuple[str, str], object]:
    return {(r.phrase, r.context): r for r in bayes_table(_texts())}


def test_corpus_totals() -> None:
    _, total, n_f = count_occurrences(_texts())
    assert (total, n_f) == (18519, 220)


def test_worked_case(table: dict) -> None:
    r = table[("so", "initial")]
    assert (r.n_phrase_ctx, r.n_phrase_ctx_filler) == (78, 43)
    assert r.posterior == pytest.approx(43 / 78) == pytest.approx(r.direct)
    assert r.posterior == pytest.approx(0.551, abs=5e-4)
    m = table[("so", "medial")]
    assert (m.n_phrase_ctx, m.n_phrase_ctx_filler) == (135, 25)
    assert table[("like", "medial")].n_phrase_ctx_filler == 0


def test_bayes_identity_everywhere(table: dict) -> None:
    for r in table.values():
        assert r.posterior == pytest.approx(r.direct)


def test_toy_corpus_and_cli(capsys: pytest.CaptureFixture[str]) -> None:
    rows = bayes_table(["um, write a sort. So, basically list it", "so many rows fail"])
    got = {(r.phrase, r.context): r.direct for r in rows}
    assert got[("um", "initial")] == 1.0 and got[("so", "initial")] == 0.5
    with pytest.raises(ValueError):
        bayes_table(["write code"])
    assert main(["--prompts", str(PROMPTS)]) == 0
    assert "P(filler)=0.01188" in capsys.readouterr().out


def test_doc_quotes_the_computed_numbers(table: dict) -> None:
    text = (ROOT / "docs" / "derivations" / "bayes.md").read_text(encoding="utf-8")
    _, total, n_f = count_occurrences(_texts())
    assert f"$N = {total:,}$".replace(",", "{,}") in text
    assert f"$N_F = {n_f}$" in text
    assert f"{n_f / total:.5f}" in text
    r = table[("so", "initial")]
    assert f"({r.n_phrase_ctx_filler}/{n_f})" in text and f"{r.n_phrase_ctx}/" in text
