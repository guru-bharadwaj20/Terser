"""Tests for terser.phase2.judge_bias: runs over fake pilot pairs and the comparison table."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from terser.judge.rubrics import load_rubric
from terser.llm import LLMClient, TransportResult
from terser.phase2.judge_bias import PilotPair, derive_sides, protocol_rows, run_protocol


def make_client(tmp_path: Path, transport: Any) -> LLMClient:
    budget = tmp_path / "budget.yaml"
    budget.write_text(
        yaml.safe_dump(
            {
                "phases": {"phase2": {"max_usd": 0, "max_tokens": 1_000_000}},
                "providers": {"local": {"base_url": "http://127.0.0.1:9/v1", "priced": False}},
            }
        ),
        encoding="utf-8",
    )
    return LLMClient(
        "phase2",
        budget_path=budget,
        transport=transport,
        ledger_path=tmp_path / "ledger.json",
        cache_dir=tmp_path / "cache",
    )


def always_a(provider, model, messages, max_tokens, system, params):  # type: ignore[no-untyped-def]
    """Maximally position-biased pairwise judge; pointwise prefers outputs containing 'good'."""
    body = messages[0]["content"]
    if "[Response A]" in body:
        return TransportResult('{"verdict": "A", "reason": "first"}', 10, 3)
    response = re.search(r"\[Response\]\n(.*?)\n\[End of Response\]", body, re.S)
    assert response is not None
    value = 5 if "good" in response.group(1) else 3
    dims = [d.name for d in load_rubric("fact").dimensions]
    return TransportResult(json.dumps({"scores": dict.fromkeys(dims, value)}), 10, 4)


def pairs() -> list[PilotPair]:
    def pair(i: int, orig_out: str, rew_out: str) -> PilotPair:
        return PilotPair(
            f"fact-{i:03d}:F",
            f"fact-{i:03d}",
            "fact",
            "F",
            f"q{i}",
            f"q{i}!",
            "t",
            orig_out,
            rew_out,
        )

    return [pair(1, "good", "good"), pair(2, "good", "meh"), pair(3, "meh", "good")]


def pilot_records(ps: list[PilotPair]) -> list[dict[str, Any]]:
    sides = [("tie", "tie"), ("original", "rewrite"), ("rewrite", "rewrite")]
    return [
        {
            "id": p.id,
            "category": p.category,
            "judge_model": "j",
            "verdicts": {"original_first": {"side": a}, "rewrite_first": {"side": b}},
        }
        for p, (a, b) in zip(ps, sides, strict=True)
    ]


def test_runs_and_table(tmp_path: Path) -> None:
    client = make_client(tmp_path, always_a)
    ps = pairs()
    stats = run_protocol(
        [*ps, ps[0]],
        protocol="pairwise",
        client=client,
        judge_model="j",
        out_path=tmp_path / "pw.jsonl",
        workers=2,
    )
    # 3 distinct pairs x 2 orders; the identical-output pair repeats its own request.
    assert (stats.calls, stats.cached_calls, stats.generated_tokens) == (6, 1, 15)
    pw = [json.loads(x) for x in (tmp_path / "pw.jsonl").read_text().splitlines()]
    assert len(pw) == 4 and pw[0]["identical_outputs"]
    assert derive_sides(pw[1], "pairwise") == ("original", "rewrite")
    # One-hot fallback: an always-A judge averages to an exact original/rewrite tie -> original.
    assert derive_sides(pw[1], "pairwise_avg") == ("original", "original")

    run_protocol(
        ps,
        protocol="pointwise",
        client=client,
        judge_model="j",
        out_path=tmp_path / "pt.jsonl",
        workers=2,
    )
    pt = [json.loads(x) for x in (tmp_path / "pt.jsonl").read_text().splitlines()]
    assert [derive_sides(r, "pointwise")[0] for r in pt] == ["tie", "original", "rewrite"]

    df = protocol_rows(pilot_records(ps), {"pw": pw[:3], "pt": pt})
    by = {(r["run"], r["protocol"]): r for r in df.to_dict("records")}
    assert by[("pilot_v1", "pairwise")]["position_disagreement"] == pytest.approx(1 / 3)
    assert by[("pw", "pairwise")]["position_disagreement"] == pytest.approx(1.0)
    assert by[("pw", "pairwise")]["first_position_share"] == pytest.approx(1.0)
    assert by[("pt", "pointwise")]["position_disagreement"] == 0.0
    # Reference = pilot pairs 1 (preserved) and 3 (preserved); pointwise agrees on both.
    assert by[("pt", "pointwise")]["ref_n"] == 2
    assert by[("pt", "pointwise")]["ref_agreement"] == pytest.approx(1.0)
    assert by[("pt", "pointwise")]["identical_preserved"] == 1
    with pytest.raises(ValueError):
        run_protocol(
            ps, protocol="x", client=client, judge_model="j", out_path=tmp_path / "x.jsonl"
        )
