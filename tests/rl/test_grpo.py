"""Stretch S-9.3: GRPO pieces, the tiny CPU smoke run, the audit and the RL vs SFT report."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from terser.models.rl.audit import audit_table, parse_verdict, pick, run_audit
from terser.models.rl.grpo import (
    degenerate_kind,
    degenerate_rates,
    group_advantages,
    is_natural_language,
    make_reward,
)
from terser.models.rl.report import build, gate_precision, wilson


def test_group_advantages_standardise_within_groups() -> None:
    adv = group_advantages([0, 0, 10, 10, 5, 5, 5, 5], 4)
    assert adv[:4] == pytest.approx([-1, -1, 1, 1], abs=1e-3)
    assert adv[4:] == [0.0] * 4
    with pytest.raises(ValueError):
        group_advantages([1, 2, 3], 2)


def test_degenerate_monitor() -> None:
    assert degenerate_kind("  ", False) == "empty"
    assert degenerate_kind("Write a sort function.", True) == "truncated"
    assert degenerate_kind("}}{{ ### 1234 5678 !!", False) == "non_nl"
    assert degenerate_kind("the cat " * 20, False) == "non_nl"  # runaway loop
    assert degenerate_kind("Write a Python function that reverses a list.", False) is None
    assert is_natural_language("def f(x):\n    return sorted(x)  # sort the list")
    r = degenerate_rates(["empty", None, "non_nl", None])
    assert r == {"empty": 0.25, "truncated": 0.0, "non_nl": 0.25, "any": 0.5}


def test_reward_is_savings_times_gate_pass() -> None:
    count = lambda t: len(t.split())  # noqa: E731
    reward = make_reward(count, lambda rows: [0.1 if "ok" in r["rewrite"] else 0.9 for r in rows],
                         threshold=0.5)  # fmt: skip
    p = "please could you kindly sort this list ok"
    out = reward([p, p, p], ["sort list ok", "sort list", p + " and more"])
    assert out[0]["reward"] == pytest.approx(100 * (1 - 3 / 8)) and out[0]["gate_pass"]
    assert out[1]["reward"] == 0.0 and not out[1]["gate_pass"] and out[1]["savings_pct"] > 0
    assert out[2]["fell_back"] and out[2]["reward"] == 0.0 and not out[2]["gate_pass"]


def test_tiny_grpo_smoke(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    from terser.config import load_config
    from terser.models.rl.grpo import run

    cfg = load_config(
        ["experiment=grpo_tiny", "hardware=cpu", f"experiment.params.output_dir={tmp_path}"]
    )
    summary = run(cfg)
    assert summary["steps_done"] == 3
    rows = [json.loads(x) for x in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows] == [1, 2, 3]
    for key in ("reward_mean", "gate_pass_rate", "kl", "degenerate_any", "degenerate_truncated"):
        assert key in rows[0]
    assert rows[0]["kl"] == pytest.approx(0.0, abs=1e-6)  # policy == reference before step 1
    audits = sorted(p.name for p in (tmp_path / "audit").glob("*.jsonl"))
    assert audits == ["step_00001.jsonl", "step_00002.jsonl"]
    s = json.loads((tmp_path / "audit" / "step_00002.jsonl").read_text().splitlines()[0])
    assert {"prompt", "rewrite", "reward", "gate_pass", "degenerate"} <= set(s)


def test_audit_parse_pick_and_table() -> None:
    assert parse_verdict('{"preserved": true, "reason": "ok"}') is True
    assert parse_verdict('Sure. {"preserved": false}') is False
    assert parse_verdict("no idea") is None
    samples = [
        {"step": 5, "prompt": "p", "rewrite": f"r{i}", "gate_pass": i < 2, "fell_back": i == 4,
         "reward": float(i < 2), "degenerate": None}  # fmt: skip
        for i in range(6)
    ]
    chosen = pick(samples, 3)
    assert len(chosen) == 3 and all(c["gate_pass"] for c in chosen[:2])
    judged = run_audit({5: samples}, lambda p, r: f'{{"preserved": {str(r == "r0").lower()}}}', 4)
    assert len(judged) == 4 and not any(j["fell_back"] for j in judged)
    table = audit_table(judged, {5: samples})
    assert "| 5 | 6 |" in table and "50.0% (2)" in table


def _rows(tag: str, n: int, fell_back: int) -> tuple[list[dict], list[dict]]:
    rows, labels = [], []
    for i in range(n):
        rows.append({"id": f"p{i}", "category": "code" if i % 2 else "fact", "prompt": "a b c d",
                     "rewrite": "a b", "raw_rewrite": "a b", "fell_back": i < fell_back,
                     "savings": 0.5, "oracle_savings": 0.4, "latency_s": 0.1})  # fmt: skip
        for t in ("qwen", "phi"):
            labels.append({"candidate_id": f"p{i}:{tag}", "target": t,
                           "label": "preserved" if i % 3 else "degraded"})  # fmt: skip
    return rows, labels


def test_gate_precision_and_report() -> None:
    assert wilson(0, 0) == (0.0, 1.0)
    lo, hi = wilson(9, 10)
    assert lo < 0.9 < hi
    rows, labels = _rows("rl", 6, 0)
    g = gate_precision(rows, [0.1, 0.9, 0.1, 0.1, 0.1, 0.1], labels, "rl", 0.5)
    # passed: p0 (degraded), p2, p3 (degraded), p4, p5 -> 2 targets each
    assert g["gate_passed"] == 5 and g["pairs"] == 10 and g["precision"] == pytest.approx(0.6)
    sft_rows, sft_labels = _rows("sft", 6, 2)
    md, res = build(
        {"sft": sft_rows, "rl": rows}, labels + sft_labels, "rl", "sft",
        lambda rs: [0.1] * len(rs), 0.5, p2_precision=0.62,
    )  # fmt: skip
    assert res["c9_3"] and "C9.3: pass" in md and "| rl | 6 |" in md
    _, res2 = build({"sft": sft_rows, "rl": rows}, labels + sft_labels, "rl", "sft",
                    lambda rs: [0.1] * len(rs), 0.5, p2_precision=0.91)  # fmt: skip
    assert not res2["c9_3"]
