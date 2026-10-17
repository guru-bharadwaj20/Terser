"""Row 12.10: compute accounting parsers and attribution on a synthetic evidence tree."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from terser.phase12.compute_accounting import (
    build,
    busy_seconds,
    json_blobs,
    log_span,
    md_table,
    render_markdown,
    vllm_span,
)


def t(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=UTC)


def test_parsers() -> None:
    text = 'noise\n{\n  "a": 1\n}\nmore\n{\n  "wall_seconds": 2.5,\n  "records": 3\n}\n'
    assert json_blobs(text) == [{"a": 1}, {"wall_seconds": 2.5, "records": 3}]
    span = log_span(
        [
            "2026/10/01 16:58:31 INFO mlflow hint",
            "[2026-10-01 17:00:00,123][httpx][INFO] - POST",
            "no timestamp",
        ]
    )
    assert span == (t("2026-10-01T16:58:31"), t("2026-10-01T17:00:00"))
    vs = vllm_span(["(APIServer pid=1) INFO 10-01 07:19:08 [x] up", "x", "INFO 10-01 07:24:25 y"])
    assert vs == (t("2026-10-01T07:19:08"), t("2026-10-01T07:24:25"))
    times = [t("2026-10-01T00:00:00"), t("2026-10-01T00:01:00"), t("2026-10-01T01:00:00")]
    assert busy_seconds(times, gap_s=300) == 60.0
    md = "x\n\n| Stage | Stage seconds |\n| :-- | --: |\n| A | 1,000.5 |\n| B | 2 |\n\nafter"
    assert md_table(md, "Stage") == [
        {"Stage": "A", "Stage seconds": "1,000.5"},
        {"Stage": "B", "Stage seconds": "2"},
    ]


def test_build_attributes_ledger_tokens_and_reservations(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "vllm_7b.log").write_text(
        "(APIServer pid=1) INFO 10-01 17:24:48 [api] model   Qwen/Qwen2.5-7B-Instruct\n"
        "(APIServer pid=1) INFO 10-01 19:24:48 [launcher] shutdown\n",
        encoding="utf-8",
    )
    (logs / "p2_respond_qwen7b.log").write_text(
        '{\n  "generated_tokens": 30,\n  "wall_seconds": 600.0\n}\n', encoding="utf-8"
    )
    calls = [
        {
            "timestamp": "2026-10-01T17:30:00+00:00",
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "input_tokens": 5,
            "output_tokens": 10,
        },
        {
            "timestamp": "2026-10-01T17:31:00+00:00",
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "input_tokens": 5,
            "output_tokens": 20,
        },
        # another model inside the window, and a call after the v2 cutoff: neither counted
        {
            "timestamp": "2026-10-01T17:32:00+00:00",
            "model": "microsoft/Phi-3.5-mini-instruct",
            "input_tokens": 5,
            "output_tokens": 99,
        },
        {
            "timestamp": "2026-10-20T09:00:00+00:00",
            "model": "Qwen/Qwen2.5-7B-Instruct",
            "input_tokens": 5,
            "output_tokens": 7,
        },
    ]
    (tmp_path / ".cache").mkdir()
    (tmp_path / ".cache" / "llm_ledger.json").write_text(
        json.dumps({"phases": {"phase2": {"calls": calls, "tokens": 0, "usd": 0.0}}}),
        encoding="utf-8",
    )
    report = build(tmp_path)
    st = next(s for s in report["stages"] if s["id"] == "p2-respond-qwen")
    assert st["generated_tokens"] == 30 and st["reported_tokens"] == 30
    assert st["active_s"] == 600.0 and st["ledger"]["calls"] == 2
    p2 = next(p for p in report["phases"] if p["phase"] == 2)
    assert p2["reserved_s"] >= 7200.0  # the server was up for two hours
    assert abs(p2["energy_active_kwh"] - 600 / 3600 * 0.210) < 1e-9
    assert report["ledger"]["after_cutoff_output_tokens"] == 7
    assert report["ledger"]["unattributed_by_model"] == {"microsoft/Phi-3.5-mini-instruct": 99}
    # the seed-run stages have no logs here: pending, not counted, later calls held back
    pend = {e["id"] for e in report["excluded"]}
    assert "p12-seed-train-s1" in pend and "p12-seed-judge" in pend
    assert all(not s["id"].startswith("p12-seed") for s in report["stages"])
    md = render_markdown(report)
    assert "# Compute accounting (row 12.10)" in md and "## Pending" in md


def test_pending_stage_counted_once_complete(tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    logs.mkdir()
    train = logs / "p12_train_sft3bv2_s1.log"
    train.write_text("[2026-10-20 01:38:03,563][httpx][INFO] - GET\n", encoding="utf-8")
    assert "p12-seed-train-s1" in {e["id"] for e in build(tmp_path)["excluded"]}
    train.write_text(
        "[2026-10-20 01:38:03,563][httpx][INFO] - GET\n"
        "[2026-10-20 02:38:03,000][__main__][INFO] - trained Qwen/Qwen2.5-3B for 1000 steps\n",
        encoding="utf-8",
    )
    report = build(tmp_path)
    st = next(s for s in report["stages"] if s["id"] == "p12-seed-train-s1")
    assert st["active_s"] == 3600.0
    assert "p12-seed-train-s1" not in {e["id"] for e in report["excluded"]}
    p12 = next(p for p in report["phases"] if p["phase"] == 12)
    assert p12["reserved_s"] >= 3600.0


def test_single_line_and_nested_summaries() -> None:
    text = 'x\n{"rewrite": {"wall_seconds": 2}}\n'
    assert json_blobs(text) == [{"rewrite": {"wall_seconds": 2}}]
