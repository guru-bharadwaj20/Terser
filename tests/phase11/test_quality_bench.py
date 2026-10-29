"""Pure parts of the quality-path bench (C11.3 / C11.4)."""

from __future__ import annotations

import json
from pathlib import Path

from terser.phase11.quality_bench import c11_3_pass, c11_4_pass, prompt_pool, summarise


def test_prompt_pool_filters_by_token_count(tmp_path: Path) -> None:
    f = tmp_path / "train.jsonl"
    rows = [{"prompt": "a b c"}, {"casual_instruction": "x " * 300}, {"prompt": " "}]
    f.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    assert prompt_pool(f, lambda s: len(s.split()), max_tokens=256) == ["a b c"]


def test_summarise_counts_quality_share() -> None:
    recs = [{"ms": 100.0, "served": "quality", "gate_rejected": True}] * 19
    recs += [{"ms": 400.0, "served": "timeout", "gate_rejected": False}]
    s = summarise(recs, n=21, errors=1, wall=1.05)
    assert s["served"] == {"quality": 19, "timeout": 1}
    assert abs(s["quality_share"] - 19 / 21) < 1e-12
    assert s["gate_rejected"] == 19 and s["errors"] == 1
    assert abs(s["achieved_rps"] - 20.0) < 1e-9


def test_c11_verdicts_on_2026_10_28_run() -> None:
    lat = {"p95_ms": 416.06, "errors": 0, "quality_share": 0.1607}
    thr = {"achieved_rps": 19.9975, "errors": 0, "quality_share": 0.09275, "vllm_alive_after": True}
    assert c11_3_pass(lat) and c11_4_pass(thr)
    assert not c11_3_pass({**lat, "p95_ms": 451.0})
    assert not c11_3_pass({**lat, "quality_share": 0.0})
    assert not c11_4_pass({**thr, "quality_share": 0.0})
    assert not c11_4_pass({**thr, "vllm_alive_after": False})
