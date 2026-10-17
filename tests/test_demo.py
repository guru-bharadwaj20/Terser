"""Row 13.8: demo script. Cache building, cached fallback and display; no GPU, no tokenizers."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

from terser.demo import (
    CacheIndex,
    LiveEndpoints,
    build_cache,
    probe,
    run_demo,
    side_by_side,
    token_table,
)

QWEN, PHI = "Qwen/Qwen2.5-7B-Instruct", "microsoft/Phi-3.5-mini-instruct"


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _fixture(tmp: Path) -> Path:
    """Three val prompts: two eligible (one preserved, one degraded), one that fell back."""
    d = tmp / "rewrites"
    d.mkdir()
    prompts = {
        "a": ("explain", "hey um so what is a planet?? thx", "what is a planet?", "preserved"),
        "b": ("explain", "pls explain recursion like im 5", "explain recursion", "degraded"),
        "c": ("explain", "list three primes", "list three primes", "preserved"),
    }
    _write(
        d / "sft-qwen1.5b-lora.val.jsonl",
        [
            {
                "id": k,
                "category": c,
                "prompt": p,
                "rewrite": r,
                "fell_back": k == "c",
                "latency_s": 0.1,
                "model": "models/rewriter/sft-qwen1.5b-lora",
            }
            for k, (c, p, r, _) in prompts.items()
        ],
    )
    for name, model in (("val_resp_qwen2.5-7b.jsonl", QWEN), ("val_resp_phi3.5.jsonl", PHI)):
        rows = []
        for k in prompts:
            for cand in ("original", f"{k}:sft15", f"{k}:sft05"):
                rows.append(
                    {
                        "prompt_id": k,
                        "candidate_id": cand,
                        "target": model,
                        "sample": 0,
                        "output": f"{model} answer to {cand}",
                        "output_tokens": 7,
                    }
                )
        _write(d / name, rows)
    labels = []
    for k, (*_, lab) in prompts.items():
        for model in (QWEN, PHI):
            labels.append(
                {
                    "prompt_id": k,
                    "candidate_id": f"{k}:sft15",
                    "target": model,
                    "label": lab,
                    "reason": lab,
                    "judge_model": "judge",
                }
            )
    _write(d / "val_labels.jsonl", labels)
    return d


def words(text: str, _target: str) -> int:
    return len(text.split())


def fake_fast(prompt: str, target: str) -> dict[str, Any]:
    cand = " ".join(w for w in prompt.split() if w not in {"hey", "um", "so", "thx", "pls"})
    return {
        "rewrite": prompt,
        "candidate": cand,
        "gate_score": 0.9,
        "gate_rejected": True,
        "threshold": 0.26,
        "latency_ms": 1.0,
        "config": "fake",
    }


def test_build_cache_selects_eligible_examples(tmp_path: Path) -> None:
    out = tmp_path / "cache.jsonl"
    recs = build_cache(_fixture(tmp_path), out, per_category=3)
    assert {r["id"] for r in recs} == {"a", "b"}  # "c" fell back: nothing to show
    a = next(r for r in recs if r["id"] == "a")
    assert a["quality_rewrite"] == "what is a planet?"
    assert set(a["responses"]) == {"qwen2.5", "phi3.5"}
    assert a["responses"]["qwen2.5"]["original"].endswith("answer to original")
    assert a["responses"]["qwen2.5"]["rewrite"].endswith("answer to a:sft15")
    assert CacheIndex.load(out).by_id("b") is not None


def test_cached_mode_shows_rewrite_answers_and_tokens(tmp_path: Path) -> None:
    out = tmp_path / "cache.jsonl"
    build_cache(_fixture(tmp_path), out)
    cache = CacheIndex.load(out)
    buf = io.StringIO()
    shown = run_demo(
        "hey um so what is a planet?? thx  ",  # normalised lookup ignores trailing space
        cache=cache,
        fast=fake_fast,
        counter=words,
        gate=lambda p, r, t: 0.1,
        count_targets=("gpt-4o", "qwen2.5"),
        out=buf,
    )
    text = buf.getvalue()
    assert shown["mode"] == "cached" and shown["cache_id"] == "a"
    assert shown["quality"] == "what is a planet?"
    assert shown["tokens"]["original"]["qwen2.5"] == 8
    assert shown["tokens"]["quality rewrite"]["qwen2.5"] == 4
    assert set(shown["answers"]) == {"qwen2.5", "phi3.5"}
    assert "REJECTED, original served" in text and "would pass" in text
    assert "4 (-50.0%)" in text and "CACHED" in text


def test_unknown_prompt_falls_back_to_fast_path_only(tmp_path: Path) -> None:
    out = tmp_path / "cache.jsonl"
    build_cache(_fixture(tmp_path), out)
    buf = io.StringIO()
    shown = run_demo(
        "hey so um what is a mutex",
        cache=CacheIndex.load(out),
        fast=fake_fast,
        counter=words,
        nearest=True,
        out=buf,
    )
    assert shown["mode"] == "fast-only" and shown["quality"] is None and not shown["answers"]
    assert "unavailable" in buf.getvalue() and "--example a" in buf.getvalue()


def test_side_by_side_and_token_table() -> None:
    txt = side_by_side("L", "one two three four five six", "R", "x\ny", width=40, max_lines=2)
    lines = txt.splitlines()
    assert lines[0].startswith("L") and " | R" in lines[0]
    assert all(len(line) <= 40 for line in lines)
    assert len(lines) == 4  # title, rule, two body rows (wrapping truncated to max_lines)
    table = token_table([("original", "a b c d"), ("longer", "a b c d e")], ["t"], words)
    assert "5 (+25.0%)" in table


def test_no_endpoint_means_no_live_path() -> None:
    assert probe("http://127.0.0.1:9/v1", timeout_s=0.2) is None
    live = LiveEndpoints.discover(None, {"qwen2.5": "http://127.0.0.1:9/v1"}, timeout_s=0.2)
    assert not live.any
