"""CPU-safe tests for ``terser.hardware.serving_bench`` (rows 7.5, 7.6)."""

from __future__ import annotations

from terser.hardware.serving_bench import BatchResult, ServingResult, run_name, tokens_per_s
from terser.tracking.runs import RUN_NAME_RE


def test_tokens_per_s() -> None:
    assert tokens_per_s(512, 2.0) == 256.0
    assert tokens_per_s(10, 0.0) == 0.0


def test_run_name_follows_pattern() -> None:
    names = [
        run_name("Qwen/Qwen2.5-0.5B-Instruct", None, smoke=True),
        run_name("Qwen/Qwen2.5-32B-Instruct-AWQ", "awq", smoke=False),
    ]
    assert names == ["7-vllmsmoke-qwen2.50.5b-bf16-0", "7-servingbench-qwen2.532bawq-awq-0"]
    assert all(RUN_NAME_RE.match(n) for n in names)


def test_result_to_dict() -> None:
    result = ServingResult("m", None, "bfloat16", 0.9, 4096, 500.0, 22000.0, 30.0)
    result.batches.append(BatchResult(16, 4096, 8.0))
    out = result.to_dict()
    assert out["batches"][0]["tokens_per_s"] == 512.0
    assert out["peak_vram_mib"] == 22000.0
