"""CPU-safe tests for ``terser.hardware.train_bench`` (row 7.7)."""

from __future__ import annotations

from terser.hardware.train_bench import TrainResult, lora_target_modules, run_name
from terser.tracking.runs import RUN_NAME_RE


def test_run_names() -> None:
    names = [
        run_name("Qwen/Qwen2.5-0.5B-Instruct", "full", "fp32"),
        run_name("Qwen/Qwen2.5-1.5B-Instruct", "lora", "bf16"),
    ]
    assert names == ["7-trainbench-qwen2.50.5b-full-fp32-0", "7-trainbench-qwen2.51.5b-lora-r16-0"]
    assert all(RUN_NAME_RE.match(n) for n in names)


def test_lora_targets_cover_attention_and_mlp() -> None:
    targets = lora_target_modules()
    assert {"q_proj", "v_proj", "down_proj"} <= set(targets)


def test_result_summary() -> None:
    result = TrainResult("m", "full", "fp32", 1024, 4, 10, 10, 1.0, 2.0, [0.5, 1.5])
    out = result.to_dict()
    assert out["mean_step_seconds"] == 1.0
    assert out["tokens_per_s"] == 4096.0
    assert TrainResult("m", "full", "fp32", 1024, 4, 1, 1, 0.0, 0.0).tokens_per_s == 0.0
