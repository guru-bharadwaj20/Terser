"""Tests for ``terser.hardware.verify`` (row 7.4)."""

from __future__ import annotations

from pathlib import Path

import pytest

from terser.hardware.verify import VerifyResult, render_report


def _result(**overrides: object) -> VerifyResult:
    values: dict[str, object] = {
        "device_name": "NVIDIA RTX 4500 Ada Generation",
        "total_memory_gib": 23.99,
        "capability": (8, 9),
        "bf16_supported": True,
        "tflops": 150.0,
        "parity_max_rel_error": 2e-3,
        "matmul_size": 8192,
        "versions": {"torch": "2.x"},
        "timestamp": "2026-10-01 00:00 UTC",
    }
    values.update(overrides)
    return VerifyResult(**values)  # type: ignore[arg-type]


def test_all_green_report() -> None:
    result = _result()
    assert result.all_passed
    report = render_report(result, host="box")
    assert "## Checks: ALL GREEN" in report
    assert "| Host | box |" in report
    assert "❌" not in report


@pytest.mark.parametrize(
    "overrides",
    [
        {"capability": (8, 6)},
        {"total_memory_gib": 11.9},
        {"bf16_supported": False},
        {"parity_max_rel_error": 0.5},
    ],
)
def test_failing_check(overrides: dict[str, object]) -> None:
    result = _result(**overrides)
    assert not result.all_passed
    assert "## Checks: FAILED" in render_report(result)


def test_parity_on_cpu() -> None:
    torch = pytest.importorskip("torch")
    from terser.hardware.verify import bf16_parity

    err = bf16_parity("cpu", n=64)
    assert 0 < err < 1e-2
    assert torch.get_default_dtype() == torch.float32


@pytest.mark.gpu
def test_verify_on_gpu(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device visible")
    from terser.hardware.verify import main

    out = tmp_path / "hw.md"
    assert main(["--output", str(out), "--matmul-size", "2048"]) == 0
    assert "ALL GREEN" in out.read_text(encoding="utf-8")
