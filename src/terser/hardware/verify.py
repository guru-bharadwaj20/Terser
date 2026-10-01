"""GPU verification script (row 7.4): ``python -m terser.hardware.verify``.

Reports the CUDA device name, total memory and compute capability, benchmarks a bf16 matmul
(TFLOP/s) and checks bf16-vs-fp32 parity. The Markdown report is written to
``docs/checkpoints/C7/hardware.md`` by default. ``torch`` is imported lazily, so this module can be
imported (and its report rendering tested) on a machine without CUDA.
"""

from __future__ import annotations

import argparse
import datetime as dt
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "Check",
    "VerifyResult",
    "bf16_matmul_tflops",
    "bf16_parity",
    "main",
    "render_report",
    "run_verification",
]

DEFAULT_OUTPUT = Path("docs/checkpoints/C7/hardware.md")
EXPECTED_CAPABILITY = (8, 9)
MIN_TOTAL_MEMORY_GIB = 22.0
MAX_PARITY_ERROR = 1e-2


@dataclass(frozen=True)
class Check:
    """One pass/fail line of the verification report."""

    name: str
    measured: str
    target: str
    passed: bool


@dataclass
class VerifyResult:
    """Everything measured by :func:`run_verification`."""

    device_name: str
    total_memory_gib: float
    capability: tuple[int, int]
    bf16_supported: bool
    tflops: float
    parity_max_rel_error: float
    matmul_size: int
    versions: dict[str, str] = field(default_factory=dict)
    timestamp: str = ""

    @property
    def checks(self) -> list[Check]:
        cap = f"({self.capability[0]}, {self.capability[1]})"
        return [
            Check(
                "Compute capability",
                cap,
                f"{EXPECTED_CAPABILITY}",
                self.capability == EXPECTED_CAPABILITY,
            ),
            Check(
                "Total memory",
                f"{self.total_memory_gib:.2f} GiB",
                f">= {MIN_TOTAL_MEMORY_GIB:.0f} GiB",
                self.total_memory_gib >= MIN_TOTAL_MEMORY_GIB,
            ),
            Check("bf16 supported", str(self.bf16_supported), "True", self.bf16_supported),
            Check(
                "bf16 matmul throughput",
                f"{self.tflops:.1f} TFLOP/s",
                "> 0",
                self.tflops > 0,
            ),
            Check(
                "bf16 vs fp32 parity (max rel. error)",
                f"{self.parity_max_rel_error:.2e}",
                f"< {MAX_PARITY_ERROR:.0e}",
                self.parity_max_rel_error < MAX_PARITY_ERROR,
            ),
        ]

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)


def _torch() -> Any:
    import torch

    return torch


def bf16_matmul_tflops(device: str = "cuda", n: int = 8192, iters: int = 50) -> float:
    """Time ``iters`` bf16 ``n x n`` matmuls and return the achieved TFLOP/s."""
    torch = _torch()
    a = torch.randn(n, n, device=device, dtype=torch.bfloat16)
    b = torch.randn(n, n, device=device, dtype=torch.bfloat16)
    for _ in range(5):  # warm-up (cuBLAS heuristics, clocks)
        a @ b
    _sync(torch, device)
    start = time.perf_counter()
    for _ in range(iters):
        a @ b
    _sync(torch, device)
    elapsed = time.perf_counter() - start
    return 2.0 * n**3 * iters / elapsed / 1e12


def bf16_parity(device: str = "cuda", n: int = 1024, seed: int = 0) -> float:
    """Max |bf16 - fp32| / max |fp32| of a random ``n x n`` matmul (TF32 disabled)."""
    torch = _torch()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    a = torch.randn(n, n, generator=gen).to(device)
    b = torch.randn(n, n, generator=gen).to(device)
    tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        ref = a @ b
        low = (a.bfloat16() @ b.bfloat16()).float()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
    return float(((low - ref).abs().max() / ref.abs().max()).item())


def _sync(torch: Any, device: str) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def _versions(torch: Any) -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda (torch build)": str(torch.version.cuda),
        "cudnn": str(torch.backends.cudnn.version()),
    }


def run_verification(index: int = 0, matmul_size: int = 8192) -> VerifyResult:
    """Measure CUDA device ``index``; raises RuntimeError when no CUDA device is visible."""
    torch = _torch()
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device visible; run this on a GPU host.")
    device = f"cuda:{index}"
    props = torch.cuda.get_device_properties(index)
    torch.cuda.set_device(index)
    return VerifyResult(
        device_name=str(props.name),
        total_memory_gib=props.total_memory / 2**30,
        capability=(int(props.major), int(props.minor)),
        bf16_supported=bool(torch.cuda.is_bf16_supported()),
        tflops=bf16_matmul_tflops(device, n=matmul_size),
        parity_max_rel_error=bf16_parity(device),
        matmul_size=matmul_size,
        versions=_versions(torch),
        timestamp=dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M UTC"),
    )


def render_report(result: VerifyResult, host: str = "") -> str:
    """Render ``result`` as the Markdown report for ``docs/checkpoints/C7/hardware.md``."""
    verdict = "ALL GREEN" if result.all_passed else "FAILED"
    lines = [
        "# C7 hardware verification",
        "",
        "Generated by `python -m terser.hardware.verify` (row 7.4).",
        "",
        "| Field | Value |",
        "| :--- | :--- |",
        f"| Host | {host or platform.node()} |",
        f"| Measured at | {result.timestamp} |",
        f"| Device | {result.device_name} |",
        f"| Total memory | {result.total_memory_gib:.2f} GiB |",
        f"| Compute capability | {result.capability[0]}.{result.capability[1]} |",
    ]
    lines += [f"| {k} | {v} |" for k, v in result.versions.items()]
    lines += [
        "",
        f"## Checks: {verdict}",
        "",
        "| Check | Measured | Target | Pass |",
        "| :--- | :--- | :--- | :---: |",
    ]
    lines += [
        f"| {c.name} | {c.measured} | {c.target} | {'✅' if c.passed else '❌'} |"
        for c in result.checks
    ]
    lines += [
        "",
        f"Throughput: {result.matmul_size} x {result.matmul_size} bf16 matmul, 50 timed iterations "
        "after 5 warm-up iterations. Parity: 1024 x 1024 fp32-random matmul computed in bf16 vs "
        "fp32 (TF32 off), error normalised by max |fp32 result|.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--matmul-size", type=int, default=8192)
    parser.add_argument("--host", default="", help="host label for the report")
    args = parser.parse_args(argv)

    result = run_verification(args.device_index, args.matmul_size)
    report = render_report(result, host=args.host)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(report)
    return 0 if result.all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
