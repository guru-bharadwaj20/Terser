"""Serving checkpoints on the CPU: fast-path latency (**C11.2**), fallbacks (**C11.5**).

* ``latency``: starts the real API (``uvicorn terser.serving.api:app``) with
  ``configs/serving/fast_onnx.yaml`` (ONNX gate + span tagger, row 11.3.2) and drives it with an
  open-loop load generator at ``--rps`` (default 20) for ``--seconds`` over **distinct** train
  prompts (no cache hits). p50 / p95 / p99 of client-side latency; C11.2 asks p95 <= 150 ms.
* ``fallback``: fault injection on :class:`~terser.serving.service.RewriteService` over
  ``--n`` real prompts per scenario: quality path crashes, times out, or hangs; fast path
  crashes; both fail. Every request must return a response, and whenever no rewriter produced a
  result the response must be the **original prompt** (C11.5: 100%).

Report: ``reports/p11_serving_checks.md``.

CLI::

    python -m terser.phase11.serving_checks latency fallback
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from terser.serving.accounting import Pricing
from terser.serving.api import load_config
from terser.serving.paths import RewriteResult
from terser.serving.service import RewriteService

__all__ = ["main"]


def _prompts(path: Path, n: int) -> list[str]:
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    out = [r["prompt"] for r in rows if 20 <= len(r["prompt"]) <= 4000]
    return out[:n]


async def _load(url: str, prompts: list[str], rps: float, seconds: float) -> dict[str, Any]:
    import httpx

    n = int(rps * seconds)
    lat: list[float] = []
    errors = 0
    async with httpx.AsyncClient(timeout=10.0) as client:

        async def one(p: str) -> None:
            nonlocal errors
            t0 = time.perf_counter()
            try:
                r = await client.post(url, json={"prompt": p, "target": "gpt-4o", "mode": "fast"})
                r.raise_for_status()
                lat.append((time.perf_counter() - t0) * 1000)
            except Exception:  # noqa: BLE001
                errors += 1

        start = time.perf_counter()
        tasks = []
        for i in range(n):
            delay = start + i / rps - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            tasks.append(asyncio.create_task(one(prompts[i % len(prompts)])))
        await asyncio.gather(*tasks)
        wall = time.perf_counter() - start
    a = np.asarray(lat)
    return {
        "requests": n,
        "errors": errors,
        "achieved_rps": n / wall,
        "p50_ms": float(np.percentile(a, 50)),
        "p95_ms": float(np.percentile(a, 95)),
        "p99_ms": float(np.percentile(a, 99)),
    }


def check_latency(prompts: list[str], rps: float, seconds: float, port: int) -> dict[str, Any]:
    env = {**os.environ, "TERSER_SERVING_CONFIG": "configs/serving/fast_onnx.yaml"}
    srv = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "terser.serving.api:app",
            "--port",
            str(port),
            "--log-level",
            "warning",
        ],
        env=env,
    )
    try:
        import httpx

        for _ in range(120):
            try:
                httpx.get(f"http://127.0.0.1:{port}/healthz", timeout=1.0)
                break
            except Exception:  # noqa: BLE001
                time.sleep(0.5)
        # warm-up: load tokenizers and ONNX sessions before timing
        for p in prompts[:20]:
            httpx.post(f"http://127.0.0.1:{port}/rewrite", json={"prompt": p}, timeout=30.0)
        res = asyncio.run(_load(f"http://127.0.0.1:{port}/rewrite", prompts[20:], rps, seconds))
    finally:
        srv.terminate()
        srv.wait(timeout=30)
    res["c11_2"] = res["p95_ms"] <= 150 and res["errors"] == 0
    return res


class _Crash:
    def __call__(self, prompt: str, target: str) -> RewriteResult:
        raise RuntimeError("injected crash")


class _Slow:
    def __init__(self, s: float) -> None:
        self.s = s

    def __call__(self, prompt: str, target: str) -> RewriteResult:
        time.sleep(self.s)
        return RewriteResult(prompt[: len(prompt) // 2])


def check_fallback(prompts: list[str]) -> dict[str, Any]:
    pricing = Pricing.from_config(load_config("configs/serving/default.yaml")["pricing"])
    scenarios: dict[str, tuple[dict[str, Any], str]] = {
        "quality crash": ({"quality": _Crash()}, "fast"),
        "quality timeout (1 s vs 0.05 s budget)": ({"quality": _Slow(1.0)}, "fast"),
        "fast crash (fast mode)": ({"fast": _Crash()}, "original"),
        "quality crash + fast crash": ({"quality": _Crash(), "fast": _Crash()}, "original"),
        "quality timeout + fast crash": ({"quality": _Slow(1.0), "fast": _Crash()}, "original"),
    }
    rows = []
    for name, (kw, expect) in scenarios.items():
        svc = RewriteService(
            pricing=pricing, quality_timeout_s=0.05, cache_size=1, max_workers=64, **kw
        )
        mode = "fast" if name.startswith("fast crash") else "quality"
        ok = answered = 0
        for i, p in enumerate(prompts):
            r = svc.rewrite(p + f" #{i}", "gpt-4o", mode=mode)  # distinct text: no cache
            answered += 1
            if expect == "original":
                ok += r.rewrite == p + f" #{i}"
            else:
                ok += r.fallback_used and r.path in ("fast", "original")
        rows.append(
            {
                "scenario": name,
                "requests": len(prompts),
                "answered": answered,
                "expected": "original prompt" if expect == "original" else "fast-path fallback",
                "correct": ok,
                "rate": ok / len(prompts),
            }
        )
        print(json.dumps(rows[-1]), flush=True)
    return {"rows": rows, "c11_5": all(r["rate"] == 1.0 for r in rows)}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="C11.2 / C11.5 serving checks.")
    ap.add_argument("checks", nargs="+", choices=("latency", "fallback"))
    ap.add_argument("--prompts", type=Path, default=Path("data/processed/splits/train.jsonl"))
    ap.add_argument("--rps", type=float, default=20.0)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--report", type=Path, default=Path("reports/p11_serving_checks.md"))
    args = ap.parse_args(argv)
    prompts = _prompts(args.prompts, max(args.n, int(args.rps * args.seconds) + 40))
    out: dict[str, Any] = {}
    lines = [
        "# Serving checks on the CPU (C11.2, C11.5)",
        "",
        "Generated by `python -m terser.phase11.serving_checks`.",
        "",
    ]
    if "latency" in args.checks:
        lat = check_latency(prompts, args.rps, args.seconds, args.port)
        out["latency"] = lat
        lines += [
            "## C11.2 Fast-path latency",
            "",
            f"Real API (`uvicorn`, 1 worker) with `configs/serving/fast_onnx.yaml` (ONNX span "
            f"tagger + ONNX gate, no GPU). Open-loop load at {args.rps:g} RPS for "
            f"{args.seconds:g} s over {lat['requests']} **distinct** train prompts (no cache "
            "hits), client-side latency including HTTP.",
            "",
            "| requests | errors | achieved RPS | p50 ms | p95 ms | p99 ms |",
            "|---|---|---|---|---|---|",
            f"| {lat['requests']} | {lat['errors']} | {lat['achieved_rps']:.1f} | "
            f"{lat['p50_ms']:.1f} | {lat['p95_ms']:.1f} | {lat['p99_ms']:.1f} |",
            "",
            f"C11.2 (p95 <= 150 ms at 20 RPS): **{'pass' if lat['c11_2'] else 'fail'}**.",
            "",
        ]
    if "fallback" in args.checks:
        fb = check_fallback(prompts[: args.n])
        out["fallback"] = fb
        lines += [
            "## C11.5 Fallback correctness",
            "",
            f"Fault injection on `RewriteService` (quality budget 0.05 s), {args.n} real train "
            "prompts per scenario, distinct texts (no cache hits).",
            "",
            "| scenario | requests | answered | expected | correct | rate |",
            "|---|---|---|---|---|---|",
            *(
                f"| {r['scenario']} | {r['requests']} | {r['answered']} | {r['expected']} | "
                f"{r['correct']} | {r['rate']:.1%} |"
                for r in fb["rows"]
            ),
            "",
            "Whenever no rewriter produced a result, the original prompt came back. "
            f"C11.5: **{'pass' if fb['c11_5'] else 'fail'}**.",
            "",
        ]
    args.report.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(out, default=str)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
