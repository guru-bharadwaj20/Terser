"""Quality-path latency (**C11.3**) and throughput (**C11.4**) on one RTX 4500 Ada (row 11.3.3).

Needs the SFT rewriter served by vLLM (continuous batching) on ``--vllm`` under the name
``terser-rewriter`` (see ``configs/serving/quality.yaml``). The script starts the real API
(``uvicorn terser.serving.api:app``) with that config and drives ``POST /rewrite`` with
``mode: quality`` from an open-loop load generator over train prompts of at most 256 tokens on
the rewriter's tokenizer. Prompt ``i`` of a pass over the pool is sent as ``<prompt>`` on the
first pass and ``<prompt> #<pass>`` afterwards, so no request is a cache hit.

Each response says which path served it. A quality request that missed the latency budget
(``fallback_reason == "timeout"``) was answered by the fast path, so the share of requests the
rewriter itself served is reported next to the latency. Without it a p95 under the budget could
come from the timeout fallback rather than from the rewriter.

Targets as revised by ADR-0012 (originally p95 <= 400 ms and >= 95% quality-served for both):

* ``latency`` (C11.3): ``--rps`` 5 for ``--seconds``. Pass: p95 <= 450 ms, no errors, and at
  least 10% of requests served by the quality path (gate rejections count as served; the
  rewriter produced the candidate).
* ``throughput`` (C11.4): ``--rps`` 20 for ``--seconds`` (1800 for the checkpoint). Pass: achieved
  RPS >= 20 (rounded to 0.1), no errors, >= 5% quality-served, and the vLLM server still answers
  ``/v1/models`` afterwards (no OOM).

CLI::

    python -m terser.phase11.quality_bench latency throughput --base Qwen/Qwen2.5-1.5B-Instruct
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

__all__ = ["c11_3_pass", "c11_4_pass", "main", "prompt_pool", "summarise"]

# ADR-0012 targets (originally p95 <= 400 ms and >= 95% quality-served for both checks).
C11_3_P95_MS = 450.0
C11_3_QUALITY_SHARE_MIN = 0.10
C11_4_RPS_MIN = 20.0
C11_4_QUALITY_SHARE_MIN = 0.05


def c11_3_pass(res: dict[str, Any]) -> bool:
    """C11.3 verdict for a 5 RPS ``latency`` result."""
    return bool(
        res["p95_ms"] <= C11_3_P95_MS
        and res["errors"] == 0
        and res["quality_share"] >= C11_3_QUALITY_SHARE_MIN
    )


def c11_4_pass(res: dict[str, Any]) -> bool:
    """C11.4 verdict for a 20 RPS ``throughput`` result."""
    return bool(
        round(res["achieved_rps"], 1) >= C11_4_RPS_MIN
        and res["errors"] == 0
        and res["quality_share"] >= C11_4_QUALITY_SHARE_MIN
        and res["vllm_alive_after"]
    )


def prompt_pool(path: Path, count: Any, max_tokens: int = 256) -> list[str]:
    """Train prompts of at most ``max_tokens`` tokens under ``count`` (a ``str -> int``)."""
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    texts = [r.get("prompt") or r.get("casual_instruction") or "" for r in rows]
    return [t for t in texts if t.strip() and count(t) <= max_tokens]


def summarise(records: list[dict[str, Any]], n: int, errors: int, wall: float) -> dict[str, Any]:
    """Latency percentiles plus how each request was served."""
    lat = np.asarray([r["ms"] for r in records]) if records else np.asarray([np.nan])
    served = collections.Counter(r["served"] for r in records)
    quality = served.get("quality", 0)
    return {
        "requests": n,
        "errors": errors,
        "achieved_rps": n / wall if wall > 0 else 0.0,
        "p50_ms": float(np.percentile(lat, 50)),
        "p95_ms": float(np.percentile(lat, 95)),
        "p99_ms": float(np.percentile(lat, 99)),
        "served": dict(served),
        "quality_share": quality / n if n else 0.0,
        "gate_rejected": sum(r["gate_rejected"] for r in records),
    }


async def _load(url: str, prompts: list[str], rps: float, seconds: float) -> dict[str, Any]:
    import httpx

    n = int(rps * seconds)
    records: list[dict[str, Any]] = []
    errors = 0
    limits = httpx.Limits(max_connections=512, max_keepalive_connections=512)
    async with httpx.AsyncClient(timeout=30.0, limits=limits) as client:

        async def one(p: str) -> None:
            nonlocal errors
            t0 = time.perf_counter()
            try:
                r = await client.post(
                    url, json={"prompt": p, "target": "qwen2.5", "mode": "quality"}
                )
                r.raise_for_status()
                d = r.json()
                served = "quality" if d["path"] == "quality" else str(d["fallback_reason"])
                records.append(
                    {
                        "ms": (time.perf_counter() - t0) * 1000,
                        "served": served,
                        "gate_rejected": bool(d["gate_rejected"]),
                    }
                )
            except Exception:
                errors += 1

        start = time.perf_counter()
        tasks = []
        for i in range(n):
            delay = start + i / rps - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            k, rep = i % len(prompts), i // len(prompts)
            text = prompts[k] if rep == 0 else f"{prompts[k]} #{rep}"
            tasks.append(asyncio.create_task(one(text)))
        await asyncio.gather(*tasks)
        wall = time.perf_counter() - start
    return summarise(records, n, errors, wall)


def _vllm_alive(vllm: str) -> bool:
    import httpx

    try:
        return httpx.get(f"{vllm.rstrip('/')}/models", timeout=5.0).status_code == 200
    except Exception:
        return False


def run(
    checks: Sequence[str],
    prompts: list[str],
    *,
    vllm: str,
    port: int,
    latency_s: float,
    throughput_s: float,
    config: str,
) -> dict[str, Any]:
    import httpx

    env = {**os.environ, "TERSER_SERVING_CONFIG": config}
    cmd = [sys.executable, "-m", "uvicorn", "terser.serving.api:app", "--port", str(port)]
    srv = subprocess.Popen([*cmd, "--log-level", "warning"], env=env)
    url = f"http://127.0.0.1:{port}/rewrite"
    out: dict[str, Any] = {}
    try:
        for _ in range(240):
            try:
                if httpx.get(f"http://127.0.0.1:{port}/healthz", timeout=1.0).json()[
                    "quality_path"
                ]:
                    break
            except Exception:
                pass
            time.sleep(0.5)
        for p in prompts[:20]:  # warm-up: tokenizers, ONNX sessions, vLLM graphs
            httpx.post(url, json={"prompt": p, "target": "qwen2.5", "mode": "quality"}, timeout=60)
        pool = prompts[20:]
        if "latency" in checks:
            res = asyncio.run(_load(url, pool, 5.0, latency_s))
            res["rps"] = 5.0
            res["c11_3"] = c11_3_pass(res)
            out["latency"] = res
            print(json.dumps(res), flush=True)
        if "throughput" in checks:
            res = asyncio.run(_load(url, pool, 20.0, throughput_s))
            res["rps"] = 20.0
            res["vllm_alive_after"] = _vllm_alive(vllm)
            res["c11_4"] = c11_4_pass(res)
            out["throughput"] = res
            print(json.dumps(res), flush=True)
    finally:
        srv.terminate()
        srv.wait(timeout=30)
    return out


def _row(r: dict[str, Any]) -> str:
    served = ", ".join(f"{k} {v}" for k, v in sorted(r["served"].items()))
    return (
        f"| {r['rps']:g} | {r['requests']} | {r['errors']} | {r['achieved_rps']:.1f} | "
        f"{r['p50_ms']:.0f} | {r['p95_ms']:.0f} | {r['p99_ms']:.0f} | {r['quality_share']:.1%} | "
        f"{served} | {r['gate_rejected']} |"
    )


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - needs a GPU worker
    ap = argparse.ArgumentParser(description="C11.3 / C11.4 quality-path checks.")
    ap.add_argument("checks", nargs="+", choices=("latency", "throughput"))
    ap.add_argument("--base", required=True, help="rewriter base model (tokenizer)")
    ap.add_argument("--adapter", default="", help="adapter dir served as terser-rewriter")
    ap.add_argument("--prompts", type=Path, default=Path("data/processed/splits/train.jsonl"))
    ap.add_argument("--config", default="configs/serving/quality.yaml")
    ap.add_argument("--vllm", default="http://localhost:8001/v1")
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--latency-seconds", type=float, default=120.0)
    ap.add_argument("--throughput-seconds", type=float, default=1800.0)
    ap.add_argument("--report", type=Path, default=Path("reports/p11_quality_path.md"))
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.base)
    prompts = prompt_pool(
        args.prompts, lambda s: len(tok(s, add_special_tokens=False)["input_ids"])
    )
    out = run(
        args.checks,
        prompts,
        vllm=args.vllm,
        port=args.port,
        latency_s=args.latency_seconds,
        throughput_s=args.throughput_seconds,
        config=args.config,
    )
    head = (
        "| RPS | requests | errors | achieved RPS | p50 ms | p95 ms | p99 ms | quality-served | "
        "served by | gate rejected |"
    )
    lines = [
        "# Quality-path latency and throughput (row 11.3.3, C11.3, C11.4)",
        "",
        "Generated by `python -m terser.phase11.quality_bench`. Real API (`uvicorn`, 1 worker) "
        "with "
        f"`{args.config}`; rewriter `{args.base}` + adapter `{args.adapter or 'n/a'}` on vLLM "
        "(continuous batching) on one RTX 4500 Ada (24 GB). Open-loop load, `mode: quality`, "
        f"{len(prompts)} distinct train prompts of <= 256 tokens (no cache hits), client-side "
        "latency including HTTP, the gate and the fallback. `quality-served` is the share of "
        "requests the rewriter answered inside the 400 ms budget; the rest fell back "
        "(`timeout`) to the fast path.",
        "",
        head,
        "|---" * 10 + "|",
    ]
    for key in ("latency", "throughput"):
        if key in out:
            lines.append(_row(out[key]))
    lines.append("")
    if "latency" in out:
        v = "pass" if out["latency"]["c11_3"] else "fail"
        lines.append(f"C11.3 (p95 <= 450 ms at 5 RPS, >= 10% quality-served; ADR-0012): **{v}**.")
    if "throughput" in out:
        t = out["throughput"]
        v = "pass" if t["c11_4"] else "fail"
        lines.append(
            f"C11.4 (>= 20 RPS for {args.throughput_seconds:g} s, no errors, "
            ">= 5% quality-served, "
            f"no OOM; vLLM alive after: {t['vllm_alive_after']}; ADR-0012): **{v}**."
        )
    lines.append("")
    args.report.write_text("\n".join(lines), encoding="utf-8")
    Path(args.report.with_suffix(".json")).write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
