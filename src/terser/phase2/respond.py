"""``respond`` stage: target-model outputs for every prompt and candidate (plan row **2.2.3**).

For each casual prompt and each of its non-rejected candidates (row 2.2.1) the target model
answers the text as a single user turn through :class:`terser.llm.LLMClient` (vLLM on the RTX
4500 Ada, provider ``local``):

* ``code``, ``fact``, ``reason``, ``explain``, ``transform``: one greedy sample (temperature 0);
* ``creative``: ``creative_samples`` (default 3) samples at ``creative_temperature`` with fixed
  per-sample seeds. The same seeds are used for the original and every rewrite (common random
  numbers), so sample ``i`` of a rewrite is compared with sample ``i`` of the original.

A candidate whose text equals the prompt reuses the original's output by construction (identical
requests hit the cache), and identical texts across methods are requested once.

Output: one JSON line per (prompt, candidate, sample) with :data:`RESPONSE_FIELDS`. Records are
written in a deterministic order; requests run concurrently (``workers``) so vLLM can batch, and
everything is served from the disk cache on a rerun, so the stage is resumable and ``dvc repro``
rebuilds it on CPU.

CLI::

    python -m terser.phase2.respond --prompts data/interim/label_prompts.jsonl \\
        --candidates data/interim/candidates.jsonl --target Qwen/Qwen2.5-7B-Instruct \\
        --out data/interim/responses/qwen2.5-7b.jsonl --workers 64
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from terser.llm import LLMClient

__all__ = [
    "ORIGINAL_ID",
    "RESPONSE_FIELDS",
    "RespondItem",
    "RespondStats",
    "build_items",
    "main",
    "run_respond",
    "sample_plan",
]

ORIGINAL_ID = "original"
"""``candidate_id`` of the original prompt (matches :data:`terser.phase2.oracle.ORIGINAL_ID`)."""

RESPONSE_FIELDS: tuple[str, ...] = (
    "prompt_id",
    "candidate_id",
    "category",
    "target",
    "sample",
    "temperature",
    "seed",
    "text_sha256",
    "output",
    "output_tokens",
    "truncated",
)

DEFAULT_MAX_TOKENS = 768
DEFAULT_CREATIVE_SAMPLES = 3
DEFAULT_CREATIVE_TEMPERATURE = 0.7
SEED_BASE = 1000


@dataclass(frozen=True)
class RespondItem:
    """One text to answer for one (prompt, candidate)."""

    prompt_id: str
    candidate_id: str
    category: str
    text: str


def sample_plan(
    category: str,
    creative_samples: int = DEFAULT_CREATIVE_SAMPLES,
    creative_temperature: float = DEFAULT_CREATIVE_TEMPERATURE,
) -> list[tuple[int, float, int | None]]:
    """``[(sample, temperature, seed)]`` for a category: greedy once, or seeded creative samples."""
    if category == "creative" and creative_samples > 1:
        return [(i, creative_temperature, SEED_BASE + i) for i in range(creative_samples)]
    return [(0, 0.0, None)]


def _refuse_stand_in(record: Mapping[str, Any], kind: str, rid: Any) -> None:
    if record.get("stand_in") in (True, "True", "true"):
        raise ValueError(f"stand-in {kind} {rid!r} must not be used for target outputs")


def build_items(
    prompts: Iterable[Mapping[str, Any]], candidates: Iterable[Mapping[str, Any]]
) -> list[RespondItem]:
    """The original of every prompt plus every non-rejected candidate of a known prompt.

    Raises ``ValueError`` on any stand-in prompt or candidate: stand-in text must never reach
    the target outputs that labels are built from.
    """
    items: list[RespondItem] = []
    cats: dict[str, str] = {}
    for p in prompts:
        _refuse_stand_in(p, "prompt", p.get("id"))
        pid = str(p["id"])
        cats[pid] = str(p["category"])
        items.append(RespondItem(pid, ORIGINAL_ID, cats[pid], str(p["prompt"])))
    for c in candidates:
        _refuse_stand_in(c, "candidate", c.get("candidate_id"))
        pid = str(c["prompt_id"])
        if c.get("rejected") or pid not in cats or not str(c.get("text", "")).strip():
            continue
        items.append(RespondItem(pid, str(c["candidate_id"]), cats[pid], str(c["text"])))
    return items


@dataclass
class RespondStats:
    """Accounting of one run."""

    records: int = 0
    requests: int = 0
    cached: int = 0
    generated_tokens: int = 0
    truncated: int = 0
    wall_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form."""
        return dict(self.__dict__)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run_respond(
    items: Sequence[RespondItem],
    client: LLMClient,
    *,
    target: str,
    out_path: Path,
    provider: str = "local",
    max_tokens: int = DEFAULT_MAX_TOKENS,
    creative_samples: int = DEFAULT_CREATIVE_SAMPLES,
    creative_temperature: float = DEFAULT_CREATIVE_TEMPERATURE,
    workers: int = 64,
) -> RespondStats:
    """Answer every item with ``target`` and write one record per (item, sample)."""
    stats = RespondStats()
    lock = threading.Lock()
    start = time.perf_counter()
    # Distinct requests: (text, temperature, seed).
    requests: dict[tuple[str, float, int | None], dict[str, Any] | None] = {}
    for it in items:
        for _, temp, seed in sample_plan(it.category, creative_samples, creative_temperature):
            requests.setdefault((it.text, temp, seed), None)

    def call(key: tuple[str, float, int | None]) -> None:
        text, temp, seed = key
        params: dict[str, Any] = {"temperature": temp}
        if seed is not None:
            params["seed"] = seed
        resp = client.complete(
            provider, target, [{"role": "user", "content": text}], max_tokens, **params
        )
        with lock:
            requests[key] = {
                "output": resp.text,
                "output_tokens": resp.output_tokens,
                "cached": resp.cached,
            }
            stats.requests += 1
            stats.cached += int(resp.cached)
            stats.generated_tokens += 0 if resp.cached else resp.output_tokens

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(call, list(requests)))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as fh:
        for it in items:
            for sample, temp, seed in sample_plan(
                it.category, creative_samples, creative_temperature
            ):
                res = requests[(it.text, temp, seed)]
                assert res is not None
                truncated = int(res["output_tokens"]) >= max_tokens
                rec = {
                    "prompt_id": it.prompt_id,
                    "candidate_id": it.candidate_id,
                    "category": it.category,
                    "target": target,
                    "sample": sample,
                    "temperature": temp,
                    "seed": seed,
                    "text_sha256": _sha(it.text),
                    "output": res["output"],
                    "output_tokens": res["output_tokens"],
                    "truncated": truncated,
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                stats.records += 1
                stats.truncated += int(truncated)
    stats.wall_seconds = time.perf_counter() - start
    return stats


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(prog="python -m terser.phase2.respond", description=__doc__)
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--candidates", type=Path, default=None)
    ap.add_argument("--target", required=True, help="served model id")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--provider", default="local")
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--creative-samples", type=int, default=DEFAULT_CREATIVE_SAMPLES)
    ap.add_argument("--creative-temperature", type=float, default=DEFAULT_CREATIVE_TEMPERATURE)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None, help="first N prompts (smoke)")
    ap.add_argument("--budget-phase", default="phase2")
    ap.add_argument("--hardware", default="rtx4500ada")
    ap.add_argument("--track", action="store_true")
    args = ap.parse_args(argv)

    from terser.llm import LLMClient

    prompts = _read_jsonl(args.prompts)[: args.limit]
    candidates = _read_jsonl(args.candidates) if args.candidates else []
    items = build_items(prompts, candidates)
    client = LLMClient(args.budget_phase)
    kwargs: dict[str, Any] = {
        "target": args.target,
        "out_path": args.out,
        "provider": args.provider,
        "max_tokens": args.max_tokens,
        "creative_samples": args.creative_samples,
        "creative_temperature": args.creative_temperature,
        "workers": args.workers,
    }
    if args.track:
        import mlflow

        from terser.tracking import start_run

        name = args.target.split("/")[-1].replace("-", "_")
        with start_run(
            phase=2,
            row="2.2.3",
            task="respond",
            run_name=f"2-respond-{name}-full-0",
            hardware=args.hardware,
            params={
                "target": args.target,
                "n_prompts": len(prompts),
                "n_items": len(items),
                "max_tokens": args.max_tokens,
                "creative_samples": args.creative_samples,
                "creative_temperature": args.creative_temperature,
            },
        ):
            stats = run_respond(items, client, **kwargs)
            mlflow.log_metrics(
                {
                    **{k: float(v) for k, v in stats.to_dict().items()},
                    "gpu_hours": stats.wall_seconds / 3600 if args.hardware != "cpu" else 0.0,
                }
            )
    else:
        stats = run_respond(items, client, **kwargs)
    print(json.dumps({"out": str(args.out), **stats.to_dict()}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
