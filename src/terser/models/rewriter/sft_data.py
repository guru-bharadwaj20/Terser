"""SFT data for the generative rewriter (plan row **8.3.1**).

One target-agnostic rewriter (C1.5: tokenizers agree), so each prompt gets a single target:
the shortest candidate the judge labelled ``preserved`` for **every** target model. Prompts
with no such candidate keep the original prompt as an identity example, so the rewriter also
learns when not to compress. Train and val splits only; the test split is never read.

Entry point::

    python -m terser.models.rewriter.sft_data --out data/processed/sft
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

__all__ = ["build_sft_rows", "main"]


def _jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def build_sft_rows(
    prompts: Sequence[dict[str, Any]],
    candidates: Sequence[dict[str, Any]],
    labels: Sequence[dict[str, Any]],
    tokens: dict[str, int],
) -> list[dict[str, Any]]:
    """``{id, category, prompt, rewrite, candidate_id, method, savings}`` per prompt.

    ``tokens`` maps a candidate id to its token count on the reference tokenizer; the original
    prompt's count is keyed ``<prompt_id>:original``.
    """
    targets = {r["target"] for r in labels}
    preserved: dict[str, set[str]] = defaultdict(set)
    for r in labels:
        if r["label"] == "preserved" and not r.get("stand_in", False):
            preserved[r["candidate_id"]].add(r["target"])
    by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in candidates:
        if not c.get("rejected") and not c.get("stand_in", False):
            by_prompt[c["prompt_id"]].append(c)
    rows = []
    for p in prompts:
        pid = p["id"]
        n0 = tokens[f"{pid}:original"]
        ok = [
            c
            for c in by_prompt.get(pid, [])
            if preserved.get(c["candidate_id"]) == targets and tokens[c["candidate_id"]] < n0
        ]
        best = min(ok, key=lambda c: (tokens[c["candidate_id"]], c["candidate_id"]), default=None)
        n1 = tokens[best["candidate_id"]] if best else n0
        rows.append(
            {
                "id": pid,
                "category": p["category"],
                "prompt": p["prompt"],
                "rewrite": best["text"] if best else p["prompt"],
                "candidate_id": best["candidate_id"] if best else f"{pid}:original",
                "method": best["method"] if best else "original",
                "savings": 1 - n1 / n0,
            }
        )
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the row 8.3.1 SFT dataset.")
    ap.add_argument("--splits", default="data/processed/splits")
    ap.add_argument("--candidates", default="data/interim/candidates.jsonl")
    ap.add_argument("--labels", default="data/processed/labels.jsonl")
    ap.add_argument("--tokens", default="data/interim/token_counts.parquet")
    ap.add_argument("--token-target", default="qwen2.5")
    ap.add_argument("--out", default="data/processed/sft")
    args = ap.parse_args(argv)

    import pandas as pd

    tc = pd.read_parquet(args.tokens)
    tc = tc[tc.target == args.token_target]
    tokens = dict(zip(tc.candidate_id, tc.tokens.astype(int), strict=True))
    labels = list(_jsonl(Path(args.labels)))
    candidates = list(_jsonl(Path(args.candidates)))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stats: dict[str, dict[str, Any]] = {}
    for split in ("train", "val"):
        labelled = {r["prompt_id"] for r in labels if r.get("split") == split}
        prompts = [p for p in _jsonl(Path(args.splits) / f"{split}.jsonl") if p["id"] in labelled]
        rows = build_sft_rows(prompts, candidates, labels, tokens)
        with (out / f"{split}.jsonl").open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        sav = sorted(r["savings"] for r in rows)
        stats[split] = {
            "n": len(rows),
            "identity": sum(r["method"] == "original" for r in rows),
            "median_savings": sav[len(sav) // 2] if sav else 0.0,
        }
    (out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
