"""Prompts that go through ``candidates → respond → judge → label`` (plan rows 2.2.1–2.4.2).

Judging every candidate of all 7,080 casual prompts in both A/B orders would take ~100M tokens
on the 32B judge, above the phase2 token cap in ``configs/budget.yaml``. The labelled set is
therefore a deterministic subset of the split prompts: ``per_category`` prompts per category.

Order within a category: first one casual variant of every source group, then a second variant
of every group, and so on; ties broken by ``sha256(seed:id)``. So the subset covers as many
distinct source prompts as possible, and the first ``n`` prompts of a larger subset are the
subset for ``n``. Raising ``per_category`` extends the labelled set; with resumable,
cached stages the earlier work is reused. Every record gets a ``split`` field.

CLI::

    python -m terser.data.label_subset --splits-dir data/processed/splits \\
        --out data/interim/label_prompts.jsonl --per-category 300 --seed 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from terser.data.dedupe import read_jsonl, write_jsonl

__all__ = ["SPLITS", "label_subset", "main"]

SPLITS = ("train", "val", "test")


def _key(seed: int, rid: str) -> str:
    return hashlib.sha256(f"{seed}:{rid}".encode()).hexdigest()


def label_subset(
    split_records: dict[str, list[dict[str, Any]]], per_category: int, seed: int = 0
) -> list[dict[str, Any]]:
    """Pick ``per_category`` prompts per category, spreading over source groups first."""
    if per_category < 1:
        raise ValueError(f"per_category must be >= 1, got {per_category}")
    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for split, records in split_records.items():
        for r in records:
            by_cat[str(r["category"])].append({**r, "split": split})
    out: list[dict[str, Any]] = []
    for cat in sorted(by_cat):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in by_cat[cat]:
            groups[str(r.get("source_id") or r["id"])].append(r)
        ranked: list[tuple[int, str, dict[str, Any]]] = []
        for members in groups.values():
            members.sort(key=lambda r: _key(seed, str(r["id"])))
            ranked.extend((rank, _key(seed, str(r["id"])), r) for rank, r in enumerate(members))
        ranked.sort(key=lambda t: (t[0], t[1]))
        out.extend(r for _, _, r in ranked[:per_category])
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--splits-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-category", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    split_records = {s: read_jsonl(args.splits_dir / f"{s}.jsonl") for s in SPLITS}
    rows = label_subset(split_records, args.per_category, args.seed)
    write_jsonl(args.out, rows)
    cats = Counter(str(r["category"]) for r in rows)
    splits = Counter(str(r["split"]) for r in rows)
    groups = len({str(r.get("source_id") or r["id"]) for r in rows})
    print(json.dumps({"written": len(rows), "groups": groups, **cats, **splits}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
