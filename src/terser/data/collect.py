"""``collect`` stage: category-balanced, PII-scrubbed clean instructions (plan row **2.1.7**).

First stage of ``collect -> casualise -> dedupe -> split``. Reads the normalised records of the
downloaded sources (row 2.1.1, :func:`terser.data.sources.iter_records`), keeps records whose
``category_hint`` is one of the six categories, drops records outside a word-count window, runs
the strict PII scrubber (row 2.1.3) over every emitted string, including the reference answer,
and selects ``per_category`` records per category in a seeded hash order. All detected spans
are replaced with typed placeholders before ``data/interim/``.

Output record (``data/interim/clean.jsonl``)::

    {"id": "dolly:42", "source_id": "dolly:42", "source": "dolly", "category": "fact",
     "instruction": "...", "input": "..." | null, "clean": "<instruction>\\n\\n<input>",
     "reference": "...", "pii_status": "clean" | "redacted"}

CLI::

    python -m terser.data.collect --out data/interim/clean.jsonl --per-category 280
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from terser.data.categories import Category
from terser.data.pii import scrub_record
from terser.data.sources import DEFAULT_RAW_DIR, SOURCES, iter_records

__all__ = ["clean_text", "collect", "main", "select_balanced"]

DEFAULT_OUT = Path("data/interim/clean.jsonl")
DEFAULT_PER_CATEGORY = 280
DEFAULT_MIN_WORDS = 3
DEFAULT_MAX_WORDS = 350
CATEGORIES: tuple[str, ...] = tuple(c.value for c in Category)


def clean_text(instruction: str, input_text: str | None) -> str:
    """The clean prompt: the instruction, then its input (payload) after a blank line."""
    return f"{instruction}\n\n{input_text}" if input_text else instruction


def _rank(seed: int, source_id: str) -> str:
    return hashlib.sha256(f"terser-collect:{seed}:{source_id}".encode()).hexdigest()


def select_balanced(
    records: Iterable[Mapping[str, Any]], per_category: int, seed: int = 0
) -> list[dict[str, Any]]:
    """Up to ``per_category`` records per category, in seeded hash order, output sorted by id."""
    by_cat: dict[str, list[Mapping[str, Any]]] = {c: [] for c in CATEGORIES}
    for rec in records:
        by_cat[str(rec["category"])].append(rec)
    out: list[dict[str, Any]] = []
    for cat in CATEGORIES:
        ranked = sorted(by_cat[cat], key=lambda r: _rank(seed, str(r["source_id"])))
        out.extend(dict(r) for r in ranked[:per_category])
    return sorted(out, key=lambda r: (r["category"], r["id"]))


def collect(
    records: Iterable[Mapping[str, Any]],
    *,
    per_category: int = DEFAULT_PER_CATEGORY,
    seed: int = 0,
    min_words: int = DEFAULT_MIN_WORDS,
    max_words: int = DEFAULT_MAX_WORDS,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Filter, scrub and balance normalised source records; return ``(records, counts)``.

    ``counts`` tallies exclusions (``no_category``, ``length``), entities across eligible rows
    (``pii:<type>``) and selected rows per category (``kept:<category>``).
    """
    counts: Counter[str] = Counter()
    eligible: list[dict[str, Any]] = []
    for rec in records:
        cat = rec.get("category_hint")
        if cat not in CATEGORIES:
            counts["no_category"] += 1
            continue
        instruction = str(rec["instruction"]).strip()
        input_text = (str(rec["input"]).strip() or None) if rec.get("input") else None
        words = len(clean_text(instruction, input_text).split())
        if not min_words <= words <= max_words:
            counts["length"] += 1
            continue
        row, results = scrub_record(
            {
                "id": str(rec["source_id"]),
                "source_id": str(rec["source_id"]),
                "source": str(rec["source"]),
                "category": str(cat),
                "instruction": instruction,
                "input": input_text,
                "reference": rec.get("reference"),
            },
            skip=("id", "source_id", "source", "category"),
        )
        for result in results:
            counts.update({f"pii:{kind}": count for kind, count in result.counts.items()})
        row["clean"] = clean_text(row["instruction"], row["input"])
        row["pii_status"] = "redacted" if any(r.counts for r in results) else "clean"
        eligible.append(row)
    selected = select_balanced(eligible, per_category, seed)
    for rec in selected:
        counts[f"kept:{rec['category']}"] += 1
    counts["eligible"] = len(eligible)
    return selected, counts


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(prog="python -m terser.data.collect", description=__doc__)
    ap.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--per-category", type=int, default=DEFAULT_PER_CATEGORY)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-words", type=int, default=DEFAULT_MIN_WORDS)
    ap.add_argument("--max-words", type=int, default=DEFAULT_MAX_WORDS)
    ap.add_argument("--stats", type=Path, default=None, help="optional JSON stats path")
    args = ap.parse_args(argv)

    def all_records() -> Iterable[Mapping[str, Any]]:
        for name in sorted(SOURCES):
            yield from iter_records(name, args.raw_dir)

    selected, counts = collect(
        all_records(),
        per_category=args.per_category,
        seed=args.seed,
        min_words=args.min_words,
        max_words=args.max_words,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="\n") as fh:
        for rec in selected:
            fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
    stats = {"written": len(selected), **dict(sorted(counts.items()))}
    if args.stats is not None:
        args.stats.parent.mkdir(parents=True, exist_ok=True)
        args.stats.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
