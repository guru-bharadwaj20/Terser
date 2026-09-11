"""Blind human annotation batches for judge validation (plan row **2.3.5**).

Workflow (see ``docs/annotation_guideline.md``):

1. :func:`sample_pairs` draws a stratified sample, 50 pairs per category (300 in total), from
   ``respond``/``label`` stage records that carry the user's original request and both target
   outputs. Stand-in (smoke) records are refused.
2. :func:`build_batch` gives every pair an opaque ``item_id`` (so ids such as ``p12:FHC`` cannot
   leak the operator subset), then for **each annotator independently** shuffles the row order
   and randomises which output is shown as Response A. It writes one sheet per annotator
   (``annotator_<name>.csv`` and ``.jsonl``) plus a hidden ``key.jsonl`` that maps
   ``item_id`` -> source id and, per annotator, which side is A. The key must not be shared
   with annotators.
3. Annotators fill ``verdict`` (``A`` / ``B`` / ``tie`` / ``cannot_judge``) and optionally
   ``confidence`` and ``notes``.
4. :func:`import_sheet` validates a filled sheet against the key and maps positions back to sides
   (``original`` / ``rewrite`` / ``tie``) and labels; :func:`merge_annotations` combines the
   annotators into gold records (``data/gold/human_300.jsonl``).

A human label follows the judge rule for a single order: ``degraded`` iff the annotator prefers the
output of the original prompt; ``preserved`` for tie or rewrite better; ``None`` for
``cannot_judge``.

CLI::

    python -m terser.judge.annotation build --records labels.jsonl --out data/gold/batch \\
        --annotators ann1 ann2 --seed 2305
    python -m terser.judge.annotation import --batch data/gold/batch \\
        --sheet ann1=filled_ann1.csv --sheet ann2=filled_ann2.csv --out data/gold/human_300.jsonl
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

from terser.data.categories import Category, parse_category

__all__ = [
    "ANNOTATION_VERDICTS",
    "CANNOT_JUDGE",
    "SHEET_COLUMNS",
    "AnnotationImportError",
    "BatchPaths",
    "ImportedAnnotation",
    "build_batch",
    "human_label",
    "import_sheet",
    "load_jsonl",
    "load_key",
    "main",
    "merge_annotations",
    "sample_pairs",
    "write_jsonl",
]

Side = Literal["original", "rewrite", "tie"]

CANNOT_JUDGE: Final = "cannot_judge"
ANNOTATION_VERDICTS: Final[tuple[str, ...]] = ("A", "B", "tie", CANNOT_JUDGE)
SHEET_COLUMNS: Final[tuple[str, ...]] = (
    "item_id",
    "category",
    "request",
    "response_a",
    "response_b",
    "verdict",
    "confidence",
    "notes",
)
"""Columns of an annotator sheet; the last three are filled by the annotator."""

PER_CATEGORY: Final = 50
_REQUIRED_INPUT = ("id", "category", "prompt", "original_output", "rewrite_output")


class AnnotationImportError(ValueError):
    """A filled sheet failed validation; ``errors`` lists every problem found."""

    def __init__(self, path: Path, errors: Sequence[str]) -> None:
        self.errors = list(errors)
        shown = "\n  ".join(self.errors[:20])
        more = f"\n  ... and {len(self.errors) - 20} more" if len(self.errors) > 20 else ""
        super().__init__(f"{path}: {len(self.errors)} problem(s):\n  {shown}{more}")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file (blank lines skipped)."""
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(rows: Iterable[Mapping[str, Any]], path: Path) -> None:
    """Write rows as UTF-8 JSONL with LF endings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def sample_pairs(
    records: Iterable[Mapping[str, Any]],
    *,
    per_category: int = PER_CATEGORY,
    categories: Sequence[str | Category] | None = None,
    seed: int = 0,
    allow_short: bool = False,
    allow_stand_in: bool = False,
) -> list[dict[str, Any]]:
    """Stratified random sample of ``per_category`` pairs from every category.

    Args:
        records: Pair records with ``id``, ``category``, ``prompt`` (the user's original request),
            ``original_output`` and ``rewrite_output``.
        per_category: Pairs per category (50 for row 2.3.5).
        categories: Categories to sample (default: all six).
        seed: RNG seed; the sample is a deterministic function of records and seed.
        allow_short: Take all pairs of a category that has fewer than ``per_category``.
        allow_stand_in: Accept records tagged ``stand_in: true`` (tests / smoke runs only).

    Returns:
        Sampled records, ordered by category then id.

    Raises:
        ValueError: Missing fields, duplicate ids, stand-in records, or too few pairs.
    """
    wanted = [parse_category(c) for c in (categories or list(Category))]
    by_cat: dict[Category, list[dict[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    for i, rec in enumerate(records):
        missing = [k for k in _REQUIRED_INPUT if rec.get(k) in (None, "")]
        if missing:
            raise ValueError(f"record {i}: missing field(s) {missing}")
        rid = str(rec["id"])
        if rid in seen:
            raise ValueError(f"duplicate record id {rid!r}")
        seen.add(rid)
        if rec.get("stand_in") and not allow_stand_in:
            raise ValueError(f"record {rid!r} is stand_in output; it must not be annotated")
        cat = parse_category(str(rec["category"]))
        if cat in wanted:
            by_cat[cat].append(dict(rec))
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    for cat in wanted:
        pool = sorted(by_cat[cat], key=lambda r: str(r["id"]))
        if len(pool) < per_category and not allow_short:
            raise ValueError(f"category {cat.value!r} has {len(pool)} pairs < {per_category}")
        chosen = rng.sample(pool, min(per_category, len(pool)))
        out.extend(sorted(chosen, key=lambda r: str(r["id"])))
    return out


@dataclass(frozen=True)
class BatchPaths:
    """Files written by :func:`build_batch`."""

    out_dir: Path
    key: Path
    manifest: Path
    sheets_csv: dict[str, Path]
    sheets_jsonl: dict[str, Path]


def _sheet_path(out_dir: Path, annotator: str, ext: str) -> Path:
    return out_dir / f"annotator_{annotator}.{ext}"


def build_batch(
    pairs: Sequence[Mapping[str, Any]],
    out_dir: Path,
    *,
    annotators: Sequence[str] = ("ann1", "ann2"),
    seed: int = 0,
) -> BatchPaths:
    """Write blind annotator sheets and the hidden key for a sample of pairs.

    Each annotator gets an independent row order and an independent A/B assignment per item.
    """
    if len(set(annotators)) != len(annotators) or not annotators:
        raise ValueError("annotators must be non-empty and unique")
    for name in annotators:
        if not name.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"annotator name {name!r} must be alphanumeric (plus _ or -)")
    ids = [str(p["id"]) for p in pairs]
    if len(set(ids)) != len(ids):
        raise ValueError("pair ids must be unique")
    rng = random.Random(seed)
    opaque = list(range(1, len(pairs) + 1))
    rng.shuffle(opaque)
    width = max(3, len(str(len(pairs))))
    item_ids = [f"item{n:0{width}d}" for n in opaque]

    key_rows: list[dict[str, Any]] = []
    sheets: dict[str, list[dict[str, str]]] = {a: [] for a in annotators}
    for item_id, pair in zip(item_ids, pairs, strict=True):
        a_sides: dict[str, Side] = {}
        for name in annotators:
            a_side: Side = "rewrite" if rng.random() < 0.5 else "original"
            a_sides[name] = a_side
            b_side = "original" if a_side == "rewrite" else "rewrite"
            sheets[name].append(
                {
                    "item_id": item_id,
                    "category": str(parse_category(str(pair["category"]))),
                    "request": str(pair["prompt"]),
                    "response_a": str(pair[f"{a_side}_output"]),
                    "response_b": str(pair[f"{b_side}_output"]),
                    "verdict": "",
                    "confidence": "",
                    "notes": "",
                }
            )
        key_rows.append(
            {
                "item_id": item_id,
                "source_id": str(pair["id"]),
                "category": str(parse_category(str(pair["category"]))),
                "a_side": a_sides,
            }
        )
    key_rows.sort(key=lambda r: r["item_id"])

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_paths: dict[str, Path] = {}
    jsonl_paths: dict[str, Path] = {}
    for name, rows in sheets.items():
        rng.shuffle(rows)
        csv_paths[name] = _sheet_path(out_dir, name, "csv")
        with csv_paths[name].open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(SHEET_COLUMNS), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        jsonl_paths[name] = _sheet_path(out_dir, name, "jsonl")
        write_jsonl(rows, jsonl_paths[name])
    key_path = out_dir / "key.jsonl"
    write_jsonl(key_rows, key_path)
    digest = hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()
    manifest = {
        "row": "2.3.5",
        "seed": seed,
        "annotators": list(annotators),
        "n_items": len(pairs),
        "per_category": dict(sorted(Counter(r["category"] for r in key_rows).items())),
        "source_ids_sha256": digest,
        "key_file": key_path.name,
        "note": "key.jsonl reveals which response is the rewrite; never share it with annotators",
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    return BatchPaths(out_dir, key_path, manifest_path, csv_paths, jsonl_paths)


def load_key(path: Path) -> dict[str, dict[str, Any]]:
    """Load ``key.jsonl`` as ``{item_id: key row}``."""
    return {str(r["item_id"]): r for r in load_jsonl(path)}


def human_label(side: Side | None) -> str | None:
    """Single-order label: ``degraded`` iff the original's output is preferred."""
    if side is None:
        return None
    return "degraded" if side == "original" else "preserved"


@dataclass(frozen=True)
class ImportedAnnotation:
    """One validated annotation, mapped back from positions to sides."""

    item_id: str
    source_id: str
    category: str
    annotator: str
    verdict: str
    side: Side | None
    label: str | None
    confidence: int | None
    notes: str


def _read_sheet(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            return list(reader.fieldnames or []), list(reader)
    if path.suffix.lower() == ".jsonl":
        rows = load_jsonl(path)
        cols = list(dict.fromkeys(k for r in rows for k in r))
        return cols, rows
    raise ValueError(f"{path}: sheet must be .csv or .jsonl")


def import_sheet(
    path: Path, key: Mapping[str, Mapping[str, Any]], annotator: str
) -> list[ImportedAnnotation]:
    """Validate a filled sheet against the key and map verdicts to sides and labels.

    Checks: required columns; every key item present exactly once and no unknown items; category
    unchanged; verdict in ``A|B|tie|cannot_judge`` (case-insensitive); confidence empty or 1-3;
    annotator present in the key.

    Raises:
        AnnotationImportError: Listing every problem found.
    """
    errors: list[str] = []
    columns, rows = _read_sheet(path)
    missing_cols = [c for c in ("item_id", "category", "verdict") if c not in columns]
    if missing_cols:
        raise AnnotationImportError(path, [f"missing column(s) {missing_cols}"])
    counts = Counter(str(r.get("item_id", "")).strip() for r in rows)
    for item_id, n in counts.items():
        if n > 1:
            errors.append(f"item {item_id!r} appears {n} times")
    for item_id in sorted(set(key) - set(counts)):
        errors.append(f"item {item_id!r} missing from sheet")

    verdict_map = {v.lower(): v for v in ANNOTATION_VERDICTS}
    verdict_map.update({"cannot judge": CANNOT_JUDGE, "cannot-judge": CANNOT_JUDGE})
    out: list[ImportedAnnotation] = []
    for lineno, row in enumerate(rows, 2 if path.suffix.lower() == ".csv" else 1):
        item_id = str(row.get("item_id", "")).strip()
        entry = key.get(item_id)
        if entry is None:
            errors.append(f"line {lineno}: unknown item_id {item_id!r}")
            continue
        a_sides = entry.get("a_side", {})
        if annotator not in a_sides:
            raise AnnotationImportError(path, [f"annotator {annotator!r} not in key"])
        if str(row.get("category", "")).strip() != entry["category"]:
            errors.append(f"line {lineno}: category changed for {item_id!r}")
        raw_verdict = str(row.get("verdict") or "").strip()
        verdict = verdict_map.get(raw_verdict.lower())
        if verdict is None:
            errors.append(
                f"line {lineno}: verdict {raw_verdict!r} for {item_id!r} not in "
                f"{list(ANNOTATION_VERDICTS)}"
            )
            continue
        raw_conf = str(row.get("confidence") or "").strip()
        confidence: int | None = None
        if raw_conf:
            try:
                confidence = int(float(raw_conf))
            except ValueError:
                confidence = -1
            if confidence not in (1, 2, 3):
                errors.append(f"line {lineno}: confidence {raw_conf!r} must be 1, 2, 3 or empty")
                continue
        a_side: Side = a_sides[annotator]
        b_side: Side = "original" if a_side == "rewrite" else "rewrite"
        side: Side | None = {"A": a_side, "B": b_side, "tie": "tie"}.get(verdict)
        out.append(
            ImportedAnnotation(
                item_id=item_id,
                source_id=str(entry["source_id"]),
                category=str(entry["category"]),
                annotator=annotator,
                verdict=verdict,
                side=side,
                label=human_label(side),
                confidence=confidence,
                notes=str(row.get("notes") or ""),
            )
        )
    if errors:
        raise AnnotationImportError(path, errors)
    return sorted(out, key=lambda a: a.item_id)


def merge_annotations(
    annotations: Mapping[str, Sequence[ImportedAnnotation]],
    *,
    adjudications: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Combine annotators into gold records (one per source pair).

    ``gold_label`` is the shared label when all annotators gave the same non-null label, the
    adjudicated label (``adjudications[source_id]``) when given, else ``None`` (needs
    adjudication or excluded as cannot-judge).
    """
    adjudications = dict(adjudications or {})
    by_source: dict[str, dict[str, ImportedAnnotation]] = defaultdict(dict)
    for name, items in annotations.items():
        for ann in items:
            by_source[ann.source_id][name] = ann
    names = sorted(annotations)
    gold: list[dict[str, Any]] = []
    for source_id in sorted(by_source):
        per = by_source[source_id]
        if sorted(per) != names:
            absent = sorted(set(names) - set(per))
            raise ValueError(f"{source_id!r} lacks annotations from {absent}")
        first = next(iter(per.values()))
        labels = {per[n].label for n in names}
        same = len(labels) == 1
        agreed = labels.pop() if same else None
        adjudicated = source_id in adjudications
        if adjudicated and adjudications[source_id] not in ("preserved", "degraded"):
            raise ValueError(f"adjudication for {source_id!r} must be preserved|degraded")
        gold.append(
            {
                "id": source_id,
                "item_id": first.item_id,
                "category": first.category,
                "annotations": {
                    n: {
                        "verdict": per[n].verdict,
                        "side": per[n].side,
                        "label": per[n].label,
                        "confidence": per[n].confidence,
                        "notes": per[n].notes,
                    }
                    for n in names
                },
                "annotators_agree": same,
                "adjudicated": adjudicated,
                "gold_label": adjudications[source_id] if adjudicated else agreed,
            }
        )
    return gold


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m terser.judge.annotation",
        description="Row 2.3.5: build blind annotation batches and import filled sheets.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="sample pairs and write annotator sheets + hidden key")
    b.add_argument("--records", type=Path, required=True, help="JSONL pair records")
    b.add_argument("--out", type=Path, required=True, help="batch directory")
    b.add_argument("--annotators", nargs="+", default=["ann1", "ann2"])
    b.add_argument("--per-category", type=int, default=PER_CATEGORY)
    b.add_argument("--seed", type=int, default=0)
    i = sub.add_parser("import", help="validate filled sheets and write gold JSONL")
    i.add_argument("--batch", type=Path, required=True, help="batch directory (with key.jsonl)")
    i.add_argument(
        "--sheet", action="append", required=True, metavar="NAME=PATH", help="filled sheet"
    )
    i.add_argument("--adjudications", type=Path, help="JSONL of {id, gold_label}")
    i.add_argument("--out", type=Path, required=True, help="gold JSONL, e.g. human_300.jsonl")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    args = _parser().parse_args(argv)
    try:
        if args.cmd == "build":
            pairs = sample_pairs(
                load_jsonl(args.records), per_category=args.per_category, seed=args.seed
            )
            paths = build_batch(pairs, args.out, annotators=args.annotators, seed=args.seed)
            n_ann = len(args.annotators)
            print(f"wrote {len(pairs)} items for {n_ann} annotators to {paths.out_dir}")
            return 0
        key = load_key(args.batch / "key.jsonl")
        imported: dict[str, list[ImportedAnnotation]] = {}
        for spec in args.sheet:
            name, _, path = spec.partition("=")
            if not path:
                raise ValueError(f"--sheet expects NAME=PATH, got {spec!r}")
            imported[name] = import_sheet(Path(path), key, name)
        adjudications = (
            {str(r["id"]): str(r["gold_label"]) for r in load_jsonl(args.adjudications)}
            if args.adjudications
            else None
        )
        gold = merge_annotations(imported, adjudications=adjudications)
        write_jsonl(gold, args.out)
        pending = sum(1 for g in gold if g["gold_label"] is None)
        print(f"wrote {len(gold)} gold records to {args.out} ({pending} without gold label)")
        return 0
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
