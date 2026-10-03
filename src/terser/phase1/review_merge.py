"""Row 1.3.6: merge the human-reviewed sheet back into the pilot labels.

The pilot pipeline (:mod:`terser.phase1.pilot_labels`) writes ``labels.jsonl`` and a review sheet
``labels_review.csv`` with empty ``human_label`` / ``reviewer`` / ``notes`` columns. A reviewer
fills those in (``human_label`` in ``{preserved, degraded}``, ``reviewer`` = their handle). This
module copies the filled columns into ``labels.jsonl`` by ``id`` so that
:mod:`terser.phase1.version_spaces` (row 1.3.7, check C1.3) uses the human label, and prints the
review progress and judge-vs-human agreement.

Rules:

- A row with a ``human_label`` must name a ``reviewer``; a row with a reviewer but no label is
  rejected too (a half-filled row is a mistake, not a review).
- Ids must match ``labels.jsonl`` one to one; an unknown id or a changed ``rewrite`` / ``original``
  text means the sheet belongs to another labelling run and the merge is refused.
- Rows left blank keep whatever ``labels.jsonl`` already has, so the sheet can be filled in batches.

Row 1.3.6 is done when every label has a ``human_label`` (``--require-complete`` exits non-zero
otherwise). CLI::

    python -m terser.phase1.review_merge --labels data/pilot/labels.jsonl \\
        --review data/pilot/labels_review.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_LABELS = Path("data/pilot/labels.jsonl")
DEFAULT_REVIEW = Path("data/pilot/labels_review.csv")
HUMAN_LABELS = frozenset({"preserved", "degraded"})


@dataclass(frozen=True)
class MergeSummary:
    """Review progress after a merge."""

    total: int
    reviewed: int
    flagged: int
    flagged_reviewed: int
    agree: int
    reviewers: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return self.reviewed == self.total

    @property
    def agreement(self) -> float | None:
        return self.agree / self.reviewed if self.reviewed else None

    def lines(self) -> list[str]:
        agreement = "n/a" if self.agreement is None else f"{self.agreement:.1%}"
        return [
            f"reviewed: {self.reviewed}/{self.total}",
            f"flagged reviewed: {self.flagged_reviewed}/{self.flagged}",
            f"judge-human agreement: {agreement} ({self.agree}/{self.reviewed})",
            f"reviewers: {', '.join(self.reviewers) or 'none'}",
            "row 1.3.6: " + ("COMPLETE" if self.complete else "pending"),
        ]


def _read_labels(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _read_review(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def merge(labels: list[dict[str, Any]], review: Sequence[dict[str, str]]) -> MergeSummary:
    """Copy filled review rows into ``labels`` in place and summarise progress.

    Raises:
        ValueError: On an unknown or duplicate id, mismatched text, an invalid label, or a
            half-filled row. Nothing is modified when this is raised.
    """
    by_id = {str(rec["id"]): rec for rec in labels}
    updates: dict[str, dict[str, str]] = {}
    errors: list[str] = []
    for lineno, row in enumerate(review, 2):
        rid = (row.get("id") or "").strip()
        if rid not in by_id:
            errors.append(f"line {lineno}: unknown id {rid!r}")
            continue
        if rid in updates:
            errors.append(f"line {lineno}: duplicate id {rid!r}")
            continue
        rec = by_id[rid]
        for key in ("original", "rewrite"):
            if key in row and row[key] != rec.get(key):
                errors.append(f"line {lineno}: {key} differs from labels for {rid!r}")
        label = (row.get("human_label") or "").strip().lower()
        reviewer = (row.get("reviewer") or "").strip()
        if not label and not reviewer:
            continue
        if label not in HUMAN_LABELS:
            errors.append(f"line {lineno}: human_label {label!r} not in {sorted(HUMAN_LABELS)}")
        elif not reviewer:
            errors.append(f"line {lineno}: human_label without reviewer for {rid!r}")
        else:
            updates[rid] = {
                "human_label": label,
                "reviewer": reviewer,
                "review_notes": (row.get("notes") or "").strip(),
            }
    if errors:
        raise ValueError("review sheet rejected:\n" + "\n".join(errors[:50]))
    for rid, fields in updates.items():
        by_id[rid].update(fields)
    return summarise(labels)


def summarise(labels: Sequence[dict[str, Any]]) -> MergeSummary:
    reviewed = [r for r in labels if r.get("human_label")]
    flagged = [r for r in labels if r.get("needs_human_review")]
    return MergeSummary(
        total=len(labels),
        reviewed=len(reviewed),
        flagged=len(flagged),
        flagged_reviewed=sum(1 for r in flagged if r.get("human_label")),
        agree=sum(1 for r in reviewed if r["human_label"] == r.get("label")),
        reviewers=tuple(sorted({str(r["reviewer"]) for r in reviewed if r.get("reviewer")})),
    )


def _write_labels(path: Path, labels: Sequence[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        for rec in labels:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    p.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    p.add_argument("--dry-run", action="store_true", help="validate and report; write nothing")
    p.add_argument(
        "--require-complete", action="store_true", help="exit 1 unless every label is reviewed"
    )
    args = p.parse_args(argv)
    labels = _read_labels(args.labels)
    try:
        summary = merge(labels, _read_review(args.review))
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    if not args.dry_run:
        _write_labels(args.labels, labels)
    print("\n".join(summary.lines()))
    return 1 if args.require_complete and not summary.complete else 0


if __name__ == "__main__":
    raise SystemExit(main())
