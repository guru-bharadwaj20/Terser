"""Val report for SFT rewriter rungs (rows **8.3.3** / **8.3.4**, gate **C8.3**).

Inputs are the :mod:`terser.models.rewriter.generate` outputs (one file per rung) and the judge
labels of those rewrites from :mod:`terser.phase2.label_stage` (candidate ids ``<prompt>:<tag>``).
A rewrite that fell back to the original prompt is preserved by construction.

C8.3: median savings >= 25% on every target; preservation >= 92% overall and >= 85% per category.

Entry point::

    python -m terser.models.rewriter.val_report \
        --rung sft05=data/interim/rewrites/sft-qwen0.5b-full.val.jsonl \
        --labels data/interim/rewrites/val_labels.jsonl --out reports/p8_rewriter_val.md
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

__all__ = ["main", "rung_metrics"]

MIN_SAVINGS = 0.25
MIN_PRESERVED = 0.92
MIN_PRESERVED_CATEGORY = 0.85


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def rung_metrics(
    rewrites: Sequence[dict[str, Any]], labels: Sequence[dict[str, Any]], tag: str
) -> dict[str, Any]:
    """Savings and preservation for one rung; ``labels`` may hold several rungs and targets."""
    ok: dict[tuple[str, str], bool] = {}
    for r in labels:
        pid, _, t = r["candidate_id"].rpartition(":")
        if t == tag:
            ok[(pid, r["target"])] = r["label"] == "preserved"
    targets = sorted({t for _, t in ok})
    by_cat: dict[str, list[bool]] = defaultdict(list)
    missing = 0
    for w in rewrites:
        for t in targets:
            if w["fell_back"]:
                by_cat[w["category"]].append(True)
            elif (w["id"], t) in ok:
                by_cat[w["category"]].append(ok[(w["id"], t)])
            else:
                missing += 1  # not judged: left out of the rate, and reported
    flat = [x for xs in by_cat.values() for x in xs]
    per_cat = {c: sum(xs) / len(xs) for c, xs in sorted(by_cat.items())}
    m = {
        "n": len(rewrites),
        "targets": targets,
        "median_savings": statistics.median(w["savings"] for w in rewrites),
        "oracle_median_savings": statistics.median(w["oracle_savings"] for w in rewrites),
        "fell_back": sum(w["fell_back"] for w in rewrites),
        "latency_s": statistics.mean(w["latency_s"] for w in rewrites),
        "preserved": sum(flat) / len(flat) if flat else 0.0,
        "preserved_by_category": per_cat,
        "missing_labels": missing,
    }
    m["c8_3"] = (
        m["median_savings"] >= MIN_SAVINGS
        and m["preserved"] >= MIN_PRESERVED
        and all(v >= MIN_PRESERVED_CATEGORY for v in per_cat.values())
    )
    return m


def _markdown(results: dict[str, dict[str, Any]]) -> str:
    lines = [
        "# Phase 8 rewriter val report (rows 8.3.3 / 8.3.4, gate C8.3)",
        "",
        "Val split, greedy decoding, 8.3.7 safeguard on. Savings on the Qwen2.5 tokenizer;"
        " preservation judged by Qwen2.5-32B-Instruct-AWQ (`pairwise_avg`) on every target.",
        "",
        "| Rung | n | Median savings | Oracle median | Fell back | Preserved | Latency (s)"
        " | C8.3 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for tag, m in results.items():
        lines.append(
            f"| {tag} | {m['n']} | {m['median_savings']:.1%} | {m['oracle_median_savings']:.1%} "
            f"| {m['fell_back']} | {m['preserved']:.1%} | {m['latency_s']:.2f} "
            f"| {'pass' if m['c8_3'] else 'fail'} |"
        )
    cats = sorted({c for m in results.values() for c in m["preserved_by_category"]})
    lines += ["", "Preservation by category:", "", "| Rung | " + " | ".join(cats) + " |"]
    lines.append("|---" * (len(cats) + 1) + "|")
    for tag, m in results.items():
        pc = m["preserved_by_category"]
        lines.append(f"| {tag} | " + " | ".join(f"{pc.get(c, 0):.1%}" for c in cats) + " |")
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="C8.3 val report for SFT rewriter rungs.")
    ap.add_argument("--rung", action="append", required=True, help="tag=path to generate output")
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("reports/p8_rewriter_val.md"))
    args = ap.parse_args(argv)

    labels = _jsonl(args.labels)
    results = {}
    for spec in args.rung:
        tag, _, path = spec.partition("=")
        results[tag] = rung_metrics(_jsonl(Path(path)), labels, tag)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(_markdown(results), encoding="utf-8")
    args.out.with_suffix(".json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                k: {"savings": v["median_savings"], "preserved": v["preserved"], "c8_3": v["c8_3"]}
                for k, v in results.items()
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
