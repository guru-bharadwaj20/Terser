"""Failure analysis of degraded rewriter outputs (plan row **8.3.10**).

A *degraded case* is a ``(prompt, target)`` pair where the rung produced a rewrite (it did not fall
back to the original under the 8.3.7 safeguard) and the judge labelled it ``degraded``. Labels use
the candidate ids of :mod:`terser.models.rewriter.val_report` (``<prompt_id>:<rung tag>``).

``sample`` draws a seeded sample stratified by prompt category (largest-remainder allocation,
proportional to the degraded pool). Each case is then put in one failure category
(:data:`FAILURES`) with a one-line note, and ``report`` turns the annotation file into
``reports/p8_failures.md``. Anything below the ``<!-- manual -->`` marker of an existing report is
kept, so hand-written discussion survives a re-run.

Annotation records (``data/annotations/p8_failures.jsonl``, ``p12_errors.jsonl``): ``excerpt`` is
optional; annotators add it only for cases worth quoting, and ``report`` shows up to three
excerpts per failure category. ``other`` means the rewrite was judged degraded but none of the
four named failures fits (e.g. a judge disagreement the annotator could not attribute, or a
formatting change); its ``note`` says what was seen.

Entry points::

    python -m terser.models.rewriter.failures sample \
        --rung sft3b=data/interim/rewrites/sft-qwen3b-lora.val.jsonl \
        --labels data/interim/rewrites/val_labels_all_v2.jsonl \
        --prompts data/interim/rewrites/val_prompts.jsonl --n 100 --seed 0 --out sample.jsonl
    python -m terser.models.rewriter.failures report \
        --annotations data/annotations/p8_failures.jsonl --out reports/p8_failures.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

__all__ = ["FAILURES", "allocate", "degraded_cases", "main", "render", "sample"]

FAILURES = ("dropped constraint", "lost tone", "dropped example", "changed meaning", "other")
MANUAL = "<!-- manual -->"


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def short_hash(text: str) -> str:
    """First 10 hex digits of the SHA-256 of ``text`` (id-safe reference to a prompt)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]


def degraded_cases(
    rewrites: Sequence[dict[str, Any]], labels: Sequence[dict[str, Any]], tag: str
) -> list[dict[str, Any]]:
    """Non-fallback ``(prompt, target)`` pairs of rung ``tag`` that the judge labelled degraded."""
    by_id = {w["id"]: w for w in rewrites}
    seen: set[tuple[str, str]] = set()
    out = []
    for r in labels:
        pid, _, t = r["candidate_id"].rpartition(":")
        if t != tag or r["label"] != "degraded" or (pid, r["target"]) in seen:
            continue
        w = by_id.get(pid)
        if w is None or w["fell_back"]:
            continue
        seen.add((pid, r["target"]))
        out.append(
            {
                "prompt_id": pid,
                "target": r["target"],
                "category": w["category"],
                "rung": tag,
                "judge_reason": r.get("reason"),
                "savings": w["savings"],
                "rewrite": w["rewrite"],
            }
        )
    return sorted(out, key=lambda c: (c["rung"], c["prompt_id"], c["target"]))


def allocate(sizes: dict[str, int], n: int) -> dict[str, int]:
    """Proportional allocation of ``n`` over strata (largest remainder, ties by name)."""
    total = sum(sizes.values())
    if n >= total:
        return dict(sizes)
    quota = {k: n * v / total for k, v in sizes.items()}
    alloc = {k: int(q) for k, q in quota.items()}
    rest = sorted(sizes, key=lambda k: (-(quota[k] - alloc[k]), k))
    for k in rest[: n - sum(alloc.values())]:
        alloc[k] += 1
    return alloc


def sample(cases: Sequence[dict[str, Any]], n: int, seed: int = 0) -> list[dict[str, Any]]:
    """Seeded sample of ``n`` cases stratified by prompt category."""
    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for c in cases:
        strata[c["category"]].append(c)
    alloc = allocate({k: len(v) for k, v in strata.items()}, n)
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    for k in sorted(strata):
        out += rng.sample(strata[k], alloc[k])
    return sorted(out, key=lambda c: (c["category"], c["prompt_id"], c["target"]))


def _pct(k: int, n: int) -> str:
    return f"{k / n:.0%}" if n else "-"


def _crosstab(records: Sequence[dict[str, Any]], key: str) -> list[str]:
    cols = sorted({r[key] for r in records})
    counts = Counter((r[key], r["failure"]) for r in records)
    lines = [
        f"| Failure | {' | '.join(cols)} |",
        "| :--- |" + " ---: |" * len(cols),
    ]
    for f in FAILURES:
        cells = []
        for c in cols:
            tot = sum(counts[(c, g)] for g in FAILURES)
            cells.append(f"{counts[(c, f)]} ({_pct(counts[(c, f)], tot)})")
        lines.append(f"| {f} | {' | '.join(cells)} |")
    totals = [str(sum(counts[(c, g)] for g in FAILURES)) for c in cols]
    lines.append(f"| n | {' | '.join(totals)} |")
    return lines


def render(
    records: Sequence[dict[str, Any]],
    pool: dict[str, Any] | None = None,
    title: str = "Phase 8 rewriter failure analysis (row 8.3.10)",
    split: str = "val",
    annotations: str = "data/annotations/p8_failures.jsonl",
) -> str:
    """Generated part of the report (everything above the manual marker).

    Defaults reproduce ``reports/p8_failures.md``; row 12.8 passes its own title, split and
    annotation path for ``reports/final_errors.md``.
    """
    n = len(records)
    bad = [r for r in records if r["failure"] not in FAILURES]
    if bad:
        raise ValueError(f"unknown failure category in {bad[0]['prompt_id']}: {bad[0]['failure']}")
    annotators = sorted({r["annotator"] for r in records})
    rungs = sorted({r["rung"] for r in records})
    cnt = Counter(r["failure"] for r in records)
    lines = [
        f"# {title}",
        "",
        "> **Annotator caveat.** Every case below was categorised by an LLM assistant (Claude), not"
        " by a human annotator. The categories and notes have **not been human-reviewed**. Treat"
        " the shares as a first-pass triage, not as gold labels.",
        "",
        f"Annotator field: {', '.join(f'`{a}`' for a in annotators)}. Rung(s): "
        f"{', '.join(f'`{r}`' for r in rungs)}. Cases: {n} degraded (prompt, target) pairs on "
        f"{len({r['prompt_id'] for r in records})} unique {split} prompts.",
        "",
    ]
    if pool:
        lines += [
            f"Population: {pool['n_degraded']} non-fallback degraded pairs "
            f"({pool['n_prompts']} prompts); sample of {n}, stratified by prompt category, "
            f"seed {pool['seed']}. Annotations: "
            f"[{annotations}](../{annotations}).",
            "",
        ]
    sav: dict[str, list[float]] = defaultdict(list)
    for x in records:
        if x.get("savings") is not None:
            sav[x["failure"]].append(x["savings"])
    lines += [
        "## Counts",
        "",
        "| Failure category | n | Share | Median rewrite savings |",
        "| :--- | ---: | ---: | ---: |",
    ]
    for f in FAILURES:
        med = f"{statistics.median(sav[f]):.0%}" if sav[f] else "-"
        lines.append(f"| {f} | {cnt[f]} | {_pct(cnt[f], n)} | {med} |")
    lines += [f"| total | {n} | 100% | |", ""]
    tags = Counter((t, x["failure"]) for x in records for t in x.get("tags", []))
    if tags:
        lines += [
            "Secondary tags (a case can carry several):",
            "",
            "| Tag | Failure category | n |",
            "| :--- | :--- | ---: |",
        ]
        lines += [f"| `{t}` | {f} | {k} |" for (t, f), k in sorted(tags.items())]
        lines.append("")
    lines += ["## By prompt category", ""] + _crosstab(records, "category") + [""]
    lines += ["## By target", ""] + _crosstab(records, "target") + [""]
    if all("judge_reason" in x for x in records):
        lines += ["## By judge reason", ""] + _crosstab(records, "judge_reason") + [""]
    lines += ["## Examples", ""]
    for f in FAILURES:
        ex = [r for r in records if r["failure"] == f and r.get("excerpt")][:3]
        lines += [f"### {f}", ""]
        if not ex:
            lines += ["No example selected.", ""]
        for r in ex:
            sha = f" (sha `{r['prompt_sha']}`)" if "prompt_sha" in r else ""
            ref = f"`{r['prompt_id']}`{sha}"
            lines.append(f"- {ref}, {r['target'].split('/')[-1]}: {r['note']}")
            lines.append(f"  {r['excerpt']}")
        lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Row 8.3.10 failure analysis.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample", help="draw the stratified sample of degraded cases")
    s.add_argument("--rung", action="append", required=True, help="tag=path to generate output")
    s.add_argument("--labels", type=Path, action="append", required=True)
    s.add_argument("--prompts", type=Path, help="val_prompts.jsonl, adds prompt text and hash")
    s.add_argument("--n", type=int, default=100)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("report", help="render the annotation file as markdown")
    r.add_argument("--annotations", type=Path, required=True)
    r.add_argument("--pool", type=Path, help="JSON written by sample (population stats)")
    r.add_argument("--out", type=Path, default=Path("reports/p8_failures.md"))
    r.add_argument("--title", default="Phase 8 rewriter failure analysis (row 8.3.10)")
    r.add_argument("--split", default="val")
    args = ap.parse_args(argv)

    if args.cmd == "sample":
        labels = [x for p in args.labels for x in _jsonl(p)]
        cases: list[dict[str, Any]] = []
        for spec in args.rung:
            tag, _, path = spec.partition("=")
            cases += degraded_cases(_jsonl(Path(path)), labels, tag)
        picked = sample(cases, args.n, args.seed)
        if args.prompts:
            text = {p["id"]: p["prompt"] for p in _jsonl(args.prompts)}
            for c in picked:
                c["prompt"] = text[c["prompt_id"]]
                c["prompt_sha"] = short_hash(c["prompt"])
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("".join(json.dumps(c) + "\n" for c in picked), encoding="utf-8")
        pool = {
            "n_degraded": len(cases),
            "n_prompts": len({c["prompt_id"] for c in cases}),
            "by_category": dict(sorted(Counter(c["category"] for c in cases).items())),
            "by_target": dict(sorted(Counter(c["target"] for c in cases).items())),
            "rungs": sorted({c["rung"] for c in cases}),
            "n": len(picked),
            "seed": args.seed,
            "sample_by_category": dict(sorted(Counter(c["category"] for c in picked).items())),
        }
        args.out.with_suffix(".pool.json").write_text(json.dumps(pool, indent=2) + "\n")
        print(json.dumps(pool))
        return 0

    records = _jsonl(args.annotations)
    stats = json.loads(args.pool.read_text()) if args.pool else None
    md = render(records, stats, args.title, args.split, args.annotations.as_posix())
    if args.out.exists() and MANUAL in (old := args.out.read_text(encoding="utf-8")):
        md += "\n" + MANUAL + old.split(MANUAL, 1)[1]
    else:
        md += "\n" + MANUAL + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md, encoding="utf-8")
    print(json.dumps(dict(Counter(r["failure"] for r in records))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
