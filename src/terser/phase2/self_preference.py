"""Judge self-preference control (plan row **2.2.5**, ADR-0011).

The main judge (Qwen2.5-32B-Instruct-AWQ) may prefer outputs of its own family. A second judge
from another family (``microsoft/Phi-3.5-mini-instruct``, ADR-0011) re-judges a sample of pairs
with the same ``pairwise_avg`` protocol (ADR-0005) and the verdict agreement between the two
judges is compared separately for each target family. If the main judge favoured Qwen-target
outputs, agreement would differ between the Qwen-target and the second-family-target pairs.

``run``: sample ``n`` pairs from the main ``labels.jsonl`` (only pairs that needed a judge call,
i.e. not byte-identical outputs, with a parsed verdict), stratified equally over targets with a
fixed seed; re-judge them with :func:`terser.phase2.label_stage.run_labels` and the second judge.

``report``: join both label files on ``id`` and write per-target agreement (averaged side and
``preserved`` label) and the gap between target families.

CLI::

    python -m terser.phase2.self_preference run --labels data/processed/labels.jsonl \\
        --prompts data/interim/label_prompts.jsonl --responses data/interim/responses/*.jsonl \\
        --judge-model microsoft/Phi-3.5-mini-instruct --n 1000 \\
        --out data/processed/labels_judge2.jsonl
    python -m terser.phase2.self_preference report --labels data/processed/labels.jsonl \\
        --second data/processed/labels_judge2.jsonl --out reports/p2_self_preference.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = ["agreement_table", "filter_responses", "main", "select_pairs"]

GAP_THRESHOLD_PP = 5.0


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _side(rec: Mapping[str, Any]) -> str | None:
    agg = rec.get("aggregate") or {}
    side = agg.get("side")
    return None if side is None else str(side)


def select_pairs(
    labels: Sequence[Mapping[str, Any]], n: int, *, seed: int = 0
) -> list[Mapping[str, Any]]:
    """``n`` judged label records, split equally over targets (seeded, deterministic)."""
    by_target: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for rec in labels:
        if rec.get("identical_outputs") or _side(rec) is None:
            continue
        by_target[str(rec["target"])].append(rec)
    if not by_target:
        return []
    rng = random.Random(seed)
    per = n // len(by_target)
    chosen: list[Mapping[str, Any]] = []
    for target in sorted(by_target):
        pool = sorted(by_target[target], key=lambda r: str(r["id"]))
        chosen.extend(rng.sample(pool, min(per, len(pool))))
    return chosen


def filter_responses(
    responses: Iterable[Mapping[str, Any]], pairs: Sequence[Mapping[str, Any]]
) -> list[Mapping[str, Any]]:
    """Responses needed to re-judge ``pairs``: the candidate's and the original's outputs."""
    want = set()
    for rec in pairs:
        target, pid = str(rec["target"]), str(rec["prompt_id"])
        want.add((target, pid, str(rec["candidate_id"])))
        want.add((target, pid, "original"))
    return [
        r
        for r in responses
        if (str(r["target"]), str(r["prompt_id"]), str(r["candidate_id"])) in want
    ]


def agreement_table(
    main_labels: Sequence[Mapping[str, Any]], second_labels: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Per-target agreement between two judges, and the gap between target families (pp)."""
    first = {str(r["id"]): r for r in main_labels}
    rows: dict[str, dict[str, Any]] = {}
    acc: dict[str, list[tuple[bool, bool, bool]]] = defaultdict(list)
    for rec in second_labels:
        m = first.get(str(rec["id"]))
        s1, s2 = (None, None) if m is None else (_side(m), _side(rec))
        if m is None or s1 is None or s2 is None:
            continue
        acc[str(rec["target"])].append(
            (s1 == s2, m["label"] == rec["label"], s1 == "rewrite" or s1 == "tie")
        )
    for target, items in sorted(acc.items()):
        k = len(items)
        rows[target] = {
            "pairs": k,
            "side_agreement": sum(a for a, _, _ in items) / k,
            "label_agreement": sum(b for _, b, _ in items) / k,
            "main_judge_tie_or_rewrite": sum(c for _, _, c in items) / k,
        }
    gap = None
    label_gap = None
    if len(rows) >= 2:
        vals = [r["side_agreement"] for r in rows.values()]
        gap = 100.0 * (max(vals) - min(vals))
        lvals = [r["label_agreement"] for r in rows.values()]
        label_gap = 100.0 * (max(lvals) - min(lvals))
    return {
        "per_target": rows,
        "side_agreement_gap_pp": gap,
        "label_agreement_gap_pp": label_gap,
        "threshold_pp": GAP_THRESHOLD_PP,
        "passes": gap is not None and gap <= GAP_THRESHOLD_PP,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(prog="python -m terser.phase2.self_preference")
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--labels", type=Path, required=True)
    run.add_argument("--prompts", type=Path, required=True)
    run.add_argument("--responses", type=Path, nargs="+", required=True)
    run.add_argument("--judge-model", required=True)
    run.add_argument("--provider", default="local")
    run.add_argument("--n", type=int, default=1000)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--workers", type=int, default=32)
    run.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    run.add_argument("--metric-cache", type=Path, default=Path(".cache/task_metrics"))
    run.add_argument("--budget-phase", default="phase2")
    run.add_argument("--out", type=Path, required=True)
    rep = sub.add_parser("report")
    rep.add_argument("--labels", type=Path, required=True)
    rep.add_argument("--second", type=Path, required=True)
    rep.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    if args.cmd == "run":
        from terser.judge.protocols import averaged_judge
        from terser.llm import LLMClient
        from terser.phase2.label_stage import load_task_references, run_labels

        pairs = select_pairs(_read_jsonl(args.labels), args.n, seed=args.seed)
        pids = {str(r["prompt_id"]) for r in pairs}
        prompts = [p for p in _read_jsonl(args.prompts) if str(p["id"]) in pids]
        responses = filter_responses(
            (r for path in args.responses for r in _read_jsonl(path)), pairs
        )
        judge = averaged_judge(
            LLMClient(args.budget_phase), provider=args.provider, model=args.judge_model
        )
        stats = run_labels(
            prompts,
            responses,
            judge,
            out_path=args.out,
            references=load_task_references(prompts, args.raw_dir),
            metric_cache=args.metric_cache,
            workers=args.workers,
        )
        print(json.dumps({"out": str(args.out), "pairs": len(pairs), **stats.to_dict()}, indent=2))
        return 0
    table = agreement_table(_read_jsonl(args.labels), _read_jsonl(args.second))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(table, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
