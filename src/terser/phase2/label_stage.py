"""``judge`` + ``label`` stages at dataset scale (plan rows **2.3.3** and **2.3.4**).

For every (prompt, candidate, target) the target's output for the candidate is compared with its
output for the original prompt:

1. **Judge** (row 2.3.3) with the ADR-0005 protocol ``pairwise_avg``
   (:func:`terser.judge.protocols.judge_pair_averaged`): ``pairwise_v1`` with the category rubric
   in both A/B orders, A/B/tie probabilities averaged over the two orders. ``creative`` prompts
   have several sampled outputs (row 2.2.3); sample ``i`` of the candidate is judged against
   sample ``i`` of the original and the side probabilities are averaged over samples too. Pairs
   whose outputs are byte-identical are a tie without a judge call.
2. **Task metric** (row 2.3.1) for ``code`` (unit tests), ``fact`` (EM / token F1 against the
   source reference) and ``reason`` (final answer), on the greedy sample;
   :func:`terser.judge.metrics.not_worse` gives the metric half of the rule.
3. **Label** (row 2.3.4) with :func:`terser.judge.labels.make_label_record` on the averaged side
   (used for both orders), so every record is a validated ``label_v1`` record. Extra fields keep
   the per-sample raw verdicts, the averaged probabilities and margin, the task metrics and the
   review flags (``position_inconsistent``, ``low_margin``).

Output: ``labels.jsonl`` (one record per (prompt, candidate, target)) with ``candidate_id`` and
``target`` columns as :mod:`terser.phase2.oracle` expects. Judge requests are disk-cached by
:class:`terser.llm.LLMClient` and task-metric results under ``.cache/task_metrics``, so a rerun
is cheap and resumable.

CLI::

    python -m terser.phase2.label_stage --prompts data/interim/label_prompts.jsonl \\
        --responses data/interim/responses/qwen2.5-7b.jsonl \\
        data/interim/responses/phi3.5-mini.jsonl \\
        --judge-model Qwen/Qwen2.5-32B-Instruct-AWQ --out data/processed/labels.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from terser.judge.labels import make_label_record
from terser.judge.pairwise import DEFAULT_PROMPT_VERSION
from terser.judge.protocols import LOW_MARGIN, averaged_judge, judge_pair_averaged

if TYPE_CHECKING:
    from terser.judge.pairwise import PairwiseJudge, Side

__all__ = [
    "LabelStats",
    "TaskReferences",
    "aggregate_sides",
    "load_task_references",
    "main",
    "run_labels",
]

ORIGINAL_ID = "original"
METRIC_CATEGORIES = ("code", "fact", "reason")
_SIDES: tuple[Side, ...] = ("original", "rewrite", "tie")


# --------------------------------------------------------------------------------------------
# Task references


class TaskReferences:
    """Per-``source_id`` reference for the task metric: fact text, reason answer, code tests."""

    def __init__(self, refs: Mapping[str, Any]) -> None:
        self.refs = dict(refs)

    def get(self, source_id: str) -> Any:
        """The metric reference for a source id, or None."""
        return self.refs.get(source_id)


def _gsm8k_answer(solution: str) -> str:
    return solution.split("####")[-1].strip() if "####" in solution else solution.strip()


def load_task_references(
    prompts: Iterable[Mapping[str, Any]], raw_dir: Path = Path("data/raw")
) -> TaskReferences:
    """Build metric references for the prompts' sources.

    ``fact``: the source reference text; ``reason``: the GSM8K final answer (after ``####``);
    ``code``: :class:`terser.judge.code_runner.CodeTests` from the raw HumanEval / MBPP rows.
    """
    from terser.data.sources import SOURCES, read_rows
    from terser.judge.code_runner import CodeTests

    refs: dict[str, Any] = {}
    need_code = set()
    for p in prompts:
        sid, cat, ref = str(p["source_id"]), str(p["category"]), p.get("reference")
        if cat == "fact" and ref:
            refs[sid] = str(ref)
        elif cat == "reason" and ref:
            refs[sid] = _gsm8k_answer(str(ref))
        elif cat == "code":
            need_code.add(sid)
    if any(s.startswith("humaneval:") for s in need_code):
        for row in read_rows(SOURCES["humaneval"], raw_dir):
            sid = f"humaneval:{row['task_id']}"
            if sid in need_code:
                refs[sid] = CodeTests.humaneval(str(row["test"]), str(row["entry_point"]))
    if any(s.startswith("mbpp:") for s in need_code):
        for row in read_rows(SOURCES["mbpp"], raw_dir):
            sid = f"mbpp:{int(row['task_id'])}"
            if sid in need_code:
                raw_tests = row.get("test_list")
                tests = [] if raw_tests is None else [str(t) for t in raw_tests]
                setup = str(row.get("test_setup_code") or "")
                if tests:
                    refs[sid] = CodeTests.mbpp(tests, setup)
    return TaskReferences(refs)


class _MetricCache:
    """Disk cache of task-metric results keyed by (category, output, reference)."""

    def __init__(self, root: Path | None) -> None:
        self.root = root

    def _key(self, category: str, output: str, reference: Any) -> str:
        ref = repr(reference)
        blob = json.dumps([category, output, ref], ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def compute(self, category: str, output: str, reference: Any) -> Any:
        from terser.judge.metrics import MetricResult, task_metric

        path = None
        if self.root is not None:
            key = self._key(category, output, reference)
            path = self.root / key[:2] / f"{key}.json"
            if path.is_file():
                d = json.loads(path.read_text(encoding="utf-8"))
                from terser.data.categories import parse_category

                return MetricResult(
                    parse_category(d["category"]), d["name"], d["value"], d["passed"], d["details"]
                )
        result = task_metric(category, output, reference)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(result.to_dict(), default=str), encoding="utf-8")
            tmp.replace(path)
        return result


# --------------------------------------------------------------------------------------------
# Aggregation


def aggregate_sides(
    probs: Sequence[Mapping[str, float] | None],
) -> tuple[Side | None, dict[str, float] | None]:
    """Average per-sample side probabilities; argmax with ties → tie, then original."""
    if not probs or any(p is None for p in probs):
        return None, None
    avg: dict[str, float] = {
        s: sum(float(p[s]) for p in probs if p is not None) / len(probs) for s in _SIDES
    }
    preference: tuple[Side, ...] = ("tie", "original", "rewrite")
    best = max(preference, key=lambda s: (avg[s], -preference.index(s)))
    return best, avg


@dataclass
class LabelStats:
    """Accounting of one run."""

    records: int = 0
    judged_pairs: int = 0
    identical_pairs: int = 0
    calls: int = 0
    cached_calls: int = 0
    generated_tokens: int = 0
    raw_disagreements: int = 0
    raw_parsed: int = 0
    low_margin: int = 0
    wall_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form."""
        return dict(self.__dict__)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def run_labels(
    prompts: Sequence[Mapping[str, Any]],
    responses: Iterable[Mapping[str, Any]],
    judge: PairwiseJudge,
    *,
    out_path: Path,
    references: TaskReferences | None = None,
    metric_cache: Path | None = None,
    workers: int = 16,
    splits: Mapping[str, str] | None = None,
) -> LabelStats:
    """Judge and label every (prompt, candidate, target) in ``responses``; write ``out_path``."""
    stats = LabelStats()
    lock = threading.Lock()
    start = time.perf_counter()
    by_prompt = {str(p["id"]): p for p in prompts}
    # outputs[(target, prompt_id, candidate_id)][sample] = output
    outputs: dict[tuple[str, str, str], dict[int, str]] = defaultdict(dict)
    for r in responses:
        if r.get("stand_in") in (True, "True", "true"):
            raise ValueError(f"stand-in response {r.get('candidate_id')!r} must never be labelled")
        key = (str(r["target"]), str(r["prompt_id"]), str(r["candidate_id"]))
        outputs[key][int(r["sample"])] = str(r["output"])
    pairs = sorted(k for k in outputs if k[2] != ORIGINAL_ID and k[1] in by_prompt)

    # 1) judge: every (pair, sample) with distinct outputs, concurrently.
    jobs: list[tuple[str, str, str, str, str, str]] = []
    for target, pid, cid in pairs:
        orig = outputs.get((target, pid, ORIGINAL_ID), {})
        for sample, cand_out in outputs[(target, pid, cid)].items():
            if sample in orig and orig[sample] != cand_out:
                p = by_prompt[pid]
                jobs.append(
                    (str(p["prompt"]), orig[sample], cand_out, str(p["category"]), pid, cid)
                )
    verdicts: dict[tuple[str, str, str], Any] = {}

    def judge_one(job: tuple[str, str, str, str, str, str]) -> None:
        prompt, o, c, cat, _, _ = job
        res = judge_pair_averaged(judge, prompt, o, c, cat)
        with lock:
            verdicts[(prompt, o, c)] = res
            for call in (res.pairwise.original_first, res.pairwise.rewrite_first):
                stats.calls += call.attempts
                stats.cached_calls += call.attempts if call.cached else 0
                stats.generated_tokens += call.output_tokens

    unique = {(j[0], j[1], j[2]): j for j in jobs}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(judge_one, unique.values()))

    # 2) task metrics on the greedy sample, concurrently (code runs in a sandbox subprocess).
    mcache = _MetricCache(metric_cache)
    # Keyed by the reference as well: two prompts can share a byte-identical output (an empty
    # answer, a refusal) while having different references.
    metric_jobs: dict[tuple[str, str, str], tuple[str, str, Any]] = {}
    for target, pid, cid in pairs:
        p = by_prompt[pid]
        cat = str(p["category"])
        ref = references.get(str(p["source_id"])) if references else None
        if cat not in METRIC_CATEGORIES or ref is None:
            continue
        for c in (ORIGINAL_ID, cid):
            out = outputs.get((target, pid, c), {}).get(0)
            if out is not None:
                metric_jobs[(cat, out, repr(ref))] = (cat, out, ref)
    metrics: dict[tuple[str, str, str], Any] = {}

    def metric_one(key: tuple[str, str, str]) -> None:
        cat, out, ref = metric_jobs[key]
        res = mcache.compute(cat, out, ref)
        with lock:
            metrics[key] = res

    with ThreadPoolExecutor(max_workers=max(1, min(workers, 16))) as pool:
        list(pool.map(metric_one, list(metric_jobs)))

    # 3) label records in deterministic order.
    from terser.judge.metrics import not_worse

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as fh:
        for target, pid, cid in pairs:
            p = by_prompt[pid]
            cat = str(p["category"])
            orig = outputs.get((target, pid, ORIGINAL_ID), {})
            cand = outputs[(target, pid, cid)]
            per_sample: list[dict[str, Any]] = []
            probs: list[Mapping[str, float] | None] = []
            raw_inconsistent = False
            identical_all = True
            for sample in sorted(cand):
                if sample not in orig:
                    probs.append(None)
                    continue
                if orig[sample] == cand[sample]:
                    probs.append({"original": 0.0, "rewrite": 0.0, "tie": 1.0})
                    per_sample.append({"sample": sample, "identical": True})
                    continue
                identical_all = False
                res = verdicts[(str(p["prompt"]), orig[sample], cand[sample])]
                probs.append(res.probs)
                a, b = res.pairwise.original_first.side, res.pairwise.rewrite_first.side
                if a is not None and b is not None:
                    stats.raw_parsed += 1
                    if a != b:
                        stats.raw_disagreements += 1
                        raw_inconsistent = True
                per_sample.append(
                    {
                        "sample": sample,
                        "identical": False,
                        "original_first": a,
                        "rewrite_first": b,
                        "probs_original_first": res.pairwise.original_first.verdict_probs,
                        "probs_rewrite_first": res.pairwise.rewrite_first.verdict_probs,
                        "reason_original_first": res.pairwise.original_first.reason,
                        "reason_rewrite_first": res.pairwise.rewrite_first.reason,
                    }
                )
            side, avg = aggregate_sides(probs)
            margin = None
            if avg is not None:
                top, second = sorted(avg.values(), reverse=True)[:2]
                margin = top - second
            task_nw: bool | None = None
            ref_key = repr(references.get(str(p["source_id"])) if references else None)
            m_orig = metrics.get((cat, orig.get(0, "\0"), ref_key))
            m_cand = metrics.get((cat, cand.get(0, "\0"), ref_key))
            if m_orig is not None and m_cand is not None:
                task_nw = not_worse(m_orig, m_cand)
                if m_orig.value is None and m_cand.value is None:
                    task_nw = None
            reasons = []
            if raw_inconsistent:
                reasons.append("position_inconsistent")
            if not identical_all and (margin is None or margin < LOW_MARGIN):
                reasons.append("low_margin")
                stats.low_margin += 1
            stats.judged_pairs += int(not identical_all)
            stats.identical_pairs += int(identical_all)
            rec = make_label_record(
                id=f"{cid}@{target}",
                prompt_id=pid,
                category=cat,
                target_model=target,
                judge_model=judge.model,
                judge_prompt_version=judge.prompt_version,
                task_not_worse=task_nw,
                verdict_ab=side,
                verdict_ba=side,
                stand_in=False,
                extra={
                    "candidate_id": cid,
                    "target": target,
                    "split": (splits or {}).get(pid),
                    "judge_protocol": "pairwise_avg",
                    "aggregate": {"side": side, "probs": avg, "margin": margin},
                    "samples": per_sample,
                    "identical_outputs": identical_all,
                    "task_metric": {
                        "original": None if m_orig is None else m_orig.value,
                        "rewrite": None if m_cand is None else m_cand.value,
                    },
                    "needs_human_review": bool(reasons),
                    "review_reasons": reasons,
                },
            )
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            stats.records += 1
    stats.wall_seconds = time.perf_counter() - start
    return stats


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(prog="python -m terser.phase2.label_stage", description=__doc__)
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--responses", type=Path, nargs="+", required=True)
    ap.add_argument("--judge-model", default="Qwen/Qwen2.5-32B-Instruct-AWQ")
    ap.add_argument("--provider", default="local")
    ap.add_argument("--prompt-version", default=DEFAULT_PROMPT_VERSION)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--splits-dir", type=Path, default=None, help="tag records with their split")
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--metric-cache", type=Path, default=Path(".cache/task_metrics"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="first N prompts (smoke)")
    ap.add_argument("--budget-phase", default="phase2")
    ap.add_argument("--hardware", default="rtx4500ada")
    ap.add_argument("--stats", type=Path, default=None)
    ap.add_argument("--track", action="store_true")
    args = ap.parse_args(argv)

    from terser.llm import LLMClient

    prompts = _read_jsonl(args.prompts)[: args.limit]
    keep = {str(p["id"]) for p in prompts}
    responses = [r for path in args.responses for r in _read_jsonl(path) if r["prompt_id"] in keep]
    splits: dict[str, str] = {}
    if args.splits_dir is not None:
        for name in ("train", "val", "test"):
            path = args.splits_dir / f"{name}.jsonl"
            if path.is_file():
                splits.update({str(r["id"]): name for r in _read_jsonl(path)})
    client = LLMClient(args.budget_phase)
    judge = averaged_judge(
        client, provider=args.provider, model=args.judge_model, prompt_version=args.prompt_version
    )
    refs = load_task_references(prompts, args.raw_dir)
    kwargs: dict[str, Any] = {
        "out_path": args.out,
        "references": refs,
        "metric_cache": args.metric_cache,
        "workers": args.workers,
        "splits": splits,
    }
    if args.track:
        import mlflow

        from terser.tracking import start_run

        name = args.judge_model.split("/")[-1].replace("-", "_")
        with start_run(
            phase=2,
            row="2.3.3",
            task="judge",
            run_name=f"2-judge-{name}-pairwiseavg-0",
            hardware=args.hardware,
            params={"judge_model": args.judge_model, "n_prompts": len(prompts)},
        ):
            stats = run_labels(prompts, responses, judge, **kwargs)
            mlflow.log_metrics(
                {
                    **{k: float(v) for k, v in stats.to_dict().items()},
                    "gpu_hours": stats.wall_seconds / 3600 if args.hardware != "cpu" else 0.0,
                }
            )
    else:
        stats = run_labels(prompts, responses, judge, **kwargs)
    summary = stats.to_dict()
    if args.stats is not None:
        args.stats.parent.mkdir(parents=True, exist_ok=True)
        args.stats.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), **summary}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
