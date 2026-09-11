"""Judge validation report (plan row **2.3.6**; checkpoints C2.3-C2.6).

Joins the human gold file from row 2.3.5 (``data/gold/human_300.jsonl``, written by
:func:`terser.judge.annotation.merge_annotations`) with the ``label`` stage records of the same
pairs (row 2.3.4, :mod:`terser.judge.labels`) and writes ``reports/p2_judge_validation.md`` with:

* human-human Cohen's kappa (binary label; the 3-way side kappa is shown for information);
* judge-human kappa overall and per category, each with a bootstrap 95% CI;
* the judge as a degradation detector: sensitivity / specificity vs the human gold label;
* the position-bias rate (A/B-order disagreement) over all judge records and over the annotated
  subset;
* confusion tables and explicit PASS/FAIL lines for C2.3-C2.6.

The judge label compared with humans is, by default, the **judge-only** label (the 2.3.4 rule
with no task metric), because annotators judge the two outputs, not the task metric;
``--label-source final`` uses the stage's full label instead.

CLI::

    python -m terser.phase2.judge_validation --gold data/gold/human_300.jsonl \\
        --judge data/labels/labels.jsonl --out reports/p2_judge_validation.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from terser.eval.agreement import (
    CheckResult,
    KappaResult,
    checkpoint_checks,
    confusion_table,
    detector_metrics,
    kappa_with_ci,
    per_category_kappa,
    position_disagreement_rate,
)
from terser.judge.annotation import load_jsonl
from terser.judge.labels import label_pair

__all__ = [
    "DEFAULT_OUT",
    "JudgeValidationReport",
    "StandInJudgeError",
    "build_report",
    "judge_sides",
    "main",
    "run",
]

DEFAULT_OUT = Path("reports/p2_judge_validation.md")
LabelSource = Literal["judge", "final"]
_SIDES = ("original", "rewrite", "tie")


class StandInJudgeError(ValueError):
    """Judge records are stand-in (smoke) output and ``allow_stand_in`` is off."""


def judge_sides(record: Mapping[str, Any]) -> tuple[str | None, str | None]:
    """Return ``(original_first side, rewrite_first side)`` from a label record.

    Accepts sides stored as strings (``label_v1``) or as full ``JudgeCall`` dicts (pilot records).
    """
    verdicts = record.get("verdicts") or {}
    out: list[str | None] = []
    for order in ("original_first", "rewrite_first"):
        v = verdicts.get(order)
        if isinstance(v, Mapping):
            v = v.get("side")
        if v is not None and v not in _SIDES:
            raise ValueError(f"record {record.get('id')!r}: invalid side {v!r} for {order}")
        out.append(v)
    return out[0], out[1]


def _judge_label(record: Mapping[str, Any], source: LabelSource) -> str | None:
    if source == "final":
        label = record.get("label")
        return None if label is None else str(label)
    ab, ba = judge_sides(record)
    return label_pair(None, ab, ba).label  # type: ignore[arg-type]


@dataclass
class JudgeValidationReport:
    """Computed statistics, checkpoint lines and the rendered Markdown."""

    markdown: str
    checks: list[CheckResult]
    metrics: dict[str, float] = field(default_factory=dict)
    stand_in: bool = False

    @property
    def all_passed(self) -> bool:
        """True when every checkpoint line passes."""
        return all(c.passed for c in self.checks)


def _f(x: float) -> str:
    return "n/a" if math.isnan(x) else f"{x:.3f}"


def _kappa_row(name: str, k: KappaResult) -> str:
    return (
        f"| {name} | {k.n} | {k.n_dropped} | {_f(k.kappa)} | [{_f(k.ci_low)}, {_f(k.ci_high)}] "
        f"| {_f(k.observed_agreement)} |"
    )


def build_report(
    gold: Sequence[Mapping[str, Any]],
    judge_records: Sequence[Mapping[str, Any]],
    *,
    label_source: LabelSource = "judge",
    n_boot: int = 2000,
    seed: int = 0,
    allow_stand_in: bool = False,
    provenance: Mapping[str, str] | None = None,
) -> JudgeValidationReport:
    """Compute every 2.3.6 statistic and render the Markdown report.

    Raises:
        StandInJudgeError: If any judge record is stand-in output and ``allow_stand_in`` is off.
        ValueError: Duplicate ids, gold items without a judge record, or not exactly two
            annotators.
    """
    stand_in = any(bool(r.get("stand_in")) for r in judge_records)
    if stand_in and not allow_stand_in:
        raise StandInJudgeError("judge records are stand_in output; they cannot validate the judge")
    by_id: dict[str, Mapping[str, Any]] = {}
    for r in judge_records:
        rid = str(r["id"])
        if rid in by_id:
            raise ValueError(f"duplicate judge record id {rid!r}")
        by_id[rid] = r
    missing = [str(g["id"]) for g in gold if str(g["id"]) not in by_id]
    if missing:
        raise ValueError(f"{len(missing)} gold item(s) have no judge record, e.g. {missing[:3]}")
    annotators = sorted({n for g in gold for n in g["annotations"]})
    if len(annotators) != 2:
        raise ValueError(f"human-human kappa needs exactly 2 annotators, got {annotators}")
    a1, a2 = annotators

    cats = [str(g["category"]) for g in gold]
    h1 = [g["annotations"][a1]["label"] for g in gold]
    h2 = [g["annotations"][a2]["label"] for g in gold]
    s1 = [g["annotations"][a1]["side"] for g in gold]
    s2 = [g["annotations"][a2]["side"] for g in gold]
    gold_labels = [g.get("gold_label") for g in gold]
    matched = [by_id[str(g["id"])] for g in gold]
    judge_labels = [_judge_label(r, label_source) for r in matched]

    hh = kappa_with_ci(h1, h2, n_boot=n_boot, seed=seed, labels=("preserved", "degraded"))
    hh_side = kappa_with_ci(s1, s2, n_boot=n_boot, seed=seed, labels=_SIDES)
    hh_cat = per_category_kappa(
        cats, h1, h2, n_boot=n_boot, seed=seed, labels=("preserved", "degraded")
    )
    jh = kappa_with_ci(
        gold_labels, judge_labels, n_boot=n_boot, seed=seed, labels=("preserved", "degraded")
    )
    jh_cat = per_category_kappa(
        cats, gold_labels, judge_labels, n_boot=n_boot, seed=seed, labels=("preserved", "degraded")
    )
    det = detector_metrics(gold_labels, judge_labels)
    all_sides = [judge_sides(r) for r in judge_records]
    pos_rate, pos_dis, pos_n = position_disagreement_rate(
        [s[0] for s in all_sides], [s[1] for s in all_sides]
    )
    sub_sides = [judge_sides(r) for r in matched]
    sub_rate, sub_dis, sub_n = position_disagreement_rate(
        [s[0] for s in sub_sides], [s[1] for s in sub_sides]
    )
    checks = checkpoint_checks(
        human_human_kappa=hh.kappa,
        judge_human_kappa=jh.kappa,
        judge_human_kappa_per_category={c: k.kappa for c, k in jh_cat.items()},
        sensitivity=det.sensitivity,
        specificity=det.specificity,
        position_disagreement=pos_rate,
    )

    lines: list[str] = ["# Phase 2 - Judge Validation (rows 2.3.5-2.3.6)", ""]
    if stand_in:
        smoke = "> **SMOKE RUN: judge records are stand-in output. Not a checkpoint result.**"
        lines += [smoke, ""]
    lines += [
        f"Gold items: {len(gold)} · judge records: {len(judge_records)} · annotators: {a1}, {a2} "
        f"· judge label source: `{label_source}` · bootstrap: {n_boot} resamples, seed {seed}, "
        "percentile 95% CI · positive class: `degraded`.",
        "",
    ]
    for key, value in (provenance or {}).items():
        lines.append(f"- {key}: `{value}`")
    if provenance:
        lines.append("")
    lines += ["## Checkpoints", "", "```text", *[c.line() for c in checks], "```", ""]
    header = ["| Comparison | n | dropped | kappa | 95% CI | observed agreement |", "| :--- |"]
    header[1] += " ---: |" * 5
    lines += ["## Human-human agreement (C2.3)", "", *header, _kappa_row("binary label", hh)]
    lines.append(_kappa_row("3-way side (info)", hh_side))
    lines += [_kappa_row(f"binary, {c}", k) for c, k in hh_cat.items()]
    table_hh = confusion_table(h1, h2, row_name=a1, col_name=a2)
    pending = sum(1 for g in gold_labels if g is None)
    lines += [
        "",
        table_hh.markdown(),
        "",
        f"Items without a gold label (disagreement not adjudicated, or cannot judge): {pending}.",
        "",
    ]
    lines += ["## Judge-human agreement (C2.4)", "", *header, _kappa_row("overall", jh)]
    lines += [_kappa_row(c, k) for c, k in jh_cat.items()]
    table_jh = confusion_table(gold_labels, judge_labels, row_name="human", col_name="judge")
    lines += ["", table_jh.markdown(), ""]
    lines += [
        "## Judge as degradation detector (C2.5)",
        "",
        "| n | sensitivity | specificity | TP | FN | TN | FP |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| {det.tp + det.fn + det.tn + det.fp} | {_f(det.sensitivity)} | {_f(det.specificity)} "
        f"| {det.tp} | {det.fn} | {det.tn} | {det.fp} |",
        "",
        "## Position bias (C2.6)",
        "",
        "| Records | parsed pairs | disagreeing | rate |",
        "| :--- | ---: | ---: | ---: |",
        f"| all judge records | {pos_n} | {pos_dis} | {_f(pos_rate)} |",
        f"| annotated subset | {sub_n} | {sub_dis} | {_f(sub_rate)} |",
        "",
    ]
    metrics = {
        "hh_kappa": hh.kappa,
        "hh_kappa_ci_low": hh.ci_low,
        "hh_kappa_ci_high": hh.ci_high,
        "jh_kappa": jh.kappa,
        "jh_kappa_ci_low": jh.ci_low,
        "jh_kappa_ci_high": jh.ci_high,
        "sensitivity": det.sensitivity,
        "specificity": det.specificity,
        "position_disagreement": pos_rate,
        **{f"jh_kappa_{c}": k.kappa for c, k in jh_cat.items()},
        "checks_passed": float(sum(c.passed for c in checks)),
        "checks_total": float(len(checks)),
    }
    return JudgeValidationReport("\n".join(lines), checks, metrics, stand_in)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def run(
    gold_path: Path,
    judge_path: Path,
    out: Path,
    *,
    label_source: LabelSource = "judge",
    n_boot: int = 2000,
    seed: int = 0,
    allow_stand_in: bool = False,
) -> JudgeValidationReport:
    """Load both files, build the report and write it to ``out`` (LF endings)."""
    report = build_report(
        load_jsonl(gold_path),
        load_jsonl(judge_path),
        label_source=label_source,
        n_boot=n_boot,
        seed=seed,
        allow_stand_in=allow_stand_in,
        provenance={
            "gold": f"{gold_path.as_posix()} (sha256 {_sha(gold_path)})",
            "judge": f"{judge_path.as_posix()} (sha256 {_sha(judge_path)})",
        },
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(report.markdown)
    return report


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m terser.phase2.judge_validation",
        description="Row 2.3.6: human-human and judge-human agreement, position bias (C2.3-C2.6).",
    )
    p.add_argument("--gold", type=Path, required=True, help="human gold JSONL (row 2.3.5)")
    p.add_argument("--judge", type=Path, required=True, help="label stage JSONL (row 2.3.4)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="markdown report path")
    p.add_argument("--label-source", choices=["judge", "final"], default="judge")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--allow-stand-in", action="store_true", help="smoke runs only")
    p.add_argument("--no-track", action="store_true", help="do not log to MLflow")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns 0 on success (whatever the checkpoint outcome), 1-2 on error."""
    args = _parser().parse_args(argv)

    def execute() -> JudgeValidationReport:
        return run(
            args.gold,
            args.judge,
            args.out,
            label_source=args.label_source,
            n_boot=args.n_boot,
            seed=args.seed,
            allow_stand_in=args.allow_stand_in,
        )

    try:
        if args.no_track:
            report = execute()
        else:
            import mlflow

            from terser.tracking import start_run

            variant = "standin" if args.allow_stand_in else args.label_source
            with start_run(
                phase=2,
                row="2.3.6",
                task="judgevalidation",
                run_name=f"2-judgevalidation-judge-{variant}-{args.seed}",
                params={
                    "gold": args.gold.as_posix(),
                    "judge": args.judge.as_posix(),
                    "label_source": args.label_source,
                    "n_boot": args.n_boot,
                    "seed": args.seed,
                },
            ):
                report = execute()
                mlflow.set_tag("stand_in", str(report.stand_in).lower())
                mlflow.log_metrics({k: v for k, v in report.metrics.items() if not math.isnan(v)})
                mlflow.log_artifact(str(args.out), artifact_path="reports")
    except StandInJudgeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for check in report.checks:
        print(check.line())
    print(json.dumps({"out": args.out.as_posix(), "all_passed": report.all_passed}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
