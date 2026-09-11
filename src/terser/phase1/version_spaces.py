"""Row 1.3.7: Candidate-Elimination version space per task category on the pilot labels.

Input is the ``labels.jsonl`` written by :mod:`terser.phase1.pilot_labels` (one record per
(prompt, operator subset) with ``id``, ``prompt_id``, ``category``, ``ops``, ``original``,
``label`` in ``{preserved, degraded, null}`` and ``stand_in``). A non-empty ``human_label``
field, if present, overrides ``label`` (human review of every label is part of row 1.3.6).

For each record the instance is ``describe_prompt(original) + operator flags(ops)`` (see
:mod:`terser.concept.attributes`); prompt attributes are computed once per ``prompt_id``. The
target concept is "the rewrite **preserved** output quality", so ``preserved`` is a positive
example here. (The quality gate of row 1.1.3 uses the opposite convention, degraded = positive;
the version space of a concept and of its complement are not the same object, and Phase 1 learns
the "safe to apply" concept.) Records without a label are skipped and counted.

Candidate-Elimination runs per category with ``positives_first=True`` (identical boundaries on
consistent data, and it keeps ``G`` small). The markdown report lists, per category, the number
of examples, ``S``, ``G`` and the version-space size, or the collapse diagnosis: the offending
example and an inclusion-minimal conflicting set, each shown with record id, prompt id,
operators, label and attribute vector. The summary gives the C1.3 verdict: non-empty version
space for at least 4 of the 6 categories (a category without labelled examples counts as not
non-empty, since it has not been tested).

Stand-in labels (``stand_in: true``, CPU smoke models) are refused unless ``--allow-stand-in``
is given; the report is then stamped as a smoke run. CLI::

    python -m terser.phase1.version_spaces --labels data/pilot/labels.jsonl \\
        --out reports/p1_version_spaces.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from terser.concept.attributes import (
    ATTRIBUTE_NAMES,
    DOMAINS,
    OP_CODES,
    PROMPT_ATTRIBUTES,
    Remover,
    describe_prompt,
    instance,
)
from terser.concept.candidate_elimination import (
    BoundaryTooLargeError,
    CandidateEliminationResult,
    candidate_elimination,
)
from terser.concept.hypothesis import Example, Hypothesis, format_hypothesis
from terser.concept.version_space import iter_version_space
from terser.data.categories import Category

__all__ = [
    "C13_MIN_NONEMPTY",
    "DEFAULT_LABELS",
    "DEFAULT_OUT",
    "AttributeCache",
    "CategoryResult",
    "LabelRecord",
    "StandInLabelsError",
    "VersionSpaceReport",
    "analyse",
    "build_report",
    "check_stand_in",
    "load_labels",
    "main",
    "run",
]

DEFAULT_LABELS = Path("data/pilot/labels.jsonl")
DEFAULT_OUT = Path("reports/p1_version_spaces.md")
C13_MIN_NONEMPTY = 4
"""Checkpoint C1.3: minimum number of categories with a non-empty version space."""
MAX_ENUMERATE = 1_000_000
"""Stop counting version-space members beyond this (reported as ``> MAX_ENUMERATE``)."""
MAX_LISTED = 25
"""Boundary members shown per category before eliding."""
POSITIVE_LABEL = "preserved"
NEGATIVE_LABEL = "degraded"

Describe = Callable[[str, Remover | None], dict[str, str]]
"""``describe(text, remover)`` returns prompt attributes (default :func:`describe_prompt`)."""

_REQUIRED = ("id", "prompt_id", "category", "ops", "original")
_SHORT_NAMES = {
    "length": "length",
    "has_constraints": "constr.",
    "has_payload": "payload",
    "tone_cue": "tone",
    "filler_density": "filler",
}


class StandInLabelsError(RuntimeError):
    """The labels come from CPU stand-in models and ``allow_stand_in`` was not set."""


@dataclass(frozen=True)
class LabelRecord:
    """One labelled (prompt, operator subset) pair."""

    id: str
    prompt_id: str
    category: str
    ops: str
    original: str
    label: str | None
    stand_in: bool
    synthetic: bool = False


def load_labels(path: Path) -> list[LabelRecord]:
    """Read a pilot ``labels.jsonl``.

    A record without ``stand_in`` is treated as stand-in (never silently trusted).

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: On a missing field, an unknown label or a repeated id.
    """
    if not path.is_file():
        raise FileNotFoundError(f"labels not found: {path} (run row 1.3.6 first)")
    out: list[LabelRecord] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            row: dict[str, Any] = json.loads(line)
            missing = [k for k in _REQUIRED if row.get(k) in (None, "")]
            if missing:
                raise ValueError(f"{path}:{lineno}: missing field(s) {missing}")
            rid = str(row["id"])
            if rid in seen:
                raise ValueError(f"{path}:{lineno}: duplicate id {rid!r}")
            seen.add(rid)
            label = row.get("human_label") or row.get("label")
            if label not in (POSITIVE_LABEL, NEGATIVE_LABEL, None):
                raise ValueError(f"{path}:{lineno}: unknown label {label!r}")
            out.append(
                LabelRecord(
                    id=rid,
                    prompt_id=str(row["prompt_id"]),
                    category=str(row["category"]),
                    ops=str(row["ops"]),
                    original=str(row["original"]),
                    label=label,
                    stand_in=bool(row.get("stand_in", True)),
                    synthetic=bool(row.get("synthetic", False)),
                )
            )
    return out


class AttributeCache:
    """Prompt attributes computed once per ``prompt_id``.

    ``calls`` counts invocations of ``describe``. Raises ``ValueError`` if one ``prompt_id``
    appears with two different prompt texts.
    """

    def __init__(self, remover: Remover | None = None, describe: Describe | None = None) -> None:
        self.remover = remover
        self.describe: Describe = describe or describe_prompt
        self.calls = 0
        self._attrs: dict[str, dict[str, str]] = {}
        self._text: dict[str, str] = {}

    def get(self, prompt_id: str, text: str) -> dict[str, str]:
        """Attributes of ``text``, computed on first use of ``prompt_id``."""
        if prompt_id in self._attrs:
            if self._text[prompt_id] != text:
                raise ValueError(f"prompt_id {prompt_id!r} appears with two different texts")
            return self._attrs[prompt_id]
        self.calls += 1
        attrs = dict(self.describe(text, self.remover))
        self._attrs[prompt_id] = attrs
        self._text[prompt_id] = text
        return attrs


@dataclass
class CategoryResult:
    """Candidate-Elimination outcome for one category.

    ``examples[i]`` and ``records[i]`` describe the same labelled pair (file order), so the
    indices in ``result.collapse`` point into both.
    """

    category: str
    examples: list[Example] = field(default_factory=list)
    records: list[LabelRecord] = field(default_factory=list)
    n_unlabelled: int = 0
    result: CandidateEliminationResult | None = None
    size: int | None = None
    size_capped: bool = False
    error: str | None = None

    @property
    def n_positive(self) -> int:
        """Number of ``preserved`` examples."""
        return sum(1 for _, y in self.examples if y)

    @property
    def n_negative(self) -> int:
        """Number of ``degraded`` examples."""
        return len(self.examples) - self.n_positive

    @property
    def nonempty(self) -> bool:
        """Whether the category was tested and its version space is non-empty."""
        return bool(self.examples) and self.result is not None and not self.result.collapsed

    @property
    def status(self) -> str:
        """``non-empty``, ``collapsed``, ``no examples`` or ``not computed``."""
        if not self.examples:
            return "no examples"
        if self.result is None:
            return "not computed"
        return "collapsed" if self.result.collapsed else "non-empty"


@dataclass
class VersionSpaceReport:
    """Per-category results, headline metrics and the rendered markdown."""

    categories: list[CategoryResult]
    metrics: dict[str, float]
    markdown: str
    stand_in: bool
    synthetic: bool


def _count_version_space(result: CandidateEliminationResult, limit: int) -> tuple[int, bool]:
    n = 0
    for _ in iter_version_space(result.S, result.G, DOMAINS):
        n += 1
        if n > limit:
            return limit, True
    return n, False


def _category_order(names: Sequence[str]) -> list[str]:
    known = [c.value for c in Category]
    return known + sorted(set(names) - set(known))


def analyse(
    records: Sequence[LabelRecord],
    cache: AttributeCache,
    *,
    max_enumerate: int = MAX_ENUMERATE,
) -> list[CategoryResult]:
    """Build instances per category and run Candidate-Elimination (positives first)."""
    by_cat: dict[str, CategoryResult] = {
        c: CategoryResult(c) for c in _category_order([r.category for r in records])
    }
    for rec in records:
        cat = by_cat[rec.category]
        if rec.label is None:
            cat.n_unlabelled += 1
            continue
        x = instance(cache.get(rec.prompt_id, rec.original), rec.ops)
        cat.examples.append((x, rec.label == POSITIVE_LABEL))
        cat.records.append(rec)
    for cat in by_cat.values():
        if not cat.examples:
            continue
        try:
            cat.result = candidate_elimination(cat.examples, DOMAINS, positives_first=True)
        except BoundaryTooLargeError as exc:
            cat.error = str(exc)
            continue
        if not cat.result.collapsed:
            cat.size, cat.size_capped = _count_version_space(cat.result, max_enumerate)
    return list(by_cat.values())


def _cell(text: str) -> str:
    return text.replace("|", "\\|")


def _hyp(h: Hypothesis) -> str:
    return f"`{format_hypothesis(h, ATTRIBUTE_NAMES)}`"


def _boundary(name: str, members: Sequence[Hypothesis]) -> list[str]:
    if not members:
        return [f"- **{name}**: empty"]
    lines = [f"- **{name}** ({len(members)} member{'s' if len(members) != 1 else ''}):"]
    for h in members[:MAX_LISTED]:
        lines.append(f"  - {_hyp(h)}")
    if len(members) > MAX_LISTED:
        lines.append(f"  - ... and {len(members) - MAX_LISTED} more")
    return lines


def _example_table(cat: CategoryResult, rows: Sequence[tuple[str, int]]) -> list[str]:
    n_prompt = len(PROMPT_ATTRIBUTES)
    head = ["role", "ex", "record", "prompt", "ops", "label"]
    head += [_SHORT_NAMES[a] for a in PROMPT_ATTRIBUTES] + ["op flags " + OP_CODES]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for role, i in rows:
        rec = cat.records[i]
        x, y = cat.examples[i]
        flags = "".join("1" if v == "yes" else "0" for v in x[n_prompt:])
        cells = [role, str(i), f"`{rec.id}`", f"`{rec.prompt_id}`", f"`{rec.ops}`"]
        cells += [POSITIVE_LABEL if y else NEGATIVE_LABEL, *x[:n_prompt], f"`{flags}`"]
        lines.append("| " + " | ".join(_cell(c) for c in cells) + " |")
    return lines


def _category_section(cat: CategoryResult) -> list[str]:
    lines = [f"### {cat.category}", ""]
    lines.append(
        f"- Examples: **{len(cat.examples)}** ({cat.n_positive} preserved / "
        f"{cat.n_negative} degraded); unlabelled records skipped: {cat.n_unlabelled}"
    )
    lines.append(f"- Status: **{cat.status}**")
    if not cat.examples:
        return [*lines, ""]
    if cat.error is not None:
        return [*lines, f"- Candidate-Elimination did not finish: {cat.error}", ""]
    res = cat.result
    assert res is not None
    if not res.collapsed:
        size = f"> {cat.size:,}" if cat.size_capped else f"{cat.size:,}"
        lines.append(f"- Version-space size: **{size}** conjunctive hypotheses")
        lines += _boundary("S", res.S)
        lines += _boundary("G", res.G)
        return [*lines, ""]
    col = res.collapse
    assert col is not None
    step = res.order.index(col.index) + 1
    lines.append(
        f"- Collapsed at processed step {step} of {len(res.order)} (positives first), "
        f"on example {col.index} (`{cat.records[col.index].id}`)."
    )
    lines.append(
        f"- Minimal conflicting set: {len(col.examples)} examples. Its positives force "
        f"{_hyp(col.witness)}, which covers negative example {col.negative} "
        f"(`{cat.records[col.negative].id}`); no conjunction separates them."
    )
    lines.append("")
    rows = [("offending", col.index)] + [("conflicting", j) for j in col.conflicting]
    lines += _example_table(cat, rows)
    lines += ["", "Boundaries at the collapse step:", ""]
    lines += _boundary("S", res.S)
    lines += _boundary("G", res.G)
    return [*lines, ""]


def build_report(
    records: Sequence[LabelRecord],
    cache: AttributeCache,
    *,
    labels_path: str = str(DEFAULT_LABELS),
    max_enumerate: int = MAX_ENUMERATE,
) -> VersionSpaceReport:
    """Analyse ``records`` and render the markdown report (no stand-in check here)."""
    cats = analyse(records, cache, max_enumerate=max_enumerate)
    stand_in = any(r.stand_in for r in records)
    synthetic = any(r.synthetic for r in records)
    n_six = len(Category)
    core = [c for c in cats if c.category in {k.value for k in Category}]
    n_nonempty = sum(c.nonempty for c in core)
    passed = n_nonempty >= C13_MIN_NONEMPTY
    verdict = "PASS" if passed else "FAIL"
    if stand_in or synthetic:
        verdict += " (not valid: " + ("stand-in" if stand_in else "synthetic") + " labels)"

    lines = ["# Phase 1 version spaces per category", ""]
    if stand_in:
        lines += [
            "> **SMOKE RUN.** These labels come from CPU stand-in models (`stand_in: true`). "
            "They are not pilot labels and this report does not count towards C1.3.",
            "",
        ]
    if synthetic:
        lines += [
            "> **SYNTHETIC LABELS.** Records are marked `synthetic: true`; this report exercises "
            "the runner only.",
            "",
        ]
    lines += [
        "Row **1.3.7**. Generated by `python -m terser.phase1.version_spaces "
        f"--labels {labels_path}`. Candidate-Elimination over the conjunctive language of "
        "`terser.concept` (5 prompt attributes + 7 operator flags), one run per category, "
        "positives processed first. Positive example = label `preserved`.",
        "",
        "## Summary",
        "",
        f"**C1.3 Version space: {verdict}.** Non-empty for **{n_nonempty} of {n_six}** "
        f"categories (threshold >= {C13_MIN_NONEMPTY}); every collapse below names its "
        "offending example and an inclusion-minimal conflicting set.",
        "",
        "| category | examples | preserved | degraded | status | version-space size |",
        "|---|---|---|---|---|---|",
    ]
    for c in cats:
        size = "" if c.size is None else (f"> {c.size:,}" if c.size_capped else f"{c.size:,}")
        lines.append(
            f"| {c.category} | {len(c.examples)} | {c.n_positive} | {c.n_negative} | "
            f"{c.status} | {size} |"
        )
    lines += [
        "",
        "Attribute order: " + ", ".join(f"`{a}`" for a in ATTRIBUTE_NAMES) + ". In the "
        f"collapse tables `op flags {OP_CODES}` shows one bit per operator flag in that order.",
        "",
        "## Per category",
        "",
    ]
    for c in cats:
        lines += _category_section(c)
    lines += [
        "## Inductive bias (row 1.3.8)",
        "",
        "What the conjunctive language cannot express, why collapses are expected, and how this "
        "motivates the statistical learners of Phases 3-5: "
        "[docs/derivations/inductive_bias.md](../docs/derivations/inductive_bias.md).",
        "",
    ]
    metrics: dict[str, float] = {
        "n_records": float(len(records)),
        "n_examples": float(sum(len(c.examples) for c in cats)),
        "n_nonempty": float(n_nonempty),
        "n_collapsed": float(sum(c.status == "collapsed" for c in cats)),
        "c13_pass": float(passed and not stand_in and not synthetic),
        "attribute_evaluations": float(cache.calls),
    }
    for c in cats:
        if c.size is not None:
            metrics[f"vs_size_{c.category}"] = float(c.size)
    return VersionSpaceReport(
        categories=cats,
        metrics=metrics,
        markdown="\n".join(lines).rstrip() + "\n",
        stand_in=stand_in,
        synthetic=synthetic,
    )


def check_stand_in(
    records: Sequence[LabelRecord], labels: Path, *, allow_stand_in: bool = False
) -> None:
    """Raise :class:`StandInLabelsError` if any record is stand-in and that is not allowed."""
    n_stand_in = sum(r.stand_in for r in records)
    if n_stand_in and not allow_stand_in:
        raise StandInLabelsError(
            f"{labels}: {n_stand_in} of {len(records)} records have stand_in: true (CPU smoke "
            "models). Version spaces need real pilot labels; pass --allow-stand-in for a smoke run."
        )


def run(
    labels: Path,
    out: Path,
    *,
    allow_stand_in: bool = False,
    remover: Remover | None = None,
    describe: Describe | None = None,
) -> VersionSpaceReport:
    """Load labels, refuse stand-in labels unless allowed, write the report to ``out``.

    Raises:
        StandInLabelsError: If any record has ``stand_in: true`` and ``allow_stand_in`` is False.
    """
    records = load_labels(labels)
    check_stand_in(records, labels, allow_stand_in=allow_stand_in)
    cache = AttributeCache(remover=remover, describe=describe)
    report = build_report(records, cache, labels_path=labels.as_posix())
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(report.markdown)
    return report


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m terser.phase1.version_spaces",
        description="Row 1.3.7: Candidate-Elimination version space per category.",
    )
    p.add_argument("--labels", type=Path, default=DEFAULT_LABELS, help="pilot labels.jsonl")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="markdown report path")
    p.add_argument(
        "--allow-stand-in",
        action="store_true",
        help="run on stand-in (smoke) labels and stamp the report as a smoke run",
    )
    p.add_argument("--no-track", action="store_true", help="do not log to MLflow")
    return p


def _tracked(args: argparse.Namespace, execute: Callable[[], VersionSpaceReport]) -> None:
    import mlflow

    from terser.tracking import start_run

    params: Mapping[str, Any] = {
        "labels": args.labels.as_posix(),
        "out": args.out.as_posix(),
        "allow_stand_in": args.allow_stand_in,
        "positives_first": True,
    }
    variant = "standin" if args.allow_stand_in else "pilot"
    with start_run(
        phase=1,
        row="1.3.7",
        task="versionspaces",
        run_name=f"1-versionspaces-conjunctive-{variant}-0",
        params=dict(params),
    ):
        report = execute()
        mlflow.set_tag("stand_in", str(report.stand_in).lower())
        mlflow.set_tag("synthetic", str(report.synthetic).lower())
        mlflow.log_metrics(report.metrics)
        mlflow.log_artifact(str(args.out), artifact_path="reports")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    args = _parser().parse_args(argv)
    result: list[VersionSpaceReport] = []

    def execute() -> VersionSpaceReport:
        report = run(args.labels, args.out, allow_stand_in=args.allow_stand_in)
        result.append(report)
        return report

    try:
        # Refuse stand-in labels before an MLflow run is opened.
        check_stand_in(load_labels(args.labels), args.labels, allow_stand_in=args.allow_stand_in)
        if args.no_track:
            execute()
        else:
            _tracked(args, execute)
    except StandInLabelsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except (FileNotFoundError, ValueError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({**result[0].metrics, "out": args.out.as_posix()}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
