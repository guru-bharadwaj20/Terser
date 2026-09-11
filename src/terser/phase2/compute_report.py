"""Row 2.4.3: compute report, generated tokens and GPU-seconds per Phase 2 stage vs estimates.

Stages: ``casualise`` (row 2.1.4), ``candidates`` (2.2.1), ``respond`` (2.2.3), ``judge`` (2.3.3).

Inputs:

* **LLM ledger(s)** written by :class:`terser.llm.budget.Ledger` (``{"phases": {phase: {"calls":
  [...]}}}``). Each call gives ``input_tokens`` / ``output_tokens`` (generated tokens) and
  ``usage_reported``. Cache hits are never ledgered, so the ledger is compute actually spent.
  Entries carry no stage; attribution order: ``--ledger STAGE=PATH`` > an entry ``stage`` key >
  ``attribution.ledger_phases`` > ``attribution.models`` > ``unattributed``. Calls to
  ``stand_in_models`` (CPU smoke tests) are excluded from totals.
* **Run durations** from MLflow (experiment ``mlflow.experiment``; stage from the run's ``task``
  tag) or from a runs table (``--runs``: ``run_id, task, hardware, duration_s`` and optionally
  ``stand_in, status, n_gpus``). GPU-seconds = duration x ``n_gpus`` (tag/param, default 1) for runs
  whose ``hardware`` tag is a GPU; ``cpu`` runs count as CPU-seconds; ``auto`` / missing hardware is
  reported as unknown and not counted. Runs tagged ``stand_in=true`` and unfinished runs (no end
  time) are excluded and listed.
* **Labelled pairs**: the labels table (row 2.3.4; non-empty, non-stand-in labels) or a count.
* **Estimates**: ``configs/experiment/p2_compute_estimates.yaml`` (``params.stages.<stage>
  .generated_tokens`` / ``gpu_seconds``, ``labelled_pairs``, ``tolerance``, ``placeholder``).

Report (``reports/p2_cost.md`` + ``reports/p2_cost.csv`` by default): per-stage calls, input and
generated tokens, GPU-seconds / GPU-hours, CPU-seconds, GPU-seconds per labelled pair; estimate vs
actual per stage; the **C2.9** cost line (total GPU-seconds within +/-``tolerance`` of the estimate;
the ``dvc repro`` hash half of C2.9 is not checked here); every ``usage_reported: false`` entry.

CLI::

    python -m terser.phase2.compute_report --ledger .cache/llm_ledger.json \\
        --labels data/processed/labels.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import numbers
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from terser.phase2.oracle import Phase2InputError, _to_bool, read_table, validate_labels

__all__ = [
    "DEFAULT_CSV",
    "DEFAULT_ESTIMATES",
    "DEFAULT_MD",
    "STAGES",
    "UNATTRIBUTED",
    "ComputeEstimates",
    "ComputeReport",
    "build_report",
    "check_c29",
    "count_labelled_pairs",
    "load_estimates",
    "load_ledger_calls",
    "load_mlflow_runs",
    "main",
    "normalise_runs",
    "write_report",
]

STAGES: tuple[str, ...] = ("casualise", "candidates", "respond", "judge")
UNATTRIBUTED = "unattributed"
DEFAULT_ESTIMATES = Path("configs/experiment/p2_compute_estimates.yaml")
DEFAULT_MD = Path("reports/p2_cost.md")
DEFAULT_CSV = Path("reports/p2_cost.csv")

CALL_COLUMNS: tuple[str, ...] = (
    "stage",
    "ledger",
    "phase",
    "timestamp",
    "provider",
    "model",
    "input_tokens",
    "output_tokens",
    "cache_key",
    "usage_reported",
    "stand_in",
)
RUN_COLUMNS: tuple[str, ...] = (
    "run_id",
    "run_name",
    "task",
    "stage",
    "hardware",
    "hw_class",
    "status",
    "stand_in",
    "n_gpus",
    "duration_s",
)
STAGE_COLUMNS: tuple[str, ...] = (
    "stage",
    "calls",
    "input_tokens",
    "generated_tokens",
    "unreported_calls",
    "runs",
    "gpu_seconds",
    "gpu_hours",
    "cpu_seconds",
    "unknown_hw_seconds",
    "gpu_seconds_per_pair",
)
_NON_GPU = {"cpu"}
_UNKNOWN_HW = {"", "auto", "none", "nan", "unknown"}


# ---------------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ComputeEstimates:
    """Pre-run estimates and attribution rules (see ``p2_compute_estimates.yaml``)."""

    stages: dict[str, dict[str, float]]
    labelled_pairs: float | None = None
    tolerance: float = 0.20
    placeholder: bool = True
    experiment: str = "phase2"
    tasks: dict[str, str] = field(default_factory=lambda: {s: s for s in STAGES})
    ledger_phases: dict[str, str] = field(default_factory=dict)
    models: dict[str, str] = field(default_factory=dict)
    stand_in_models: tuple[str, ...] = ()

    @property
    def total_gpu_seconds(self) -> float:
        """Sum of the per-stage GPU-second estimates."""
        return float(sum(float(v.get("gpu_seconds", 0) or 0) for v in self.stages.values()))


def load_estimates(path: str | Path = DEFAULT_ESTIMATES) -> ComputeEstimates:
    """Read the estimates YAML (keys under ``params`` if present, else top level).

    Raises:
        Phase2InputError: Unknown stages, negative numbers or a non-positive tolerance.
    """
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise Phase2InputError(f"estimates {path}: expected a mapping")
    cfg: dict[str, Any] = raw.get("params") or raw
    stages: dict[str, dict[str, float]] = {}
    for stage, vals in (cfg.get("stages") or {}).items():
        if stage not in STAGES:
            raise Phase2InputError(f"estimates: unknown stage {stage!r}; expected {STAGES}")
        entry = {k: float(vals.get(k, 0) or 0) for k in ("generated_tokens", "gpu_seconds")}
        if any(v < 0 for v in entry.values()):
            raise Phase2InputError(f"estimates: negative value for stage {stage!r}")
        stages[stage] = entry
    tolerance = float(cfg.get("tolerance", 0.20))
    if tolerance <= 0:
        raise Phase2InputError("estimates: tolerance must be positive")
    attribution = cfg.get("attribution") or {}
    mlflow_cfg = cfg.get("mlflow") or {}
    for name, mapping in (
        ("ledger_phases", attribution.get("ledger_phases") or {}),
        ("models", attribution.get("models") or {}),
    ):
        bad = sorted({v for v in mapping.values() if v not in STAGES})
        if bad:
            raise Phase2InputError(f"estimates: attribution.{name} maps to unknown stage(s) {bad}")
    pairs = cfg.get("labelled_pairs")
    return ComputeEstimates(
        stages=stages,
        labelled_pairs=float(pairs) if pairs is not None else None,
        tolerance=tolerance,
        placeholder=bool(cfg.get("placeholder", True)),
        experiment=str(mlflow_cfg.get("experiment", "phase2")),
        tasks={
            str(k): str(v) for k, v in (mlflow_cfg.get("tasks") or {s: s for s in STAGES}).items()
        },
        ledger_phases={str(k): str(v) for k, v in (attribution.get("ledger_phases") or {}).items()},
        models={str(k): str(v) for k, v in (attribution.get("models") or {}).items()},
        stand_in_models=tuple(str(m) for m in cfg.get("stand_in_models") or ()),
    )


def load_ledger_calls(
    path: str | Path, estimates: ComputeEstimates, *, stage: str | None = None
) -> pd.DataFrame:
    """Flatten one ledger file into call rows (:data:`CALL_COLUMNS`) with a resolved stage.

    Entries without ``usage_reported`` (older ledgers) count as reported.

    Raises:
        Phase2InputError: Malformed ledger or an unknown ``stage``.
    """
    if stage is not None and stage not in STAGES:
        raise Phase2InputError(f"ledger {path}: unknown stage {stage!r}; expected {STAGES}")
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    phases = data.get("phases") if isinstance(data, dict) else None
    if not isinstance(phases, dict):
        raise Phase2InputError(f"ledger {p}: expected an object with a 'phases' mapping")
    rows: list[dict[str, Any]] = []
    for phase in sorted(phases):
        for call in (phases[phase] or {}).get("calls", []):
            model = str(call.get("model", ""))
            resolved = (
                stage
                or call.get("stage")
                or estimates.ledger_phases.get(phase)
                or estimates.models.get(model)
                or UNATTRIBUTED
            )
            try:
                tokens_in = int(call.get("input_tokens", 0) or 0)
                tokens_out = int(call.get("output_tokens", 0) or 0)
            except (TypeError, ValueError) as exc:
                raise Phase2InputError(f"ledger {p}: bad token count in {call!r}") from exc
            rows.append(
                {
                    "stage": str(resolved),
                    "ledger": p.as_posix(),
                    "phase": str(phase),
                    "timestamp": str(call.get("timestamp", "")),
                    "provider": str(call.get("provider", "")),
                    "model": model,
                    "input_tokens": tokens_in,
                    "output_tokens": tokens_out,
                    "cache_key": str(call.get("cache_key") or ""),
                    "usage_reported": _to_bool(call.get("usage_reported", True), "usage_reported"),
                    "stand_in": model in estimates.stand_in_models,
                }
            )
    return pd.DataFrame(rows, columns=list(CALL_COLUMNS))


def _hw_class(hardware: Any) -> str:
    hw = (
        ""
        if hardware is None or (isinstance(hardware, float) and math.isnan(hardware))
        else hardware
    )
    key = str(hw).strip().lower()
    if key in _NON_GPU:
        return "cpu"
    if key in _UNKNOWN_HW:
        return "unknown"
    return "gpu"


def normalise_runs(runs: pd.DataFrame, estimates: ComputeEstimates) -> pd.DataFrame:
    """Validate a runs table and add ``stage``, ``hw_class``; columns :data:`RUN_COLUMNS`.

    ``duration_s`` may be missing/NaN for unfinished runs (they are excluded later).

    Raises:
        Phase2InputError: Missing columns, duplicate run ids or negative durations.
    """
    missing = [c for c in ("run_id", "task", "hardware", "duration_s") if c not in runs.columns]
    if missing:
        raise Phase2InputError(f"runs: missing column(s) {missing}; got {list(runs.columns)}")
    out = runs.copy()
    out["run_id"] = out["run_id"].astype(str)
    if out["run_id"].duplicated().any():
        raise Phase2InputError("runs: duplicate run_id")
    task_to_stage = {task: stage for stage, task in estimates.tasks.items()}
    out["task"] = out["task"].fillna("").astype(str)
    out["stage"] = out["task"].map(lambda t: task_to_stage.get(t, ""))
    out["hw_class"] = [_hw_class(h) for h in out["hardware"]]
    out["hardware"] = out["hardware"].fillna("").astype(str)
    out["duration_s"] = pd.to_numeric(out["duration_s"], errors="coerce")
    if (out["duration_s"] < 0).any():
        raise Phase2InputError("runs: negative duration_s")
    for col, default in (("run_name", ""), ("status", "FINISHED")):
        out[col] = out[col].fillna(default).astype(str) if col in out.columns else default
    if "stand_in" in out.columns:
        out["stand_in"] = [_to_bool(v, "stand_in") for v in out["stand_in"]]
    else:
        out["stand_in"] = False
    if "n_gpus" in out.columns:
        out["n_gpus"] = pd.to_numeric(out["n_gpus"], errors="coerce").fillna(1).astype(int)
    else:
        out["n_gpus"] = 1
    return out[list(RUN_COLUMNS)].sort_values("run_id", kind="mergesort").reset_index(drop=True)


def load_mlflow_runs(experiment: str, tracking_uri: str | None = None) -> pd.DataFrame:
    """Runs of ``experiment`` as a raw runs table (``duration_s`` NaN while a run has no end).

    Uses ``tracking_uri`` if given, else :func:`terser.tracking.runs.setup_tracking`. A missing
    experiment yields an empty table.
    """
    import mlflow

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        from terser.tracking.runs import setup_tracking

        setup_tracking()
    columns = [
        "run_id",
        "run_name",
        "task",
        "hardware",
        "status",
        "stand_in",
        "n_gpus",
        "duration_s",
    ]
    if mlflow.get_experiment_by_name(experiment) is None:
        return pd.DataFrame(columns=columns)
    found = mlflow.search_runs(experiment_names=[experiment], output_format="list")
    rows: list[dict[str, Any]] = []
    for run in found:
        tags, params = run.data.tags, run.data.params
        end = run.info.end_time
        rows.append(
            {
                "run_id": run.info.run_id,
                "run_name": run.info.run_name or "",
                "task": tags.get("task", ""),
                "hardware": tags.get("hardware", ""),
                "status": run.info.status,
                "stand_in": tags.get("stand_in", "false"),
                "n_gpus": tags.get("n_gpus", params.get("n_gpus", 1)),
                "duration_s": (end - run.info.start_time) / 1000.0 if end else float("nan"),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def count_labelled_pairs(labels: pd.DataFrame) -> tuple[int, int]:
    """Return ``(labelled, stand_in_excluded)`` for a labels table (row 2.3.4 schema)."""
    lab = validate_labels(labels)
    n_stand = int(lab["stand_in"].sum())
    return len(lab) - n_stand, n_stand


# ---------------------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckResult:
    """Outcome of the C2.9 cost check."""

    name: str
    passed: bool
    detail: str

    def line(self) -> str:
        """The markdown PASS/FAIL line."""
        return f"- **{self.name}: {'PASS' if self.passed else 'FAIL'}.** {self.detail}"


@dataclass
class ComputeReport:
    """Rendered report plus its tables."""

    markdown: str
    table: pd.DataFrame
    stages: pd.DataFrame
    c29: CheckResult
    unreported: pd.DataFrame
    metrics: dict[str, float] = field(default_factory=dict)


def check_c29(
    actual: float, estimate: float, tolerance: float, *, placeholder: bool
) -> CheckResult:
    """C2.9 (cost half): actual GPU-seconds within ``+/- tolerance`` of the estimate."""
    if estimate > 0:
        ratio = actual / estimate
        passed = bool(actual > 0 and abs(ratio - 1) <= tolerance + 1e-12)
        dev = f"{100 * (ratio - 1):+.1f}%"
    else:
        passed, dev = False, "n/a (no estimate)"
    detail = (
        f"Total GPU-seconds {actual:,.0f} vs estimate {estimate:,.0f} ({dev}; need within "
        f"+/-{tolerance:.0%}). Covers the spend half of C2.9 only; the `dvc repro` "
        "dataset-hash half is checked separately."
    )
    if placeholder:
        detail += " Estimates are PLACEHOLDERS (placeholder: true), so this verdict is not valid."
    return CheckResult("C2.9 Cost within estimate", passed, detail)


def _fmt(v: Any, digits: int = 0) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, numbers.Real):
        return f"{v:,.{digits}f}"
    return str(v)


def _md(df: pd.DataFrame, digits: Mapping[str, int] | None = None) -> list[str]:
    digits = digits or {}
    lines = ["| " + " | ".join(df.columns) + " |", "|" + " :--- |" * len(df.columns)]
    for rec in df.to_dict(orient="records"):
        lines.append("| " + " | ".join(_fmt(rec[c], digits.get(c, 0)) for c in df.columns) + " |")
    return lines


def build_report(
    calls: pd.DataFrame,
    runs: pd.DataFrame,
    estimates: ComputeEstimates,
    *,
    labelled_pairs: int | None,
    stand_in_pairs: int = 0,
    max_unreported_listed: int = 50,
) -> ComputeReport:
    """Aggregate ledger calls and run durations per stage and render the report.

    Args:
        calls: Output of :func:`load_ledger_calls` (possibly concatenated over ledgers).
        runs: Output of :func:`normalise_runs`.
        estimates: Estimates and attribution rules.
        labelled_pairs: Number of real labelled pairs (``None`` = unknown).
        stand_in_pairs: Stand-in labels excluded from ``labelled_pairs`` (reported only).
        max_unreported_listed: Cap on ``usage_reported: false`` rows listed in the markdown (all
            are in the CSV).
    """
    calls = calls.sort_values(
        ["stage", "timestamp", "cache_key", "model"], kind="mergesort"
    ).reset_index(drop=True)
    live_calls = calls.loc[~calls["stand_in"].astype(bool)]
    finished = runs["duration_s"].notna()
    run_stand_in = runs["stand_in"].astype(bool)
    live_runs = runs.loc[finished & ~run_stand_in & (runs["stage"] != "")].copy()
    live_runs["seconds"] = live_runs["duration_s"] * live_runs["n_gpus"]

    stage_names = list(STAGES) + (
        [UNATTRIBUTED] if (live_calls["stage"] == UNATTRIBUTED).any() else []
    )
    pairs = labelled_pairs if labelled_pairs else None
    rows: list[dict[str, Any]] = []
    for stage in stage_names:
        c = live_calls.loc[live_calls["stage"] == stage]
        r = live_runs.loc[live_runs["stage"] == stage]
        gpu = float(r.loc[r["hw_class"] == "gpu", "seconds"].sum())
        rows.append(
            {
                "stage": stage,
                "calls": len(c),
                "input_tokens": int(c["input_tokens"].sum()),
                "generated_tokens": int(c["output_tokens"].sum()),
                "unreported_calls": int((~c["usage_reported"].astype(bool)).sum()),
                "runs": len(r),
                "gpu_seconds": gpu,
                "gpu_hours": gpu / 3600,
                "cpu_seconds": float(r.loc[r["hw_class"] == "cpu", "duration_s"].sum()),
                "unknown_hw_seconds": float(r.loc[r["hw_class"] == "unknown", "duration_s"].sum()),
                "gpu_seconds_per_pair": gpu / pairs if pairs else float("nan"),
            }
        )
    per_stage = pd.DataFrame(rows, columns=list(STAGE_COLUMNS))
    total = {
        "stage": "total",
        **{
            col: per_stage[col].sum()
            for col in STAGE_COLUMNS
            if col not in ("stage", "gpu_seconds_per_pair")
        },
    }
    total["gpu_seconds_per_pair"] = total["gpu_seconds"] / pairs if pairs else float("nan")
    per_stage = pd.concat([per_stage, pd.DataFrame([total])], ignore_index=True)

    est_rows: list[dict[str, Any]] = []
    for stage in (*STAGES, "total"):
        actual = per_stage.loc[per_stage["stage"] == stage].iloc[0]
        for metric in ("generated_tokens", "gpu_seconds"):
            if stage == "total":
                est = float(sum(v.get(metric, 0.0) for v in estimates.stages.values()))
                act = float(per_stage.loc[per_stage["stage"].isin(STAGES), metric].sum())
            else:
                est = float(estimates.stages.get(stage, {}).get(metric, float("nan")))
                act = float(actual[metric])
            ratio = act / est if est and not math.isnan(est) else float("nan")
            est_rows.append(
                {
                    "stage": stage,
                    "metric": metric,
                    "estimate": est,
                    "actual": act,
                    "actual_over_estimate": ratio,
                    "within_tolerance": (
                        bool(abs(ratio - 1) <= estimates.tolerance + 1e-12)
                        if not math.isnan(ratio)
                        else False
                    ),
                }
            )
    if estimates.labelled_pairs:
        est_pair = estimates.total_gpu_seconds / estimates.labelled_pairs
        act_pair = float(total["gpu_seconds_per_pair"])
        ratio = act_pair / est_pair if est_pair and not math.isnan(act_pair) else float("nan")
        est_rows.append(
            {
                "stage": "total",
                "metric": "gpu_seconds_per_pair",
                "estimate": est_pair,
                "actual": act_pair,
                "actual_over_estimate": ratio,
                "within_tolerance": (
                    bool(abs(ratio - 1) <= estimates.tolerance + 1e-12)
                    if not math.isnan(ratio)
                    else False
                ),
            }
        )
    est_table = pd.DataFrame(est_rows)

    actual_gpu = float(per_stage.loc[per_stage["stage"].isin(STAGES), "gpu_seconds"].sum())
    c29 = check_c29(
        actual_gpu,
        estimates.total_gpu_seconds,
        estimates.tolerance,
        placeholder=estimates.placeholder,
    )

    unreported = calls.loc[~calls["usage_reported"].astype(bool)].reset_index(drop=True)
    excluded_runs = runs.loc[~finished | run_stand_in | (runs["stage"] == "")]
    excluded_calls = calls.loc[calls["stand_in"].astype(bool)]

    lines = [
        "# Phase 2 compute report (row 2.4.3)",
        "",
        "Generated by `python -m terser.phase2.compute_report`. Generated tokens come from the "
        "LLM ledger (cache hits are not ledgered); GPU-seconds are MLflow run durations x GPUs for "
        "runs whose `hardware` tag is a GPU. No USD cost: every call goes to a local open model.",
        "",
    ]
    if estimates.placeholder:
        lines += [
            "> **PLACEHOLDER ESTIMATES.** `p2_compute_estimates.yaml` has `placeholder: true`; "
            "the estimate columns and the C2.9 verdict are not valid until real estimates are "
            "committed before the GPU run.",
            "",
        ]
    lines += [
        f"- Labelled pairs: **{_fmt(labelled_pairs)}**"
        + (f" ({stand_in_pairs} stand-in label(s) excluded)" if stand_in_pairs else "")
        + f"; ledger calls counted: {len(live_calls)}; runs counted: {len(live_runs)}.",
        f"- Total GPU-hours: **{total['gpu_hours']:,.2f}**; GPU-seconds per labelled pair: "
        f"**{_fmt(total['gpu_seconds_per_pair'], 3)}**.",
        "",
        "## Checkpoint",
        "",
        c29.line(),
        "",
        "## Per stage",
        "",
        *_md(
            per_stage,
            {"gpu_seconds": 1, "gpu_hours": 3, "cpu_seconds": 1, "unknown_hw_seconds": 1}
            | {"gpu_seconds_per_pair": 3},
        ),
        "",
        f"## Estimate vs actual (tolerance +/-{estimates.tolerance:.0%})",
        "",
        *_md(est_table, {"actual_over_estimate": 3, "estimate": 3, "actual": 3}),
        "",
        "## Ledger entries with `usage_reported: false`",
        "",
    ]
    if unreported.empty:
        lines.append("None: every ledger entry has server-reported token usage.")
    else:
        lines.append(
            f"**{len(unreported)} entr{'y' if len(unreported) == 1 else 'ies'}** had no usage from "
            "the server; their token counts are ~4 chars/token approximations "
            "(`terser.llm.budget.approx_tokens`), so generated-token totals are approximate."
        )
        lines.append("")
        cols = ["stage", "phase", "model", "timestamp", "output_tokens", "cache_key"]
        lines += _md(unreported[cols].head(max_unreported_listed))
        if len(unreported) > max_unreported_listed:
            lines.append(f"\n({len(unreported) - max_unreported_listed} more in the CSV.)")
    if (live_calls["stage"] == UNATTRIBUTED).any():
        lines += [
            "",
            f"> **{int((live_calls['stage'] == UNATTRIBUTED).sum())} ledger call(s) could not be "
            "attributed to a stage.** Use per-stage ledgers (`--ledger STAGE=PATH`) or "
            "`attribution` in the estimates file.",
        ]
    lines += ["", "## Excluded", ""]
    lines.append(
        f"- Stand-in ledger calls (models {', '.join(estimates.stand_in_models) or 'none'}): "
        f"{len(excluded_calls)}."
    )
    if excluded_runs.empty:
        lines.append("- Runs: none.")
    else:
        lines.append("- Runs (stand-in, unfinished or not a Phase 2 stage):")
        lines.append("")
        why = [
            "stand-in" if s else ("unfinished" if pd.isna(d) else "not a stage")
            for s, d in zip(excluded_runs["stand_in"], excluded_runs["duration_s"], strict=True)
        ]
        ex = excluded_runs[["run_id", "run_name", "task", "hardware", "status"]].assign(reason=why)
        lines += _md(ex)
    unknown = live_runs.loc[live_runs["hw_class"] == "unknown"]
    if not unknown.empty:
        lines += [
            "",
            f"> **{len(unknown)} run(s) have hardware `auto`/missing** and are not counted as "
            "GPU-seconds; re-tag them with the concrete hardware name.",
        ]
    lines.append("")

    table = pd.concat(
        [
            per_stage.assign(section="per_stage"),
            est_table.assign(section="estimate_vs_actual"),
            unreported.assign(section="usage_unreported"),
        ],
        ignore_index=True,
        sort=False,
    )
    lead = ["section", "stage", "metric"]
    for col in lead:
        if col not in table.columns:
            table[col] = pd.NA
    table = table[lead + [c for c in table.columns if c not in lead]]

    metrics = {
        "gpu_hours_total": float(total["gpu_hours"]),
        "gpu_seconds_per_pair": float(total["gpu_seconds_per_pair"]),
        "generated_tokens_total": float(total["generated_tokens"]),
        "usage_unreported_calls": float(len(unreported)),
        "c29_cost_pass": float(c29.passed),
    }
    return ComputeReport("\n".join(lines), table, per_stage, c29, unreported, metrics)


def write_report(
    report: ComputeReport, md_path: str | Path = DEFAULT_MD, csv_path: str | Path = DEFAULT_CSV
) -> tuple[Path, Path]:
    """Write markdown and CSV (UTF-8, LF); return both paths."""
    md, csv = Path(md_path), Path(csv_path)
    for p in (md, csv):
        p.parent.mkdir(parents=True, exist_ok=True)
    with md.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(report.markdown)
    report.table.to_csv(csv, index=False, lineterminator="\n", float_format="%.6f")
    return md, csv


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------


def _parse_ledger_arg(item: str) -> tuple[str | None, Path]:
    stage, sep, path = item.partition("=")
    if sep and stage in STAGES:
        return stage, Path(path)
    if sep and "/" not in stage and "\\" not in stage and ":" not in stage:
        raise SystemExit(f"--ledger: unknown stage {stage!r}; expected one of {STAGES}")
    return None, Path(item)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; writes the report and prints the C2.9 line."""
    ap = argparse.ArgumentParser(prog="python -m terser.phase2.compute_report", description=__doc__)
    ap.add_argument("--estimates", type=Path, default=DEFAULT_ESTIMATES)
    ap.add_argument(
        "--ledger",
        nargs="+",
        required=True,
        metavar="[STAGE=]PATH",
        help="ledger JSON file(s); prefix with a stage to attribute every call in that file",
    )
    runs_src = ap.add_mutually_exclusive_group()
    runs_src.add_argument("--runs", type=Path, default=None, help="runs table instead of MLflow")
    runs_src.add_argument("--no-mlflow", action="store_true", help="no run durations at all")
    ap.add_argument("--tracking-uri", default=None)
    ap.add_argument("--experiment", default=None, help="override mlflow.experiment")
    pairs_src = ap.add_mutually_exclusive_group()
    pairs_src.add_argument("--labels", type=Path, default=None)
    pairs_src.add_argument("--labelled-pairs", type=int, default=None)
    ap.add_argument("--out-md", type=Path, default=DEFAULT_MD)
    ap.add_argument("--out-csv", type=Path, default=DEFAULT_CSV)
    args = ap.parse_args(argv)

    estimates = load_estimates(args.estimates)
    frames = []
    for item in args.ledger:
        stage, path = _parse_ledger_arg(item)
        frames.append(load_ledger_calls(path, estimates, stage=stage))
    calls = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=CALL_COLUMNS)

    if args.runs is not None:
        raw_runs = read_table(args.runs)
    elif args.no_mlflow:
        raw_runs = pd.DataFrame(columns=["run_id", "task", "hardware", "duration_s"])
    else:
        raw_runs = load_mlflow_runs(args.experiment or estimates.experiment, args.tracking_uri)
    runs = normalise_runs(raw_runs, estimates)

    stand_in_pairs = 0
    if args.labels is not None:
        pairs, stand_in_pairs = count_labelled_pairs(read_table(args.labels))
    else:
        pairs = args.labelled_pairs

    report = build_report(
        calls, runs, estimates, labelled_pairs=pairs, stand_in_pairs=stand_in_pairs
    )
    md, csv = write_report(report, args.out_md, args.out_csv)
    print(report.c29.line())
    n_unrep = len(report.unreported)
    if n_unrep:
        print(
            f"WARNING: {n_unrep} ledger entr{'y' if n_unrep == 1 else 'ies'} usage_reported=false"
        )
    print(f"wrote {md} and {csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
