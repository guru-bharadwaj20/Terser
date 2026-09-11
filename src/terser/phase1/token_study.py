"""Row 1.4.2: per-operator token savings on the pilot prompt set.

For every (prompt, operator set, target) the prompt is rewritten with
``terser.operators.apply_ops(text, codes, target)`` and both versions are priced with the target's
real tokenizer (``terser.tokenizers.count_tokens``). ``S`` is tokenizer-aware, so every rewrite is
recomputed per target. Savings are ``100 * (tokens_before - tokens_after) / tokens_before``
(percent; negative when a rewrite costs more tokens).

Operator sets: the seven singletons ``F H D I L C S`` and the stacks ``FHC``, ``FHCI``, ``FHCID``,
``FHCIDS`` and ``FHCIDSL`` (all operators).

CLI (runs rows 1.4.2, 1.4.3 and 1.4.4 and writes ``reports/p1_tokenizer_study.{csv,md}`` plus the
figures)::

    python -m terser.phase1.token_study experiment=p1_token_study
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import DictConfig

from terser.config import hydra_main

__all__ = [
    "ALL_OPS",
    "OP_CODES",
    "OP_SETS",
    "RESULT_COLUMNS",
    "STACKS",
    "TARGETS",
    "ApplyFn",
    "CountFn",
    "StudyOutputs",
    "all_ops_by",
    "best_single_operator",
    "compute_savings",
    "plot_savings_by_operator",
    "run_study",
    "savings_pct",
    "summarize_by_operator",
    "write_report",
]

OP_CODES: tuple[str, ...] = ("F", "H", "D", "I", "L", "C", "S")
"""Single operators: filler, hedging, dedupe, imperative, list-ify, meta-talk, substitution."""

STACKS: tuple[str, ...] = ("FHC", "FHCI", "FHCID", "FHCIDS", "FHCIDSL")
"""Operator stacks, applied left to right."""

ALL_OPS: str = "FHCIDSL"
"""The stack containing every operator."""

OP_SETS: tuple[str, ...] = OP_CODES + STACKS
"""Every operator set studied."""

TARGETS: tuple[str, ...] = ("gpt-4o", "qwen2.5", "llama-3.1")
"""Targets priced in the study (gpt-4o is tokenizer-only; no model is called)."""

RESULT_COLUMNS: tuple[str, ...] = (
    "id",
    "category",
    "style",
    "op_set",
    "n_ops",
    "target",
    "tokens_before",
    "tokens_after",
    "savings",
    "changed",
)
"""Columns of the per-(prompt, op set, target) results table."""

ApplyFn = Callable[[str, str, str | None], str]
"""``apply_fn(text, codes, target)`` returns the rewritten text."""

CountFn = Callable[[str, str], int]
"""``count_fn(text, target)`` returns the token count."""


def _default_apply() -> ApplyFn:
    """The real operator pipeline (imported lazily so tests can inject a fake registry)."""
    import importlib

    apply_ops = importlib.import_module("terser.operators").apply_ops

    def apply(text: str, codes: str, target: str | None) -> str:
        return str(apply_ops(text, codes, target=target))

    return apply


def _default_count() -> CountFn:
    """The ground-truth tokenizer counter."""
    from terser.tokenizers import count_tokens

    return count_tokens


def savings_pct(before: int, after: int) -> float:
    """Percent of tokens saved; 0.0 for an empty prompt."""
    return 0.0 if before <= 0 else 100.0 * (before - after) / before


def compute_savings(
    prompts: Sequence[dict[str, Any]],
    *,
    targets: Sequence[str] = TARGETS,
    op_sets: Sequence[str] = OP_SETS,
    apply_fn: ApplyFn | None = None,
    count_fn: CountFn | None = None,
) -> pd.DataFrame:
    """Price every (prompt, operator set, target) rewrite.

    Args:
        prompts: Validated pilot records (see :mod:`terser.phase1.prompts`).
        targets: Tokenizer targets.
        op_sets: Operator code strings, e.g. ``"F"`` or ``"FHCI"``.
        apply_fn: Rewrite function; defaults to ``terser.operators.apply_ops``.
        count_fn: Token counter; defaults to ``terser.tokenizers.count_tokens``.

    Returns:
        A frame with :data:`RESULT_COLUMNS`, one row per (prompt, op set, target).
    """
    apply = apply_fn or _default_apply()
    count = count_fn or _default_count()
    memo: dict[tuple[str, str], int] = {}

    def cached_count(text: str, target: str) -> int:
        key = (text, target)
        if key not in memo:
            memo[key] = int(count(text, target))
        return memo[key]

    rows: list[dict[str, Any]] = []
    for rec in prompts:
        text = rec["prompt"]
        for target in targets:
            before = cached_count(text, target)
            for ops in op_sets:
                rewritten = apply(text, ops, target)
                after = cached_count(rewritten, target)
                rows.append(
                    {
                        "id": rec["id"],
                        "category": rec["category"],
                        "style": rec.get("style", ""),
                        "op_set": ops,
                        "n_ops": len(ops),
                        "target": target,
                        "tokens_before": before,
                        "tokens_after": after,
                        "savings": savings_pct(before, after),
                        "changed": rewritten != text,
                    }
                )
    return pd.DataFrame(rows, columns=list(RESULT_COLUMNS))


def _ordered(df: pd.DataFrame, column: str, order: Iterable[str]) -> pd.DataFrame:
    """Sort rows by a known value order in ``column`` (unknown values last)."""
    rank = {v: i for i, v in enumerate(order)}
    return df.assign(_rank=df[column].map(lambda v: rank.get(v, len(rank)))).sort_values(
        ["_rank"], kind="stable"
    )


def summarize_by_operator(df: pd.DataFrame) -> pd.DataFrame:
    """Median, mean, p25, p75 savings and changed share per (op set, target)."""
    grouped = df.groupby(["op_set", "target"], sort=False)
    out = grouped["savings"].agg(
        median="median",
        mean="mean",
        p25=lambda s: s.quantile(0.25),
        p75=lambda s: s.quantile(0.75),
    )
    out["changed_share"] = grouped["changed"].mean()
    out["n"] = grouped.size()
    out = out.reset_index()
    out = _ordered(out, "op_set", OP_SETS).drop(columns="_rank")
    return out.reset_index(drop=True)


def best_single_operator(df: pd.DataFrame, op_codes: Sequence[str] = OP_CODES) -> pd.DataFrame:
    """Best single operator per (prompt, target).

    Ties are broken by the order of ``op_codes``. A prompt no singleton improves gets
    ``best_op = "none"`` and its best (non-positive) savings.

    Returns:
        Columns ``id, category, style, target, best_op, best_savings``.
    """
    singles = df[df["op_set"].isin(op_codes)]
    singles = _ordered(singles, "op_set", op_codes)
    idx = singles.groupby(["id", "target"], sort=False)["savings"].idxmax()
    best = singles.loc[idx, ["id", "category", "style", "target", "op_set", "savings"]]
    best = best.rename(columns={"op_set": "best_op", "savings": "best_savings"})
    best.loc[best["best_savings"] <= 0, "best_op"] = "none"
    return best.reset_index(drop=True)


def all_ops_by(df: pd.DataFrame, by: str, op_set: str = ALL_OPS) -> pd.DataFrame:
    """Median and mean savings of ``op_set`` per ``by`` value (e.g. category or style) × target."""
    sub = df[df["op_set"] == op_set]
    out = sub.pivot_table(index=by, columns="target", values="savings", aggfunc="median")
    return out.reindex(columns=[t for t in TARGETS if t in out.columns] or list(out.columns))


def plot_savings_by_operator(summary: pd.DataFrame, path: str | Path) -> Path:
    """Grouped bars of median savings per op set × target, with p25-p75 whiskers."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ("#2a78d6", "#eb6834", "#1baf7a")
    targets = list(dict.fromkeys(summary["target"]))
    ops = [o for o in OP_SETS if o in set(summary["op_set"])]
    width = 0.8 / max(len(targets), 1)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for i, target in enumerate(targets):
        sub = summary[summary["target"] == target].set_index("op_set").reindex(ops)
        xs = [j + (i - (len(targets) - 1) / 2) * width for j in range(len(ops))]
        err = [
            (sub["median"] - sub["p25"]).clip(lower=0),
            (sub["p75"] - sub["median"]).clip(lower=0),
        ]
        ax.bar(
            xs,
            sub["median"],
            width=width * 0.9,
            color=colors[i % len(colors)],
            label=target,
            yerr=err,
            error_kw={"elinewidth": 1, "ecolor": "#52514e", "capsize": 2},
        )
    ax.axhline(0, color="#52514e", linewidth=0.8)
    ax.set_xticks(range(len(ops)), ops)
    ax.set_ylabel("Token savings, % (median; whiskers p25-p75)")
    ax.set_xlabel("Operator set")
    ax.set_title("Pilot token savings by operator set and tokenizer")
    ax.grid(axis="y", color="#e4e3df", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, ncols=len(targets), loc="upper left")
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _md(df: pd.DataFrame, floatfmt: str = ".1f", index: bool = False) -> str:
    """Render a frame as a GitHub markdown table."""
    return str(df.to_markdown(index=index, floatfmt=floatfmt))


@dataclass
class StudyOutputs:
    """Everything the study computed, for reporting and MLflow logging."""

    results: pd.DataFrame
    summary: pd.DataFrame
    best: pd.DataFrame
    metrics: dict[str, float]
    report_md: str


def write_report(
    results: pd.DataFrame,
    *,
    n_bootstrap: int = 1000,
    seed: int = 0,
    figures_dir: str | Path | None = None,
    savings_threshold: float = 10.0,
    rho_threshold: float = 0.9,
) -> StudyOutputs:
    """Build the full Phase 1 tokenizer-study report (rows 1.4.2-1.4.4) from raw results.

    Args:
        results: Output of :func:`compute_savings`.
        n_bootstrap: Bootstrap resamples for the Spearman confidence intervals.
        seed: Bootstrap RNG seed.
        figures_dir: Where to save figures; ``None`` skips them.
        savings_threshold: C1.4 threshold on median best-single-operator savings (percent).
        rho_threshold: C1.5 threshold on Spearman rho.

    Returns:
        A :class:`StudyOutputs` with the markdown report and headline metrics.
    """
    from terser.phase1.applicability import (
        applicability_markdown,
        applicability_table,
        plot_applicability,
    )
    from terser.phase1.tokenizer_agreement import (
        agreement_markdown,
        best_op_disagreement,
        c15_verdict,
        pairwise_spearman,
    )

    summary = summarize_by_operator(results)
    best = best_single_operator(results)
    targets = [t for t in TARGETS if t in set(results["target"])] or sorted(set(results["target"]))
    n_prompts = results["id"].nunique()
    metrics: dict[str, float] = {"n_prompts": float(n_prompts)}

    best_median = best.groupby("target")["best_savings"].median().reindex(targets)
    all_median = (
        results[results["op_set"] == ALL_OPS].groupby("target")["savings"].median().reindex(targets)
    )
    for t in targets:
        metrics[f"best_single_median_{t}"] = float(best_median[t])
        metrics[f"all_ops_median_{t}"] = float(all_median[t])
    c14_pass = bool((best_median >= savings_threshold).all())
    metrics["c14_pass"] = float(c14_pass)

    lines: list[str] = [
        "# Phase 1 pilot tokenizer study",
        "",
        "Rows **1.4.2** (per-operator savings), **1.4.3** (operator applicability) and "
        "**1.4.4** (cross-tokenizer agreement). Generated by "
        "`python -m terser.phase1.token_study experiment=p1_token_study`; raw rows in "
        "`reports/p1_tokenizer_study.csv`.",
        "",
        f"- Prompts: **{n_prompts}**; targets: {', '.join(targets)} (token counts only, "
        "no model calls; gpt-4o via tiktoken `o200k_base`).",
        "- Savings = `100 * (tokens_before - tokens_after) / tokens_before`; negative means the "
        "rewrite costs more tokens. `S` is tokenizer-aware, so every rewrite is computed per "
        "target.",
        "",
        "## Checkpoint verdicts",
        "",
    ]
    verdict14 = "PASS" if c14_pass else "FAIL"
    detail14 = ", ".join(f"{t} {best_median[t]:.1f}%" for t in targets)
    lines.append(
        f"- **C1.4 Savings exist: {verdict14}.** Median best-single-operator savings "
        f"(threshold >= {savings_threshold:g}% on every target): {detail14}."
    )
    agreement = pairwise_spearman(results, targets=targets, n_bootstrap=n_bootstrap, seed=seed)
    verdict15, detail15 = c15_verdict(agreement, threshold=rho_threshold)
    lines.append(f"- **C1.5 Tokenizer dependence: {verdict15}.** {detail15}")
    if not agreement.empty:
        metrics["min_rho"] = float(agreement["rho"].min())
    metrics["c15_pass"] = float(verdict15.startswith("PASS"))
    lines += [
        "",
        "## 1.4.2 Per-operator savings",
        "",
        "Savings (%) per operator set and target over all prompts; `changed` is the share of "
        "prompts whose text the operator set altered.",
        "",
        _md(summary, floatfmt=".2f"),
        "",
        "![Savings by operator](figures/p1_savings_by_operator.png)",
        "",
        "### Best single operator per prompt",
        "",
    ]
    best_tbl = best.groupby("target")["best_savings"].describe(percentiles=[0.25, 0.5, 0.75])
    lines += [_md(best_tbl.reindex(targets), floatfmt=".2f", index=True), ""]
    counts = best.pivot_table(
        index="best_op", columns="target", values="id", aggfunc="count", fill_value=0
    ).reindex(columns=targets)
    lines += [
        "How often each operator is the best single operator:",
        "",
        _md(counts, floatfmt=".0f", index=True),
        "",
        f"### All operators (`{ALL_OPS}`) median savings by category",
        "",
        _md(all_ops_by(results, "category"), floatfmt=".2f", index=True),
        "",
        f"### All operators (`{ALL_OPS}`) median savings by style",
        "",
        _md(all_ops_by(results, "style"), floatfmt=".2f", index=True),
        "",
    ]

    by_cat = applicability_table(results, by="category")
    by_style = applicability_table(results, by="style")
    lines += [applicability_markdown(by_cat, by_style), ""]

    disagreement = best_op_disagreement(best)
    metrics["best_op_disagreement"] = float(disagreement["differs"].sum())
    lines += [agreement_markdown(agreement, disagreement, n_bootstrap=n_bootstrap, seed=seed), ""]

    if figures_dir is not None:
        fig_dir = Path(figures_dir)
        plot_savings_by_operator(summary, fig_dir / "p1_savings_by_operator.png")
        plot_applicability(by_cat, fig_dir / "p1_applicability.png")

    return StudyOutputs(
        results=results,
        summary=summary,
        best=best,
        metrics=metrics,
        report_md="\n".join(lines).rstrip() + "\n",
    )


def run_study(
    prompts: Sequence[dict[str, Any]],
    reports_dir: str | Path,
    *,
    targets: Sequence[str] = TARGETS,
    op_sets: Sequence[str] = OP_SETS,
    n_bootstrap: int = 1000,
    seed: int = 0,
    apply_fn: ApplyFn | None = None,
    count_fn: CountFn | None = None,
) -> StudyOutputs:
    """Compute savings, write the CSV, markdown report and figures under ``reports_dir``."""
    results = compute_savings(
        prompts, targets=targets, op_sets=op_sets, apply_fn=apply_fn, count_fn=count_fn
    )
    root = Path(reports_dir)
    root.mkdir(parents=True, exist_ok=True)
    out = write_report(results, n_bootstrap=n_bootstrap, seed=seed, figures_dir=root / "figures")
    results.to_csv(root / "p1_tokenizer_study.csv", index=False, lineterminator="\n")
    with (root / "p1_tokenizer_study.md").open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(out.report_md)
    return out


@hydra_main()
def main(cfg: DictConfig) -> None:
    """Run rows 1.4.2-1.4.4 on the pilot set and log the run to MLflow."""
    import mlflow

    from terser.phase1.prompts import load_prompt_dir, load_prompt_file
    from terser.tracking import start_run

    p = cfg.experiment.params
    source = Path(str(p.prompts))
    prompts = load_prompt_dir(source) if source.is_dir() else load_prompt_file(source)
    reports_dir = Path(str(p.reports_dir))
    params = {
        "prompts": source.as_posix(),
        "n_prompts": len(prompts),
        "targets": ",".join(p.targets),
        "op_sets": ",".join(p.op_sets),
        "n_bootstrap": int(p.n_bootstrap),
        "seed": int(cfg.seed),
    }
    with start_run(
        phase=int(cfg.experiment.phase),
        row=str(cfg.experiment.row),
        task=str(cfg.experiment.task),
        run_name=f"1-tokenstudy-operators-pilot-{int(cfg.seed)}",
        params=params,
    ):
        out = run_study(
            prompts,
            reports_dir,
            targets=list(p.targets),
            op_sets=list(p.op_sets),
            n_bootstrap=int(p.n_bootstrap),
            seed=int(cfg.seed),
        )
        mlflow.log_metrics(out.metrics)
        for name in ("p1_tokenizer_study.csv", "p1_tokenizer_study.md"):
            mlflow.log_artifact(str(reports_dir / name), artifact_path="reports")
        for name in ("p1_savings_by_operator.png", "p1_applicability.png"):
            mlflow.log_artifact(str(reports_dir / "figures" / name), artifact_path="figures")
    print(json.dumps(out.metrics, indent=2))


if __name__ == "__main__":
    main()
