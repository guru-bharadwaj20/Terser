"""Row 2.4.2: headroom report, oracle savings per target x category and degraded share.

Inputs: the oracle table from row 2.4.1 (:mod:`terser.phase2.oracle`; recomputed here when only
candidates, token counts and labels are given), the candidates and labels tables (for degraded
shares) and the prompts table (``prompt_id, category[, style]``) when the oracle table carries no
category.

Report contents (``reports/p2_headroom.md`` + ``reports/p2_headroom.csv`` by default):

* Oracle savings per target x category and per target overall: n, median, p25, p75, mean and a
  percentile-bootstrap 95% CI of the median (``n_bootstrap`` resamples, fixed ``seed``, the same
  RNG stream for every group so results do not depend on group order).
* Share of prompts with **any** preserved (non-rejected) candidate on the target.
* Degraded share among labelled, non-rejected candidates: overall, per target, per method, per
  category, and per style when the prompts carry a ``style``. Oracle savings per target x style too.
* Checkpoint lines:

  - **C2.7** (go / no-go): median oracle savings >= 25% on >= 2 targets **and** >= 15% in >= 5 of
    the 6 categories. The per-category median pools every ``(prompt, target)`` row of the category,
    and a category with no rows counts as failing. On failure the plan's instruction is printed.
  - **C2.8** (class balance): overall degraded share among candidates in [15%, 60%].

Savings are fractions in the tables/CSV and percentages in the markdown.

CLI::

    python -m terser.phase2.headroom --oracle data/processed/oracle.csv \\
        --candidates data/interim/candidates.jsonl --labels data/processed/labels.jsonl \\
        --prompts data/interim/label_prompts.jsonl
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from terser.data.categories import Category
from terser.eval.labels import DEGRADED
from terser.phase2.oracle import (
    Phase2InputError,
    StandInDataError,
    _parse_map,
    _to_bool,
    compute_oracle,
    read_table,
    validate_candidates,
    validate_labels,
    validate_prompts,
)

__all__ = [
    "C27_CATEGORY_MIN",
    "C27_MIN_CATEGORIES",
    "C27_MIN_TARGETS",
    "C27_TARGET_MEDIAN",
    "C28_RANGE",
    "C27_FAILURE_INSTRUCTION",
    "CheckResult",
    "HeadroomReport",
    "bootstrap_median_ci",
    "build_report",
    "check_c27",
    "check_c28",
    "degraded_pool",
    "degraded_share_table",
    "main",
    "savings_table",
    "validate_oracle",
    "write_report",
]

C27_TARGET_MEDIAN: float = 0.25
C27_MIN_TARGETS: int = 2
C27_CATEGORY_MIN: float = 0.15
C27_MIN_CATEGORIES: int = 5
C28_RANGE: tuple[float, float] = (0.15, 0.60)

C27_FAILURE_INSTRUCTION: str = (
    "C2.7 failed (go / no-go). Write a negative-result report and narrow the "
    "scope (e.g. to the targets and categories that clear the bar) before continuing past Phase 2."
)

SAVINGS_COLUMNS: tuple[str, ...] = (
    "n",
    "median",
    "p25",
    "p75",
    "mean",
    "ci_low",
    "ci_high",
    "share_any_preserved",
)
SHARE_COLUMNS: tuple[str, ...] = ("n_labelled", "n_degraded", "degraded_share")

ORACLE_REQUIRED: tuple[str, ...] = ("prompt_id", "target", "savings", "has_preserved")

DEFAULT_MD = Path("reports/p2_headroom.md")
DEFAULT_CSV = Path("reports/p2_headroom.csv")


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one checkpoint check."""

    name: str
    passed: bool
    detail: str

    @property
    def verdict(self) -> str:
        """``PASS`` or ``FAIL``."""
        return "PASS" if self.passed else "FAIL"

    def line(self) -> str:
        """The markdown PASS/FAIL line."""
        return f"- **{self.name}: {self.verdict}.** {self.detail}"


@dataclass
class HeadroomReport:
    """Rendered report plus the tables behind it."""

    markdown: str
    table: pd.DataFrame
    c27: CheckResult
    c28: CheckResult
    metrics: dict[str, float] = field(default_factory=dict)
    stand_in: bool = False


# ---------------------------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------------------------


def bootstrap_median_ci(
    values: Sequence[float] | npt.NDArray[np.floating],
    *,
    n_bootstrap: int = 1000,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap CI of the median. ``(nan, nan)`` for an empty sample."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_bootstrap, arr.size))
    medians = np.median(arr[idx], axis=1)
    tail = 100 * (1 - level) / 2
    lo, hi = np.percentile(medians, [tail, 100 - tail])
    return (float(lo), float(hi))


def _blank(series: pd.Series) -> pd.Series:
    return series.isna() | (series.astype(str).str.strip() == "")


def validate_oracle(oracle: pd.DataFrame) -> pd.DataFrame:
    """Validate the oracle table (row 2.4.1) and normalise types.

    Required: ``prompt_id, target, savings, has_preserved``; ``stand_in`` defaults to ``False``.

    Raises:
        Phase2InputError: Missing columns, duplicate ``(prompt_id, target)`` rows, or savings
            outside ``[0, 1)``.
    """
    missing = [c for c in ORACLE_REQUIRED if c not in oracle.columns]
    if missing:
        raise Phase2InputError(f"oracle: missing column(s) {missing}; got {list(oracle.columns)}")
    orc = oracle.copy()
    for col in ("prompt_id", "target"):
        orc[col] = orc[col].astype(str)
    dup = orc.duplicated(["prompt_id", "target"], keep=False)
    if dup.any():
        raise Phase2InputError(f"oracle: {int(dup.sum())} duplicate (prompt_id, target) row(s)")
    savings = pd.to_numeric(orc["savings"], errors="coerce")
    if savings.isna().any() or (savings < 0).any() or (savings >= 1).any():
        raise Phase2InputError("oracle: 'savings' must be numbers in [0, 1)")
    orc["savings"] = savings.astype(float)
    orc["has_preserved"] = [_to_bool(v, "has_preserved") for v in orc["has_preserved"]]
    if "stand_in" in orc.columns:
        orc["stand_in"] = [_to_bool(v, "stand_in") for v in orc["stand_in"]]
    else:
        orc["stand_in"] = False
    return orc


def _category_order(values: Sequence[str]) -> list[str]:
    known = [c.value for c in Category]
    present = set(values)
    return [c for c in known if c in present] + sorted(present - set(known))


def savings_table(
    oracle: pd.DataFrame, by: Sequence[str], *, n_bootstrap: int = 1000, seed: int = 0
) -> pd.DataFrame:
    """Oracle savings summary grouped by ``by`` (empty ``by`` = one overall row)."""
    rows: list[dict[str, object]] = []
    groups = oracle.groupby(list(by), sort=True, dropna=False) if by else [((), oracle)]
    for key, grp in groups:
        keys = key if isinstance(key, tuple) else (key,)
        s = grp["savings"].to_numpy(dtype=float)
        lo, hi = bootstrap_median_ci(s, n_bootstrap=n_bootstrap, seed=seed)
        rows.append(
            {
                **dict(zip(by, keys, strict=True)),
                "n": int(s.size),
                "median": float(np.median(s)),
                "p25": float(np.percentile(s, 25)),
                "p75": float(np.percentile(s, 75)),
                "mean": float(np.mean(s)),
                "ci_low": lo,
                "ci_high": hi,
                "share_any_preserved": float(grp["has_preserved"].astype(bool).mean()),
            }
        )
    return pd.DataFrame(rows, columns=[*by, *SAVINGS_COLUMNS])


def degraded_pool(
    candidates: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    prompts: pd.DataFrame | None = None,
    target_map: Mapping[str, str] | None = None,
    targets: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Labelled, non-rejected candidates with ``method``, ``category`` and optional ``style``."""
    cand = validate_candidates(candidates)
    lab = validate_labels(labels, target_map=target_map)
    if targets is not None:
        lab = lab.loc[lab["target"].isin(list(targets))]
    lab_cols = ["prompt_id", "candidate_id", "target", "label_int", "stand_in"]
    if prompts is None and "category" in lab.columns:
        lab_cols.append("category")
    live = cand.loc[~cand["rejected"], ["prompt_id", "candidate_id", "method", "stand_in"]]
    pool = lab[lab_cols].merge(
        live, on=["prompt_id", "candidate_id"], how="inner", suffixes=("", "_cand")
    )
    pool["stand_in"] = pool["stand_in"] | pool.pop("stand_in_cand")
    if prompts is not None:
        pool = pool.merge(validate_prompts(prompts), on="prompt_id", how="left")
    if "category" not in pool.columns:
        raise Phase2InputError("degraded shares need a category: pass prompts or label categories")
    if _blank(pool["category"]).any():
        n_bad = int(_blank(pool["category"]).sum())
        raise Phase2InputError(f"degraded shares: {n_bad} labelled candidate(s) without a category")
    pool["category"] = pool["category"].astype(str).str.strip().str.lower()
    pool["degraded"] = pool["label_int"] == DEGRADED
    return pool


def degraded_share_table(pool: pd.DataFrame, by: Sequence[str]) -> pd.DataFrame:
    """Degraded share among labelled candidates grouped by ``by`` (empty = overall)."""
    rows: list[dict[str, object]] = []
    groups = pool.groupby(list(by), sort=True, dropna=False) if by else [((), pool)]
    for key, grp in groups:
        keys = key if isinstance(key, tuple) else (key,)
        n = len(grp)
        k = int(grp["degraded"].sum())
        rows.append(
            {
                **dict(zip(by, keys, strict=True)),
                "n_labelled": n,
                "n_degraded": k,
                "degraded_share": k / n if n else float("nan"),
            }
        )
    return pd.DataFrame(rows, columns=[*by, *SHARE_COLUMNS])


# ---------------------------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------------------------


def _pct(x: float) -> str:
    return "n/a" if pd.isna(x) else f"{100 * x:.1f}%"


def check_c27(oracle: pd.DataFrame) -> CheckResult:
    """C2.7: median >= 25% on >= 2 targets and >= 15% in >= 5 of the 6 categories."""
    per_target = oracle.groupby("target", sort=True)["savings"].median()
    ok_targets = [t for t, m in per_target.items() if m >= C27_TARGET_MEDIAN]
    per_cat = oracle.groupby("category", sort=True)["savings"].median()
    cats = [c.value for c in Category]
    ok_cats = [c for c in cats if c in per_cat.index and per_cat[c] >= C27_CATEGORY_MIN]
    missing = [c for c in cats if c not in per_cat.index]
    targets_ok = len(ok_targets) >= C27_MIN_TARGETS
    cats_ok = len(ok_cats) >= C27_MIN_CATEGORIES
    t_detail = ", ".join(f"{t} {_pct(m)}" for t, m in per_target.items())
    c_detail = ", ".join(f"{c} {_pct(per_cat.get(c, float('nan')))}" for c in cats)
    detail = (
        f"Targets with median >= {C27_TARGET_MEDIAN:.0%}: {len(ok_targets)} "
        f"(need >= {C27_MIN_TARGETS}; {t_detail}) [{'ok' if targets_ok else 'fail'}]. "
        f"Categories with median >= {C27_CATEGORY_MIN:.0%} (pooled over targets): "
        f"{len(ok_cats)} of {len(cats)} (need >= {C27_MIN_CATEGORIES}; {c_detail}) "
        f"[{'ok' if cats_ok else 'fail'}]."
    )
    if missing:
        detail += f" Categories with no rows (counted as failing): {', '.join(missing)}."
    return CheckResult("C2.7 Go / no-go: oracle savings", targets_ok and cats_ok, detail)


def check_c28(share: float, n_labelled: int) -> CheckResult:
    """C2.8: overall degraded share among candidates in [15%, 60%]."""
    lo, hi = C28_RANGE
    passed = bool(n_labelled > 0 and lo <= share <= hi)
    detail = (
        f"Degraded share among {n_labelled} labelled candidates: {_pct(share)} "
        f"(need {lo:.0%} to {hi:.0%})."
    )
    return CheckResult("C2.8 Class balance", passed, detail)


# ---------------------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------------------


def _md_table(df: pd.DataFrame, keys: Sequence[str]) -> list[str]:
    pct_cols = {"median", "p25", "p75", "mean", "share_any_preserved", "degraded_share"}
    cols = [str(c) for c in keys] + [str(c) for c in df.columns if c not in keys]
    labels = {"ci_low": "95% CI (median)", "share_any_preserved": "any preserved"}
    header = [labels.get(c, c) for c in cols if c != "ci_high"]
    lines = ["| " + " | ".join(header) + " |", "|" + " :--- |" * len(header)]
    for rec in df.to_dict(orient="records"):
        cells: list[str] = []
        for c in cols:
            if c == "ci_high":
                continue
            v = rec[c]
            if c == "ci_low":
                cells.append(f"[{_pct(v)}, {_pct(rec['ci_high'])}]")
            elif c in pct_cols:
                cells.append(_pct(float(v)))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _sort_categories(df: pd.DataFrame) -> pd.DataFrame:
    if "category" not in df.columns or df.empty:
        return df
    order = {c: i for i, c in enumerate(_category_order(list(df["category"].astype(str))))}
    rank = df["category"].astype(str).map(order)
    other = [c for c in df.columns if c in ("target",)]
    return (
        df.assign(_r=rank).sort_values([*other, "_r"], kind="mergesort").drop(columns="_r")
    ).reset_index(drop=True)


def build_report(
    oracle: pd.DataFrame,
    candidates: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    prompts: pd.DataFrame | None = None,
    target_map: Mapping[str, str] | None = None,
    n_bootstrap: int = 1000,
    seed: int = 0,
    allow_stand_in: bool = False,
) -> HeadroomReport:
    """Compute every headroom table and render the markdown report.

    Raises:
        StandInDataError: If stand-in data is present and ``allow_stand_in`` is false.
        Phase2InputError: If the oracle table has no category and no prompts table is given.
    """
    orc = validate_oracle(oracle)
    if prompts is not None:
        meta = validate_prompts(prompts)
        orc = orc.drop(columns=[c for c in ("category", "style") if c in orc.columns])
        orc = orc.merge(meta, on="prompt_id", how="left")
    if "category" not in orc.columns or _blank(orc["category"]).any():
        raise Phase2InputError("oracle rows without a category: pass --prompts")
    orc["category"] = orc["category"].astype(str).str.strip().str.lower()
    if "style" in orc.columns:
        orc.loc[_blank(orc["style"]), "style"] = pd.NA
    has_style = "style" in orc.columns and bool(orc["style"].notna().any())
    if prompts is None:
        # Categories/styles for the degraded shares come from the oracle table itself.
        meta_cols = ["prompt_id", "category"] + (["style"] if has_style else [])
        prompts = orc[meta_cols].drop_duplicates("prompt_id")

    pool = degraded_pool(
        candidates,
        labels,
        prompts=prompts,
        target_map=target_map,
        targets=sorted(set(orc["target"])),
    )
    raw_cand_stand_in = validate_candidates(candidates)["stand_in"].any()
    raw_label_stand_in = "stand_in" in labels.columns and any(
        _to_bool(v, "stand_in") for v in labels["stand_in"]
    )
    stand_in = bool(
        orc["stand_in"].any() or pool["stand_in"].any() or raw_cand_stand_in or raw_label_stand_in
    )
    if stand_in and not allow_stand_in:
        raise StandInDataError(
            "stand-in data in the oracle table, candidates or labels; stand-in outputs are never "
            "labels (row 2.2.4). Pass allow_stand_in=True / --allow-stand-in for smoke tests only"
        )

    kw = {"n_bootstrap": n_bootstrap, "seed": seed}
    sav_target = savings_table(orc, ["target"], **kw)
    sav_tc = _sort_categories(savings_table(orc, ["target", "category"], **kw))
    sav_cat = _sort_categories(savings_table(orc, ["category"], **kw))
    share_all = degraded_share_table(pool, [])
    share_target = degraded_share_table(pool, ["target"])
    share_method = degraded_share_table(pool, ["method"])
    share_cat = _sort_categories(degraded_share_table(pool, ["category"]))
    sav_style = savings_table(orc, ["target", "style"], **kw) if has_style else None
    share_style = (
        degraded_share_table(pool, ["style"]) if has_style and "style" in pool.columns else None
    )

    c27 = check_c27(orc)
    overall_share = float(share_all["degraded_share"].iloc[0]) if len(pool) else float("nan")
    c28 = check_c28(overall_share, len(pool))

    sections: list[tuple[str, pd.DataFrame, list[str]]] = [
        ("savings_by_target", sav_target, ["target"]),
        ("savings_by_target_category", sav_tc, ["target", "category"]),
        ("savings_by_category", sav_cat, ["category"]),
        ("degraded_overall", share_all, []),
        ("degraded_by_target", share_target, ["target"]),
        ("degraded_by_method", share_method, ["method"]),
        ("degraded_by_category", share_cat, ["category"]),
    ]
    if sav_style is not None:
        sections.append(("savings_by_target_style", sav_style, ["target", "style"]))
    if share_style is not None:
        sections.append(("degraded_by_style", share_style, ["style"]))
    table = pd.concat(
        [df.assign(section=name) for name, df, _ in sections], ignore_index=True, sort=False
    )
    key_cols = ["section", "target", "category", "method", "style"]
    for col in key_cols:
        if col not in table.columns:
            table[col] = pd.NA
    table = table[key_cols + [*SAVINGS_COLUMNS, *SHARE_COLUMNS]]

    n_prompts = orc["prompt_id"].nunique()
    lines = [
        "# Phase 2 headroom report (rows 2.4.1-2.4.2)",
        "",
        "Generated by `python -m terser.phase2.headroom`. Oracle = shortest non-rejected candidate "
        "labelled `preserved` on the target and shorter than the original on the target's "
        "tokenizer; prompts with none get the original (savings 0). Savings "
        "`(n_T(x) - n_T(x')) / n_T(x)` per ADR-0002.",
        "",
    ]
    if stand_in:
        lines += [
            "> **STAND-IN DATA.** This report includes CPU smoke-test outputs "
            "(`stand_in: true`). The numbers and verdicts below are not valid for C2.7 / C2.8.",
            "",
        ]
    lines += [
        f"- Prompts: **{n_prompts}**; targets: {', '.join(sorted(set(orc['target'])))}; "
        f"labelled candidates: **{len(pool)}**; bootstrap: {n_bootstrap} resamples, seed {seed}.",
        "",
        "## Checkpoints",
        "",
        c27.line(),
        c28.line(),
    ]
    if not c27.passed:
        lines += ["", f"> **{C27_FAILURE_INSTRUCTION}**"]
    titles = {
        "savings_by_target": "Oracle savings per target",
        "savings_by_target_category": "Oracle savings per target x category",
        "savings_by_category": "Oracle savings per category (pooled over targets, used by C2.7)",
        "degraded_overall": "Degraded share among candidates (overall, used by C2.8)",
        "degraded_by_target": "Degraded share per target",
        "degraded_by_method": "Degraded share per method",
        "degraded_by_category": "Degraded share per category",
        "savings_by_target_style": "Oracle savings per target x style",
        "degraded_by_style": "Degraded share per style",
    }
    for name, df, keys in sections:
        lines += ["", f"## {titles[name]}", ""] + _md_table(df, keys)
    lines.append("")

    metrics: dict[str, float] = {
        f"median_savings_{t}": float(m)
        for t, m in zip(sav_target["target"], sav_target["median"], strict=True)
    }
    metrics["degraded_share"] = overall_share
    metrics["c27_pass"] = float(c27.passed)
    metrics["c28_pass"] = float(c28.passed)
    return HeadroomReport("\n".join(lines), table, c27, c28, metrics, stand_in)


def write_report(
    report: HeadroomReport, md_path: str | Path = DEFAULT_MD, csv_path: str | Path = DEFAULT_CSV
) -> tuple[Path, Path]:
    """Write the markdown and CSV (UTF-8, LF line endings); return both paths."""
    md, csv = Path(md_path), Path(csv_path)
    for p in (md, csv):
        p.parent.mkdir(parents=True, exist_ok=True)
    with md.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(report.markdown)
    report.table.to_csv(csv, index=False, lineterminator="\n", float_format="%.6f")
    return md, csv


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Exit code 0 even when a check fails (the verdict is in the report)."""
    ap = argparse.ArgumentParser(prog="python -m terser.phase2.headroom", description=__doc__)
    ap.add_argument("--oracle", type=Path, default=None, help="oracle table; else recomputed")
    ap.add_argument("--candidates", required=True, type=Path)
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--token-counts", type=Path, default=None, help="needed without --oracle")
    ap.add_argument("--prompts", type=Path, default=None)
    ap.add_argument("--target-map", nargs="*", default=None, metavar="LABEL=TOKENS")
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-stand-in", action="store_true")
    ap.add_argument("--out-md", type=Path, default=DEFAULT_MD)
    ap.add_argument("--out-csv", type=Path, default=DEFAULT_CSV)
    args = ap.parse_args(argv)

    candidates = read_table(args.candidates)
    labels = read_table(args.labels)
    prompts = read_table(args.prompts) if args.prompts else None
    target_map = _parse_map(args.target_map)
    if args.oracle is not None:
        oracle = read_table(args.oracle)
    elif args.token_counts is not None:
        oracle = compute_oracle(
            candidates,
            read_table(args.token_counts),
            labels,
            prompts=prompts,
            target_map=target_map,
            allow_stand_in=args.allow_stand_in,
        )
    else:
        ap.error("pass --oracle or --token-counts")
    report = build_report(
        oracle,
        candidates,
        labels,
        prompts=prompts,
        target_map=target_map,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        allow_stand_in=args.allow_stand_in,
    )
    md, csv = write_report(report, args.out_md, args.out_csv)
    if report.stand_in:
        print("STAND-IN DATA: verdicts below are not valid for C2.7 / C2.8")
    print(report.c27.line())
    print(report.c28.line())
    if not report.c27.passed:
        print(C27_FAILURE_INSTRUCTION)
    print(f"wrote {md} and {csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
