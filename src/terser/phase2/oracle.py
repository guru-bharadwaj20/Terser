"""Row 2.4.1: the oracle rewrite, the shortest candidate still ``preserved``, per prompt and target.

Definition (``docs/decisions/ADR-0002-problem-framing.md``, sub-problems 2, 3 and 5):

* For every ``(prompt_id, target)`` the **eligible** candidates are those that are **not rejected**,
  are labelled ``preserved`` on that target (row 2.3.4), and are **strictly shorter** than the
  original prompt on the target's own tokenizer (row 2.2.2 counts). Rewrites that are not shorter
  are never served (row 8.3.7), so they cannot be the oracle.
* The oracle is the eligible candidate with the fewest target tokens. Ties are broken
  deterministically by (1) fewer tokens, (2) method priority (``method_priority``, earlier wins;
  methods not listed rank after all listed ones, alphabetically), (3) ``candidate_id``
  (lexicographic).
* **If no candidate is eligible, the oracle is the original prompt** (``oracle_candidate_id =
  "original"``, ``method = "original"``) with **savings 0**. This matches the ADR: a system that
  returns the original has savings 0, and the savings predictor's label for such prompts is 0.
* ``savings = (n_T(x) - n_T(x')) / n_T(x)`` as a fraction in ``[0, 1)``.

Inputs (validated here; the producing modules are not imported):

* **candidates** (``candidates`` stage, row 2.2.1): ``prompt_id, candidate_id, method, text,
  generator, stand_in, rejected, rejection_reason``.
* **token counts** (row 2.2.2): ``prompt_id, candidate_id, target, tokens``; the original prompt
  uses ``candidate_id = "original"``.
* **labels** (row 2.3.4): ``prompt_id, candidate_id, target, label`` with label ``preserved`` /
  ``degraded`` (``None``/empty = unlabelled, ignored). ``target_model`` is accepted in place of
  ``target``; ``target_map`` translates label targets (e.g. HF model ids) to token-count targets.
  An optional ``stand_in`` column is honoured.
* **prompts** (optional): ``prompt_id, category`` and optionally ``style``, copied onto the table.

Stand-in data (CPU smoke-test outputs, row 2.2.4) is refused unless ``allow_stand_in=True``; every
oracle row records whether stand-in data was involved.

CLI::

    python -m terser.phase2.oracle --candidates data/interim/candidates.jsonl \\
        --token-counts data/interim/token_counts.parquet --labels data/processed/labels.jsonl \\
        --prompts data/interim/label_prompts.jsonl --out data/processed/oracle.csv
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from terser.eval.labels import LABEL_NAMES, PRESERVED, to_binary

__all__ = [
    "CANDIDATE_COLUMNS",
    "DEFAULT_METHOD_PRIORITY",
    "LABEL_COLUMNS",
    "ORACLE_COLUMNS",
    "ORIGINAL_ID",
    "PROMPT_COLUMNS",
    "TOKEN_COUNT_COLUMNS",
    "Phase2InputError",
    "StandInDataError",
    "compute_oracle",
    "main",
    "method_rank",
    "read_table",
    "validate_candidates",
    "validate_labels",
    "validate_prompts",
    "validate_token_counts",
    "write_table",
]

ORIGINAL_ID: str = "original"
"""``candidate_id`` of the original prompt in the token-count table, and the oracle fallback."""

CANDIDATE_COLUMNS: tuple[str, ...] = (
    "prompt_id",
    "candidate_id",
    "method",
    "text",
    "generator",
    "stand_in",
    "rejected",
    "rejection_reason",
)
TOKEN_COUNT_COLUMNS: tuple[str, ...] = ("prompt_id", "candidate_id", "target", "tokens")
LABEL_COLUMNS: tuple[str, ...] = ("prompt_id", "candidate_id", "target", "label")
PROMPT_COLUMNS: tuple[str, ...] = ("prompt_id", "category")

DEFAULT_METHOD_PRIORITY: tuple[str, ...] = (
    "operators",
    "operator_stack",
    "llmlingua2",
    "llm_low",
    "llm_medium",
    "llm_high",
)
"""Tie-break order among equally short candidates: deterministic, cheaper methods first."""

ORACLE_COLUMNS: tuple[str, ...] = (
    "prompt_id",
    "target",
    "category",
    "style",
    "oracle_candidate_id",
    "method",
    "tokens_original",
    "tokens_oracle",
    "savings",
    "is_original",
    "has_preserved",
    "n_candidates",
    "n_labelled",
    "n_preserved",
    "stand_in",
)
"""Oracle table columns. ``category``/``style`` are empty when no prompts table is given."""


class Phase2InputError(ValueError):
    """An input table does not match the expected schema."""


class StandInDataError(Phase2InputError):
    """Stand-in (CPU smoke-test) data was found and ``allow_stand_in`` is false."""


# ---------------------------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------------------------


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a ``.jsonl``, ``.json`` (list of records), ``.csv`` or ``.parquet`` table.

    ID-like columns are read as strings for CSV so ``"001"`` stays ``"001"``.
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".jsonl":
        rows = [
            json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        return pd.DataFrame(rows)
    if suffix == ".json":
        return pd.DataFrame(json.loads(p.read_text(encoding="utf-8")))
    if suffix == ".csv":
        return pd.read_csv(
            p, dtype={"prompt_id": str, "candidate_id": str, "oracle_candidate_id": str}
        )
    if suffix == ".parquet":
        return pd.read_parquet(p)
    raise Phase2InputError(f"unsupported table format {suffix!r} for {p}")


def write_table(df: pd.DataFrame, path: str | Path) -> Path:
    """Write ``df`` as CSV (LF line endings) or JSONL, chosen by extension; return the path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() == ".jsonl":
        with p.open("w", encoding="utf-8", newline="\n") as fh:
            for rec in df.to_dict(orient="records"):
                fh.write(json.dumps(rec, sort_keys=False, default=str) + "\n")
    else:
        df.to_csv(p, index=False, lineterminator="\n")
    return p


# ---------------------------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------------------------

_TRUE = {"true", "1", "yes", "y", "t"}
_FALSE = {"false", "0", "no", "n", "f", ""}


def _to_bool(value: Any, column: str) -> bool:
    """Coerce JSON/CSV booleans (``True``, ``"false"``, ``0``); missing values are ``False``."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    if isinstance(value, bool | np.bool_):
        return bool(value)
    if isinstance(value, int | float | np.integer) and value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in _TRUE | _FALSE:
        return value.strip().lower() in _TRUE
    raise Phase2InputError(f"column {column!r}: cannot interpret {value!r} as a boolean")


def _require(df: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise Phase2InputError(f"{name}: missing column(s) {missing}; got {list(df.columns)}")


def _require_keys(df: pd.DataFrame, keys: Sequence[str], name: str) -> None:
    for key in keys:
        bad = df[key].isna() | (df[key].astype(str).str.strip() == "")
        if bad.any():
            raise Phase2InputError(f"{name}: {int(bad.sum())} row(s) with empty {key!r}")
    dup = df.duplicated(list(keys), keep=False)
    if dup.any():
        example = df.loc[dup, list(keys)].iloc[0].to_dict()
        raise Phase2InputError(
            f"{name}: {int(dup.sum())} duplicate row(s) on {list(keys)}, e.g. {example}"
        )


def validate_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalise the candidates table (ids as str, ``stand_in``/``rejected`` bool).

    ``text``, ``generator`` and ``rejection_reason`` must be present as columns but are not used.

    Raises:
        Phase2InputError: Missing columns, empty/duplicate keys, a candidate named ``original``,
            or uninterpretable booleans.
    """
    _require(df, CANDIDATE_COLUMNS, "candidates")
    out = df.copy()
    out["prompt_id"] = out["prompt_id"].astype(str)
    out["candidate_id"] = out["candidate_id"].astype(str)
    _require_keys(out, ("prompt_id", "candidate_id"), "candidates")
    if (out["candidate_id"] == ORIGINAL_ID).any():
        raise Phase2InputError(f"candidates: candidate_id {ORIGINAL_ID!r} is reserved")
    if out["method"].isna().any():
        raise Phase2InputError("candidates: empty 'method'")
    out["method"] = out["method"].astype(str)
    out["stand_in"] = [_to_bool(v, "stand_in") for v in out["stand_in"]]
    out["rejected"] = [_to_bool(v, "rejected") for v in out["rejected"]]
    return out


def validate_token_counts(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the token-count table: unique keys, integer ``tokens >= 0``, originals ``> 0``.

    The token-count stage (:mod:`terser.data.token_counts`) ids originals ``<prompt_id>:original``;
    those are normalised to ``ORIGINAL_ID``.

    Raises:
        Phase2InputError: On any schema problem.
    """
    _require(df, TOKEN_COUNT_COLUMNS, "token_counts")
    out = df.copy()
    for col in ("prompt_id", "candidate_id", "target"):
        out[col] = out[col].astype(str)
    stage_orig = out["candidate_id"] == out["prompt_id"] + ":" + ORIGINAL_ID
    out.loc[stage_orig, "candidate_id"] = ORIGINAL_ID
    _require_keys(out, ("prompt_id", "candidate_id", "target"), "token_counts")
    tokens = pd.to_numeric(out["tokens"], errors="coerce")
    if tokens.isna().any() or (tokens % 1 != 0).any() or (tokens < 0).any():
        raise Phase2InputError("token_counts: 'tokens' must be non-negative integers")
    out["tokens"] = tokens.astype("int64")
    orig = out["candidate_id"] == ORIGINAL_ID
    if (out.loc[orig, "tokens"] <= 0).any():
        raise Phase2InputError("token_counts: original prompts must have tokens > 0")
    return out


def validate_labels(
    df: pd.DataFrame, *, target_map: Mapping[str, str] | None = None
) -> pd.DataFrame:
    """Validate labels; add ``label_int`` (0 preserved, 1 degraded) and drop unlabelled rows.

    Accepts ``target_model`` when ``target`` is absent. ``target_map`` rewrites label targets to
    the token-count target names. An optional ``stand_in`` column is coerced to bool (default
    ``False``).

    Raises:
        Phase2InputError: On missing columns, duplicate keys or unknown label values.
    """
    out = df.copy()
    if "target" not in out.columns and "target_model" in out.columns:
        out = out.rename(columns={"target_model": "target"})
    _require(out, LABEL_COLUMNS, "labels")
    for col in ("prompt_id", "candidate_id", "target"):
        out[col] = out[col].astype(str)
    if target_map:
        out["target"] = out["target"].map(lambda t: target_map.get(t, t))
    unlabelled = out["label"].isna() | (out["label"].astype(str).str.strip() == "")
    out = out.loc[~unlabelled].copy()
    _require_keys(out, ("prompt_id", "candidate_id", "target"), "labels")
    try:
        out["label_int"] = [to_binary(v) for v in out["label"]]
    except (TypeError, ValueError) as exc:
        raise Phase2InputError(f"labels: {exc}") from exc
    out["label"] = out["label_int"].map(LABEL_NAMES)
    if "stand_in" in out.columns:
        out["stand_in"] = [_to_bool(v, "stand_in") for v in out["stand_in"]]
    else:
        out["stand_in"] = False
    return out


def validate_prompts(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the prompts table (``prompt_id`` unique, ``category`` present; ``style`` optional).

    A casualised-prompt table (``label_prompts.jsonl``) keys prompts by ``id``; that column is
    used as ``prompt_id`` when ``prompt_id`` itself is absent.

    Raises:
        Phase2InputError: On missing columns or duplicate/empty keys.
    """
    if "prompt_id" not in df.columns and "id" in df.columns:
        df = df.rename(columns={"id": "prompt_id"})
    _require(df, PROMPT_COLUMNS, "prompts")
    cols = ["prompt_id", "category"] + (["style"] if "style" in df.columns else [])
    out = df[cols].copy()
    out["prompt_id"] = out["prompt_id"].astype(str)
    _require_keys(out, ("prompt_id",), "prompts")
    if out["category"].isna().any():
        raise Phase2InputError("prompts: empty 'category'")
    out["category"] = out["category"].astype(str).str.strip().str.lower()
    return out


# ---------------------------------------------------------------------------------------------
# Oracle
# ---------------------------------------------------------------------------------------------


def method_rank(method: str, priority: Sequence[str]) -> tuple[int, str]:
    """Sort key for ``method``: its position in ``priority``, else after all of them by name."""
    try:
        return (list(priority).index(method), "")
    except ValueError:
        return (len(priority), method)


def compute_oracle(
    candidates: pd.DataFrame,
    token_counts: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    prompts: pd.DataFrame | None = None,
    targets: Sequence[str] | None = None,
    method_priority: Sequence[str] = DEFAULT_METHOD_PRIORITY,
    target_map: Mapping[str, str] | None = None,
    allow_stand_in: bool = False,
) -> pd.DataFrame:
    """Build the oracle table (one row per ``(prompt_id, target)``, columns :data:`ORACLE_COLUMNS`).

    Args:
        candidates: Candidates table (see module docstring).
        token_counts: Token counts including ``candidate_id = "original"`` rows.
        labels: Label records.
        prompts: Optional prompt metadata (``category``, ``style``). Prompts listed here without
            candidates still get an oracle row (the original, savings 0).
        targets: Targets to build; default = every target present in ``labels``.
        method_priority: Tie-break order of methods (see module docstring).
        target_map: Label-target to token-count-target renaming.
        allow_stand_in: Accept stand-in candidates/labels (rows are flagged ``stand_in``).

    Returns:
        The oracle table sorted by ``target, prompt_id``.

    Raises:
        StandInDataError: Stand-in data present and ``allow_stand_in`` is false.
        Phase2InputError: Schema problems, labels for unknown candidates, or missing token counts
            for the original or for a labelled, non-rejected candidate.
    """
    cand = validate_candidates(candidates)
    tc = validate_token_counts(token_counts)
    lab = validate_labels(labels, target_map=target_map)
    meta = validate_prompts(prompts) if prompts is not None else None

    if not allow_stand_in:
        n_c = int(cand["stand_in"].sum())
        # Count on the raw table: an unlabelled stand-in record is still stand-in data.
        raw = labels["stand_in"] if "stand_in" in labels.columns else ()
        n_l = sum(_to_bool(v, "stand_in") for v in raw)
        if n_c or n_l:
            raise StandInDataError(
                f"stand-in data found ({n_c} candidate(s), {n_l} label(s)); stand-in outputs are "
                "never labels (row 2.2.4). Pass allow_stand_in=True / --allow-stand-in for "
                "smoke tests only."
            )

    target_list = sorted(set(lab["target"])) if targets is None else list(dict.fromkeys(targets))
    lab = lab.loc[lab["target"].isin(target_list)]

    known = set(zip(cand["prompt_id"], cand["candidate_id"], strict=True))
    unknown = [k for k in zip(lab["prompt_id"], lab["candidate_id"], strict=True) if k not in known]
    if unknown:
        raise Phase2InputError(
            f"labels: {len(unknown)} label(s) refer to unknown candidates, e.g. {unknown[0]}"
        )

    prompt_ids = set(cand["prompt_id"])
    if meta is not None:
        prompt_ids |= set(meta["prompt_id"])
    universe = pd.DataFrame(
        [(p, t) for t in target_list for p in sorted(prompt_ids)], columns=["prompt_id", "target"]
    )

    originals = tc.loc[tc["candidate_id"] == ORIGINAL_ID, ["prompt_id", "target", "tokens"]]
    universe = universe.merge(originals, on=["prompt_id", "target"], how="left")
    if universe["tokens"].isna().any():
        miss = universe.loc[universe["tokens"].isna(), ["prompt_id", "target"]].iloc[0].to_dict()
        n_miss = int(universe["tokens"].isna().sum())
        raise Phase2InputError(f"token_counts: {n_miss} original count(s) missing, e.g. {miss}")
    universe = universe.rename(columns={"tokens": "tokens_original"})
    universe["tokens_original"] = universe["tokens_original"].astype("int64")

    live = cand.loc[~cand["rejected"], ["prompt_id", "candidate_id", "method", "stand_in"]]
    pool = lab.merge(live, on=["prompt_id", "candidate_id"], how="inner", suffixes=("", "_cand"))
    pool = pool.merge(tc, on=["prompt_id", "candidate_id", "target"], how="left")
    if pool["tokens"].isna().any():
        miss = pool.loc[pool["tokens"].isna(), ["prompt_id", "candidate_id", "target"]]
        raise Phase2InputError(
            f"token_counts: {len(miss)} labelled candidate count(s) missing, "
            f"e.g. {miss.iloc[0].to_dict()}"
        )
    pool = pool.merge(
        universe[["prompt_id", "target", "tokens_original"]], on=["prompt_id", "target"]
    )
    pool["tokens"] = pool["tokens"].astype("int64")
    pool["preserved"] = pool["label_int"] == PRESERVED

    # Counts per (prompt, target).
    n_live = live.groupby("prompt_id").size().rename("n_candidates")
    stats = (
        pool.groupby(["prompt_id", "target"])
        .agg(n_labelled=("candidate_id", "size"), n_preserved=("preserved", "sum"))
        .reset_index()
    )

    eligible = pool.loc[pool["preserved"] & (pool["tokens"] < pool["tokens_original"])].copy()
    ranks = {m: method_rank(m, method_priority) for m in set(eligible["method"])}
    eligible["_rank_pos"] = eligible["method"].map(lambda m: ranks[m][0])
    eligible["_rank_name"] = eligible["method"].map(lambda m: ranks[m][1])
    eligible = eligible.sort_values(
        ["prompt_id", "target", "tokens", "_rank_pos", "_rank_name", "candidate_id"],
        kind="mergesort",
    )
    best = eligible.drop_duplicates(["prompt_id", "target"], keep="first")[
        ["prompt_id", "target", "candidate_id", "method", "tokens"]
    ].rename(columns={"candidate_id": "oracle_candidate_id", "tokens": "tokens_oracle"})

    out = universe.merge(best, on=["prompt_id", "target"], how="left")
    out = out.merge(stats, on=["prompt_id", "target"], how="left")
    out = out.merge(n_live, left_on="prompt_id", right_index=True, how="left")
    for col in ("n_labelled", "n_preserved", "n_candidates"):
        out[col] = out[col].fillna(0).astype("int64")
    out["is_original"] = out["oracle_candidate_id"].isna()
    out["oracle_candidate_id"] = out["oracle_candidate_id"].fillna(ORIGINAL_ID)
    out["method"] = out["method"].fillna(ORIGINAL_ID)
    out["tokens_oracle"] = out["tokens_oracle"].fillna(out["tokens_original"]).astype("int64")
    out["savings"] = (out["tokens_original"] - out["tokens_oracle"]) / out["tokens_original"]
    out["has_preserved"] = out["n_preserved"] > 0

    stand_cand = set(cand.loc[cand["stand_in"], "prompt_id"])
    stand_lab = set(
        zip(*[lab.loc[lab["stand_in"], c] for c in ("prompt_id", "target")], strict=True)
    )
    out["stand_in"] = [
        (p in stand_cand) or ((p, t) in stand_lab)
        for p, t in zip(out["prompt_id"], out["target"], strict=True)
    ]

    if meta is not None:
        out = out.merge(meta, on="prompt_id", how="left")
    if "category" not in out.columns:
        out["category"] = pd.NA
    if "style" not in out.columns:
        out["style"] = pd.NA
    out = out[list(ORACLE_COLUMNS)].sort_values(["target", "prompt_id"], kind="mergesort")
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------


def _parse_map(items: Sequence[str] | None) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in items or ():
        key, sep, value = item.partition("=")
        if not sep or not key or not value:
            raise SystemExit(f"--target-map expects LABEL_TARGET=TOKEN_TARGET, got {item!r}")
        mapping[key] = value
    return mapping


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; writes the oracle table and prints a one-line summary per target."""
    ap = argparse.ArgumentParser(prog="python -m terser.phase2.oracle", description=__doc__)
    ap.add_argument("--candidates", required=True, type=Path)
    ap.add_argument("--token-counts", required=True, type=Path)
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--prompts", type=Path, default=None)
    ap.add_argument("--targets", nargs="*", default=None)
    ap.add_argument("--target-map", nargs="*", default=None, metavar="LABEL=TOKENS")
    ap.add_argument("--method-priority", nargs="*", default=list(DEFAULT_METHOD_PRIORITY))
    ap.add_argument("--allow-stand-in", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("data/processed/oracle.csv"))
    args = ap.parse_args(argv)

    oracle = compute_oracle(
        read_table(args.candidates),
        read_table(args.token_counts),
        read_table(args.labels),
        prompts=read_table(args.prompts) if args.prompts else None,
        targets=args.targets,
        method_priority=args.method_priority,
        target_map=_parse_map(args.target_map),
        allow_stand_in=args.allow_stand_in,
    )
    path = write_table(oracle, args.out)
    for target, grp in oracle.groupby("target", sort=True):
        print(
            f"{target}: {len(grp)} prompts, median savings {100 * grp['savings'].median():.1f}%, "
            f"{100 * grp['has_preserved'].mean():.1f}% with a preserved candidate"
            + (" [STAND-IN]" if grp["stand_in"].any() else "")
        )
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
