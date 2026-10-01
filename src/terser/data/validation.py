"""pandera schemas for the labelled pair table (plan row **11.1.4**).

The pair table (one row per original prompt x candidate rewrite x target) feeds every model. A
corrupted batch must stop the pipeline before training, so :func:`validate_pairs` checks:

* **nulls**: ``id``, ``category``, ``prompt``, ``rewrite``, ``target``, token counts and
  ``label`` are non-null; prompts are non-empty;
* **token-ratio ranges**: ``tokens_before >= 1``, ``tokens_after >= 0``,
  ``token_ratio = tokens_after / tokens_before`` lies in ``[min_ratio, max_ratio]`` and agrees
  with the two counts;
* **category balance**: every category is one of the six (:class:`terser.data.categories.Category`)
  and each present category's share is in ``[min_share, max_share]``; with ``require_all`` every
  category must appear.

CLI: ``python -m terser.data.validation pairs.jsonl`` (exit 1 on failure).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import Any

from terser.data.categories import Category

__all__ = ["LABELS", "pair_schema", "validate_pairs"]

LABELS: tuple[str, ...] = ("preserved", "degraded")


def pair_schema(
    *,
    min_ratio: float = 0.05,
    max_ratio: float = 1.5,
    min_share: float = 0.05,
    max_share: float = 0.5,
    require_all: bool = False,
) -> Any:
    """Build the :class:`pandera.DataFrameSchema` for the pair table."""
    import pandera.pandas as pa

    cats = [c.value for c in Category]

    def balance(df: Any) -> bool:
        share = df["category"].value_counts(normalize=True)
        if require_all and set(share.index) != set(cats):
            return False
        return bool(((share >= min_share) & (share <= max_share)).all())

    def ratio_consistent(df: Any) -> Any:
        expected = df["tokens_after"] / df["tokens_before"]
        return (df["token_ratio"] - expected).abs() <= 1e-6

    return pa.DataFrameSchema(
        {
            "id": pa.Column(str, nullable=False, unique=True),
            "category": pa.Column(str, pa.Check.isin(cats), nullable=False),
            "prompt": pa.Column(str, pa.Check.str_length(min_value=1), nullable=False),
            "rewrite": pa.Column(str, nullable=False),
            "target": pa.Column(str, pa.Check.str_length(min_value=1), nullable=False),
            "tokens_before": pa.Column(int, pa.Check.ge(1), nullable=False),
            "tokens_after": pa.Column(int, pa.Check.ge(0), nullable=False),
            "token_ratio": pa.Column(
                float, pa.Check.in_range(min_ratio, max_ratio), nullable=False
            ),
            "label": pa.Column(str, pa.Check.isin(list(LABELS)), nullable=False),
        },
        checks=[
            pa.Check(ratio_consistent, name="token_ratio_matches_counts"),
            pa.Check(balance, name="category_balance"),
        ],
        strict=False,
        coerce=False,
    )


def validate_pairs(frame: Any, **schema_kwargs: Any) -> Any:
    """Validate ``frame``; raise :class:`pandera.errors.SchemaErrors` listing every failure."""
    return pair_schema(**schema_kwargs).validate(frame, lazy=True)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: validate a JSONL pair table."""
    import pandas as pd
    import pandera.errors

    ap = argparse.ArgumentParser(description="Validate a labelled pair table (row 11.1.4)")
    ap.add_argument("path")
    ap.add_argument("--require-all", action="store_true")
    args = ap.parse_args(argv)
    df = pd.read_json(args.path, lines=True)
    try:
        validate_pairs(df, require_all=args.require_all)
    except pandera.errors.SchemaErrors as exc:
        print(exc.failure_cases.to_string(), file=sys.stderr)
        return 1
    print(f"ok: {len(df)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
