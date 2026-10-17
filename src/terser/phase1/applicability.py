"""Row 1.4.3: operator applicability, the share of prompts each operator changes.

A prompt counts as changed by an operator when the rewrite differs from the input. Only ``S`` is
tokenizer-aware, so by default a prompt counts as changed if the operator altered it for **any**
target; pass ``target`` to restrict to one tokenizer.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from terser.data.categories import Category
from terser.phase1.token_study import OP_CODES

__all__ = ["OVERALL", "applicability_markdown", "applicability_table", "plot_applicability"]

OVERALL = "all"
"""Row label of the overall share across every group."""


def applicability_table(
    results: pd.DataFrame,
    by: str = "category",
    *,
    target: str | None = None,
    op_codes: Sequence[str] = OP_CODES,
) -> pd.DataFrame:
    """Share of prompts (0-1) each single operator changes, per ``by`` value.

    Args:
        results: Output of :func:`terser.phase1.token_study.compute_savings`.
        by: Grouping column, ``"category"`` or ``"style"``.
        target: Restrict to one target; ``None`` means changed on any target.
        op_codes: Operators to tabulate (columns, in this order).

    Returns:
        Index: group values (categories in canonical order) plus a final :data:`OVERALL` row.
        Columns: ``op_codes`` then ``n`` (number of prompts in the group).
    """
    sub = results[results["op_set"].isin(op_codes)]
    if target is not None:
        sub = sub[sub["target"] == target]
    per_prompt = sub.groupby(["id", by, "op_set"], sort=False)["changed"].any().reset_index()
    wide = per_prompt.pivot_table(
        index=["id", by], columns="op_set", values="changed", aggfunc="any", fill_value=False
    )
    wide = wide.reindex(columns=list(op_codes), fill_value=False).astype(float)
    table = wide.groupby(level=by).mean()
    table["n"] = wide.groupby(level=by).size()
    if by == "category":
        order = [c.value for c in Category if c.value in table.index]
        table = table.reindex(order + [i for i in table.index if i not in order])
    overall = wide.mean()
    overall["n"] = float(len(wide))
    table.loc[OVERALL] = overall
    table["n"] = table["n"].astype(int)
    table.index.name = by
    table.columns.name = None
    return table


def applicability_markdown(
    by_category: pd.DataFrame, by_style: pd.DataFrame, target: str | None = None
) -> str:
    """Report section for row 1.4.3 (shares rendered as percentages).

    ``target`` names the tokenizer target the tables were restricted to (None: any target).
    """
    scope = "on any target" if target is None else f"on target `{target}`"

    def pct(table: pd.DataFrame) -> str:
        shown = table.copy()
        ops = [c for c in shown.columns if c != "n"]
        shown[ops] = shown[ops] * 100
        return str(shown.to_markdown(floatfmt=".0f"))

    return "\n".join(
        [
            "## 1.4.3 Operator applicability",
            "",
            f"Share of prompts (%) each single operator changes ({scope}), per category and "
            "per style; `n` is the number of prompts.",
            "",
            "### By category",
            "",
            pct(by_category),
            "",
            "![Applicability heatmap](figures/p1_applicability.png)",
            "",
            "### By style",
            "",
            pct(by_style),
        ]
    )


def plot_applicability(table: pd.DataFrame, path: str | Path) -> Path:
    """Heatmap of operator × group applicability shares, with each cell labelled."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    data = table.drop(columns="n")
    cmap = LinearSegmentedColormap.from_list(
        "terser_blue", ["#f5f9fe", "#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"]
    )
    fig, ax = plt.subplots(figsize=(1.0 * len(data.columns) + 2.5, 0.5 * len(data) + 1.6))
    im = ax.imshow(data.to_numpy() * 100, cmap=cmap, vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(data.columns)), list(data.columns))
    ax.set_yticks(range(len(data)), [f"{i} (n={n})" for i, n in table["n"].items()])
    for (r, c), v in pd.DataFrame(data.to_numpy()).stack().items():
        ax.text(
            int(c),
            int(r),
            f"{100 * v:.0f}",
            ha="center",
            va="center",
            fontsize=8,
            color="#ffffff" if v >= 0.55 else "#0b0b0b",
        )
    ax.set_xlabel("Operator")
    ax.set_title("Share of prompts each operator changes (%)")
    for side in ax.spines.values():
        side.set_visible(False)
    fig.colorbar(im, ax=ax, label="% of prompts changed", fraction=0.04)
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
