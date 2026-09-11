"""Row 1.4.4: cross-tokenizer agreement of per-(prompt, operator set) savings.

For every tokenizer pair, Spearman's rho is computed over the paired savings of each
(prompt, operator set). The 95% confidence interval is a percentile bootstrap that resamples
**prompts** (all operator-set rows of a prompt move together, since they are not independent),
with a seeded generator. Two variants are reported: all rows, and only rows the operator set
changed on at least one target (unchanged rows are 0-vs-0 ties that inflate agreement). The C1.5
verdict uses the all-rows rho, matching the plan wording.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

__all__ = [
    "agreement_markdown",
    "best_op_disagreement",
    "bootstrap_spearman",
    "c15_verdict",
    "pairwise_spearman",
    "spearman",
]


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho; ``nan`` when fewer than 3 pairs or either side is constant."""
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def bootstrap_spearman(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    *,
    n_bootstrap: int = 1000,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, float, float]:
    """Spearman rho with a percentile cluster-bootstrap CI.

    Args:
        x: Savings on the first tokenizer.
        y: Savings on the second tokenizer, aligned with ``x``.
        groups: Cluster label (prompt id) per row; clusters are resampled with replacement.
        n_bootstrap: Number of resamples.
        seed: Seed for ``numpy.random.default_rng``.
        level: Confidence level.

    Returns:
        ``(rho, ci_low, ci_high)``; CI bounds are ``nan`` if no resample gave a finite rho.
    """
    rho = spearman(x, y)
    codes, uniques = pd.factorize(groups)
    members = [np.flatnonzero(codes == k) for k in range(len(uniques))]
    rng = np.random.default_rng(seed)
    stats: list[float] = []
    for _ in range(n_bootstrap):
        pick = rng.integers(0, len(members), size=len(members))
        idx = np.concatenate([members[k] for k in pick]) if len(members) else np.array([], int)
        value = spearman(x[idx], y[idx])
        if np.isfinite(value):
            stats.append(value)
    if not stats:
        return rho, float("nan"), float("nan")
    alpha = (1 - level) / 2
    lo, hi = np.quantile(stats, [alpha, 1 - alpha])
    return rho, float(lo), float(hi)


def pairwise_spearman(
    results: pd.DataFrame,
    *,
    targets: Sequence[str] | None = None,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Spearman rho with 95% bootstrap CI for every tokenizer pair.

    Returns:
        Columns ``target_a, target_b, n, rho, ci_low, ci_high, n_changed, rho_changed,
        ci_low_changed, ci_high_changed``.
    """
    names = list(targets) if targets is not None else list(dict.fromkeys(results["target"]))
    wide = results.pivot_table(index=["id", "op_set"], columns="target", values="savings")
    changed = results.groupby(["id", "op_set"])["changed"].any().reindex(wide.index)
    rows = []
    for a, b in itertools.combinations(names, 2):
        pair = wide[[a, b]].dropna()
        row: dict[str, object] = {"target_a": a, "target_b": b}
        for suffix, mask in (("", None), ("_changed", changed.reindex(pair.index).to_numpy())):
            sub = pair if mask is None else pair[mask.astype(bool)]
            rho, lo, hi = bootstrap_spearman(
                sub[a].to_numpy(float),
                sub[b].to_numpy(float),
                sub.index.get_level_values("id").to_numpy(),
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
            row[f"n{suffix}"] = len(sub)
            row[f"rho{suffix}"] = rho
            row[f"ci_low{suffix}"] = lo
            row[f"ci_high{suffix}"] = hi
        rows.append(row)
    columns = [
        "target_a",
        "target_b",
        "n",
        "rho",
        "ci_low",
        "ci_high",
        "n_changed",
        "rho_changed",
        "ci_low_changed",
        "ci_high_changed",
    ]
    return pd.DataFrame(rows, columns=columns)


def best_op_disagreement(best: pd.DataFrame) -> pd.DataFrame:
    """Per prompt, the best single operator on each target and whether they differ.

    Args:
        best: Output of :func:`terser.phase1.token_study.best_single_operator`.

    Returns:
        One row per prompt: ``category``, one column per target with its best operator, and
        ``differs`` (True when the targets do not all pick the same operator).
    """
    wide = best.pivot_table(
        index=["id", "category"], columns="target", values="best_op", aggfunc="first"
    )
    wide.columns.name = None
    wide["differs"] = wide.nunique(axis=1) > 1
    return wide.reset_index()


def c15_verdict(agreement: pd.DataFrame, threshold: float = 0.9) -> tuple[str, str]:
    """Checkpoint C1.5 verdict from :func:`pairwise_spearman` output.

    Passes by dependence when some pair has rho < ``threshold``; otherwise passes as a documented
    finding that the tokenizers agree (Phase 8 then ships one target-agnostic rewriter).

    Returns:
        ``(verdict, explanation)``.
    """
    finite = agreement.dropna(subset=["rho"])
    if finite.empty:
        return "UNDETERMINED", "No tokenizer pair had enough varying savings to compute rho."
    low = finite.loc[finite["rho"].idxmin()]
    desc = (
        f"lowest rho = {low['rho']:.3f} (95% CI {low['ci_low']:.3f}-{low['ci_high']:.3f}) for "
        f"{low['target_a']} vs {low['target_b']}"
    )
    below = finite[finite["rho"] < threshold]
    if not below.empty:
        pairs = ", ".join(f"{r.target_a}/{r.target_b}" for r in below.itertuples())
        return (
            "PASS (tokenizer dependence)",
            f"rho < {threshold:g} for {len(below)} pair(s) ({pairs}); {desc}.",
        )
    return (
        "PASS (documented agreement)",
        f"every pair has rho >= {threshold:g}; {desc}. Finding: savings rankings agree across "
        "tokenizers, so Phase 8 can ship a single target-agnostic rewriter.",
    )


def agreement_markdown(
    agreement: pd.DataFrame, disagreement: pd.DataFrame, *, n_bootstrap: int, seed: int
) -> str:
    """Report section for row 1.4.4."""
    n = len(disagreement)
    k = int(disagreement["differs"].sum())
    by_cat = disagreement.groupby("category")["differs"].agg(["sum", "size"])
    by_cat.columns = ["differs", "prompts"]
    target_cols = [c for c in disagreement.columns if c not in ("id", "category", "differs")]
    examples = disagreement[disagreement["differs"]].head(10)[["id", *target_cols]]
    lines = [
        "## 1.4.4 Cross-tokenizer agreement",
        "",
        "Spearman rho of per-(prompt, operator set) savings for each tokenizer pair. 95% CI: "
        f"percentile bootstrap over prompts, {n_bootstrap} resamples, seed {seed}. `_changed` "
        "columns restrict to rows the operator set changed on at least one target.",
        "",
        str(agreement.to_markdown(index=False, floatfmt=".3f")),
        "",
        f"Prompts whose best single operator differs across tokenizers: **{k} of {n}** "
        f"({100 * k / n if n else 0:.1f}%). `none` means no single operator saves tokens.",
        "",
        str(by_cat.to_markdown()),
    ]
    if not examples.empty:
        lines += ["", "Examples (first 10):", "", str(examples.to_markdown(index=False))]
    return "\n".join(lines)
