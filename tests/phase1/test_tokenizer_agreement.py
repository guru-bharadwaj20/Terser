"""Tests for row 1.4.4: cross-tokenizer Spearman agreement and best-operator disagreement."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from terser.phase1.tokenizer_agreement import (
    agreement_markdown,
    best_op_disagreement,
    bootstrap_spearman,
    c15_verdict,
    pairwise_spearman,
    spearman,
)


def _results(n_prompts: int = 20, seed: int = 1) -> pd.DataFrame:
    """Target ``b`` is a monotone copy of ``a``; ``c`` is independent noise."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_prompts):
        for op in ("F", "H", "FH"):
            base = float(rng.uniform(0, 30))
            for target, value in (("a", base), ("b", 2 * base + 1), ("c", float(rng.normal()))):
                rows.append(
                    {
                        "id": f"p{i}",
                        "category": "code",
                        "op_set": op,
                        "target": target,
                        "savings": value,
                        "changed": True,
                    }
                )
    return pd.DataFrame(rows)


def test_spearman_edge_cases() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert spearman(x, x * 3) == pytest.approx(1.0)
    assert spearman(x, -x) == pytest.approx(-1.0)
    assert np.isnan(spearman(x, np.ones(4)))
    assert np.isnan(spearman(x[:2], x[:2]))


def test_bootstrap_is_seeded_and_brackets_rho() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=60)
    y = x + rng.normal(scale=1.0, size=60)
    groups = np.repeat(np.arange(20), 3)
    first = bootstrap_spearman(x, y, groups, n_bootstrap=200, seed=7)
    again = bootstrap_spearman(x, y, groups, n_bootstrap=200, seed=7)
    other = bootstrap_spearman(x, y, groups, n_bootstrap=200, seed=8)
    assert first == again
    assert first != other
    rho, lo, hi = first
    assert lo <= rho <= hi
    assert -1 <= lo < hi <= 1


def test_pairwise_spearman_all_pairs() -> None:
    table = pairwise_spearman(_results(), targets=["a", "b", "c"], n_bootstrap=100, seed=0)
    assert list(zip(table["target_a"], table["target_b"], strict=True)) == [
        ("a", "b"),
        ("a", "c"),
        ("b", "c"),
    ]
    ab = table.iloc[0]
    assert ab["rho"] == pytest.approx(1.0)
    assert ab["ci_low"] == pytest.approx(1.0)
    assert ab["n"] == 60
    assert abs(table.iloc[1]["rho"]) < 0.5


def test_changed_only_variant_drops_unchanged_rows() -> None:
    df = _results(n_prompts=10)
    df.loc[df["op_set"] == "FH", "changed"] = False
    table = pairwise_spearman(df, targets=["a", "b"], n_bootstrap=20)
    assert table.iloc[0]["n"] == 30
    assert table.iloc[0]["n_changed"] == 20


def test_c15_verdicts() -> None:
    dep = pairwise_spearman(_results(), targets=["a", "b", "c"], n_bootstrap=50)
    verdict, detail = c15_verdict(dep)
    assert verdict == "PASS (tokenizer dependence)"
    assert "a/c" in detail
    agree = pairwise_spearman(_results(), targets=["a", "b"], n_bootstrap=50)
    verdict, detail = c15_verdict(agree)
    assert verdict == "PASS (documented agreement)"
    assert "target-agnostic" in detail
    empty = agree.assign(rho=np.nan)
    assert c15_verdict(empty)[0] == "UNDETERMINED"


def test_best_op_disagreement_and_markdown() -> None:
    best = pd.DataFrame(
        [
            {"id": "p1", "category": "code", "target": "a", "best_op": "F", "best_savings": 5.0},
            {"id": "p1", "category": "code", "target": "b", "best_op": "F", "best_savings": 4.0},
            {"id": "p2", "category": "fact", "target": "a", "best_op": "S", "best_savings": 3.0},
            {"id": "p2", "category": "fact", "target": "b", "best_op": "H", "best_savings": 3.0},
        ]
    )
    dis = best_op_disagreement(best)
    assert dis.set_index("id")["differs"].to_dict() == {"p1": False, "p2": True}
    agreement = pairwise_spearman(_results(), targets=["a", "b"], n_bootstrap=10)
    md = agreement_markdown(agreement, dis, n_bootstrap=10, seed=0)
    assert "## 1.4.4" in md
    assert "**1 of 2**" in md
    assert "p2" in md
