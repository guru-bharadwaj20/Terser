"""Tests for row 2.4.2: the headroom report (synthetic data only)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from terser.data.categories import Category
from terser.phase2.headroom import (
    C27_FAILURE_INSTRUCTION,
    HeadroomReport,
    bootstrap_median_ci,
    build_report,
    check_c27,
    check_c28,
    main,
    savings_table,
    validate_oracle,
    write_report,
)
from terser.phase2.oracle import (
    Phase2InputError,
    StandInDataError,
    compute_oracle,
    write_table,
)

TARGETS = ("llama", "qwen")
CATS = [c.value for c in Category]
# candidate_id -> (method, tokens on a 100-token original)
CANDS = {"c1": ("operators", 90), "c2": ("llm_high", 50), "c3": ("llm_low", 70)}

LabelFn = Callable[[str, str, str, str], str]


def _default_label(pid: str, cat: str, cid: str, target: str) -> str:
    # c2 (50 tokens) degraded, c1/c3 preserved -> oracle is c3, savings 30%; degraded share 1/3.
    return "degraded" if cid == "c2" else "preserved"


def make_tables(
    label_fn: LabelFn = _default_label,
    *,
    per_cat: int = 3,
    style: bool = False,
    stand_in: bool = False,
) -> dict[str, pd.DataFrame]:
    cands, tc, labs, prompts = [], [], [], []
    for cat in CATS:
        for i in range(per_cat):
            pid = f"{cat}-{i}"
            prompt = {"prompt_id": pid, "category": cat}
            if style:
                prompt["style"] = ("rambling", "texting")[i % 2]
            prompts.append(prompt)
            for target in TARGETS:
                tc.append((pid, "original", target, 100))
            for cid, (method, tokens) in CANDS.items():
                cands.append(
                    {
                        "prompt_id": pid,
                        "candidate_id": cid,
                        "method": method,
                        "text": "t",
                        "generator": "synthetic",
                        "stand_in": stand_in,
                        "rejected": False,
                        "rejection_reason": None,
                    }
                )
                for target in TARGETS:
                    tc.append((pid, cid, target, tokens))
                    labs.append((pid, cid, target, label_fn(pid, cat, cid, target)))
    return {
        "candidates": pd.DataFrame(cands),
        "token_counts": pd.DataFrame(tc, columns=["prompt_id", "candidate_id", "target", "tokens"]),
        "labels": pd.DataFrame(labs, columns=["prompt_id", "candidate_id", "target", "label"]),
        "prompts": pd.DataFrame(prompts),
    }


def _report(
    tables: dict[str, pd.DataFrame], *, allow_stand_in: bool = False, seed: int = 0
) -> HeadroomReport:
    allow = allow_stand_in
    oracle = compute_oracle(
        tables["candidates"],
        tables["token_counts"],
        tables["labels"],
        prompts=tables["prompts"],
        allow_stand_in=allow,
    )
    return build_report(
        oracle,
        tables["candidates"],
        tables["labels"],
        prompts=tables["prompts"],
        n_bootstrap=200,
        seed=seed,
        allow_stand_in=allow,
    )


# --- statistics ------------------------------------------------------------------------------


def test_bootstrap_ci_seeded_and_brackets_median() -> None:
    values = np.random.default_rng(0).uniform(0.0, 0.5, size=37)
    a = bootstrap_median_ci(values, n_bootstrap=500, seed=3)
    assert a == bootstrap_median_ci(values, n_bootstrap=500, seed=3)
    assert a[0] <= float(np.median(values)) <= a[1]
    lo, hi = bootstrap_median_ci([], n_bootstrap=10)
    assert np.isnan(lo) and np.isnan(hi)
    with pytest.raises(ValueError):
        bootstrap_median_ci([0.1], n_bootstrap=0)


def test_savings_table_quantiles() -> None:
    df = pd.DataFrame(
        {
            "target": ["a"] * 5 + ["b"],
            "savings": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            "has_preserved": [False, True, True, True, True, True],
        }
    )
    out = savings_table(df, ["target"], n_bootstrap=50)
    a = out.set_index("target").loc["a"]
    assert a["n"] == 5
    assert a["median"] == pytest.approx(0.2)
    assert a["p25"] == pytest.approx(0.1)
    assert a["p75"] == pytest.approx(0.3)
    assert a["share_any_preserved"] == pytest.approx(0.8)
    overall = savings_table(df, [], n_bootstrap=50)
    assert len(overall) == 1 and overall["n"].iloc[0] == 6


# --- checkpoints -----------------------------------------------------------------------------


def _oracle(per_target: dict[str, float], per_cat: dict[str, float]) -> pd.DataFrame:
    rows = []
    for t in per_target:
        for c, s in per_cat.items():
            rows.append({"target": t, "category": c, "savings": min(s, per_target[t])})
    return pd.DataFrame(rows)


def test_c27_pass_and_fail_modes() -> None:
    all_cats = dict.fromkeys(CATS, 0.30)
    assert check_c27(_oracle({"a": 0.3, "b": 0.3}, all_cats)).passed
    # Only one target clears 25%.
    assert not check_c27(_oracle({"a": 0.3, "b": 0.2}, all_cats)).passed
    # Two categories under 15% -> 4 of 6.
    low = {**all_cats, "code": 0.1, "fact": 0.1}
    res = check_c27(_oracle({"a": 0.3, "b": 0.3}, low))
    assert not res.passed and "4 of 6" in res.detail
    # One category under 15% is allowed.
    assert check_c27(_oracle({"a": 0.3, "b": 0.3}, {**all_cats, "code": 0.0})).passed
    # A missing category counts as failing.
    two_missing = {c: 0.3 for c in CATS[:4]}
    res = check_c27(_oracle({"a": 0.3, "b": 0.3}, two_missing))
    assert not res.passed and "no rows" in res.detail


def test_c28_bounds() -> None:
    assert check_c28(0.15, 10).passed
    assert check_c28(0.60, 10).passed
    assert not check_c28(0.149, 10).passed
    assert not check_c28(0.61, 10).passed
    assert not check_c28(float("nan"), 0).passed


# --- report ----------------------------------------------------------------------------------


def test_report_passing_synthetic() -> None:
    rep = _report(make_tables())
    assert rep.c27.passed and rep.c28.passed
    assert "C2.7 Go / no-go: oracle savings: PASS" in rep.markdown
    assert "C2.8 Class balance: PASS" in rep.markdown
    assert C27_FAILURE_INSTRUCTION not in rep.markdown
    assert "STAND-IN" not in rep.markdown
    t = rep.table
    tc = t[t["section"] == "savings_by_target_category"]
    assert len(tc) == len(TARGETS) * len(CATS)
    assert tc["median"].tolist() == pytest.approx([0.30] * len(tc))
    assert list(tc["category"].iloc[:6]) == CATS  # enum order, not alphabetical
    overall = t[t["section"] == "degraded_overall"].iloc[0]
    assert overall["degraded_share"] == pytest.approx(1 / 3)
    by_method = t[t["section"] == "degraded_by_method"].set_index("method")["degraded_share"]
    assert by_method.to_dict() == pytest.approx({"llm_high": 1.0, "llm_low": 0.0, "operators": 0.0})
    assert not (t["section"] == "degraded_by_style").any()
    assert rep.metrics["median_savings_qwen"] == pytest.approx(0.30)


def test_report_failing_prints_instruction() -> None:
    def only_ops(pid: str, cat: str, cid: str, target: str) -> str:
        # Only the 10%-savings operator candidate survives: C2.7 fails; 2/3 degraded: C2.8 fails.
        return "preserved" if cid == "c1" else "degraded"

    rep = _report(make_tables(only_ops))
    assert not rep.c27.passed and not rep.c28.passed
    assert "C2.7 Go / no-go: oracle savings: FAIL" in rep.markdown
    assert "C2.8 Class balance: FAIL" in rep.markdown
    assert C27_FAILURE_INSTRUCTION in rep.markdown
    assert "negative-result report" in C27_FAILURE_INSTRUCTION


def test_per_category_and_style_sections() -> None:
    def creative_hard(pid: str, cat: str, cid: str, target: str) -> str:
        if cat in ("creative", "code") and cid != "c1":
            return "degraded"
        return _default_label(pid, cat, cid, target)

    rep = _report(make_tables(creative_hard, style=True))
    t = rep.table
    cat = t[t["section"] == "savings_by_category"].set_index("category")["median"]
    assert cat["creative"] == pytest.approx(0.10) and cat["fact"] == pytest.approx(0.30)
    assert not rep.c27.passed  # 4 of 6 categories
    styles = t[t["section"] == "degraded_by_style"]
    assert set(styles["style"]) == {"rambling", "texting"}
    assert (t["section"] == "savings_by_target_style").any()
    assert "Degraded share per style" in rep.markdown


def test_no_preserved_candidate_share() -> None:
    def none_for_fact0(pid: str, cat: str, cid: str, target: str) -> str:
        return "degraded" if pid == "fact-0" else _default_label(pid, cat, cid, target)

    rep = _report(make_tables(none_for_fact0))
    t = rep.table
    row = t[(t["section"] == "savings_by_target_category") & (t["category"] == "fact")].iloc[0]
    assert row["share_any_preserved"] == pytest.approx(2 / 3)


def test_stand_in_refused_unless_flagged() -> None:
    tables = make_tables(stand_in=True)
    oracle = compute_oracle(
        tables["candidates"], tables["token_counts"], tables["labels"], allow_stand_in=True
    )
    with pytest.raises(StandInDataError):
        build_report(oracle, tables["candidates"], tables["labels"], prompts=tables["prompts"])
    # Labels-only stand-in flag is caught too.
    clean = make_tables()
    labels = clean["labels"].assign(stand_in=True)
    oracle = compute_oracle(
        clean["candidates"], clean["token_counts"], labels, allow_stand_in=True
    ).assign(stand_in=False)
    with pytest.raises(StandInDataError):
        build_report(oracle, clean["candidates"], labels, prompts=clean["prompts"])
    rep = _report(tables, allow_stand_in=True)
    assert rep.stand_in and "STAND-IN DATA" in rep.markdown


def test_categories_from_oracle_when_no_prompts() -> None:
    tables = make_tables()
    oracle = compute_oracle(
        tables["candidates"], tables["token_counts"], tables["labels"], prompts=tables["prompts"]
    )
    rep = build_report(oracle, tables["candidates"], tables["labels"], n_bootstrap=50)
    assert rep.c28.passed
    no_cat = oracle.drop(columns=["category"])
    with pytest.raises(Phase2InputError, match="category"):
        build_report(no_cat, tables["candidates"], tables["labels"], n_bootstrap=50)


def test_validate_oracle_rejects_bad_tables() -> None:
    good = pd.DataFrame(
        {"prompt_id": ["p"], "target": ["t"], "savings": [0.2], "has_preserved": ["true"]}
    )
    out = validate_oracle(good)
    assert bool(out["has_preserved"].iloc[0]) and not bool(out["stand_in"].iloc[0])
    with pytest.raises(Phase2InputError, match="missing"):
        validate_oracle(good.drop(columns=["savings"]))
    with pytest.raises(Phase2InputError, match="duplicate"):
        validate_oracle(pd.concat([good, good]))
    with pytest.raises(Phase2InputError, match="savings"):
        validate_oracle(good.assign(savings=1.0))


def test_write_report_deterministic_lf(tmp_path: Path) -> None:
    tables = make_tables(style=True)
    paths = []
    for run in ("a", "b"):
        rep = _report(tables, seed=7)
        paths.append(write_report(rep, tmp_path / run / "h.md", tmp_path / run / "h.csv"))
    for i in range(2):
        first, second = paths[0][i].read_bytes(), paths[1][i].read_bytes()
        assert first == second
        assert b"\r\n" not in first


def test_cli_recompute_and_from_oracle(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    tables = make_tables()
    files = {name: write_table(df, tmp_path / f"{name}.jsonl") for name, df in tables.items()}
    common = [
        "--candidates",
        str(files["candidates"]),
        "--labels",
        str(files["labels"]),
        "--prompts",
        str(files["prompts"]),
        "--n-bootstrap",
        "100",
    ]
    md1, csv1 = tmp_path / "r1.md", tmp_path / "r1.csv"
    tc_args = ["--token-counts", str(files["token_counts"])]
    assert main([*common, *tc_args, "--out-md", str(md1), "--out-csv", str(csv1)]) == 0
    assert "C2.7 Go / no-go: oracle savings: PASS" in capsys.readouterr().out

    oracle = compute_oracle(
        tables["candidates"], tables["token_counts"], tables["labels"], prompts=tables["prompts"]
    )
    oracle_csv = write_table(oracle, tmp_path / "oracle.csv")
    md2, csv2 = tmp_path / "r2.md", tmp_path / "r2.csv"
    args = [*common, "--oracle", str(oracle_csv), "--out-md", str(md2), "--out-csv", str(csv2)]
    assert main(args) == 0
    assert md1.read_bytes() == md2.read_bytes()
    assert csv1.read_bytes() == csv2.read_bytes()

    with pytest.raises(SystemExit):
        main(common)

    stand = make_tables(stand_in=True)
    sfiles = {name: write_table(df, tmp_path / f"s_{name}.jsonl") for name, df in stand.items()}
    sargs = [
        "--candidates",
        str(sfiles["candidates"]),
        "--labels",
        str(sfiles["labels"]),
        "--token-counts",
        str(sfiles["token_counts"]),
        "--prompts",
        str(sfiles["prompts"]),
        "--n-bootstrap",
        "20",
        "--out-md",
        str(tmp_path / "s.md"),
        "--out-csv",
        str(tmp_path / "s.csv"),
    ]
    with pytest.raises(StandInDataError):
        main(sargs)
    assert main([*sargs, "--allow-stand-in"]) == 0
    out = capsys.readouterr().out
    assert "STAND-IN" in out and C27_FAILURE_INSTRUCTION not in out
