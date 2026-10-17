"""Tests for row 1.4.2: pilot prompt loading and per-operator token savings."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from terser.config import load_config
from terser.phase1.prompts import (
    PromptValidationError,
    load_prompt_dir,
    load_prompt_file,
    missing_category_files,
    validate_prompts,
)
from terser.phase1.token_study import (
    ALL_OPS,
    OP_SETS,
    RESULT_COLUMNS,
    all_ops_by,
    best_single_operator,
    compute_savings,
    run_study,
    savings_pct,
    summarize_by_operator,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "pilot_prompts_small.jsonl"
FAKE_TARGETS = ("words", "chars4")

FAKE_RULES: dict[str, list[tuple[str, str]]] = {
    "F": [(r"\b(?:um|basically|like|so|ok|yo|hey) ", "")],
    "H": [(r"\b(?:maybe|perhaps|possibly|I think|I guess|I might be wrong but) ", "")],
    "D": [],
    "I": [(r"\bcould you ", "")],
    "L": [],
    "C": [(r"\b(?:please|pls|kindly) ", "")],
    "S": [(r"\bin order to\b", "to")],
}


def fake_apply(text: str, codes: str, target: str | None) -> str:
    """A fake operator registry: regex deletions; ``S`` only fires on the ``words`` target."""
    out = text
    for code in codes:
        if code == "S" and target != "words":
            continue
        for pattern, repl in FAKE_RULES[code]:
            out = re.sub(pattern, repl, out)
    return out


def fake_count(text: str, target: str) -> int:
    """Word count for ``words``; ceil(chars / 4) for ``chars4``."""
    return len(text.split()) if target == "words" else -(-len(text) // 4)


@pytest.fixture(scope="module")
def prompts() -> list[dict[str, Any]]:
    return load_prompt_file(FIXTURE)


@pytest.fixture(scope="module")
def results(prompts: list[dict[str, Any]]) -> pd.DataFrame:
    return compute_savings(prompts, targets=FAKE_TARGETS, apply_fn=fake_apply, count_fn=fake_count)


# --- prompts --------------------------------------------------------------------------------


def test_fixture_is_balanced(prompts: list[dict[str, Any]]) -> None:
    assert len(prompts) == 12
    assert pd.Series([p["category"] for p in prompts]).value_counts().eq(2).all()


def _record(**overrides: Any) -> dict[str, Any]:
    rec: dict[str, Any] = {
        "id": "x-1",
        "category": "code",
        "prompt": "write code",
        "origin": "handwritten",
        "style": "texting",
        "has_payload": False,
        "constraints": [],
    }
    rec.update(overrides)
    return rec


@pytest.mark.parametrize(
    "bad",
    [
        {"category": "poetry"},
        {"prompt": "   "},
        {"has_payload": "no"},
        {"constraints": [1]},
    ],
)
def test_validate_rejects_bad_records(bad: dict[str, Any]) -> None:
    with pytest.raises(PromptValidationError):
        validate_prompts([_record(**bad)])


def test_validate_rejects_missing_field_and_duplicates() -> None:
    rec = _record()
    del rec["origin"]
    with pytest.raises(PromptValidationError, match="origin"):
        validate_prompts([rec])
    with pytest.raises(PromptValidationError, match="duplicate"):
        validate_prompts([_record(), _record()])


def test_validate_normalises_category() -> None:
    assert validate_prompts([_record(category=" Code ")])[0]["category"] == "code"


def _write_dir(root: Path, prompts: list[dict[str, Any]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for cat in {p["category"] for p in prompts}:
        with (root / f"{cat}.jsonl").open("w", encoding="utf-8") as fh:
            for p in prompts:
                if p["category"] == cat:
                    fh.write(json.dumps(p) + "\n")


def test_load_prompt_dir_merges(tmp_path: Path, prompts: list[dict[str, Any]]) -> None:
    _write_dir(tmp_path, prompts)
    assert missing_category_files(tmp_path) == []
    assert {p["id"] for p in load_prompt_dir(tmp_path)} == {p["id"] for p in prompts}


def test_load_prompt_dir_missing_and_mismatch(
    tmp_path: Path, prompts: list[dict[str, Any]]
) -> None:
    _write_dir(tmp_path, prompts)
    (tmp_path / "fact.jsonl").unlink()
    with pytest.raises(FileNotFoundError, match=r"fact\.jsonl"):
        load_prompt_dir(tmp_path)
    _write_dir(
        tmp_path,
        [
            dict(p, category="fact") if p["category"] == "code" else p
            for p in prompts
            if p["category"] in ("code", "fact")
        ],
    )
    with (tmp_path / "fact.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(_record(id="odd", category="code")) + "\n")
    with pytest.raises(PromptValidationError):
        load_prompt_dir(tmp_path)


# --- savings --------------------------------------------------------------------------------


def test_savings_pct() -> None:
    assert savings_pct(10, 8) == pytest.approx(20.0)
    assert savings_pct(10, 12) == pytest.approx(-20.0)
    assert savings_pct(0, 0) == 0.0


def test_compute_savings_shape(results: pd.DataFrame) -> None:
    assert tuple(results.columns) == RESULT_COLUMNS
    assert len(results) == 12 * len(OP_SETS) * len(FAKE_TARGETS)
    assert not results.duplicated(["id", "op_set", "target"]).any()


def test_unchanged_rows_have_zero_savings(results: pd.DataFrame) -> None:
    unchanged = results[~results["changed"]]
    assert (unchanged["savings"] == 0).all()
    assert (unchanged["tokens_before"] == unchanged["tokens_after"]).all()
    assert not results[results["op_set"].isin(["D", "L"])]["changed"].any()


def test_tokenizer_aware_operator_differs_per_target(results: pd.DataFrame) -> None:
    s = results[(results["op_set"] == "S")].set_index(["id", "target"])["changed"]
    assert s.loc[("code-002", "words")]
    assert not s.loc[("code-002", "chars4")]


def test_stack_saves_at_least_each_member(results: pd.DataFrame) -> None:
    wide = results.pivot_table(index=["id", "target"], columns="op_set", values="savings")
    for op in ALL_OPS:
        assert (wide[ALL_OPS] >= wide[op] - 1e-9).all()


def test_summary_and_best(results: pd.DataFrame) -> None:
    summary = summarize_by_operator(results)
    assert list(summary["op_set"].unique()) == list(OP_SETS)
    assert {"median", "mean", "p25", "p75", "changed_share", "n"} <= set(summary.columns)
    assert (summary["p25"] <= summary["median"]).all()
    assert (summary["median"] <= summary["p75"]).all()

    best = best_single_operator(results)
    assert len(best) == 12 * len(FAKE_TARGETS)
    wide = results[results["op_set"].str.len() == 1].pivot_table(
        index=["id", "target"], columns="op_set", values="savings"
    )
    merged = best.set_index(["id", "target"])["best_savings"]
    assert (merged == wide.max(axis=1).reindex(merged.index)).all()
    assert (best.loc[best["best_savings"] <= 0, "best_op"] == "none").all()


def test_best_single_tie_break_follows_operator_order() -> None:
    df = pd.DataFrame(
        [
            {
                "id": "a",
                "category": "fact",
                "style": "s",
                "op_set": op,
                "target": "t",
                "savings": 5.0,
            }
            for op in ("H", "F")
        ]
    )
    assert best_single_operator(df)["best_op"].tolist() == ["F"]


def test_all_ops_by_category(results: pd.DataFrame) -> None:
    table = all_ops_by(results, "category")
    assert len(table) == 6
    assert set(table.columns) == set(FAKE_TARGETS)


def test_run_study_writes_outputs(tmp_path: Path, prompts: list[dict[str, Any]]) -> None:
    out = run_study(
        prompts,
        tmp_path,
        targets=FAKE_TARGETS,
        n_bootstrap=50,
        apply_fn=fake_apply,
        count_fn=fake_count,
    )
    csv = pd.read_csv(tmp_path / "p1_tokenizer_study.csv")
    assert len(csv) == len(out.results)
    md = (tmp_path / "p1_tokenizer_study.md").read_bytes()
    assert b"\r\n" not in md
    text = md.decode("utf-8")
    for heading in ("C1.4", "C1.5", "## 1.4.2", "## 1.4.3", "## 1.4.4"):
        assert heading in text
    assert (tmp_path / "figures" / "p1_savings_by_operator.png").stat().st_size > 0
    assert (tmp_path / "figures" / "p1_applicability.png").stat().st_size > 0
    assert "best_single_median_words" in out.metrics
    assert out.metrics["c14_pass"] in (0.0, 1.0)


def test_experiment_config_composes() -> None:
    cfg = load_config(["experiment=p1_token_study"])
    assert cfg.experiment.phase == 1
    assert cfg.experiment.row == "1.4.2"
    assert list(cfg.experiment.params.op_sets) == list(OP_SETS)
    assert cfg.experiment.params.n_bootstrap == 1000
    assert str(cfg.experiment.params.prompts).endswith("data/pilot/prompts")
