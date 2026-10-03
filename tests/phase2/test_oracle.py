"""Tests for row 2.4.1: the oracle rewrite table."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from terser.phase2.oracle import (
    ORACLE_COLUMNS,
    Phase2InputError,
    StandInDataError,
    compute_oracle,
    main,
    method_rank,
    read_table,
    write_table,
)


def _cand(pid: str, cid: str, method: str, *, rejected: bool = False, stand_in: bool = False):
    return {
        "prompt_id": pid,
        "candidate_id": cid,
        "method": method,
        "text": f"text {cid}",
        "generator": "synthetic",
        "stand_in": stand_in,
        "rejected": rejected,
        "rejection_reason": "too long" if rejected else None,
    }


def _tc(rows: list[tuple[str, str, str, int]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["prompt_id", "candidate_id", "target", "tokens"])


def _lab(rows: list[tuple[str, str, str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["prompt_id", "candidate_id", "target", "label"])


@pytest.fixture()
def tables() -> dict[str, pd.DataFrame]:
    cands = pd.DataFrame(
        [
            _cand("p1", "c1", "llm_high"),
            _cand("p1", "c2", "operators"),
            _cand("p1", "c3", "llm_low"),
            _cand("p1", "c4", "llm_medium", rejected=True),
            _cand("p2", "c1", "llm_high"),
        ]
    )
    tc = _tc(
        [
            ("p1", "original", "qwen", 100),
            ("p1", "c1", "qwen", 60),
            ("p1", "c2", "qwen", 60),
            ("p1", "c3", "qwen", 80),
            ("p1", "c4", "qwen", 10),
            ("p1", "original", "llama", 50),
            ("p1", "c1", "llama", 30),
            ("p1", "c2", "llama", 40),
            ("p1", "c3", "llama", 45),
            ("p1", "c4", "llama", 5),
            ("p2", "original", "qwen", 40),
            ("p2", "c1", "qwen", 20),
            ("p2", "original", "llama", 40),
            ("p2", "c1", "llama", 20),
        ]
    )
    labels = _lab(
        [
            ("p1", "c1", "qwen", "preserved"),
            ("p1", "c2", "qwen", "preserved"),
            ("p1", "c3", "qwen", "preserved"),
            ("p1", "c4", "qwen", "preserved"),  # rejected: must be ignored
            ("p1", "c1", "llama", "degraded"),
            ("p1", "c2", "llama", "preserved"),
            ("p1", "c3", "llama", "preserved"),
            ("p2", "c1", "qwen", "degraded"),
            ("p2", "c1", "llama", None),  # unlabelled
        ]
    )
    prompts = pd.DataFrame(
        {"prompt_id": ["p1", "p2"], "category": ["Code", "fact"], "style": ["texting", "hedging"]}
    )
    return {"candidates": cands, "token_counts": tc, "labels": labels, "prompts": prompts}


def _row(df: pd.DataFrame, pid: str, target: str) -> pd.Series:
    return df.loc[(df["prompt_id"] == pid) & (df["target"] == target)].iloc[0]


def test_oracle_picks_shortest_preserved_with_method_tiebreak(tables):
    out = compute_oracle(**{k: tables[k] for k in ("candidates", "token_counts", "labels")})
    assert tuple(out.columns) == ORACLE_COLUMNS
    assert len(out) == 4
    r = _row(out, "p1", "qwen")
    # c1 and c2 tie at 60 tokens; operators ranks before llm_high. c4 (10 tokens) is rejected.
    assert r["oracle_candidate_id"] == "c2"
    assert r["method"] == "operators"
    assert r["savings"] == pytest.approx(0.4)
    assert r["n_candidates"] == 3 and r["n_labelled"] == 3 and r["n_preserved"] == 3
    assert not r["is_original"] and not r["stand_in"]
    # llama: c1 degraded, so c2 (40 tokens) wins.
    r = _row(out, "p1", "llama")
    assert r["oracle_candidate_id"] == "c2"
    assert r["savings"] == pytest.approx(0.2)


def test_no_preserved_candidate_falls_back_to_original(tables):
    out = compute_oracle(**{k: tables[k] for k in ("candidates", "token_counts", "labels")})
    for target in ("qwen", "llama"):
        r = _row(out, "p2", target)
        assert r["oracle_candidate_id"] == "original" and r["method"] == "original"
        assert r["savings"] == 0.0 and r["is_original"] and not r["has_preserved"]
        assert r["tokens_oracle"] == r["tokens_original"] == 40
    assert _row(out, "p2", "llama")["n_labelled"] == 0


def test_candidate_id_tiebreak_and_unknown_methods():
    cands = pd.DataFrame(
        [_cand("p", "b", "zeta"), _cand("p", "a", "zeta"), _cand("p", "c", "alpha")]
    )
    tc = _tc(
        [("p", "original", "t", 10), ("p", "a", "t", 5), ("p", "b", "t", 5), ("p", "c", "t", 5)]
    )
    labels = _lab([("p", x, "t", "preserved") for x in "abc"])
    out = compute_oracle(cands, tc, labels, method_priority=())
    assert out.loc[0, "oracle_candidate_id"] == "c"  # alpha < zeta
    out = compute_oracle(cands, tc, labels, method_priority=("zeta",))
    assert out.loc[0, "oracle_candidate_id"] == "a"
    assert method_rank("x", ("y",)) == (1, "x") and method_rank("y", ("y",)) == (0, "")


def test_preserved_but_not_shorter_is_not_eligible():
    cands = pd.DataFrame([_cand("p", "a", "llm_low")])
    tc = _tc([("p", "original", "t", 10), ("p", "a", "t", 10)])
    out = compute_oracle(cands, tc, _lab([("p", "a", "t", "preserved")]))
    assert out.loc[0, "is_original"] and out.loc[0, "savings"] == 0.0
    assert out.loc[0, "has_preserved"]


def test_prompts_metadata_and_target_map(tables):
    labels = tables["labels"].rename(columns={"target": "target_model"})
    labels["target_model"] = labels["target_model"].map({"qwen": "Qwen/7B", "llama": "Llama/8B"})
    out = compute_oracle(
        tables["candidates"],
        tables["token_counts"],
        labels,
        prompts=tables["prompts"],
        target_map={"Qwen/7B": "qwen", "Llama/8B": "llama"},
        targets=["qwen"],
    )
    assert list(out["target"].unique()) == ["qwen"]
    assert _row(out, "p1", "qwen")["category"] == "code"
    assert _row(out, "p2", "qwen")["style"] == "hedging"


def test_stand_in_refused_unless_allowed(tables):
    cands = tables["candidates"].copy()
    cands.loc[cands["prompt_id"] == "p2", "stand_in"] = True
    with pytest.raises(StandInDataError):
        compute_oracle(cands, tables["token_counts"], tables["labels"])
    labels = tables["labels"].assign(stand_in=[False] * 8 + [True])
    with pytest.raises(StandInDataError):
        compute_oracle(tables["candidates"], tables["token_counts"], labels)
    out = compute_oracle(cands, tables["token_counts"], tables["labels"], allow_stand_in=True)
    assert out.set_index(["prompt_id", "target"])["stand_in"].to_dict() == {
        ("p1", "llama"): False,
        ("p1", "qwen"): False,
        ("p2", "llama"): True,
        ("p2", "qwen"): True,
    }


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda t: t["candidates"].drop(columns=["rejected"]), "missing column"),
        (lambda t: pd.concat([t["candidates"], t["candidates"].iloc[:1]]), "duplicate"),
        (lambda t: t["candidates"].assign(rejected="maybe"), "boolean"),
    ],
)
def test_candidate_validation(tables, mutate, match):
    with pytest.raises(Phase2InputError, match=match):
        compute_oracle(mutate(tables), tables["token_counts"], tables["labels"])


def test_token_count_and_label_validation(tables):
    tc = tables["token_counts"]
    with pytest.raises(Phase2InputError, match="original count"):
        compute_oracle(tables["candidates"], tc.iloc[1:], tables["labels"])
    with pytest.raises(Phase2InputError, match="labelled candidate count"):
        compute_oracle(tables["candidates"], tc.drop(index=1), tables["labels"])
    with pytest.raises(Phase2InputError, match="non-negative integers"):
        compute_oracle(tables["candidates"], tc.assign(tokens=1.5), tables["labels"])
    bad = tables["labels"].copy()
    bad.loc[0, "label"] = "fine"
    with pytest.raises(Phase2InputError, match="unknown label"):
        compute_oracle(tables["candidates"], tc, bad)
    ghost = pd.concat([tables["labels"], _lab([("p9", "c1", "qwen", "preserved")])])
    with pytest.raises(Phase2InputError, match="unknown candidates"):
        compute_oracle(tables["candidates"], tc, ghost)


def test_deterministic_under_row_shuffle(tables):
    base = compute_oracle(**{k: tables[k] for k in ("candidates", "token_counts", "labels")})
    shuffled = compute_oracle(
        tables["candidates"].sample(frac=1, random_state=1),
        tables["token_counts"].sample(frac=1, random_state=2),
        tables["labels"].sample(frac=1, random_state=3),
    )
    pd.testing.assert_frame_equal(base, shuffled)


def test_io_roundtrip_and_cli(tables, tmp_path: Path, capsys):
    paths = {}
    for name in ("candidates", "labels", "prompts"):
        paths[name] = write_table(tables[name], tmp_path / f"{name}.jsonl")
    paths["token_counts"] = write_table(tables["token_counts"], tmp_path / "tc.csv")
    assert read_table(paths["token_counts"]).shape == tables["token_counts"].shape
    out = tmp_path / "oracle.csv"
    rc = main(
        [
            "--candidates", str(paths["candidates"]),
            "--token-counts", str(paths["token_counts"]),
            "--labels", str(paths["labels"]),
            "--prompts", str(paths["prompts"]),
            "--out", str(out),
        ]
    )  # fmt: skip
    assert rc == 0
    table = read_table(out)
    assert len(table) == 4 and set(table["oracle_candidate_id"]) == {"c2", "original"}
    assert b"\r\n" not in out.read_bytes()
    assert "median savings" in capsys.readouterr().out


def test_validate_prompts_accepts_id_column() -> None:
    import pandas as pd

    from terser.phase2.oracle import validate_prompts

    out = validate_prompts(pd.DataFrame({"id": ["p1", "p2"], "category": ["Code", "fact"]}))
    assert out["prompt_id"].tolist() == ["p1", "p2"]
    assert out["category"].tolist() == ["code", "fact"]


def test_validate_token_counts_normalises_stage_original_ids() -> None:
    import pandas as pd

    from terser.phase2.oracle import ORIGINAL_ID, validate_token_counts

    df = pd.DataFrame(
        {
            "prompt_id": ["p1", "p1"],
            "candidate_id": ["p1:original", "p1:c1"],
            "target": ["qwen2.5", "qwen2.5"],
            "tokens": [10, 7],
        }
    )
    assert validate_token_counts(df)["candidate_id"].tolist() == [ORIGINAL_ID, "p1:c1"]
