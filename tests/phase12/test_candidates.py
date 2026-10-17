"""Candidate build plumbing and the span-deletion rule (synthetic data, no models)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from terser.phase12 import candidates as C
from terser.phase12.common import read_jsonl, write_jsonl
from terser.phase12.taggers import delete_tokens


def _prompts(n: int, prefix: str = "v") -> list[dict[str, Any]]:
    return [
        {
            "id": f"{prefix}{i}",
            "prompt": f"um so please write a list of {i} things",
            "category": "fact",
            "source_id": f"src{i // 2}",
        }
        for i in range(n)
    ]


def test_val_reads_val_prompts_only(tmp_path: Path) -> None:
    p = tmp_path / "val_prompts.jsonl"
    write_jsonl(
        p, [*_prompts(3), {"id": "bad", "prompt": "x", "category": "fact", "rejected": True}]
    )
    rows = C.load_split_prompts("val", val_prompts=p)
    assert [r["id"] for r in rows] == ["v0", "v1", "v2"]
    assert {r["slice"] for r in rows} == {"main"} and {r["split"] for r in rows} == {"val"}
    t = tmp_path / "test_prompts.jsonl"
    write_jsonl(t, _prompts(1))
    with pytest.raises(ValueError):
        C.load_split_prompts("val", val_prompts=t)


def test_test_split_goes_through_reader() -> None:
    calls: list[tuple[str, str]] = []

    def reader(name: str, purpose: str) -> list[dict[str, Any]]:
        calls.append((name, purpose))
        return _prompts(2, "t") if name == "test" else _prompts(1, "real-")

    rows = C.load_split_prompts("test", reader=reader)
    assert [c[0] for c in calls] == ["test", "real"]
    assert [(r["id"], r["slice"]) for r in rows] == [
        ("t0", "main"),
        ("t1", "main"),
        ("real-0", "real"),
    ]


def test_build_writes_files_and_merges(tmp_path: Path) -> None:
    vp = tmp_path / "val_prompts.jsonl"
    write_jsonl(vp, _prompts(4))

    def fake(prompts: Any, res: Any, **kw: Any) -> tuple[dict[str, str], dict[str, Any]]:
        texts = {str(p["id"]): str(p["prompt"]).replace("um so please ", "") for p in prompts}
        return texts, {"generator": f"fake{kw.get('seed', '')}", "latency_mean_s": 0.001}

    class Res:
        train_ids: set[str] = set()

    out = tmp_path / "out"
    meta = C.build(
        "val",
        out,
        ["ops_FHC", "textcnn_tagger"],
        textcnn_seeds=(0, 1),
        val_prompts=vp,
        resources=Res(),  # type: ignore[arg-type]
        builders={"ops_FHC": fake, "textcnn_tagger": fake},
    )
    assert set(meta["methods"]) == {"ops_FHC", "textcnn_tagger", "textcnn_tagger_s1"}
    cands = read_jsonl(out / "val_candidates_cpu.jsonl")
    assert len(cands) == 12
    c0 = cands[0]
    assert c0["candidate_id"] == f"{c0['prompt_id']}:{c0['method']}" and c0["rejected"] is False
    sft_in = read_jsonl(out / "val_sft_input.jsonl")
    assert sft_in[0]["savings"] is None and set(sft_in[0]) == {
        "id",
        "category",
        "prompt",
        "savings",
    }
    # rebuilding one method replaces only its records
    C.build("val", out, ["ops_FHC"], val_prompts=vp, resources=Res(), builders={"ops_FHC": fake})  # type: ignore[arg-type]
    assert len(read_jsonl(out / "val_candidates_cpu.jsonl")) == 12
    with pytest.raises(ValueError):
        C.build("val", out, ["nope"], val_prompts=vp)


def test_leak_check(tmp_path: Path) -> None:
    vp = tmp_path / "val_prompts.jsonl"
    write_jsonl(vp, _prompts(2))

    class Res:
        train_ids = {"v1"}

    with pytest.raises(ValueError, match="train span table"):
        C.build("val", tmp_path / "o", ["hmm"], val_prompts=vp, resources=Res())  # type: ignore[arg-type]


def test_import_sft_keeps_raw_rewrite(tmp_path: Path) -> None:
    gen = tmp_path / "gen.jsonl"
    write_jsonl(
        gen,
        [
            {
                "id": "a",
                "prompt": "one two three",
                "raw_rewrite": "one two three four",
                "latency_s": 0.5,
            },
            {"id": "b", "prompt": "one two three", "raw_rewrite": " ", "latency_s": 0.7},
        ],
    )
    out = tmp_path / "gpu.jsonl"
    C.import_sft("sftx", gen, out, lambda t: len(t.split()))
    rows = {r["prompt_id"]: r for r in read_jsonl(out)}
    assert rows["a"]["text"] == "one two three four"  # longer: safeguard applies later
    assert rows["a"]["rewriter_tokens_in"] == 4 and rows["a"]["rewriter_tokens_out"] == 5
    assert rows["b"]["text"] == "one two three" and rows["b"]["empty_raw"] is True


def test_delete_tokens_rules() -> None:
    text = 'Um so please do not drop 3 "um quoted" words\n    indented code stays'
    toks = ["Um", "so", "please", "do", "not", "drop", "3", '"', "um", "quoted", '"', "words"]
    toks += ["indented", "code", "stays"]
    drop = [t in {"Um", "so", "please", "not", "3", "um"} for t in toks]
    out = delete_tokens(text, drop)
    first, second = out.split("\n")
    assert first.startswith("Do not drop 3")  # negation and number kept, capital restored
    assert '"um quoted"' in first  # payload span untouched
    assert second == "    indented code stays"  # untouched line byte for byte
    assert delete_tokens(text, [False] * len(toks)) == text
    with pytest.raises(ValueError):
        delete_tokens(text, [True])
    assert json.dumps(out)
