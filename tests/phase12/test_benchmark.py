"""Row 12.11 benchmark release: layout, manifest, licence/NOTICE, WildChat exclusion, provenance."""

from __future__ import annotations

import functools
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from terser.data import real_slice as rs
from terser.data.splits import ALLOW_TEST_ENV, freeze_test
from terser.phase12 import benchmark as B
from terser.phase12 import unlock as U

WILD_MARKER = "zebrawildmarker"
WILD_ROWS: list[dict[str, Any]] = [
    {
        "conversation_hash": "a" * 32,
        "language": "English",
        "toxic": False,
        "redacted": False,
        "conversation": [
            {
                "role": "user",
                "content": f"Can you write a python function that reverses a string {WILD_MARKER}",
            }
        ],
    },
    {
        "conversation_hash": "b" * 32,
        "language": "English",
        "toxic": False,
        "redacted": False,
        "conversation": [
            {
                "role": "user",
                "content": f"Please fix this python code for my school project {WILD_MARKER} now",
            }
        ],
    },
]


def _jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _prompt(src: str, n: int, style: str, cat: str) -> dict[str, Any]:
    sid = f"{src}:{n}"
    return {
        "id": f"{sid}:{style}",
        "source": src,
        "source_id": sid,
        "category": cat,
        "style": style,
        "prompt": f"hey so um {src} {n} {style} pls",
        "clean": f"Clean {src} {n}.",
        "generator_model": "Qwen/Qwen2.5-32B-Instruct-AWQ",
        "prompt_version": "casualise_v1",
        "reference": "UPSTREAM-REFERENCE-ANSWER",
        "stand_in": False,
        "rejected": False,
    }


@dataclass
class Env:
    root: Path
    inputs: B.BuildInputs
    log: Path
    out: Path
    prompt_ids: list[str]


def _candidate(pid: str, method: str, generator: str) -> dict[str, Any]:
    return {
        "prompt_id": pid,
        "candidate_id": f"{pid}:{method}",
        "method": method,
        "text": f"short {pid}",
        "generator": generator,
        "stand_in": False,
        "rejected": False,
        "rejection_reason": None,
    }


def _label(pid: str, split: str, judge: str = "Qwen/Qwen2.5-32B-Instruct-AWQ") -> dict[str, Any]:
    return {
        "schema_version": "label_v1",
        "id": f"{pid}:llm_low@Qwen/Qwen2.5-7B-Instruct",
        "prompt_id": pid,
        "candidate_id": f"{pid}:llm_low",
        "category": "fact",
        "target_model": "Qwen/Qwen2.5-7B-Instruct",
        "judge_model": judge,
        "judge_protocol": "pairwise_avg",
        "judge_prompt_version": "pairwise_v1",
        "label": "preserved",
        "label_int": 0,
        "split": split,
        "stand_in": False,
        "needs_human_review": False,
        "position_consistent": True,
    }


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Env:
    monkeypatch.delenv(ALLOW_TEST_ENV, raising=False)
    monkeypatch.setattr(U, "git_state", lambda repo=Path("."), exclude=(): ("abc123", False))
    splits = tmp_path / "splits"
    rows = {
        "train": [_prompt("dolly", 1, "texting", "fact"), _prompt("dolly", 1, "hedging", "fact")],
        "val": [_prompt("gsm8k", 2, "rambling", "reason")],
        "test": [_prompt("humaneval", 3, "repetitive", "code")],
    }
    for split, rs_ in rows.items():
        _jsonl(splits / f"{split}.jsonl", rs_)
    (splits / "splits.json").write_text(
        json.dumps({"summary": {"seed": 0}, "leakage": {"ok": True, "n_shared_groups": 0}}),
        encoding="utf-8",
    )
    freeze_test(splits)
    real = tmp_path / "real"
    real_recs = rs.build_records(WILD_ROWS, rs.SliceSummary())
    assert len(real_recs) == 2
    _jsonl(real / "test.jsonl", real_recs)
    freeze_test(real)
    dirs = {"test": splits, "real": real}
    log = tmp_path / "log" / "access.jsonl"
    U.unlock("final eval", who="tester", log_path=log, split_dirs=dirs)

    pids = [r["id"] for rs_ in rows.values() for r in rs_]
    cands = [_candidate(p, "llm_low", "Qwen/Qwen2.5-32B-Instruct-AWQ|rewrite_v1") for p in pids]
    cands.append(
        _candidate(
            pids[0],
            "llmlingua2",
            "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank" "?rate=0.5",
        )
    )
    cands.append(_candidate(pids[0], "ops_FHC", "operators:FHC"))
    real_cand = _candidate(real_recs[0]["id"], "llm_low", "Qwen/Qwen2.5-32B-Instruct-AWQ|x")
    real_cand["text"] = f"rewrite of {WILD_MARKER}"
    cands.append(real_cand)
    _jsonl(tmp_path / "candidates.jsonl", cands)
    split_of = {r["id"]: s for s, rs_ in rows.items() for r in rs_}
    _jsonl(tmp_path / "labels.jsonl", [_label(p, split_of[p]) for p in pids])

    rub = tmp_path / "rubrics"
    rub.mkdir()
    (rub / "code.yaml").write_text("category: code\nversion: 1\n", encoding="utf-8")
    prm = tmp_path / "prompts"
    prm.mkdir()
    (prm / "pairwise_v1.txt").write_text("Compare A and B.\n", encoding="utf-8")

    inputs = B.BuildInputs(
        splits_dir=splits,
        candidates=tmp_path / "candidates.jsonl",
        labels=tmp_path / "labels.jsonl",
        rubrics_dir=rub,
        judge_prompts_dir=prm,
        reader=functools.partial(U.read_locked, log_path=log, split_dirs=dirs),
        lock_dirs=dirs,
        repo=tmp_path,
        code_ref=("deadbeef", False),
    )
    return Env(tmp_path, inputs, log, tmp_path / "release", pids)


def _files(out: Path) -> dict[str, bytes]:
    return {p.relative_to(out).as_posix(): p.read_bytes() for p in out.rglob("*") if p.is_file()}


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def test_manifest_hashes_every_file(env: Env) -> None:
    manifest = B.build(env.out, env.inputs)
    files = _files(env.out)
    listed = {f["path"]: f for f in manifest["files"]}
    assert set(files) - set(listed) == {"MANIFEST.json", "SHA256SUMS"}
    for rel, f in listed.items():
        assert f["sha256"] == hashlib.sha256(files[rel]).hexdigest()
        assert f["bytes"] == len(files[rel])
    assert listed["prompts/train.jsonl"]["records"] == 2
    assert listed[B.CANDIDATES_FILE]["records"] == 6  # the WildChat candidate is dropped
    assert listed[B.REAL_IDS_FILE]["records"] == 2
    pairs = [line.split("  ", 1) for line in files["SHA256SUMS"].decode().splitlines()]
    sums = {path: digest for digest, path in pairs}
    assert set(sums) == set(files) - {"SHA256SUMS"}
    assert sums["MANIFEST.json"] == hashlib.sha256(files["MANIFEST.json"]).hexdigest()
    assert manifest["code"] == {
        "repo": B.REPO_URL,
        "commit": "deadbeef",
        "dirty": False,
        "paths": list(B.CODE_PATHS),
        "project_licence_file": None,
    }
    assert not any(p.endswith(".py") for p in files)  # no evaluation code is copied
    assert B.verify(env.out)["ok"]
    (env.out / "prompts" / "val.jsonl").write_text("{}\n", encoding="utf-8")
    res = B.verify(env.out)
    assert not res["ok"] and res["problems"] == ["changed prompts/val.jsonl"]


def test_records_carry_source_and_source_id(env: Env) -> None:
    B.build(env.out, env.inputs)
    for split in ("train", "val", "test"):
        for r in _read(env.out / "prompts" / f"{split}.jsonl"):
            assert r["split"] == split and r["source"] and r["source_id"]
            assert r["prompt"] and r["clean"] and r["category"] and r["style"]
            assert "reference" not in r
    for rel in (B.CANDIDATES_FILE, B.LABELS_FILE):
        recs = _read(env.out / rel)
        assert recs and all(r["source"] and r["source_id"] and r["split"] for r in recs)
    test_c = [c for c in _read(env.out / B.CANDIDATES_FILE) if c["split"] == "test"]
    assert {c["source_id"] for c in test_c} == {"humaneval:3"}


def test_wildchat_text_is_excluded(env: Env) -> None:
    manifest = B.build(env.out, env.inputs)
    for rel, data in _files(env.out).items():
        assert WILD_MARKER.encode() not in data, rel
    real = _read(env.out / B.REAL_IDS_FILE)
    assert {r["source_id"] for r in real} == {"wildchat:" + "a" * 32, "wildchat:" + "b" * 32}
    for r in real:
        assert set(r) == {*B.REAL_ID_FIELDS, "split", "prompt_sha256"}
        assert len(r["prompt_sha256"]) == 64
    assert manifest["stats"]["candidates"]["dropped"] == 1


def test_locked_splits_read_through_logged_reader(env: Env) -> None:
    B.build(env.out, env.inputs)
    reads = [e for e in U.read_log(env.log) if e["event"] == "read"]
    assert [e["purpose"] for e in reads] == [B.PURPOSE, B.PURPOSE]
    assert {Path(e["file"]).parent.name for e in reads} == {"splits", "real"}


def test_license_and_notice(env: Env) -> None:
    B.build(env.out, env.inputs)
    lic = (env.out / "LICENSE").read_text(encoding="utf-8")
    assert "CC-BY-SA-4.0" in lic and B.DATA_LICENSE_URL in lic
    assert "No code is included" in lic and "no LICENSE file" in lic
    notice = (env.out / "NOTICE").read_text(encoding="utf-8")
    for name in ("dolly", "gsm8k", "humaneval"):
        src = B.SOURCES[name]
        assert f"{src.repo_id} @ {src.revision}" in notice
        assert f"Licence: {src.license_spdx}" in notice
    assert "Wikipedia editors and contributors" in notice
    assert "Copyright (c) OpenAI (https://openai.com)" in notice
    assert "Copyright (c) 2021 OpenAI" in notice
    assert notice.count("Permission is hereby granted") == 2
    assert notice.count("Changes made:") == 4  # three sources + WildChat
    assert "allenai/WildChat-1M @ " + rs.REVISION in notice
    assert "no record in this release (no notice needed): mbpp, oasst1" in notice
    assert "* Qwen/Qwen2.5-32B-Instruct-AWQ\n  Licence: Apache-2.0" in notice
    assert "wrote the casual prompts" in notice and "judged target outputs" in notice
    assert (
        "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank\n  Licence: Apache-2.0"
        in notice
    )
    assert "outputs are NOT in this release" in notice
    assert "Llama 3.1 Community Licence does not apply" in notice


def test_card_states_known_limitations(env: Env) -> None:
    B.build(env.out, env.inputs)
    card = (env.out / "README.md").read_text(encoding="utf-8")
    assert card.startswith("# Data Card: `terser-bench-v1`")
    assert "Known limitations" in card
    assert "0 residual PII" in card
    assert "judge estimates" in card
    assert "`deadbeef`" in card and B.REPO_URL in card
    assert "rebuild-real" in card
    for heading in ("## 1. Source", "## 6. Splits", "## 7. PII handling", "## 11. Maintenance"):
        assert heading in card


def test_llama_text_is_refused(env: Env) -> None:
    _jsonl(
        env.inputs.labels,
        [_label(env.prompt_ids[0], "train", judge="meta-llama/Llama-3.1-8B-Instruct")],
    )
    with pytest.raises(ValueError, match="Llama"):
        B.build(env.out, env.inputs)


def test_refuses_to_overwrite_foreign_directory(env: Env) -> None:
    env.out.mkdir()
    (env.out / "keep.txt").write_text("x", encoding="utf-8")
    with pytest.raises(FileExistsError):
        B.build(env.out, env.inputs)
    with pytest.raises(FileExistsError, match="refusing"):
        B.build(env.out, env.inputs, force=True)
    assert (env.out / "keep.txt").is_file()


def test_rebuild_real_restores_and_checks_text(env: Env) -> None:
    B.build(env.out, env.inputs)
    out = env.root / "rebuilt.jsonl"
    res = B.rebuild_real(env.out, out, rows=WILD_ROWS)
    assert res["rebuilt"] == 2 and not res["missing"] and not res["sha256_mismatch"]
    assert all(WILD_MARKER in r["prompt"] for r in _read(out))
    changed = [dict(WILD_ROWS[0]), WILD_ROWS[1]]
    changed[0]["conversation"] = [
        {"role": "user", "content": "Can you write a python function that sorts a list quickly"}
    ]
    res = B.rebuild_real(env.out, out, rows=changed)
    assert res["rebuilt"] == 1 and len(res["sha256_mismatch"]) == 1


def test_rebuild_real_restores_frozen_text_after_scrubber_drift(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    B.build(env.out, env.inputs)
    monkeypatch.setattr(
        rs, "scrub_text", lambda text: SimpleNamespace(scrubbed=text.replace("python", "[X]"))
    )
    out = env.root / "rebuilt.jsonl"
    res = B.rebuild_real(env.out, out, rows=WILD_ROWS)
    assert res["rebuilt"] == 2 and len(res["scrub_drift"]) == 2 and not res["sha256_mismatch"]
    recs = _read(out)
    assert all(r["scrub_drift"] and "python" in r["prompt"] for r in recs)


def test_model_of() -> None:
    assert B.model_of("Qwen/Qwen2.5-32B-Instruct-AWQ|rewrite_v1") == "Qwen/Qwen2.5-32B-Instruct-AWQ"
    assert B.model_of("microsoft/llmlingua-2-x@rev?rate=0.5") == "microsoft/llmlingua-2-x"
    assert B.model_of("operators:FHC") is None
