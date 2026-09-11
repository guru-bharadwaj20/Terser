"""Tests for ``terser.data.sources`` (row 2.1.1). No network unless marked ``network``."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from terser.data.categories import Category
from terser.data.sources import (
    DOLLY_CATEGORY_MAP,
    MANIFEST_NAME,
    SOURCES,
    dolly_category_hint,
    download_source,
    iter_records,
    main,
    parse_dolly,
    parse_gsm8k,
    parse_humaneval,
    parse_mbpp,
    parse_oasst1,
    sha256_file,
    verify_manifest,
)

RECORD_KEYS = {"source", "source_id", "instruction", "input", "reference", "category_hint"}
NAN = float("nan")


def test_registry_is_pinned_and_licensed() -> None:
    assert set(SOURCES) == {"dolly", "oasst1", "humaneval", "mbpp", "gsm8k"}
    for name, src in SOURCES.items():
        assert src.name == name
        assert re.fullmatch(r"[0-9a-f]{40}", src.revision), name
        assert src.files and src.license_spdx and src.card_url.startswith("https://")
        assert all(c in {x.value for x in Category} for c in src.category_coverage)


def test_dolly_mapping_targets_valid_categories() -> None:
    assert set(DOLLY_CATEGORY_MAP) == {
        "creative_writing",
        "brainstorming",
        "summarization",
        "information_extraction",
        "closed_qa",
        "general_qa",
        "open_qa",
        "classification",
    }
    assert dolly_category_hint("classification", "x") is None
    assert dolly_category_hint("not_a_category", "x") is None
    assert dolly_category_hint("open_qa", "Canberra") is Category.FACT
    assert dolly_category_hint("open_qa", "word " * 31) is Category.EXPLAIN
    assert dolly_category_hint("summarization", "x") is Category.TRANSFORM


def test_parse_dolly() -> None:
    rows = [
        {
            "instruction": "Capital of France?",
            "context": "",
            "response": "Paris",
            "category": "open_qa",
        },
        {"instruction": "  ", "context": "", "response": "skip", "category": "open_qa"},
        {
            "instruction": "Summarise this.",
            "context": "Long text.",
            "response": "Short.",
            "category": "summarization",
        },
        {
            "instruction": "Fish or not?",
            "context": "",
            "response": "Tope",
            "category": "classification",
        },
    ]
    recs = list(parse_dolly(rows))
    assert [r["source_id"] for r in recs] == ["dolly:0", "dolly:2", "dolly:3"]
    assert all(set(r) == RECORD_KEYS for r in recs)
    assert recs[0]["input"] is None and recs[0]["category_hint"] == "fact"
    assert recs[1]["input"] == "Long text." and recs[1]["category_hint"] == "transform"
    assert recs[2]["category_hint"] is None


def test_parse_oasst1_first_turn_english_with_best_reply() -> None:
    def msg(mid: str, parent: object, role: str, text: str, **kw: object) -> dict[str, object]:
        row = {"message_id": mid, "parent_id": parent, "role": role, "text": text, "lang": "en"}
        return {"deleted": False, "review_result": True, "rank": NAN, **row, **kw}

    rows = [
        msg("p1", NAN, "prompter", "Hi?"),
        msg("a1", "p1", "assistant", "second", rank=1.0),
        msg("a0", "p1", "assistant", "best", rank=0.0),
        msg("p2", None, "prompter", "Hallo", lang="de"),
        msg("p3", None, "prompter", "deleted", deleted=True),
        msg("p4", None, "prompter", "rejected", review_result=False),
        msg("f1", "a0", "prompter", "follow-up"),
        msg("p5", None, "prompter", "unanswered"),
    ]
    recs = list(parse_oasst1(rows))
    assert [r["source_id"] for r in recs] == ["oasst1:p1", "oasst1:p5"]
    assert recs[0]["reference"] == "best" and recs[0]["category_hint"] is None
    assert recs[1]["reference"] is None


def test_parse_code_and_math_sources() -> None:
    he = next(
        parse_humaneval(
            [{"task_id": "HumanEval/0", "prompt": "def f():\n", "canonical_solution": " return 1"}]
        )
    )
    assert he["source_id"] == "humaneval:HumanEval/0"
    assert he["input"] == "def f():\n" and he["category_hint"] == "code"

    mb = next(
        parse_mbpp(
            [
                {
                    "task_id": 7,
                    "text": "Write f.",
                    "code": "def f(): ...",
                    "test_list": ["assert f()"],
                }
            ]
        )
    )
    assert mb["source_id"] == "mbpp:7" and mb["input"] == "assert f()"
    assert mb["category_hint"] == "code"

    gs = next(
        parse_gsm8k([{"question": "1+1?", "answer": "2\n#### 2", "split": "test", "index": 3}])
    )
    assert gs["source_id"] == "gsm8k:test:3" and gs["category_hint"] == "reason"
    assert set(gs) == RECORD_KEYS


def _fake_source_dir(tmp_path: Path) -> Path:
    d = tmp_path / "raw" / "dolly"
    d.mkdir(parents=True)
    data = d / "databricks-dolly-15k.jsonl"
    rows = [
        {"instruction": "Why?", "context": "", "response": "Because.", "category": "general_qa"}
    ]
    data.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    manifest = {
        "source": "dolly",
        "files": [
            {
                "path": "databricks-dolly-15k.jsonl",
                "sha256": sha256_file(data),
                "bytes": data.stat().st_size,
            }
        ],
    }
    (d / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")
    return d


def test_manifest_verification_detects_tampering(tmp_path: Path) -> None:
    d = _fake_source_dir(tmp_path)
    assert verify_manifest(d) == []
    data = d / "databricks-dolly-15k.jsonl"
    content = data.read_bytes()
    data.write_bytes(content.replace(b"Because", b"Becausf"))
    assert any("sha256 mismatch" in p for p in verify_manifest(d))
    data.write_bytes(content + b"\n")
    assert any("size mismatch" in p for p in verify_manifest(d))
    data.unlink()
    assert any("missing file" in p for p in verify_manifest(d))
    assert verify_manifest(tmp_path / "nope") == [f"missing {tmp_path / 'nope' / MANIFEST_NAME}"]


def test_iter_records_and_cli_from_local_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _fake_source_dir(tmp_path)
    raw = tmp_path / "raw"
    recs = list(iter_records("dolly", raw))
    assert recs == [
        {
            "source": "dolly",
            "source_id": "dolly:0",
            "instruction": "Why?",
            "input": None,
            "reference": "Because.",
            "category_hint": "explain",
        }
    ]
    assert main(["verify", "--source", "dolly", "--raw-dir", str(raw)]) == 0
    assert main(["stats", "--source", "dolly", "--raw-dir", str(raw)]) == 0
    assert "dolly: 1 records (explain=1)" in capsys.readouterr().out
    with pytest.raises(ValueError, match="unknown source"):
        iter_records("nope", raw)


@pytest.mark.network
def test_download_humaneval(tmp_path: Path) -> None:
    manifest = download_source("humaneval", tmp_path)
    assert verify_manifest(manifest.parent) == []
    assert len(list(iter_records("humaneval", tmp_path))) == 164
