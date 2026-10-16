"""``exclude_splits`` keeps held-out prompts out of candidate generation."""

from __future__ import annotations

import json
from pathlib import Path

from terser.data.candidates import load_prompts


def _write(path: Path) -> None:
    rows = [
        {"id": "a", "prompt": "x", "split": "train"},
        {"id": "b", "prompt": "y", "split": "test"},
        {"id": "c", "prompt": "z", "split": "val"},
        {"id": "d", "prompt": "w"},
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def test_exclude_test(tmp_path: Path) -> None:
    f = tmp_path / "p.jsonl"
    _write(f)
    assert [r["id"] for r in load_prompts(f, exclude_splits=("test",))] == ["a", "c", "d"]


def test_limit_counts_kept_prompts(tmp_path: Path) -> None:
    f = tmp_path / "p.jsonl"
    _write(f)
    assert [r["id"] for r in load_prompts(f, limit=2, exclude_splits=("test",))] == ["a", "c"]


def test_default_keeps_all(tmp_path: Path) -> None:
    f = tmp_path / "p.jsonl"
    _write(f)
    assert len(load_prompts(f)) == 4
