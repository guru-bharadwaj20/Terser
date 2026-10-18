"""Tests for the on-disk LLM response cache."""

from __future__ import annotations

from pathlib import Path

import pytest

from terser.llm import cache as cache_mod
from terser.llm.cache import ResponseCache


def test_put_get_round_trip(tmp_path: Path) -> None:
    c = ResponseCache(tmp_path)
    c.put("abcd", {"text": "hi"})
    assert c.get("abcd") == {"text": "hi"}
    assert c.get("missing") is None


def test_put_tolerates_a_lost_rename_race(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    c = ResponseCache(tmp_path)
    c.put("abcd", {"text": "first"})

    def denied(*_: object, **__: object) -> None:
        raise PermissionError("[WinError 5] Access is denied")

    monkeypatch.setattr(cache_mod, "atomic_write_json", denied)
    c.put("abcd", {"text": "first"})  # another writer already stored it: no error
    with pytest.raises(PermissionError):
        c.put("ef01", {"text": "new"})  # nothing on disk: a real failure
