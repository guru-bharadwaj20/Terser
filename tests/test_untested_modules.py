"""Direct tests for modules that were only exercised indirectly (audit item 257)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from terser.llm.cache import ResponseCache, cache_key


def test_cache_key_is_canonical_and_request_sensitive() -> None:
    msgs = [{"role": "user", "content": "hi"}]
    a = cache_key("local", "m", msgs, {"temperature": 0.0, "max_tokens": 8})
    b = cache_key("local", "m", msgs, {"max_tokens": 8, "temperature": 0.0})
    assert a == b and len(a) == 64
    assert a != cache_key("local", "m", msgs, {"temperature": 0.0, "max_tokens": 9})
    assert a != cache_key("local", "m", msgs, {"temperature": 0.0, "max_tokens": 8}, system="s")


def test_response_cache_roundtrip_layout_and_corrupt_entry(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path)
    key = cache_key("local", "m", [], {})
    assert cache.get(key) is None
    cache.put(key, {"text": "ok"})
    path = tmp_path / key[:2] / f"{key}.json"
    assert path.is_file() and cache.get(key) == {"text": "ok"}
    path.write_text("{not json", encoding="utf-8")
    assert cache.get(key) is None  # a corrupt entry is a miss, not a crash


def test_serving_checks_prompt_filter(tmp_path: Path) -> None:
    from terser.phase11 import serving_checks as sc

    rows = [{"prompt": c * n} for c, n in (("x", 10), ("y", 30), ("z", 5000), ("w", 40))]
    path = tmp_path / "p.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert sc._prompts(path, 5) == ["y" * 30, "w" * 40]
    assert sc._prompts(path, 1) == ["y" * 30]


def test_serving_checks_load_refuses_repeated_prompts() -> None:
    import asyncio

    pytest.importorskip("httpx")
    from terser.phase11 import serving_checks as sc

    with pytest.raises(ValueError, match="distinct prompts"):
        asyncio.run(sc._load("http://127.0.0.1:9/rewrite", ["a", "b"], rps=2.0, seconds=2.0))
