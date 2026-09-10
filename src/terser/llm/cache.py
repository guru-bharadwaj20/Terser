"""Disk cache for LLM responses, keyed by sha256 of the canonical request JSON."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from terser.llm.budget import atomic_write_json


def cache_key(
    provider: str,
    model: str,
    messages: list[dict[str, Any]],
    params: dict[str, Any],
    system: str | None = None,
) -> str:
    """sha256 over canonical JSON (sorted keys, compact separators) of the full request."""
    payload = {
        "provider": provider,
        "model": model,
        "params": params,
        "messages": messages,
        "system": system,
    }
    canon = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


class ResponseCache:
    """One JSON file per key under ``root/<key[:2]>/<key>.json``."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def put(self, key: str, payload: dict[str, Any]) -> None:
        atomic_write_json(self._path(key), payload)
