"""Shared constants and small I/O helpers for the Phase 12 tooling."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any, Final

__all__ = [
    "MAIN_SLICE",
    "REAL_SLICE",
    "TARGETS",
    "TOKEN_TARGET",
    "VAL_PROMPTS",
    "group_of",
    "read_jsonl",
    "short_target",
    "text_sha",
    "write_jsonl",
]

TARGETS: Final[dict[str, str]] = {
    "Qwen/Qwen2.5-7B-Instruct": "qwen2.5",
    "microsoft/Phi-3.5-mini-instruct": "phi3.5",
}
"""Judged target models and the tokenizer key (``terser.tokenizers.count``) of each."""

TOKEN_TARGET: Final = "qwen2.5"
"""Tokenizer of the rewriters, the gates' cost features and the Phase 8 val tables."""

VAL_PROMPTS: Final = Path("data/interim/rewrites/val_prompts.jsonl")
"""The 285 val prompts every Phase 8 rewrite and baseline was judged on."""

MAIN_SLICE: Final = "main"
REAL_SLICE: Final = "real"


def read_jsonl(
    path: str | Path, keep: Callable[[dict[str, Any]], bool] | None = None
) -> list[dict[str, Any]]:
    """Records of a JSONL file (blank lines skipped); ``keep`` filters while parsing."""
    out: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                if keep is None or keep(rec):
                    out.append(rec)
    return out


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> int:
    """Write ``rows`` as JSONL (LF line ends, UTF-8); returns the number written."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with p.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def text_sha(text: str) -> str:
    """sha256 of the UTF-8 text (the ``text_sha256`` of :mod:`terser.phase2.respond`)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def group_of(record: Mapping[str, Any]) -> str:
    """Bootstrap / split group: ``source_id`` (all casual variants of one prompt), else ``id``."""
    sid = record.get("source_id")
    return str(sid) if sid else str(record["id"])


def short_target(target: str) -> str:
    """``qwen2.5`` / ``phi3.5`` for a served model id (unknown ids map to themselves)."""
    return TARGETS.get(target, target)
