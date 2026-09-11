"""Load, merge and validate the Phase 1 pilot prompt set.

The pilot set (row 1.4.1) is stored as one JSON Lines file per category under
``data/pilot/prompts/{code,fact,reason,explain,creative,transform}.jsonl``. Every record carries
``id``, ``category``, ``prompt``, ``origin``, ``style``, ``has_payload`` and ``constraints``.
:func:`load_prompt_dir` merges the six files; :func:`load_prompt_file` reads a single file (e.g. a
test fixture). Both validate every record and reject duplicate ids.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from terser.data.categories import Category, parse_category

__all__ = [
    "REQUIRED_FIELDS",
    "PromptValidationError",
    "category_files",
    "load_prompt_dir",
    "load_prompt_file",
    "missing_category_files",
    "validate_prompts",
    "validate_record",
]

REQUIRED_FIELDS: dict[str, type | tuple[type, ...]] = {
    "id": str,
    "category": str,
    "prompt": str,
    "origin": str,
    "style": str,
    "has_payload": bool,
    "constraints": list,
}
"""Required record fields and their JSON types."""


class PromptValidationError(ValueError):
    """A pilot prompt record or file is malformed."""


def validate_record(record: Any, where: str = "record") -> dict[str, Any]:
    """Validate one prompt record and return it with ``category`` normalised.

    Args:
        record: A decoded JSON object.
        where: Location used in error messages, e.g. ``"code.jsonl:3"``.

    Returns:
        A shallow copy of the record whose ``category`` is the canonical lowercase name.

    Raises:
        PromptValidationError: If a field is missing, has the wrong type, is empty, or the
            category is unknown.
    """
    if not isinstance(record, dict):
        raise PromptValidationError(f"{where}: expected a JSON object, got {type(record).__name__}")
    for name, typ in REQUIRED_FIELDS.items():
        if name not in record:
            raise PromptValidationError(f"{where}: missing field {name!r}")
        if not isinstance(record[name], typ):
            raise PromptValidationError(
                f"{where}: field {name!r} must be {getattr(typ, '__name__', typ)}, "
                f"got {type(record[name]).__name__}"
            )
    for name in ("id", "prompt", "style"):
        if not record[name].strip():
            raise PromptValidationError(f"{where}: field {name!r} is empty")
    if not all(isinstance(c, str) for c in record["constraints"]):
        raise PromptValidationError(f"{where}: 'constraints' must be a list of strings")
    try:
        category = parse_category(record["category"])
    except ValueError as exc:
        raise PromptValidationError(f"{where}: {exc}") from None
    out = dict(record)
    out["category"] = category.value
    return out


def validate_prompts(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate records and reject duplicate ids; return the normalised list."""
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for i, rec in enumerate(records):
        clean = validate_record(rec, where=f"record {i}")
        if clean["id"] in seen:
            raise PromptValidationError(f"duplicate prompt id {clean['id']!r}")
        seen.add(clean["id"])
        out.append(clean)
    if not out:
        raise PromptValidationError("prompt set is empty")
    return out


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read and validate every non-blank line of a JSON Lines file."""
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            where = f"{path.name}:{lineno}"
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PromptValidationError(f"{where}: invalid JSON ({exc.msg})") from None
            records.append(validate_record(obj, where=where))
    return records


def load_prompt_file(path: str | Path) -> list[dict[str, Any]]:
    """Load one JSON Lines prompt file (any mix of categories) and validate it."""
    return validate_prompts(_read_jsonl(Path(path)))


def category_files(prompt_dir: str | Path) -> dict[Category, Path]:
    """Expected per-category file paths under ``prompt_dir``."""
    root = Path(prompt_dir)
    return {cat: root / f"{cat.value}.jsonl" for cat in Category}


def missing_category_files(prompt_dir: str | Path) -> list[Path]:
    """Per-category prompt files that do not exist yet."""
    return [p for p in category_files(prompt_dir).values() if not p.is_file()]


def load_prompt_dir(prompt_dir: str | Path) -> list[dict[str, Any]]:
    """Merge the six per-category files, checking each record matches its file's category.

    Raises:
        FileNotFoundError: If any category file is missing.
        PromptValidationError: On malformed records, category mismatches or duplicate ids.
    """
    missing = missing_category_files(prompt_dir)
    if missing:
        raise FileNotFoundError(
            "missing pilot prompt files: " + ", ".join(p.as_posix() for p in missing)
        )
    merged: list[dict[str, Any]] = []
    for cat, path in category_files(prompt_dir).items():
        for rec in _read_jsonl(path):
            if rec["category"] != cat.value:
                raise PromptValidationError(
                    f"{path.name}: record {rec['id']!r} has category {rec['category']!r}"
                )
            merged.append(rec)
    return validate_prompts(merged)
