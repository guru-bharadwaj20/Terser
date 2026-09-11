"""Registry, downloader and record parsers for the upstream prompt sources.

Plan row **2.1.1**. Each :class:`Source` pins a Hugging Face dataset repository to a commit hash,
lists the files Terser uses, records the licence (SPDX id, audited in
``reports/license_audit.md``, row 2.1.2) and knows how to turn the raw rows into normalised records:

.. code-block:: python

    {"source": "dolly", "source_id": "dolly:42", "instruction": "...", "input": None,
     "reference": "...", "category_hint": "fact"}

``category_hint`` is one of the six Terser categories (``docs/categories.md``) or ``None`` when the
source gives no reliable signal. It is a *hint* for balanced sampling; the final label is assigned
later (rows 2.1.4 and 2.3.5). The mapping is documented in ``data/SOURCES.md``.

Downloads use :func:`huggingface_hub.hf_hub_download` (no ``datasets`` dependency) and are copied to
``data/raw/<source>/`` together with ``MANIFEST.json`` holding the sha256 of every file.

CLI::

    python -m terser.data.sources download [--source dolly --source gsm8k] [--raw-dir data/raw]
    python -m terser.data.sources verify   [--source ...]
    python -m terser.data.sources stats    [--source ...]
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, TypedDict

from terser.data.categories import Category

__all__ = [
    "DEFAULT_RAW_DIR",
    "DOLLY_CATEGORY_MAP",
    "MANIFEST_NAME",
    "SOURCES",
    "NormalisedRecord",
    "Source",
    "dolly_category_hint",
    "download_source",
    "iter_records",
    "main",
    "parse_dolly",
    "parse_gsm8k",
    "parse_humaneval",
    "parse_mbpp",
    "parse_oasst1",
    "read_rows",
    "sha256_file",
    "verify_manifest",
]

DEFAULT_RAW_DIR = Path("data/raw")
MANIFEST_NAME = "MANIFEST.json"
_CHUNK = 1 << 20


class NormalisedRecord(TypedDict):
    """One upstream prompt in Terser's common schema."""

    source: str
    source_id: str
    instruction: str
    input: str | None
    reference: str | None
    category_hint: str | None


Parser = Callable[[Iterable[Mapping[str, Any]]], Iterator[NormalisedRecord]]


def _is_missing(value: Any) -> bool:
    """True for ``None`` and float NaN (pandas' marker for a null in a parquet column)."""
    return value is None or (isinstance(value, float) and value != value)


def _clean(text: Any) -> str | None:
    """Return ``text`` stripped, or ``None`` when it is missing or blank."""
    if _is_missing(text):
        return None
    s = str(text).strip()
    return s or None


def _record(
    source: str,
    source_id: str,
    instruction: str,
    input_: str | None,
    reference: str | None,
    category: Category | None,
) -> NormalisedRecord:
    return NormalisedRecord(
        source=source,
        source_id=source_id,
        instruction=instruction,
        input=input_,
        reference=reference,
        category_hint=None if category is None else category.value,
    )


# --------------------------------------------------------------------------------------------------
# Dolly
# --------------------------------------------------------------------------------------------------

DOLLY_CATEGORY_MAP: Mapping[str, Category | None] = MappingProxyType(
    {
        "creative_writing": Category.CREATIVE,
        "brainstorming": Category.CREATIVE,
        "summarization": Category.TRANSFORM,
        "information_extraction": Category.TRANSFORM,
        "closed_qa": Category.FACT,
        "general_qa": Category.EXPLAIN,
        "open_qa": Category.FACT,  # refined by reference length, see dolly_category_hint
        "classification": None,
    }
)
"""Dolly ``category`` → Terser category hint (``data/SOURCES.md`` § Category mapping)."""

DOLLY_OPEN_QA_FACT_MAX_WORDS = 30
"""``open_qa`` rows whose reference answer has more words than this are hinted ``explain``."""


def dolly_category_hint(dolly_category: str, response: str | None) -> Category | None:
    """Map a Dolly category (plus, for ``open_qa``, the reference length) to a Terser hint.

    Args:
        dolly_category: The Dolly ``category`` field.
        response: The reference response; only used for ``open_qa``.

    Returns:
        The category hint, or ``None`` for unknown or deliberately unmapped categories.
    """
    hint = DOLLY_CATEGORY_MAP.get(dolly_category)
    if dolly_category == "open_qa":
        words = len((response or "").split())
        return Category.FACT if words <= DOLLY_OPEN_QA_FACT_MAX_WORDS else Category.EXPLAIN
    return hint


def parse_dolly(rows: Iterable[Mapping[str, Any]]) -> Iterator[NormalisedRecord]:
    """Parse ``databricks-dolly-15k.jsonl`` rows. ``source_id`` is the 0-based line index."""
    for i, row in enumerate(rows):
        instruction = _clean(row.get("instruction"))
        if instruction is None:
            continue
        response = _clean(row.get("response"))
        yield _record(
            "dolly",
            f"dolly:{i}",
            instruction,
            _clean(row.get("context")),
            response,
            dolly_category_hint(str(row.get("category", "")), response),
        )


# --------------------------------------------------------------------------------------------------
# OpenAssistant oasst1
# --------------------------------------------------------------------------------------------------


def parse_oasst1(rows: Iterable[Mapping[str, Any]]) -> Iterator[NormalisedRecord]:
    """Parse oasst1 messages into English first-turn prompts.

    Keeps messages with ``role == "prompter"``, no parent, ``lang == "en"``, not deleted, and not
    rejected in review (``review_result`` is not ``False``). The reference is the best-ranked
    (lowest ``rank``) assistant reply to that prompt when one exists. No category hint: oasst1 has
    no task-type label. ``user_id`` and every other metadata field are dropped.
    """
    prompts: dict[str, Mapping[str, Any]] = {}
    best_reply: dict[str, tuple[float, str]] = {}
    for row in rows:
        if row.get("deleted"):
            continue
        parent = None if _is_missing(row.get("parent_id")) else str(row["parent_id"])
        if row.get("role") == "prompter" and not parent:
            if row.get("lang") == "en" and row.get("review_result") is not False:
                prompts[str(row["message_id"])] = row
        elif row.get("role") == "assistant" and parent:
            rank = row.get("rank")
            text = _clean(row.get("text"))
            if text is None:
                continue
            key = float("inf") if _is_missing(rank) else float(str(rank))
            prev = best_reply.get(parent)
            if prev is None or key < prev[0]:
                best_reply[parent] = (key, text)
    for message_id in sorted(prompts):
        instruction = _clean(prompts[message_id].get("text"))
        if instruction is None:
            continue
        reply = best_reply.get(message_id)
        yield _record(
            "oasst1",
            f"oasst1:{message_id}",
            instruction,
            None,
            None if reply is None else reply[1],
            None,
        )


# --------------------------------------------------------------------------------------------------
# HumanEval, MBPP, GSM8K
# --------------------------------------------------------------------------------------------------

HUMANEVAL_INSTRUCTION = "Complete the following Python function."


def parse_humaneval(rows: Iterable[Mapping[str, Any]]) -> Iterator[NormalisedRecord]:
    """Parse HumanEval rows: the signature and docstring become the ``input`` payload."""
    for row in rows:
        yield _record(
            "humaneval",
            f"humaneval:{row['task_id']}",
            HUMANEVAL_INSTRUCTION,
            str(row["prompt"]),
            _clean(row.get("canonical_solution")),
            Category.CODE,
        )


def parse_mbpp(rows: Iterable[Mapping[str, Any]]) -> Iterator[NormalisedRecord]:
    """Parse MBPP (``full`` config) rows: the problem text is the instruction, the assert-based
    tests (which fix the function name) are the ``input`` payload."""
    for row in rows:
        instruction = _clean(row.get("text"))
        if instruction is None:
            continue
        raw_tests = row.get("test_list")
        tests = [] if raw_tests is None else [str(t) for t in raw_tests]
        yield _record(
            "mbpp",
            f"mbpp:{int(row['task_id'])}",
            instruction,
            "\n".join(tests) or None,
            _clean(row.get("code")),
            Category.CODE,
        )


def parse_gsm8k(rows: Iterable[Mapping[str, Any]]) -> Iterator[NormalisedRecord]:
    """Parse GSM8K (``main`` config) rows. Rows must carry ``split`` and ``index`` keys, which
    :func:`read_rows` adds, so that ``source_id`` is stable (``gsm8k:train:17``)."""
    for row in rows:
        instruction = _clean(row.get("question"))
        if instruction is None:
            continue
        yield _record(
            "gsm8k",
            f"gsm8k:{row['split']}:{row['index']}",
            instruction,
            None,
            _clean(row.get("answer")),
            Category.REASON,
        )


# --------------------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Source:
    """One upstream dataset pinned to a Hugging Face commit."""

    name: str
    repo_id: str
    revision: str
    files: tuple[str, ...]
    license_spdx: str
    card_url: str
    redistribution: str
    category_coverage: tuple[str, ...]
    parser: Parser


SOURCES: Mapping[str, Source] = MappingProxyType(
    {
        s.name: s
        for s in (
            Source(
                name="dolly",
                repo_id="databricks/databricks-dolly-15k",
                revision="bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a",
                files=("databricks-dolly-15k.jsonl",),
                license_spdx="CC-BY-SA-3.0",
                card_url="https://huggingface.co/datasets/databricks/databricks-dolly-15k",
                redistribution="yes, attribution + share-alike",
                category_coverage=("fact", "explain", "creative", "transform"),
                parser=parse_dolly,
            ),
            Source(
                name="oasst1",
                repo_id="OpenAssistant/oasst1",
                revision="fdf72ae0827c1cda404aff25b6603abec9e3399b",
                files=(
                    "data/train-00000-of-00001-b42a775f407cee45.parquet",
                    "data/validation-00000-of-00001-134b8fd0c89408b6.parquet",
                ),
                license_spdx="Apache-2.0",
                card_url="https://huggingface.co/datasets/OpenAssistant/oasst1",
                redistribution="yes, licence + notice",
                category_coverage=(),
                parser=parse_oasst1,
            ),
            Source(
                name="humaneval",
                repo_id="openai/openai_humaneval",
                revision="7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",
                files=("openai_humaneval/test-00000-of-00001.parquet",),
                license_spdx="MIT",
                card_url="https://huggingface.co/datasets/openai/openai_humaneval",
                redistribution="yes, copyright notice",
                category_coverage=("code",),
                parser=parse_humaneval,
            ),
            Source(
                name="mbpp",
                repo_id="google-research-datasets/mbpp",
                revision="4bb6404fdc6cacfda99d4ac4205087b89d32030c",
                files=(
                    "full/train-00000-of-00001.parquet",
                    "full/validation-00000-of-00001.parquet",
                    "full/test-00000-of-00001.parquet",
                    "full/prompt-00000-of-00001.parquet",
                ),
                license_spdx="CC-BY-4.0",
                card_url="https://huggingface.co/datasets/google-research-datasets/mbpp",
                redistribution="yes, attribution",
                category_coverage=("code",),
                parser=parse_mbpp,
            ),
            Source(
                name="gsm8k",
                repo_id="openai/gsm8k",
                revision="740312add88f781978c0658806c59bc2815b9866",
                files=("main/train-00000-of-00001.parquet", "main/test-00000-of-00001.parquet"),
                license_spdx="MIT",
                card_url="https://huggingface.co/datasets/openai/gsm8k",
                redistribution="yes, copyright notice",
                category_coverage=("reason",),
                parser=parse_gsm8k,
            ),
        )
    }
)
"""All registered sources, keyed by short name."""


# --------------------------------------------------------------------------------------------------
# Files, manifest, download
# --------------------------------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    """Return the hex sha256 digest of the file at ``path``."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _get_source(name: str) -> Source:
    try:
        return SOURCES[name]
    except KeyError:
        raise ValueError(f"unknown source {name!r}; expected one of {', '.join(SOURCES)}") from None


def download_source(name: str, raw_dir: Path = DEFAULT_RAW_DIR) -> Path:
    """Download a source's pinned files into ``raw_dir/<name>/`` and write its manifest.

    Files are fetched into the Hugging Face cache and copied, so ``raw_dir`` holds only the data
    files (at their repository-relative paths) and ``MANIFEST.json``.

    Args:
        name: Registered source name.
        raw_dir: Root of the raw data directory.

    Returns:
        Path of the written manifest.
    """
    from huggingface_hub import hf_hub_download  # imported lazily: network-only path

    source = _get_source(name)
    dest = raw_dir / source.name
    entries = []
    for filename in source.files:
        cached = Path(
            hf_hub_download(source.repo_id, filename, repo_type="dataset", revision=source.revision)
        )
        target = dest / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached, target)
        entries.append(
            {"path": filename, "sha256": sha256_file(target), "bytes": target.stat().st_size}
        )
    manifest = {
        "source": source.name,
        "repo_id": source.repo_id,
        "revision": source.revision,
        "license": source.license_spdx,
        "retrieved": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "files": entries,
    }
    path = dest / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    return path


def verify_manifest(source_dir: Path) -> list[str]:
    """Check every file listed in ``source_dir/MANIFEST.json`` against its size and sha256.

    Args:
        source_dir: A directory such as ``data/raw/dolly``.

    Returns:
        A list of problems (missing manifest, missing file, size or hash mismatch). Empty if valid.
    """
    manifest_path = source_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        return [f"missing {manifest_path}"]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    problems = []
    for entry in manifest.get("files", []):
        path = source_dir / entry["path"]
        if not path.is_file():
            problems.append(f"missing file {path}")
        elif path.stat().st_size != entry["bytes"]:
            problems.append(f"size mismatch {path}")
        elif sha256_file(path) != entry["sha256"]:
            problems.append(f"sha256 mismatch {path}")
    return problems


def read_rows(source: Source, raw_dir: Path = DEFAULT_RAW_DIR) -> Iterator[dict[str, Any]]:
    """Yield raw rows from a downloaded source's files, in registry file order.

    ``.jsonl`` and ``.jsonl.gz`` are read line by line; ``.parquet`` is read with pandas (pyarrow
    engine). Every row gains ``split`` (from the file name) and ``index`` (row number in that file).
    """
    for filename in source.files:
        path = raw_dir / source.name / filename
        split = Path(filename).name.split("-")[0].split(".")[0]
        if filename.endswith(".parquet"):
            import pandas as pd

            frame = pd.read_parquet(path)
            for i, rec in enumerate(frame.to_dict(orient="records")):
                yield {**{str(k): v for k, v in rec.items()}, "split": split, "index": i}
        else:
            opener = gzip.open if filename.endswith(".gz") else open
            with opener(path, "rt", encoding="utf-8") as fh:
                for i, line in enumerate(fh):
                    if line.strip():
                        yield {**json.loads(line), "split": split, "index": i}


def iter_records(name: str, raw_dir: Path = DEFAULT_RAW_DIR) -> Iterator[NormalisedRecord]:
    """Yield normalised records for a downloaded source."""
    source = _get_source(name)
    return source.parser(read_rows(source, raw_dir))


# --------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``python -m terser.data.sources``."""
    parser = argparse.ArgumentParser(prog="python -m terser.data.sources", description=__doc__)
    parser.add_argument("command", choices=("download", "verify", "stats"))
    parser.add_argument(
        "--source",
        action="append",
        choices=list(SOURCES),
        help="Source to act on (repeatable). Default: all.",
    )
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    args = parser.parse_args(argv)
    names: list[str] = args.source or list(SOURCES)
    status = 0
    for name in names:
        if args.command == "download":
            print(f"{name}: wrote {download_source(name, args.raw_dir)}")
        elif args.command == "verify":
            problems = verify_manifest(args.raw_dir / name)
            print(f"{name}: {'OK' if not problems else '; '.join(problems)}")
            status |= bool(problems)
        else:
            counts = Counter(r["category_hint"] for r in iter_records(name, args.raw_dir))
            total = sum(counts.values())
            detail = ", ".join(f"{k}={v}" for k, v in sorted(counts.items(), key=str))
            print(f"{name}: {total} records ({detail})")
    return status


if __name__ == "__main__":
    sys.exit(main())
