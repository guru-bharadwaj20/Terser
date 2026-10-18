"""Real-prompt evaluation slice from WildChat-1M (plan row **2.1.8**).

Casual prompts in the main dataset are produced by an LLM from clean instructions (row 2.1.4), so
a rewriter could score well merely by undoing the verbosity that pipeline injected. This slice
guards against that: about 1,000 **first-turn prompts written by real users**, never casualised
and **never used for training**. It is used only by the headroom report (2.4.2) and the final
evaluation (12.3), and it is hash-locked like the test split.

Pipeline (deterministic for a fixed revision and seed):

1. **Download** one pinned shard of ``allenai/WildChat-1M`` (ODC-BY) into ``data/raw/wildchat/``
   with a ``MANIFEST.json`` (same format as :mod:`terser.data.sources`).
2. **Filter** first user turns: ``language == "English"``, not ``toxic``, not ``redacted`` (the
   authors' Presidio pass fired, so the text carries placeholders), 5-300 words, mostly ASCII,
   mostly letters, and not a pasted log or forum dump.
   All user metadata (hashed IP, country, state, headers) is dropped at parse time.
3. **Deduplicate** on whitespace- and case-normalised text.
4. **PII-scrub** with :func:`terser.data.pii.scrub_text` (strict typed redaction).
   This is on top of the authors' de-identification.
5. **Category-tag** with :func:`tag_category`, an ordered keyword rule set; prompts matching no
   rule are dropped. Tags are heuristic and marked ``category_source: "rules"``.
6. **Sample** a category-balanced slice (quota per category, shortfalls redistributed) in
   ``sha256("terser-real-slice:<seed>:<source_id>")`` order, write
   ``data/processed/real_slice/test.jsonl`` and freeze it with
   :func:`terser.data.splits.freeze_test`.

CLI::

    python -m terser.data.real_slice download
    python -m terser.data.real_slice build [--size 1000] [--seed 0] [--force]
    python -m terser.data.real_slice verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from terser.data.categories import Category
from terser.data.pii import scrub_text
from terser.data.sources import MANIFEST_NAME, sha256_file, verify_manifest
from terser.data.splits import LOCK_NAME, FrozenTestSetError, freeze_test, verify_frozen

__all__ = [
    "REPO_ID",
    "REVISION",
    "SHARDS",
    "SliceSummary",
    "build_slice",
    "download",
    "normalise_text",
    "parse_wildchat",
    "sample_balanced",
    "tag_category",
]

REPO_ID: Final[str] = "allenai/WildChat-1M"
REVISION: Final[str] = "7d6490e462285cf85d91eabea0f9a954fbddcd1f"
SHARDS: Final[tuple[str, ...]] = ("data/train-00000-of-00014.parquet",)
LICENSE_SPDX: Final[str] = "ODC-By-1.0"
SOURCE_NAME: Final[str] = "wildchat"
DEFAULT_RAW_DIR: Final[Path] = Path("data/raw")
DEFAULT_OUT_DIR: Final[Path] = Path("data/processed/real_slice")
MIN_WORDS: Final[int] = 5
MAX_WORDS: Final[int] = 300
MIN_ASCII_SHARE: Final[float] = 0.95
MIN_ALPHA_SHARE: Final[float] = 0.6
_DUMP_RE = re.compile(
    r"\d{2,4}[/-]\d{2}[/-]\d{2,4}[ T(]|\[(error|warn|info|debug)\]|traceback \(most recent|"
    r"^\s*anonymous \d",
    re.IGNORECASE | re.MULTILINE,
)
"""Log lines, timestamps and forum dumps: pasted material rather than a request."""


# --------------------------------------------------------------------------------------------------
# Download
# --------------------------------------------------------------------------------------------------


def download(raw_dir: Path = DEFAULT_RAW_DIR) -> Path:
    """Fetch the pinned shard(s) into ``raw_dir/wildchat/`` and write ``MANIFEST.json``."""
    from huggingface_hub import hf_hub_download  # network-only path

    dest = raw_dir / SOURCE_NAME
    entries = []
    for filename in SHARDS:
        cached = Path(hf_hub_download(REPO_ID, filename, repo_type="dataset", revision=REVISION))
        target = dest / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cached, target)
        entries.append(
            {"path": filename, "sha256": sha256_file(target), "bytes": target.stat().st_size}
        )
    manifest = {
        "source": SOURCE_NAME,
        "repo_id": REPO_ID,
        "revision": REVISION,
        "license": LICENSE_SPDX,
        "retrieved": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "files": entries,
    }
    path = dest / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    return path


def read_raw_rows(raw_dir: Path = DEFAULT_RAW_DIR) -> Iterator[dict[str, Any]]:
    """Yield raw conversation rows (only the columns the parser needs)."""
    import pandas as pd

    for filename in SHARDS:
        frame = pd.read_parquet(
            raw_dir / SOURCE_NAME / filename,
            columns=["conversation_hash", "conversation", "language", "toxic", "redacted"],
        )
        yield from frame.to_dict(orient="records")


# --------------------------------------------------------------------------------------------------
# Parse and filter
# --------------------------------------------------------------------------------------------------

_WS_RE = re.compile(r"\s+")


def normalise_text(text: str) -> str:
    """Lower-case, whitespace-collapsed text used as the de-duplication key."""
    return _WS_RE.sub(" ", text).strip().lower()


def _ascii_share(text: str) -> float:
    return sum(ch.isascii() for ch in text) / max(len(text), 1)


def parse_wildchat(rows: Iterable[Mapping[str, Any]]) -> Iterator[dict[str, Any]]:
    """First user turns that pass the language, safety, redaction and length filters.

    Yields ``{"source", "source_id", "instruction"}``; every user-metadata field is dropped.
    """
    for row in rows:
        if row.get("language") != "English" or row.get("toxic") or row.get("redacted"):
            continue
        conversation = row.get("conversation")
        if conversation is None or len(conversation) == 0:
            continue
        first = conversation[0]
        if first.get("role") != "user":
            continue
        text = str(first.get("content") or "").strip()
        n_words = len(text.split())
        if not MIN_WORDS <= n_words <= MAX_WORDS or _ascii_share(text) < MIN_ASCII_SHARE:
            continue
        alpha = sum(ch.isalpha() for ch in text) / len(text)
        if alpha < MIN_ALPHA_SHARE or _DUMP_RE.search(text):
            continue
        yield {
            "source": SOURCE_NAME,
            "source_id": f"{SOURCE_NAME}:{row['conversation_hash']}",
            "instruction": text,
        }


# --------------------------------------------------------------------------------------------------
# Category rules
# --------------------------------------------------------------------------------------------------


def _rx(*patterns: str) -> re.Pattern[str]:
    return re.compile("|".join(patterns), re.IGNORECASE)


_LANGS = (
    r"python|javascript|typescript|java|c\+\+|c#|golang|rust|sql|html|css|bash|powershell|kotlin|"
    r"swift|php|matlab|vba|react|django|flask|pandas|numpy|unity|arduino"
)

CATEGORY_RULES: Final[tuple[tuple[Category, re.Pattern[str]], ...]] = (
    (
        Category.CODE,
        _rx(
            r"```",
            rf"\b(in|using|with) ({_LANGS})\b",
            rf"\b({_LANGS}) (code|script|function|program|class|query|error)\b",
            r"\b(write|fix|debug|optimi[sz]e|refactor) (a |an |the |this |my )?"
            r"(code|program|function|class|query|regex)\b",
            r"\b(syntax error|stack ?trace|traceback|compile error|null pointer)\b",
            r"\bdef \w+\(|\bfunction \w+\(|#include|\bpublic static\b|\bSELECT .+ FROM\b",
        ),
    ),
    (
        Category.TRANSFORM,
        _rx(
            r"\b(rewrite|re-write|paraphrase|rephrase|reword|summari[sz]e|summary of|translate|"
            r"proofread|shorten|condense|simplify)\b",
            r"\b(correct|fix|improve|check) (the |my |this )?(grammar|spelling|wording|text)\b",
            r"\bmake (it|this) (more|less|sound)\b",
            r"\bconvert (this|the following|it|below)\b",
        ),
    ),
    (
        Category.CREATIVE,
        _rx(
            r"\b(write|compose|create|generate) (me )?(a |an )?(short |funny |long )?(story|poem|"
            r"song|lyrics|haiku|limerick|script|screenplay|dialogue|scene|fanfic|novel|chapter|joke|"
            r"slogan|tagline|rap|essay|article|blog post|speech|letter)\b",
            r"\b(imagine|fictional|roleplay|role-play|once upon)\b",
        ),
    ),
    (
        Category.REASON,
        _rx(
            r"\b(solve|calculate|compute|probability|equation|how many|how much|find the|"
            r"what is the (sum|product|value|area|average))\b(?=.*\d)",
            r"^(?=.*\d).*\b(solve|calculate|compute|probability|how many|how much)\b",
            r"\b(riddle|logic puzzle)\b",
        ),
    ),
    (
        Category.FACT,
        _rx(
            r"^\s*(who|when|where|which) (is|was|are|were|did|does)\b",
            r"^\s*how (many|much|old|tall|long|far)\b",
            r"\bwhat year\b",
            r"\bcapital of\b",
            r"^\s*what (is|was) the (name|date|population|capital|height|length|size|birthday|"
            r"author|price)\b",
        ),
    ),
    (
        Category.EXPLAIN,
        _rx(
            r"\b(explain|describe|elaborate)\b",
            r"^\s*(why|how) (do|does|did|is|are|can|would|should)\b",
            r"\bwhat(?:'s| is) the difference\b",
            r"^\s*what (is|are) ",
            r"\bhow does\b",
        ),
    ),
)
"""Ordered rules: the first matching category wins (code before transform, and so on)."""


AGENT_PROMPT_RE: Final[re.Pattern[str]] = _rx(
    r"^\s*CONSTRAINTS:",
    r"ignore (all )?(previous|prior) instructions",
    r"\b(i want you to act as|you are going to pretend|jailbreak|DAN mode)\b",
)
"""Agent scaffolds, persona and jailbreak set-ups: real traffic, but not a single user request."""


def tag_category(text: str) -> Category | None:
    """Heuristic task category for a real prompt, or ``None`` when no rule matches.

    Agent/persona/jailbreak set-ups (:data:`AGENT_PROMPT_RE`) are never tagged.
    """
    if AGENT_PROMPT_RE.search(text):
        return None
    for category, pattern in CATEGORY_RULES:
        if pattern.search(text):
            return category
    return None


# --------------------------------------------------------------------------------------------------
# Sample and build
# --------------------------------------------------------------------------------------------------


def _order_key(seed: int, record_id: str) -> str:
    return hashlib.sha256(f"terser-real-slice:{seed}:{record_id}".encode()).hexdigest()


def sample_balanced(
    records: Sequence[Mapping[str, Any]], size: int, seed: int = 0
) -> list[dict[str, Any]]:
    """Category-balanced sample of ``size`` records (fewer if the pool is smaller).

    Each category gets ``size // K`` slots (remainder to the first categories in
    :class:`Category` order); slots a category cannot fill go to categories with spare records.
    Within a category, records are taken in
    ``sha256("terser-real-slice:<seed>:<source_id>")`` order.
    """
    pools: dict[str, list[Mapping[str, Any]]] = {c.value: [] for c in Category}
    for rec in records:
        pools[str(rec["category"])].append(rec)
    for pool in pools.values():
        pool.sort(key=lambda r: _order_key(seed, str(r["source_id"])))
    cats = [c for c in pools if pools[c]]
    quota = dict.fromkeys(pools, 0)
    remaining = min(size, sum(len(p) for p in pools.values()))
    while remaining > 0:
        open_cats = [c for c in cats if quota[c] < len(pools[c])]
        share, extra = divmod(remaining, len(open_cats))
        for i, c in enumerate(open_cats):
            add = min(share + (1 if i < extra else 0), len(pools[c]) - quota[c])
            quota[c] += add
            remaining -= add
    picked = [dict(r) for c in pools for r in pools[c][: quota[c]]]
    return sorted(picked, key=lambda r: str(r["id"]))


@dataclass
class SliceSummary:
    """Counts at every stage of :func:`build_slice`."""

    parsed: int = 0
    duplicates: int = 0
    pii_scrubbed: int = 0
    untagged: int = 0
    pool_by_category: dict[str, int] = field(default_factory=dict)
    slice_by_category: dict[str, int] = field(default_factory=dict)
    out_path: str = ""
    lock_path: str = ""

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def build_records(rows: Iterable[Mapping[str, Any]], summary: SliceSummary) -> list[dict[str, Any]]:
    """Parse, deduplicate, PII-scrub and tag; returns the tagged pool."""
    seen: set[str] = set()
    pool: list[dict[str, Any]] = []
    for rec in parse_wildchat(rows):
        summary.parsed += 1
        key = normalise_text(rec["instruction"])
        if key in seen:
            summary.duplicates += 1
            continue
        seen.add(key)
        result = scrub_text(rec["instruction"])
        assert result.scrubbed is not None  # scrub_text redacts; it never drops a record
        if result.scrubbed != rec["instruction"]:
            summary.pii_scrubbed += 1
        category = tag_category(result.scrubbed)
        if category is None:
            summary.untagged += 1
            continue
        pool.append(
            {
                "id": f"real-{rec['source_id'].split(':', 1)[1][:16]}",
                "source": rec["source"],
                "source_id": rec["source_id"],
                "category": category.value,
                "category_source": "rules",
                "prompt": result.scrubbed,
                "pii_scrubbed": result.scrubbed != rec["instruction"],
                "casualised": False,
                "use": "eval_only",
            }
        )
    summary.pool_by_category = dict(sorted(Counter(r["category"] for r in pool).items()))
    return pool


def build_slice(
    raw_dir: Path = DEFAULT_RAW_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    size: int = 1000,
    seed: int = 0,
    force: bool = False,
    rows: Iterable[Mapping[str, Any]] | None = None,
) -> SliceSummary:
    """Build, write and freeze the slice. ``rows`` overrides reading the raw shard (tests)."""
    if rows is None:
        problems = verify_manifest(raw_dir / SOURCE_NAME)
        if problems:
            raise RuntimeError("raw WildChat shard failed verification: " + "; ".join(problems))
        rows = read_raw_rows(raw_dir)
    summary = SliceSummary()
    picked = sample_balanced(build_records(rows, summary), size, seed)
    summary.slice_by_category = dict(sorted(Counter(r["category"] for r in picked).items()))
    # Refuse before writing: the frozen file must survive a rebuild attempt without --force.
    if (out_dir / LOCK_NAME).exists() and not force:
        raise FrozenTestSetError(f"{out_dir / LOCK_NAME} exists; refusing to overwrite the slice")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "test.jsonl"
    with out_path.open("w", encoding="utf-8", newline="\n") as fh:
        for rec in picked:
            fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
    summary.out_path = str(out_path)
    summary.lock_path = str(freeze_test(out_dir, force=force))
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``python -m terser.data.real_slice``."""
    parser = argparse.ArgumentParser(prog="python -m terser.data.real_slice", description=__doc__)
    parser.add_argument("command", choices=("download", "build", "verify"))
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="overwrite an existing lock")
    args = parser.parse_args(argv)
    if args.command == "download":
        print(download(args.raw_dir))
    elif args.command == "build":
        summary = build_slice(args.raw_dir, args.out_dir, args.size, args.seed, args.force)
        print(json.dumps(summary.as_dict(), indent=2))
    else:
        lock = verify_frozen(args.out_dir)
        print(f"OK: {lock['n_records']} records, sha256 {lock['sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
