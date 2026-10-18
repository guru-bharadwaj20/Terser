"""Token counts of every prompt and candidate on every target tokenizer (plan row **2.2.2**).

The table has one row per ``(prompt_id, candidate_id, target)`` with an integer ``tokens``.
Original prompts appear with ``candidate_id = "<prompt_id>:original"``. Counts match
:func:`terser.tokenizers.count_tokens` exactly (no special tokens, no chat framing) but are
computed in batches: ``tiktoken``'s ``encode_batch`` and Hugging Face ``tokenizers``'
``encode_batch``. Identical texts are counted once, and counts are cached on disk per target
(``.cache/token_counts/batch/<target>.json``, keyed by the sha256 of the text), so re-runs only
tokenize new texts. ``claude`` is included only when :func:`available_targets` lists it.

:func:`check_complete` fails loudly unless every expected (text, target) cell has a count.

CLI::

    python -m terser.data.token_counts \\
        --prompts data/pilot/prompts.jsonl \\
        --candidates data/interim/pilot_candidates.jsonl \\
        --out data/interim/pilot_token_counts.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from terser.llm.budget import atomic_write_json

__all__ = [
    "COLUMNS",
    "ORIGINAL_SUFFIX",
    "BatchCounter",
    "MissingTokenCountsError",
    "build_table",
    "check_complete",
    "default_targets",
    "load_texts",
    "main",
    "make_batch_counter",
    "write_table",
]

COLUMNS: tuple[str, ...] = ("prompt_id", "candidate_id", "target", "tokens")
ORIGINAL_SUFFIX = ":original"

BatchCounter = Callable[[Sequence[str]], list[int]]
"""Counts a batch of texts on one target, returning one count per text in order."""


class MissingTokenCountsError(RuntimeError):
    """The counts table lacks some (text, target) cells."""


def default_targets() -> list[str]:
    """The local target tokenizers (no ``claude``).

    ``claude`` counts need a network call and an API key, so a default that included it whenever
    ``ANTHROPIC_API_KEY`` happened to be set made the DVC output machine-dependent. Ask for it
    explicitly with ``--targets ... claude``.
    """
    from terser.tokenizers import available_targets

    return [t for t in available_targets() if t != "claude"]


def make_batch_counter(target: str) -> BatchCounter:
    """Batched counter for ``target`` equivalent to :func:`terser.tokenizers.count_tokens`."""
    from terser.tokenizers import count as tc

    if target not in tc.TARGETS:
        raise KeyError(f"unknown target {target!r}; choose from {sorted(tc.TARGETS)}")
    spec = tc.TARGETS[target]
    if spec.kind == "tiktoken":
        import tiktoken

        enc = tiktoken.get_encoding(spec.ref)
        threads = max(1, min(8, os.cpu_count() or 1))

        def tiktoken_batch(texts: Sequence[str]) -> list[int]:
            ids = enc.encode_batch(list(texts), num_threads=threads, disallowed_special=())
            return [len(x) for x in ids]

        return tiktoken_batch
    if spec.kind == "hf":
        tok = tc.load_hf_tokenizer(spec.ref)

        def hf_batch(texts: Sequence[str]) -> list[int]:
            encs = tok.encode_batch(list(texts), add_special_tokens=False)
            return [len(e.ids) for e in encs]

        return hf_batch
    # Remote counters (claude) have no batch endpoint; count_tokens caches per text on disk.
    return lambda texts: [tc.count_tokens(t, target) for t in texts]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _ref_tag(target: str) -> str:
    """Short digest of the tokenizer behind ``target`` (the Claude model for ``claude``)."""
    from terser.tokenizers import count as tc

    spec = tc.TARGETS.get(target)
    if spec is None:
        ref = target
    elif spec.kind == "anthropic":
        ref = tc.claude_model()
    else:
        ref = spec.ref
    return _sha(ref)[:10]


class _CountCache:
    """Per-target JSON map ``sha256(text) -> tokens``.

    The file name carries the tokenizer reference (HF repo, tiktoken encoding, Claude model), so
    changing the model behind a target name starts a new cache instead of reusing stale counts.
    """

    def __init__(self, root: Path | None, target: str) -> None:
        self.path = None if root is None else root / f"{target}-{_ref_tag(target)}.json"
        self.data: dict[str, int] = {}
        if self.path is not None and self.path.is_file():
            self.data = {k: int(v) for k, v in json.loads(self.path.read_text("utf-8")).items()}
        self.dirty = False

    def save(self) -> None:
        if self.path is not None and self.dirty:
            atomic_write_json(self.path, self.data)
            self.dirty = False


def load_texts(prompts_path: Path, candidates_path: Path) -> pd.DataFrame:
    """One row per text to count: ``prompt_id, candidate_id, text`` (originals first).

    Raises:
        FileNotFoundError: If an input file is missing.
        ValueError: On a duplicate ``candidate_id`` or a candidate whose prompt is unknown.
    """
    for path in (prompts_path, candidates_path):
        if not path.is_file():
            raise FileNotFoundError(f"input not found: {path}")
    rows: list[dict[str, str]] = []
    prompt_ids: set[str] = set()
    with prompts_path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                pid = str(rec["id"])
                prompt_ids.add(pid)
                rows.append(
                    {"prompt_id": pid, "candidate_id": pid + ORIGINAL_SUFFIX, "text": rec["prompt"]}
                )
    with candidates_path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            rec = json.loads(line)
            if str(rec["prompt_id"]) not in prompt_ids:
                raise ValueError(
                    f"{candidates_path}:{lineno}: prompt_id {rec['prompt_id']!r} not in prompts"
                )
            rows.append(
                {
                    "prompt_id": str(rec["prompt_id"]),
                    "candidate_id": str(rec["candidate_id"]),
                    "text": str(rec["text"]),
                }
            )
    df = pd.DataFrame(rows, columns=["prompt_id", "candidate_id", "text"])
    dup = df["candidate_id"].duplicated()
    if dup.any():
        raise ValueError(f"duplicate candidate_id(s): {sorted(df.loc[dup, 'candidate_id'])[:5]}")
    return df


def build_table(
    texts: pd.DataFrame,
    targets: Iterable[str],
    *,
    counter_factory: Callable[[str], BatchCounter] = make_batch_counter,
    cache_dir: Path | None = None,
    batch_size: int = 1024,
) -> pd.DataFrame:
    """Count every text on every target; return a long table with :data:`COLUMNS`.

    Unique texts are counted once per target; cached counts are reused and new ones saved.
    """
    unique = list(dict.fromkeys(texts["text"].tolist()))
    keys = [_sha(t) for t in unique]
    frames = []
    for target in targets:
        cache = _CountCache(cache_dir, target)
        todo = [i for i, k in enumerate(keys) if k not in cache.data]
        if todo:
            counter = counter_factory(target)
            for start in range(0, len(todo), batch_size):
                idx = todo[start : start + batch_size]
                counts = counter([unique[i] for i in idx])
                if len(counts) != len(idx):
                    raise RuntimeError(f"{target}: counter returned {len(counts)} for {len(idx)}")
                for i, n in zip(idx, counts, strict=True):
                    cache.data[keys[i]] = int(n)
            cache.dirty = True
            cache.save()
        by_text = {t: cache.data[k] for t, k in zip(unique, keys, strict=True)}
        frame = texts[["prompt_id", "candidate_id"]].copy()
        frame["target"] = target
        frame["tokens"] = texts["text"].map(by_text).astype("int64")
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=list(COLUMNS))
    return pd.concat(frames, ignore_index=True)[list(COLUMNS)]


def check_complete(table: pd.DataFrame, texts: pd.DataFrame, targets: Sequence[str]) -> int:
    """Return 0 if every (candidate_id, target) has a non-negative integer count.

    Raises:
        MissingTokenCountsError: Listing how many cells are missing (and a few examples).
    """
    expected = pd.MultiIndex.from_product(
        [texts["candidate_id"].unique(), list(targets)], names=["candidate_id", "target"]
    )
    valid = table[pd.to_numeric(table["tokens"], errors="coerce").ge(0)]
    have = pd.MultiIndex.from_frame(valid[["candidate_id", "target"]].drop_duplicates())
    missing = expected.difference(have)
    if len(missing) or len(table) != len(expected):
        examples = [f"{c}@{t}" for c, t in list(missing)[:5]]
        raise MissingTokenCountsError(
            f"{len(missing)} missing token counts (table has {len(table)} rows, expected "
            f"{len(expected)}); e.g. {examples}"
        )
    return 0


def write_table(table: pd.DataFrame, out: Path) -> Path:
    """Write parquet if ``pyarrow`` is installed, else CSV (suffix adjusted); return the path."""
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        path = out.with_suffix(".csv")
        table.to_csv(path, index=False, lineterminator="\n")
        return path
    path = out.with_suffix(".parquet")
    table.to_parquet(path, index=False)
    return path


def main(argv: list[str] | None = None) -> int:
    """CLI: build, check and write the counts table; print a JSON summary."""
    from dotenv import load_dotenv

    load_dotenv()
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--prompts", type=Path, default=Path("data/pilot/prompts.jsonl"))
    ap.add_argument("--candidates", type=Path, default=Path("data/interim/pilot_candidates.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/interim/pilot_token_counts.parquet"))
    ap.add_argument("--targets", nargs="*", default=None, help="default: all available")
    ap.add_argument("--cache-dir", type=Path, default=Path(".cache/token_counts/batch"))
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args(argv)

    start = time.perf_counter()
    targets = args.targets or default_targets()
    texts = load_texts(args.prompts, args.candidates)
    table = build_table(texts, targets, cache_dir=None if args.no_cache else args.cache_dir)
    try:
        missing = check_complete(table, texts, targets)
    except MissingTokenCountsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    path = write_table(table, args.out)
    summary: dict[str, Any] = {
        "rows": len(table),
        "texts": len(texts),
        "unique_texts": int(texts["text"].nunique()),
        "targets": targets,
        "missing": missing,
        "seconds": round(time.perf_counter() - start, 3),
        "output": path.as_posix(),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
