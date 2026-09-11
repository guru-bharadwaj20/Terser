"""Near-duplicate removal with MinHash LSH, keeping paraphrase groups whole.

Plan row **2.1.5** (checkpoint C2.1 requires < 1% near-duplicates after deduplication).

Pipeline
--------
1. **Normalisation** (:func:`normalise`): Unicode NFKC, lowercase, every character that is not a
   letter, digit, underscore or whitespace is replaced by a space (i.e. punctuation is stripped
   and does not glue words together), then runs of whitespace collapse to one space and the ends
   are trimmed.
2. **Shingling** (:func:`shingles`): the set of contiguous word 5-grams of the normalised text.
   A text with fewer than 5 words yields a single shingle (the whole text), so short prompts are
   only duplicates of texts that normalise to the same words. An empty text has no shingles and
   is never a duplicate of anything.
3. **MinHash** (:class:`MinHasher`): each shingle is hashed to 32 bits with BLAKE2b, then passed
   through ``num_perm`` seeded universal hash functions
   ``((a * x + b) mod (2**61 - 1)) & 0xFFFFFFFF`` (uint64 arithmetic); the signature is the
   per-function minimum. The fraction of equal signature slots estimates the Jaccard similarity
   of the shingle sets. Implemented here in NumPy (no ``datasketch`` dependency) so it is
   deterministic given ``seed``.
4. **LSH banding** (:func:`optimal_bands`): the signature is cut into ``b`` bands of ``r`` slots;
   two records are *candidates* when any band matches exactly. ``(b, r)`` is chosen to minimise a
   weighted false-negative / false-positive area around the threshold (recall-heavy weights),
   because every candidate is then **verified by exact Jaccard** on the shingle sets. Estimated
   similarity is therefore only used to find candidates, never to decide a duplicate.
5. **Clustering**: union-find over *groups*. A group is all records sharing a ``source_id`` (the
   casual variants of one clean prompt, row 2.1.4); records without ``source_id`` form a group of
   their own keyed by their id. Records in the same group are never compared with each other.
   Two groups join a cluster when some record of one has exact Jaccard >= ``threshold`` with some
   record of the other; clusters are the connected components.
6. **Selection**: per cluster one whole group is kept: the one whose best (lowest) source
   priority is earliest, ties broken by the lexicographically lowest group id. All other groups
   in the cluster are removed *entirely*, so paraphrase groups are never split.

Record fields are read leniently so both the pilot schema (``id``, ``prompt``, ``origin``,
``category``) and the normalised source schema from ``terser.data.sources`` (``source``,
``source_id``, ``instruction``, ``input``, ``category_hint``) work; see :func:`record_text`,
:func:`record_id`, :func:`group_id` and :func:`record_source`.

CLI::

    python -m terser.data.dedupe --in data.jsonl --out kept.jsonl --report report.md \
        [--threshold 0.8] [--source-priority dolly,oasst] [--seed 0]

``--report`` is written as Markdown when the path ends in ``.md``, otherwise as JSON.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = [
    "DedupeResult",
    "DuplicateCluster",
    "DuplicateReport",
    "MinHasher",
    "dedupe",
    "exact_jaccard",
    "group_id",
    "main",
    "near_duplicate_pairs",
    "normalise",
    "optimal_bands",
    "read_jsonl",
    "record_id",
    "record_source",
    "record_text",
    "render_markdown",
    "shingles",
    "write_jsonl",
]

Record = Mapping[str, Any]

SHINGLE_SIZE = 5
"""Word n-gram size used for shingles."""

DEFAULT_THRESHOLD = 0.8
"""Exact Jaccard at or above which two records are near-duplicates."""

DEFAULT_NUM_PERM = 128
"""Number of MinHash permutations."""

_MERSENNE_61 = np.uint64((1 << 61) - 1)
_MASK_32 = np.uint64(0xFFFFFFFF)
_NON_WORD = re.compile(r"[^\w\s]+", flags=re.UNICODE)
_WHITESPACE = re.compile(r"\s+", flags=re.UNICODE)


# --------------------------------------------------------------------------------------------
# Record accessors
# --------------------------------------------------------------------------------------------


def record_text(record: Record) -> str:
    """Return the text used for duplicate detection.

    ``prompt`` when present (casual / pilot records), else ``instruction`` followed by ``input``
    (normalised source records), else ``text``; empty string if none exists.
    """
    prompt = record.get("prompt")
    if isinstance(prompt, str) and prompt:
        return prompt
    parts = [record.get("instruction"), record.get("input")]
    joined = "\n".join(p for p in parts if isinstance(p, str) and p)
    if joined:
        return joined
    text = record.get("text")
    return text if isinstance(text, str) else ""


def record_id(record: Record) -> str:
    """Return a record's unique id: ``id`` if present, else ``source_id``.

    Raises:
        KeyError: If the record has neither field.
    """
    for key in ("id", "source_id"):
        value = record.get(key)
        if value is not None and value != "":
            return str(value)
    raise KeyError("record has neither 'id' nor 'source_id'")


def group_id(record: Record) -> str:
    """Return the paraphrase-group key: ``source_id`` if present, else the record id.

    ``source_id`` values are assumed globally unique across sources (e.g. ``"dolly:123"``).
    """
    value = record.get("source_id")
    if value is not None and value != "":
        return str(value)
    return record_id(record)


def record_source(record: Record) -> str:
    """Return the record's source name: ``source``, else ``origin``, else ``""``."""
    for key in ("source", "origin"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


# --------------------------------------------------------------------------------------------
# Text -> shingles -> MinHash
# --------------------------------------------------------------------------------------------


def normalise(text: str) -> str:
    """NFKC, lowercase, punctuation -> space, collapse whitespace, strip (see module docstring)."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = _NON_WORD.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def shingles(text: str, size: int = SHINGLE_SIZE) -> frozenset[str]:
    """Return the set of word ``size``-grams of the normalised text.

    Texts shorter than ``size`` words give one shingle (the whole normalised text); empty
    texts give the empty set.
    """
    words = normalise(text).split(" ")
    if words == [""]:
        return frozenset()
    if len(words) < size:
        return frozenset({" ".join(words)})
    return frozenset(" ".join(words[i : i + size]) for i in range(len(words) - size + 1))


def exact_jaccard(a: frozenset[str] | set[str], b: frozenset[str] | set[str]) -> float:
    """Exact Jaccard similarity ``|a & b| / |a | b|``; 0.0 when both sets are empty."""
    if not a and not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def _hash32(token: str) -> int:
    """Stable 32-bit hash of a string (BLAKE2b, independent of ``PYTHONHASHSEED``)."""
    return int.from_bytes(hashlib.blake2b(token.encode("utf-8"), digest_size=4).digest(), "little")


class MinHasher:
    """Seeded MinHash signature generator.

    Args:
        num_perm: Number of hash functions (signature length).
        seed: Seed for the hash-function coefficients.
    """

    def __init__(self, num_perm: int = DEFAULT_NUM_PERM, seed: int = 0) -> None:
        if num_perm < 2:
            raise ValueError("num_perm must be >= 2")
        self.num_perm = num_perm
        self.seed = seed
        gen = np.random.default_rng(seed)
        self._a = gen.integers(1, int(_MERSENNE_61), size=num_perm, dtype=np.uint64)
        self._b = gen.integers(0, int(_MERSENNE_61), size=num_perm, dtype=np.uint64)

    def signature(self, shingle_set: Iterable[str]) -> npt.NDArray[np.uint64]:
        """Return the MinHash signature (``num_perm`` uint64 values) of a shingle set.

        An empty set yields an all-``2**32`` signature that matches nothing real.
        """
        hashes = np.fromiter((_hash32(s) for s in shingle_set), dtype=np.uint64)
        if hashes.size == 0:
            return np.full(self.num_perm, 1 << 32, dtype=np.uint64)
        with np.errstate(over="ignore"):
            permuted = (self._a[None, :] * hashes[:, None] + self._b[None, :]) % _MERSENNE_61
        result: npt.NDArray[np.uint64] = (permuted & _MASK_32).min(axis=0)
        return result

    @staticmethod
    def estimate(sig_a: npt.NDArray[np.uint64], sig_b: npt.NDArray[np.uint64]) -> float:
        """Estimated Jaccard: the fraction of equal signature slots."""
        return float(np.mean(sig_a == sig_b))


def _trapezoid(ys: npt.NDArray[np.float64], xs: npt.NDArray[np.float64]) -> float:
    return float(np.sum((ys[1:] + ys[:-1]) * np.diff(xs)) / 2.0)


@functools.lru_cache(maxsize=32)
def optimal_bands(
    threshold: float,
    num_perm: int,
    fp_weight: float = 0.05,
    fn_weight: float = 0.95,
) -> tuple[int, int]:
    """Choose LSH ``(bands, rows_per_band)`` minimising weighted FP/FN probability areas.

    ``P(candidate | s) = 1 - (1 - s**rows) ** bands``. False positives are cheap (exact Jaccard
    verifies every candidate), so false negatives are weighted more heavily by default.
    """
    lo_xs = np.linspace(0.0, threshold, 201)
    hi_xs = np.linspace(threshold, 1.0, 201)
    best: tuple[float, int, int] | None = None
    for bands in range(1, num_perm + 1):
        for rows in range(1, num_perm // bands + 1):
            fp = _trapezoid(1.0 - (1.0 - lo_xs**rows) ** bands, lo_xs)
            fn = _trapezoid((1.0 - hi_xs**rows) ** bands, hi_xs)
            err = fp_weight * fp + fn_weight * fn
            if best is None or err < best[0] - 1e-12:
                best = (err, bands, rows)
    assert best is not None
    return best[1], best[2]


# --------------------------------------------------------------------------------------------
# Near-duplicate search
# --------------------------------------------------------------------------------------------


def near_duplicate_pairs(
    texts: Sequence[str],
    groups: Sequence[str] | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    num_perm: int = DEFAULT_NUM_PERM,
    seed: int = 0,
) -> list[tuple[int, int, float]]:
    """Find index pairs ``(i, j, jaccard)`` with ``i < j`` and exact Jaccard >= ``threshold``.

    Candidates come from MinHash LSH; each is verified by exact Jaccard. Pairs whose ``groups``
    entries are equal are skipped. The result is sorted.
    """
    if groups is not None and len(groups) != len(texts):
        raise ValueError("groups must have the same length as texts")
    shingle_sets = [shingles(t) for t in texts]
    hasher = MinHasher(num_perm=num_perm, seed=seed)
    bands, rows = optimal_bands(threshold, num_perm)
    buckets: dict[tuple[int, bytes], list[int]] = defaultdict(list)
    for idx, sset in enumerate(shingle_sets):
        if not sset:
            continue
        sig = hasher.signature(sset)
        for band in range(bands):
            buckets[(band, sig[band * rows : (band + 1) * rows].tobytes())].append(idx)

    candidates: set[tuple[int, int]] = set()
    for members in buckets.values():
        if len(members) < 2:
            continue
        for x in range(len(members)):
            for y in range(x + 1, len(members)):
                i, j = members[x], members[y]
                if groups is not None and groups[i] == groups[j]:
                    continue
                candidates.add((i, j) if i < j else (j, i))

    pairs: list[tuple[int, int, float]] = []
    for i, j in sorted(candidates):
        sim = exact_jaccard(shingle_sets[i], shingle_sets[j])
        if sim >= threshold:
            pairs.append((i, j, sim))
    return pairs


class _UnionFind:
    def __init__(self, items: Iterable[str]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:
            self.parent[item], item = root, self.parent[item]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            lo, hi = sorted((ra, rb))
            self.parent[hi] = lo


# --------------------------------------------------------------------------------------------
# Deduplication
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class DuplicateCluster:
    """A set of >= 2 groups that are near-duplicates of each other."""

    kept_group: str
    removed_groups: list[str]
    group_sizes: dict[str, int]
    """Number of records per group in the cluster."""
    max_jaccard: float
    """Highest verified exact Jaccard between records of different groups in the cluster."""

    @property
    def size(self) -> int:
        """Number of groups in the cluster."""
        return 1 + len(self.removed_groups)


@dataclass
class DuplicateReport:
    """Summary of one deduplication run."""

    threshold: float
    num_perm: int
    bands: int
    rows_per_band: int
    n_records_in: int
    n_groups_in: int
    n_records_out: int
    n_groups_out: int
    near_duplicate_pairs: int
    """Verified cross-group record pairs at or above the threshold (before removal)."""
    clusters: list[DuplicateCluster] = field(default_factory=list)
    removed_group_ids: list[str] = field(default_factory=list)
    removed_record_ids: list[str] = field(default_factory=list)
    near_duplicate_rate_before: float = 0.0
    """Fraction of input records that were redundant (i.e. were removed)."""
    near_duplicate_rate_after: float = 0.0
    """Fraction of kept records still having a near-duplicate in another kept group."""

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation."""
        data = asdict(self)
        for cluster, raw in zip(self.clusters, data["clusters"], strict=True):
            raw["size"] = cluster.size
        return data


@dataclass
class DedupeResult:
    """Kept records (input order preserved) plus the :class:`DuplicateReport`."""

    kept: list[dict[str, Any]]
    report: DuplicateReport


def _source_rank(source: str, priority: Sequence[str]) -> tuple[int, str]:
    try:
        return (list(priority).index(source), source)
    except ValueError:
        return (len(priority), source)


def dedupe(
    records: Sequence[Record],
    threshold: float = DEFAULT_THRESHOLD,
    source_priority: Sequence[str] = (),
    num_perm: int = DEFAULT_NUM_PERM,
    seed: int = 0,
) -> DedupeResult:
    """Remove near-duplicate groups, keeping one whole group per duplicate cluster.

    Args:
        records: Input records (see module docstring for accepted fields).
        threshold: Exact Jaccard at or above which records are near-duplicates.
        source_priority: Source names from most to least preferred; unlisted sources rank after
            all listed ones, alphabetically.
        num_perm: MinHash signature length.
        seed: Seed for the MinHash hash functions.

    Returns:
        :class:`DedupeResult` with kept records in input order and the report.

    Raises:
        ValueError: On an out-of-range threshold or duplicate record ids.
    """
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
    ids = [record_id(r) for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("record ids must be unique")
    gids = [group_id(r) for r in records]
    pairs = near_duplicate_pairs(
        [record_text(r) for r in records], gids, threshold=threshold, num_perm=num_perm, seed=seed
    )

    uf = _UnionFind(sorted(set(gids)))
    best_sim: dict[tuple[str, str], float] = {}
    for i, j, sim in pairs:
        uf.union(gids[i], gids[j])
        a, b = sorted((gids[i], gids[j]))
        best_sim[(a, b)] = max(best_sim.get((a, b), 0.0), sim)

    members: dict[str, list[str]] = defaultdict(list)
    for g in sorted(set(gids)):
        members[uf.find(g)].append(g)
    group_size: dict[str, int] = defaultdict(int)
    group_rank: dict[str, tuple[int, str]] = {}
    for rec, g in zip(records, gids, strict=True):
        group_size[g] += 1
        rank = _source_rank(record_source(rec), source_priority)
        group_rank[g] = min(group_rank.get(g, rank), rank)

    clusters: list[DuplicateCluster] = []
    removed_groups: set[str] = set()
    for root in sorted(members):
        groups_in = members[root]
        if len(groups_in) < 2:
            continue
        ordered = sorted(groups_in, key=lambda g: (group_rank[g], g))
        kept, removed = ordered[0], ordered[1:]
        removed_groups.update(removed)
        in_cluster = set(groups_in)
        max_sim = max(
            (s for (a, b), s in best_sim.items() if a in in_cluster and b in in_cluster),
            default=0.0,
        )
        clusters.append(
            DuplicateCluster(
                kept_group=kept,
                removed_groups=sorted(removed),
                group_sizes={g: group_size[g] for g in sorted(groups_in)},
                max_jaccard=round(max_sim, 6),
            )
        )

    kept_records = [dict(r) for r, g in zip(records, gids, strict=True) if g not in removed_groups]
    removed_ids = sorted(i for i, g in zip(ids, gids, strict=True) if g in removed_groups)
    bands, rows = optimal_bands(threshold, num_perm)

    after_pairs = near_duplicate_pairs(
        [record_text(r) for r in kept_records],
        [group_id(r) for r in kept_records],
        threshold=threshold,
        num_perm=num_perm,
        seed=seed,
    )
    residual = {i for p in after_pairs for i in p[:2]}
    n_in = len(records)
    report = DuplicateReport(
        threshold=threshold,
        num_perm=num_perm,
        bands=bands,
        rows_per_band=rows,
        n_records_in=n_in,
        n_groups_in=len(set(gids)),
        n_records_out=len(kept_records),
        n_groups_out=len(set(gids)) - len(removed_groups),
        near_duplicate_pairs=len(pairs),
        clusters=clusters,
        removed_group_ids=sorted(removed_groups),
        removed_record_ids=removed_ids,
        near_duplicate_rate_before=len(removed_ids) / n_in if n_in else 0.0,
        near_duplicate_rate_after=len(residual) / len(kept_records) if kept_records else 0.0,
    )
    return DedupeResult(kept=kept_records, report=report)


# --------------------------------------------------------------------------------------------
# I/O and CLI
# --------------------------------------------------------------------------------------------


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSON-lines file, skipping blank lines."""
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> None:
    """Write records as UTF-8 JSON lines with LF endings, creating parent directories."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def render_markdown(report: DuplicateReport, input_name: str = "") -> str:
    """Render a human-readable Markdown duplicate report."""
    lines = [
        "# Duplicate report" + (f": `{input_name}`" if input_name else ""),
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Threshold (exact Jaccard, word {SHINGLE_SIZE}-grams) | {report.threshold} |",
        f"| MinHash permutations / LSH bands x rows | {report.num_perm} / "
        f"{report.bands} x {report.rows_per_band} |",
        f"| Records in / out | {report.n_records_in} / {report.n_records_out} |",
        f"| Groups in / out | {report.n_groups_in} / {report.n_groups_out} |",
        f"| Verified near-duplicate record pairs | {report.near_duplicate_pairs} |",
        f"| Duplicate clusters | {len(report.clusters)} |",
        f"| Near-duplicate rate before (removed / in) | {report.near_duplicate_rate_before:.2%} |",
        f"| Near-duplicate rate after | {report.near_duplicate_rate_after:.2%} |",
        "",
    ]
    if report.clusters:
        lines += [
            "## Clusters",
            "",
            "| Kept group | Removed groups | Group sizes | Max Jaccard |",
            "| --- | --- | --- | --- |",
        ]
        for c in report.clusters:
            sizes = ", ".join(f"{g}: {n}" for g, n in c.group_sizes.items())
            lines.append(
                f"| {c.kept_group} | {', '.join(c.removed_groups)} | {sizes} | "
                f"{c.max_jaccard:.3f} |"
            )
        lines.append("")
    else:
        lines += ["No near-duplicate clusters found.", ""]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--in", dest="inp", required=True, type=Path, help="input JSONL")
    parser.add_argument("--out", required=True, type=Path, help="output JSONL of kept records")
    parser.add_argument("--report", required=True, type=Path, help="report path (.md or .json)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--num-perm", type=int, default=DEFAULT_NUM_PERM)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--source-priority", default="", help="comma-separated sources, most preferred first"
    )
    args = parser.parse_args(argv)

    records = read_jsonl(args.inp)
    priority = [s.strip() for s in args.source_priority.split(",") if s.strip()]
    result = dedupe(
        records,
        threshold=args.threshold,
        source_priority=priority,
        num_perm=args.num_perm,
        seed=args.seed,
    )
    write_jsonl(args.out, result.kept)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    if args.report.suffix.lower() == ".md":
        text = render_markdown(result.report, args.inp.as_posix())
    else:
        text = json.dumps(result.report.to_dict(), indent=2, ensure_ascii=False) + "\n"
    args.report.write_text(text, encoding="utf-8", newline="\n")
    r = result.report
    print(
        f"{r.n_records_in} -> {r.n_records_out} records; {len(r.clusters)} clusters; "
        f"rate before {r.near_duplicate_rate_before:.2%}, after {r.near_duplicate_rate_after:.2%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
