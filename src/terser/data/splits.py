"""Source-grouped train / val / test splits, test-set freezing and leakage checks.

Plan row **2.1.6** (checkpoint C2.2: 0 source prompts shared across splits).

Grouping
--------
A *group* is every record sharing a ``source_id`` (a clean prompt and all its casual variants,
row 2.1.4); a record without ``source_id`` is its own group keyed by its id (see
:func:`terser.data.dedupe.group_id`). Whole groups are assigned to one split, so no paraphrase of
a test prompt can appear in train or val.

Stable hash-based assignment
----------------------------
Each group gets ``u = sha256(f"terser-split:{seed}:{group_id}")`` mapped to ``[0, 1)``; it goes
to ``train`` if ``u < train_frac``, ``val`` if ``u < train_frac + val_frac``, else ``test``. The
assignment of a group depends only on the seed and its own id, so:

* appending new groups later never moves an existing group (existing test prompts stay in test);
* the result is identical across machines and Python processes (no ``hash()``, no RNG state).

Trade-off: this stratifies by category (and by source) only *in expectation*. Because ``u`` is
independent of category and source, every (category, source) stratum receives each split with
the target probability, but the achieved proportions are binomial around the target rather than
exact. With ``n`` groups in a stratum the standard deviation of the test share is about
``sqrt(0.15 * 0.85 / n)`` (≈ 2.5 points at n = 200, ≈ 1.2 points at n = 900). An exact
rank-within-stratum split would hit the proportions exactly but reshuffle groups whenever data
is added, which would silently leak once-test prompts into training. :class:`SplitSummary`
reports the achieved proportions per category and per source so the deviation is visible.

Test freezing
-------------
:func:`freeze_test` writes ``test.lock.json`` next to ``test.jsonl`` holding the sorted test group
ids and a sha256 of the test file; :func:`verify_frozen` raises :class:`FrozenTestSetError` if
either changes. :func:`assert_no_test_access` is a guard for tuning code (row 3.x rule "never
tune on test"): it raises :class:`TestAccessError` for a path that is a frozen test split unless
``TERSER_ALLOW_TEST_ACCESS=1`` is set (final evaluation only).

CLI::

    python -m terser.data.splits --in data.jsonl --out-dir data/processed/splits --seed 0 [--freeze]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from terser.data.dedupe import group_id, read_jsonl, record_id, record_source, write_jsonl

__all__ = [
    "LOCK_NAME",
    "SPLIT_NAMES",
    "FrozenTestSetError",
    "LeakageReport",
    "SplitSummary",
    "TestAccessError",
    "assert_no_test_access",
    "assign_split",
    "check_leakage",
    "freeze_test",
    "main",
    "record_category",
    "split_records",
    "summarise",
    "verify_frozen",
    "write_splits",
]

SPLIT_NAMES: tuple[str, str, str] = ("train", "val", "test")
"""Split names, in assignment order."""

LOCK_NAME = "test.lock.json"
"""File name of the test-set lock written by :func:`freeze_test`."""

ALLOW_TEST_ENV = "TERSER_ALLOW_TEST_ACCESS"
"""Environment variable that, when ``"1"``, lets :func:`assert_no_test_access` pass."""

Record = Mapping[str, Any]


class FrozenTestSetError(RuntimeError):
    """The frozen test set is missing, or differs from its lock."""


class TestAccessError(PermissionError):
    """Code tried to read the frozen test split outside final evaluation."""

    __test__ = False  # not a pytest test class


def record_category(record: Record) -> str:
    """Return ``category``, else ``category_hint``, else ``"unknown"``."""
    for key in ("category", "category_hint"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return value
    return "unknown"


def assign_split(gid: str, seed: int = 0, train_frac: float = 0.70, val_frac: float = 0.15) -> str:
    """Return the split of a group from a stable hash of ``seed`` and the group id."""
    if train_frac <= 0 or val_frac < 0 or train_frac + val_frac >= 1:
        raise ValueError("need train_frac > 0, val_frac >= 0 and train_frac + val_frac < 1")
    digest = hashlib.sha256(f"terser-split:{seed}:{gid}".encode()).digest()
    u = int.from_bytes(digest[:8], "big") / 2**64
    if u < train_frac:
        return "train"
    if u < train_frac + val_frac:
        return "val"
    return "test"


def split_records(
    records: Sequence[Record],
    seed: int = 0,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> dict[str, list[dict[str, Any]]]:
    """Assign whole groups to splits; input order is preserved within each split.

    Raises:
        ValueError: On duplicate record ids.
    """
    ids = [record_id(r) for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("record ids must be unique")
    out: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLIT_NAMES}
    cache: dict[str, str] = {}
    for rec in records:
        gid = group_id(rec)
        if gid not in cache:
            cache[gid] = assign_split(gid, seed, train_frac, val_frac)
        out[cache[gid]].append(dict(rec))
    return out


@dataclass
class SplitSummary:
    """Record / group counts and achieved group proportions overall, per category and source."""

    seed: int
    target: dict[str, float]
    records: dict[str, int]
    groups: dict[str, int]
    proportions: dict[str, float]
    """Share of groups per split."""
    by_category: dict[str, dict[str, float]] = field(default_factory=dict)
    """Category -> split -> share of that category's groups (plus ``"n_groups"``)."""
    by_source: dict[str, dict[str, float]] = field(default_factory=dict)
    """Source -> split -> share of that source's groups (plus ``"n_groups"``)."""


def _group_strata(records: Sequence[Record], key: Any) -> dict[str, str]:
    """Map group id -> most common value of ``key(record)`` in the group (ties: smallest)."""
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for rec in records:
        counts[group_id(rec)][key(rec)] += 1
    return {g: min(c.items(), key=lambda kv: (-kv[1], kv[0]))[0] for g, c in counts.items()}


def _shares(group_split: Mapping[str, str], group_stratum: Mapping[str, str]) -> dict[str, Any]:
    table: dict[str, Counter[str]] = defaultdict(Counter)
    for g, s in group_split.items():
        table[group_stratum[g]][s] += 1
    result: dict[str, dict[str, float]] = {}
    for stratum in sorted(table):
        n = sum(table[stratum].values())
        row: dict[str, float] = {name: round(table[stratum][name] / n, 4) for name in SPLIT_NAMES}
        row["n_groups"] = n
        result[stratum] = row
    return result


def summarise(
    splits: Mapping[str, Sequence[Record]],
    seed: int = 0,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> SplitSummary:
    """Compute a :class:`SplitSummary` for already-split records."""
    group_split: dict[str, str] = {}
    all_records: list[Record] = []
    for name in SPLIT_NAMES:
        for rec in splits.get(name, ()):
            group_split[group_id(rec)] = name
            all_records.append(rec)
    n_groups = len(group_split)
    group_counts = Counter(group_split.values())
    return SplitSummary(
        seed=seed,
        target={"train": train_frac, "val": val_frac, "test": round(1 - train_frac - val_frac, 6)},
        records={name: len(splits.get(name, ())) for name in SPLIT_NAMES},
        groups={name: group_counts[name] for name in SPLIT_NAMES},
        proportions={
            name: round(group_counts[name] / n_groups, 4) if n_groups else 0.0
            for name in SPLIT_NAMES
        },
        by_category=_shares(group_split, _group_strata(all_records, record_category)),
        by_source=_shares(group_split, _group_strata(all_records, lambda r: record_source(r))),
    )


# --------------------------------------------------------------------------------------------
# Leakage
# --------------------------------------------------------------------------------------------


@dataclass
class LeakageReport:
    """Result of :func:`check_leakage`."""

    shared_groups: dict[str, list[str]]
    """``"a/b"`` split pair -> sorted group ids present in both."""
    near_duplicate_pairs: list[dict[str, Any]]
    """Cross-split record pairs with exact Jaccard >= threshold (different groups)."""
    near_duplicate_checked: bool
    threshold: float

    @property
    def n_shared_groups(self) -> int:
        """Total number of distinct groups present in more than one split."""
        return len({g for ids in self.shared_groups.values() for g in ids})

    @property
    def ok(self) -> bool:
        """True when no group is shared and no cross-split near-duplicate was found."""
        return self.n_shared_groups == 0 and not self.near_duplicate_pairs

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation."""
        data = asdict(self)
        data["n_shared_groups"] = self.n_shared_groups
        data["ok"] = self.ok
        return data


def check_leakage(
    splits: Mapping[str, Sequence[Record]],
    near_duplicates: bool = True,
    threshold: float = 0.8,
    seed: int = 0,
) -> LeakageReport:
    """Check that no group spans two splits and (optionally) no cross-split near-duplicates.

    Near-duplicate checking uses :func:`terser.data.dedupe.near_duplicate_pairs` (imported
    lazily); if that import fails the check is skipped and ``near_duplicate_checked`` is False.
    """
    names = sorted(splits)
    groups = {name: {group_id(r) for r in splits[name]} for name in names}
    shared: dict[str, list[str]] = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            common = sorted(groups[a] & groups[b])
            if common:
                shared[f"{a}/{b}"] = common

    pairs_out: list[dict[str, Any]] = []
    checked = False
    if near_duplicates:
        try:
            from terser.data.dedupe import near_duplicate_pairs, record_text
        except ImportError:  # pragma: no cover - dedupe ships with the package
            pass
        else:
            checked = True
            flat = [(name, rec) for name in names for rec in splits[name]]
            found = near_duplicate_pairs(
                [record_text(r) for _, r in flat],
                [group_id(r) for _, r in flat],
                threshold=threshold,
                seed=seed,
            )
            for i, j, sim in found:
                (sa, ra), (sb, rb) = flat[i], flat[j]
                if sa != sb:
                    pairs_out.append(
                        {
                            "a": {"split": sa, "id": record_id(ra), "group": group_id(ra)},
                            "b": {"split": sb, "id": record_id(rb), "group": group_id(rb)},
                            "jaccard": round(sim, 6),
                        }
                    )
    return LeakageReport(
        shared_groups=shared,
        near_duplicate_pairs=pairs_out,
        near_duplicate_checked=checked,
        threshold=threshold,
    )


# --------------------------------------------------------------------------------------------
# Writing, freezing, guarding
# --------------------------------------------------------------------------------------------


def write_splits(splits: Mapping[str, Sequence[Record]], out_dir: str | Path) -> dict[str, Path]:
    """Write ``train.jsonl``, ``val.jsonl`` and ``test.jsonl`` into ``out_dir``.

    Raises:
        FrozenTestSetError: If ``out_dir`` already holds a test lock (refuse to overwrite).
    """
    out = Path(out_dir)
    if (out / LOCK_NAME).exists():
        raise FrozenTestSetError(f"{out / LOCK_NAME} exists; refusing to overwrite frozen splits")
    paths = {}
    for name in SPLIT_NAMES:
        paths[name] = out / f"{name}.jsonl"
        write_jsonl(paths[name], splits.get(name, ()))
    return paths


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _test_group_ids(test_path: Path) -> list[str]:
    return sorted({group_id(r) for r in read_jsonl(test_path)})


def freeze_test(split_dir: str | Path, force: bool = False) -> Path:
    """Hash-lock ``split_dir/test.jsonl`` by writing ``split_dir/test.lock.json``.

    Args:
        split_dir: Directory containing ``test.jsonl``.
        force: Overwrite an existing lock (only for deliberate re-freezing).

    Returns:
        Path of the lock file.

    Raises:
        FileNotFoundError: If ``test.jsonl`` is missing.
        FrozenTestSetError: If a lock already exists and ``force`` is False.
    """
    directory = Path(split_dir)
    test_path = directory / "test.jsonl"
    if not test_path.is_file():
        raise FileNotFoundError(test_path)
    lock_path = directory / LOCK_NAME
    if lock_path.exists() and not force:
        raise FrozenTestSetError(f"{lock_path} already exists")
    groups = _test_group_ids(test_path)
    lock = {
        "file": test_path.name,
        "sha256": _sha256(test_path),
        "n_records": len(read_jsonl(test_path)),
        "n_groups": len(groups),
        "group_ids": groups,
    }
    lock_path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8", newline="\n")
    return lock_path


def verify_frozen(split_dir: str | Path) -> dict[str, Any]:
    """Verify ``test.jsonl`` against ``test.lock.json``; return the lock on success.

    Raises:
        FrozenTestSetError: If the lock or test file is missing, or the hash or group ids differ.
    """
    directory = Path(split_dir)
    lock_path = directory / LOCK_NAME
    if not lock_path.is_file():
        raise FrozenTestSetError(f"no test lock at {lock_path}")
    lock: dict[str, Any] = json.loads(lock_path.read_text(encoding="utf-8"))
    test_path = directory / str(lock.get("file", "test.jsonl"))
    if not test_path.is_file():
        raise FrozenTestSetError(f"frozen test file {test_path} is missing")
    digest = _sha256(test_path)
    if digest != lock.get("sha256"):
        raise FrozenTestSetError(
            f"test set changed: sha256 {digest} != locked {lock.get('sha256')}"
        )
    if _test_group_ids(test_path) != lock.get("group_ids"):
        raise FrozenTestSetError("test set group ids differ from the lock")
    return lock


def assert_no_test_access(path: str | Path) -> None:
    """Raise :class:`TestAccessError` if ``path`` is a frozen test split.

    A path counts as a frozen test split when a sibling ``test.lock.json`` exists and names it.
    Directories, the lock file itself, and train/val files next to the lock are allowed. Set
    ``TERSER_ALLOW_TEST_ACCESS=1`` to bypass for final evaluation.
    """
    if os.environ.get(ALLOW_TEST_ENV) == "1":
        return
    p = Path(path).resolve()
    if p.is_dir():
        return
    lock_path = p.parent / LOCK_NAME
    if p.name == LOCK_NAME:
        return
    if lock_path.is_file():
        try:
            locked_name = str(json.loads(lock_path.read_text(encoding="utf-8")).get("file"))
        except (OSError, ValueError):
            locked_name = "test.jsonl"
        if p.name == locked_name:
            raise TestAccessError(
                f"{p} is the frozen test split; set {ALLOW_TEST_ENV}=1 only for final evaluation"
            )


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns 0 on success, 1 when leakage is detected."""
    parser = argparse.ArgumentParser(description="Source-grouped train/val/test splits.")
    parser.add_argument("--in", dest="inp", required=True, type=Path, help="input JSONL")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--freeze", action="store_true", help="hash-lock the test split")
    parser.add_argument(
        "--no-near-dup-check", action="store_true", help="skip MinHash leakage check"
    )
    parser.add_argument(
        "--expect-test-sha256",
        default=None,
        help="fail unless the rebuilt test.jsonl has this sha256 (the frozen hash); none = skip",
    )
    args = parser.parse_args(argv)

    records = read_jsonl(args.inp)
    splits = split_records(records, args.seed, args.train_frac, args.val_frac)
    write_splits(splits, args.out_dir)
    summary = summarise(splits, args.seed, args.train_frac, args.val_frac)
    leakage = check_leakage(splits, near_duplicates=not args.no_near_dup_check, seed=args.seed)
    manifest = {
        "input": args.inp.as_posix(),
        "summary": asdict(summary),
        "leakage": leakage.to_dict(),
    }
    (args.out_dir / "splits.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    if args.expect_test_sha256 and args.expect_test_sha256.lower() != "none":
        digest = _sha256(args.out_dir / "test.jsonl")
        if digest != args.expect_test_sha256:
            raise FrozenTestSetError(
                f"rebuilt test split sha256 {digest} != frozen {args.expect_test_sha256}"
            )
    if args.freeze:
        freeze_test(args.out_dir)
        verify_frozen(args.out_dir)
    print(
        f"records {summary.records}; groups {summary.groups}; proportions {summary.proportions}; "
        f"shared groups {leakage.n_shared_groups}; "
        f"cross-split near-duplicates {len(leakage.near_duplicate_pairs)}"
    )
    return 0 if leakage.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
