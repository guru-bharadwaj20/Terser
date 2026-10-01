"""Test-set isolation audit for hyperparameter search (plan row **10.8**, checkpoint C10.4).

Two automated checks, both run in CI:

1. **Runtime log check.** HPO objectives read data only through :class:`DataAccessLog`, which
   records every path read (and refuses a frozen test split via
   :func:`terser.data.splits.assert_no_test_access`). The log is written as JSONL next to the
   study's outputs; :func:`audit_access_log` fails if any entry is a test split, i.e. its split
   is not ``train``/``val``.
2. **Static check.** :func:`scan_sources` greps ``src/terser`` for references to
   ``data/processed/test*`` (or a ``test.jsonl`` split file) outside ``terser.eval.final``, the
   only module allowed to read the test split (CONTRIBUTING "Never tune on test").

CLI::

    python -m terser.hpo.audit --log runs/hpo/data_access.jsonl   # exit 1 on a violation
    python -m terser.hpo.audit --scan src/terser
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from terser.data.splits import assert_no_test_access

__all__ = [
    "ALLOWED_SPLITS",
    "AuditResult",
    "DataAccessLog",
    "audit_access_log",
    "infer_split",
    "scan_sources",
]

ALLOWED_SPLITS: frozenset[str] = frozenset({"train", "val"})
_SPLIT_RE = re.compile(r"(?:^|[\\/_.-])(train|val|valid|validation|dev|test)(?:[\\/_.-]|$)")
_TEST_REF_RE = re.compile(r"""data/processed/[^"'\s]*test|["'/\\]test\.jsonl["']""")
# eval/final may read test; the split builders write and freeze it; this file names the pattern.
_ALLOWED_MODULES: tuple[str, ...] = (
    "eval/final",
    "data/splits.py",
    "data/real_slice.py",
    "hpo/audit.py",
)


def infer_split(path: str | Path) -> str:
    """Split name a data path belongs to (``train``/``val``/``test``/``unknown``).

    The file name wins over directories, so ``test_runs/train.jsonl`` is ``train``.
    """
    p = Path(path)
    for part in (p.name, *reversed(p.parent.parts)):
        m = _SPLIT_RE.search(f"/{part.lower()}/")
        if m:
            name = m.group(1)
            return "val" if name in {"valid", "validation", "dev"} else name
    return "unknown"


class DataAccessLog:
    """Records every data file an HPO run reads; refuses frozen test splits outright."""

    def __init__(self, out: str | Path | None = None) -> None:
        self._records: list[dict[str, str]] = []
        self.out = Path(out) if out is not None else None

    def record(self, path: str | Path, purpose: str = "read") -> Path:
        """Log ``path`` (raising for a frozen test split) and return it."""
        p = Path(path)
        assert_no_test_access(p)
        rec = {"path": p.as_posix(), "split": infer_split(p), "purpose": purpose}
        self._records.append(rec)
        if self.out is not None:
            self.out.parent.mkdir(parents=True, exist_ok=True)
            with self.out.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")
        return p

    def read_jsonl(self, path: str | Path) -> list[dict[str, Any]]:
        """Read a JSONL file through the log."""
        p = self.record(path)
        return [json.loads(x) for x in p.read_text(encoding="utf-8").split("\n") if x.strip()]

    def records(self) -> list[dict[str, str]]:
        """A copy of the recorded accesses."""
        return list(self._records)


@dataclass(frozen=True)
class AuditResult:
    """Outcome of an audit; ``ok`` is False if there is any violation."""

    checked: int
    violations: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.violations


def audit_access_log(records: Iterable[dict[str, Any]] | str | Path) -> AuditResult:
    """Fail on any access whose split is not ``train`` or ``val``.

    ``unknown`` also fails: an HPO run must read only files whose split is evident from the path.
    """
    if isinstance(records, str | Path):
        lines = Path(records).read_text(encoding="utf-8").split("\n")
        records = [json.loads(x) for x in lines if x.strip()]
    n, bad = 0, []
    for rec in records:
        n += 1
        split = rec.get("split") or infer_split(rec["path"])
        if split not in ALLOWED_SPLITS:
            bad.append(f"{rec['path']} (split={split})")
    return AuditResult(n, tuple(bad))


def scan_sources(root: str | Path) -> AuditResult:
    """Static check: no module outside ``terser.eval.final`` references the test split."""
    root = Path(root)
    n, bad = 0, []
    for f in sorted(root.rglob("*.py")):
        rel = f.relative_to(root).as_posix()
        n += 1
        if any(rel.startswith(a) or rel.endswith(a) for a in _ALLOWED_MODULES):
            continue
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if _TEST_REF_RE.search(line):
                bad.append(f"{rel}:{i}: {line.strip()}")
    return AuditResult(n, tuple(bad))


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: exit 0 when clean, 1 on any violation."""
    ap = argparse.ArgumentParser(description="HPO test-set isolation audit (row 10.8)")
    ap.add_argument("--log", type=Path, action="append", default=[], help="data-access JSONL")
    ap.add_argument("--scan", type=Path, default=None, help="source tree to scan")
    args = ap.parse_args(argv)
    results = [audit_access_log(p) for p in args.log]
    if args.scan is not None:
        results.append(scan_sources(args.scan))
    if not results:
        ap.error("give --log and/or --scan")
    violations = [v for r in results for v in r.violations]
    for v in violations:
        print(f"VIOLATION {v}")
    print(f"checked {sum(r.checked for r in results)} items, {len(violations)} violations")
    return 1 if violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
