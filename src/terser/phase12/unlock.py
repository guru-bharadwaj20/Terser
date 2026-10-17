"""One-time, logged unlock of the frozen test split (plan row **12.1**).

The frozen test split in ``data/processed/splits/`` and the real-prompt slice in
``data/processed/real_slice/`` (row 2.1.8) are hash-locked by
:func:`terser.data.splits.freeze_test`, and :func:`terser.data.splits.assert_no_test_access`
refuses to open them unless ``TERSER_ALLOW_TEST_ACCESS=1``. This module is the only sanctioned
way to set that switch:

* :func:`unlock` runs **once**. It re-hashes both locked files against their locks
  (:func:`~terser.data.splits.verify_frozen`), refuses a dirty git tree unless told otherwise,
  and appends an ``unlock`` entry to the committed access log :data:`ACCESS_LOG`: who, when
  (UTC and IST), git SHA, dirty flag, reason, and the sha256 / record count of every locked file.
  A second call raises :class:`AlreadyUnlockedError`; there is no relock.
* :func:`read_locked` is the only reader. It needs that unlock entry, re-verifies the hash (the
  file must be the one that was unlocked), opens the file with the access switch set for that one
  read only, and appends a ``read`` entry (who, when, SHA, purpose, file) to the same log.

The log is JSONL so every access is one appended line; commit it after the unlock and after the
final run. ADR-0001 calls the reader ``terser.eval.final``; it lives here.

CLI::

    python -m terser.phase12.unlock status
    python -m terser.phase12.unlock unlock --reason "Phase 12 final evaluation (rows 12.1-12.9)"
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import subprocess
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Final

from terser.data.splits import (
    ALLOW_TEST_ENV,
    LOCK_NAME,
    FrozenTestSetError,
    assert_no_test_access,
    verify_frozen,
)
from terser.phase12.common import read_jsonl

__all__ = [
    "ACCESS_LOG",
    "LOCKED_SPLITS",
    "AlreadyUnlockedError",
    "DirtyTreeError",
    "TestLockedError",
    "git_state",
    "main",
    "read_locked",
    "read_log",
    "status",
    "unlock",
    "unlock_entry",
]

ACCESS_LOG: Final = Path("docs/checkpoints/C12/test_access_log.jsonl")
"""Committed access log: one JSON line per unlock or read."""

LOCKED_SPLITS: Final[dict[str, Path]] = {
    "test": Path("data/processed/splits"),
    "real": Path("data/processed/real_slice"),
}
"""Name -> directory holding a frozen ``test.jsonl`` and its ``test.lock.json``."""

IST: Final = timezone(timedelta(hours=5, minutes=30), "IST")


class TestLockedError(PermissionError):
    """A locked split was requested before the row 12.1 unlock (or after its file changed)."""

    __test__ = False  # not a pytest test class


class AlreadyUnlockedError(RuntimeError):
    """The access log already holds an unlock entry; the unlock is one-time."""


class DirtyTreeError(RuntimeError):
    """The git tree has uncommitted changes, so the logged SHA would not identify the code."""


def git_state(repo: Path = Path(".")) -> tuple[str | None, bool | None]:
    """``(HEAD sha, dirty)`` of ``repo``; ``(None, None)`` outside a git checkout."""
    try:
        sha = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return sha, dirty


def _who(repo: Path) -> str:
    try:
        name = subprocess.run(
            ["git", "-C", str(repo), "config", "user.name"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        name = ""
    return name or getpass.getuser()


def _now(now: datetime | None) -> dict[str, str]:
    t = (now or datetime.now(UTC)).astimezone(UTC)
    return {"when": t.isoformat(timespec="seconds"), "when_ist": t.astimezone(IST).isoformat()}


def read_log(log_path: Path = ACCESS_LOG) -> list[dict[str, Any]]:
    """Entries of the access log (missing file -> empty)."""
    return read_jsonl(log_path) if log_path.is_file() else []


def _append(log_path: Path, entry: Mapping[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def unlock_entry(log_path: Path = ACCESS_LOG) -> dict[str, Any] | None:
    """The (single) ``unlock`` entry of the log, if any."""
    entries = [e for e in read_log(log_path) if e.get("event") == "unlock"]
    return entries[0] if entries else None


def _locked_file(directory: Path, lock: Mapping[str, Any]) -> Path:
    """The locked file a lock names (:func:`~terser.data.splits.freeze_test` always records it).

    The name comes from the lock, never from this module, so no source here spells the test file
    (row 10.8 audit, :func:`terser.hpo.audit.scan_sources`).
    """
    if not lock.get("file"):
        raise FrozenTestSetError(f"lock in {directory} names no file")
    return directory / str(lock["file"])


def _lock_summary(split_dirs: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    """Verify every locked file against its lock; ``{name: {file, sha256, n_records}}``."""
    out = {}
    for name, d in split_dirs.items():
        lock = verify_frozen(d)  # raises FrozenTestSetError on a missing or changed file
        out[name] = {
            "file": _locked_file(d, lock).as_posix(),
            "sha256": lock["sha256"],
            "n_records": lock.get("n_records"),
        }
    return out


def unlock(
    reason: str,
    *,
    who: str | None = None,
    log_path: Path = ACCESS_LOG,
    split_dirs: Mapping[str, Path] = LOCKED_SPLITS,
    repo: Path = Path("."),
    allow_dirty: bool = False,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Unlock the test split once and log it; returns the appended entry.

    Raises:
        ValueError: On an empty reason.
        AlreadyUnlockedError: If the log already holds an unlock.
        DirtyTreeError: If the git tree is dirty and ``allow_dirty`` is False.
        terser.data.splits.FrozenTestSetError: If a locked file is missing or changed.
    """
    if not reason.strip():
        raise ValueError("the unlock needs a reason")
    prior = unlock_entry(log_path)
    if prior is not None:
        raise AlreadyUnlockedError(
            f"test split already unlocked at {prior.get('when')} by {prior.get('who')} "
            f"(git {prior.get('git_sha')}); the unlock is one-time, see {log_path}"
        )
    sha, dirty = git_state(repo)
    if dirty and not allow_dirty:
        raise DirtyTreeError(
            "uncommitted changes: commit the evaluation code first so the logged SHA identifies "
            "it (or pass allow_dirty, which is recorded)"
        )
    entry = {
        "event": "unlock",
        "row": "12.1",
        "who": who or _who(repo),
        **_now(now),
        "git_sha": sha,
        "git_dirty": dirty,
        "reason": reason.strip(),
        "locked": _lock_summary(split_dirs),
    }
    _append(log_path, entry)
    return entry


def status(
    log_path: Path = ACCESS_LOG, split_dirs: Mapping[str, Path] = LOCKED_SPLITS
) -> dict[str, Any]:
    """Unlock state and per-file lock checks (never reads a locked file's records)."""
    entry = unlock_entry(log_path)
    files: dict[str, Any] = {}
    for name, d in split_dirs.items():
        lock_path = d / LOCK_NAME
        if not lock_path.is_file():
            files[name] = {"lock": "missing"}
            continue
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        unlocked = entry is not None and entry["locked"].get(name, {}).get("sha256") == lock.get(
            "sha256"
        )
        files[name] = {"sha256": lock.get("sha256"), "unlocked": unlocked}
    reads = [e for e in read_log(log_path) if e.get("event") == "read"]
    return {
        "unlocked": entry is not None,
        "unlock": entry,
        "files": files,
        "reads": len(reads),
        "log": log_path.as_posix(),
    }


@contextmanager
def _allow_access() -> Iterator[None]:
    old = os.environ.get(ALLOW_TEST_ENV)
    os.environ[ALLOW_TEST_ENV] = "1"
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(ALLOW_TEST_ENV, None)
        else:
            os.environ[ALLOW_TEST_ENV] = old


def read_locked(
    name: str,
    purpose: str,
    *,
    log_path: Path = ACCESS_LOG,
    split_dirs: Mapping[str, Path] = LOCKED_SPLITS,
    repo: Path = Path("."),
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    """Records of the locked split ``name`` (``test`` or ``real``), after the unlock; logged.

    Raises:
        TestLockedError: Before the unlock, or if the file no longer matches the unlocked hash.
        KeyError: For an unknown ``name``.
    """
    if not purpose.strip():
        raise ValueError("a read needs a purpose")
    directory = split_dirs[name]
    entry = unlock_entry(log_path)
    if entry is None:
        raise TestLockedError(
            f"{name} is locked until the row 12.1 unlock: python -m terser.phase12.unlock unlock "
            "--reason ..."
        )
    lock = verify_frozen(directory)
    unlocked_sha = entry.get("locked", {}).get(name, {}).get("sha256")
    if unlocked_sha != lock["sha256"]:
        raise TestLockedError(
            f"{name}: current sha256 {lock['sha256']} is not the unlocked {unlocked_sha}"
        )
    path = _locked_file(directory, lock)
    with _allow_access():
        assert_no_test_access(path)
        records = read_jsonl(path)
    sha, dirty = git_state(repo)
    _append(
        log_path,
        {
            "event": "read",
            "who": _who(repo),
            **_now(now),
            "git_sha": sha,
            "git_dirty": dirty,
            "file": path.as_posix(),
            "sha256": lock["sha256"],
            "n_records": len(records),
            "purpose": purpose.strip(),
        },
    )
    return records


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point (``status`` / ``unlock``)."""
    ap = argparse.ArgumentParser(prog="python -m terser.phase12.unlock", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status", help="show the unlock state (reads no locked records)")
    u = sub.add_parser("unlock", help="the one-time, logged row 12.1 unlock")
    u.add_argument("--reason", required=True)
    u.add_argument("--who", default=None, help="default: git user.name, else the OS user")
    u.add_argument("--allow-dirty", action="store_true", help="unlock with uncommitted changes")
    for p in sub.choices.values():
        p.add_argument("--log", type=Path, default=ACCESS_LOG)
    args = ap.parse_args(argv)
    if args.cmd == "status":
        print(json.dumps(status(args.log), indent=2))
        return 0
    entry = unlock(args.reason, who=args.who, log_path=args.log, allow_dirty=args.allow_dirty)
    print(json.dumps(entry, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
