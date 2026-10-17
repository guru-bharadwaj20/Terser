"""Row 12.1 unlock: one-time, logged, the only sanctioned reader of the locked splits."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import pytest

from terser.data.splits import (
    ALLOW_TEST_ENV,
    FrozenTestSetError,
    TestAccessError,
    assert_no_test_access,
    freeze_test,
)
from terser.phase12 import unlock as U


@dataclass
class Env:
    dirs: dict[str, Path]
    log: Path


def _locked(root: Path, name: str, n: int) -> Path:
    d = root / name
    d.mkdir()
    rows = [{"id": f"{name}-{i}", "prompt": f"p {i}", "category": "fact"} for i in range(n)]
    (d / "test.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    freeze_test(d)
    return d


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Env:
    monkeypatch.delenv(ALLOW_TEST_ENV, raising=False)
    monkeypatch.setattr(U, "git_state", lambda repo=Path("."), exclude=(): ("abc123", False))
    dirs = {"test": _locked(tmp_path, "splits", 3), "real": _locked(tmp_path, "real", 2)}
    return Env(dirs, tmp_path / "log" / "access.jsonl")


def test_read_refused_before_unlock(env: Env) -> None:
    with pytest.raises(U.TestLockedError):
        U.read_locked("test", "x", log_path=env.log, split_dirs=env.dirs)
    assert not env.log.exists()


def test_unlock_once_and_logged(env: Env) -> None:
    entry = U.unlock("final eval", who="tester", log_path=env.log, split_dirs=env.dirs)
    assert entry["event"] == "unlock" and entry["who"] == "tester"
    assert entry["git_sha"] == "abc123" and entry["reason"] == "final eval"
    assert set(entry["locked"]) == {"test", "real"}
    assert entry["locked"]["test"]["n_records"] == 3
    assert entry["when"].endswith("+00:00") and "+05:30" in entry["when_ist"]
    with pytest.raises(U.AlreadyUnlockedError):
        U.unlock("again", log_path=env.log, split_dirs=env.dirs)
    assert len(U.read_log(env.log)) == 1
    st = U.status(env.log, env.dirs)
    assert st["unlocked"] and st["files"]["test"]["unlocked"]


def test_read_after_unlock_logs_and_restores_guard(env: Env) -> None:
    U.unlock("final eval", log_path=env.log, split_dirs=env.dirs)
    rows = U.read_locked("real", "row 12.2 build", log_path=env.log, split_dirs=env.dirs)
    assert [r["id"] for r in rows] == ["real-0", "real-1"]
    reads = [e for e in U.read_log(env.log) if e["event"] == "read"]
    assert reads[-1]["purpose"] == "row 12.2 build" and reads[-1]["n_records"] == 2
    assert os.environ.get(ALLOW_TEST_ENV) is None
    with pytest.raises(TestAccessError):  # the guard is back on outside read_locked
        assert_no_test_access(env.dirs["real"] / "test.jsonl")


def test_changed_file_is_refused(env: Env) -> None:
    U.unlock("final eval", log_path=env.log, split_dirs=env.dirs)
    with (env.dirs["test"] / "test.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"id": "new", "prompt": "x", "category": "fact"}) + "\n")
    with pytest.raises(FrozenTestSetError):
        U.read_locked("test", "x", log_path=env.log, split_dirs=env.dirs)


def test_dirty_tree_and_empty_reason(env: Env, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError):
        U.unlock("  ", log_path=env.log, split_dirs=env.dirs)
    monkeypatch.setattr(U, "git_state", lambda repo=Path("."), exclude=(): ("abc123", True))
    with pytest.raises(U.DirtyTreeError):
        U.unlock("final", log_path=env.log, split_dirs=env.dirs)
    entry = U.unlock("final", log_path=env.log, split_dirs=env.dirs, allow_dirty=True)
    assert entry["git_dirty"] is True


def test_cli_status(env: Env, capsys: pytest.CaptureFixture[str]) -> None:
    assert U.main(["status", "--log", str(env.log)]) == 0
    assert json.loads(capsys.readouterr().out)["unlocked"] is False


def test_git_state_ignores_the_access_log(tmp_path: Path) -> None:
    import subprocess

    def git(*args: str) -> None:
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)

    git("init", "-q")
    log, other = tmp_path / "log.jsonl", tmp_path / "code.py"
    log.write_text("{}\n", encoding="utf-8")
    other.write_text("x = 1\n", encoding="utf-8")
    git("add", ".")
    git("-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-qm", "init")
    log.write_text("{}\n{}\n", encoding="utf-8")
    assert U.git_state(tmp_path)[1] is True
    assert U.git_state(tmp_path, exclude=[log])[1] is False
    other.write_text("x = 2\n", encoding="utf-8")
    assert U.git_state(tmp_path, exclude=[log])[1] is True
