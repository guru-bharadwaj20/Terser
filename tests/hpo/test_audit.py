"""Tests for the HPO test-set isolation audit (row 10.8).

``test_repository_sources_clean`` is the automated CI check on the real source tree.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from terser.data.splits import TestAccessError
from terser.hpo.audit import (
    DataAccessLog,
    audit_access_log,
    infer_split,
    main,
    scan_sources,
)

SRC = Path(__file__).resolve().parents[2] / "src" / "terser"


@pytest.mark.parametrize(
    ("path", "split"),
    [
        ("data/processed/splits/train.jsonl", "train"),
        ("data/processed/splits/val.jsonl", "val"),
        ("data/processed/splits/test.jsonl", "test"),
        ("data/processed/test/part-0.jsonl", "test"),
        ("data/processed/dev_set.csv", "val"),
        ("test_runs/train.jsonl", "train"),
        ("data/processed/features.parquet", "unknown"),
    ],
)
def test_infer_split(path: str, split: str) -> None:
    assert infer_split(path) == split


def _splits(tmp_path: Path, frozen: bool) -> Path:
    d = tmp_path / "splits"
    d.mkdir()
    for name in ("train", "val", "test"):
        (d / f"{name}.jsonl").write_text(json.dumps({"id": name}) + "\n", encoding="utf-8")
    if frozen:
        (d / "test.lock.json").write_text(json.dumps({"file": "test.jsonl"}), encoding="utf-8")
    return d


def test_hpo_run_log_audits_clean(tmp_path: Path) -> None:
    pytest.importorskip("optuna")
    from terser.hpo import Float, SearchSpace, run_search

    d = _splits(tmp_path, frozen=True)
    log = DataAccessLog(tmp_path / "access.jsonl")

    def objective(p: dict[str, float]) -> float:
        log.read_jsonl(d / "train.jsonl")
        log.read_jsonl(d / "val.jsonl")
        return -abs(p["x"])

    res = run_search(objective, SearchSpace({"x": Float(-1, 1)}), "random", 3, data_log=log)
    assert len(res.extra["data_access"]) == 6
    assert audit_access_log(tmp_path / "access.jsonl").ok
    assert main(["--log", str(tmp_path / "access.jsonl")]) == 0


def test_frozen_test_read_is_refused(tmp_path: Path) -> None:
    d = _splits(tmp_path, frozen=True)
    with pytest.raises(TestAccessError):
        DataAccessLog().read_jsonl(d / "test.jsonl")


def test_unfrozen_test_read_is_caught_by_audit(tmp_path: Path) -> None:
    d = _splits(tmp_path, frozen=False)
    log = DataAccessLog(tmp_path / "access.jsonl")
    log.read_jsonl(d / "train.jsonl")
    log.read_jsonl(d / "test.jsonl")
    res = audit_access_log(log.records())
    assert not res.ok and len(res.violations) == 1 and "test.jsonl" in res.violations[0]
    assert main(["--log", str(tmp_path / "access.jsonl")]) == 1


def test_scan_flags_test_reference(tmp_path: Path) -> None:
    pkg = tmp_path / "pkg" / "hpo"
    pkg.mkdir(parents=True)
    (pkg / "bad.py").write_text('P = "data/processed/splits/test.jsonl"\n', encoding="utf-8")
    (pkg / "good.py").write_text('P = "data/processed/splits/val.jsonl"\n', encoding="utf-8")
    res = scan_sources(tmp_path / "pkg")
    assert res.checked == 2 and len(res.violations) == 1 and "bad.py:1" in res.violations[0]


def test_repository_sources_clean() -> None:
    """CI check (C10.4): nothing outside terser.eval.final / split builders names the test split."""
    res = scan_sources(SRC)
    assert res.checked > 50
    assert res.ok, res.violations
