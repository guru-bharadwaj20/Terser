"""Tests for source-grouped splits, test freezing and leakage checks (plan row 2.1.6)."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from terser.data import splits as sp
from terser.data.dedupe import group_id, read_jsonl, write_jsonl

CATEGORIES = ["code", "fact", "reason", "explain", "creative", "transform"]
SOURCES = ["dolly", "oasst", "gsm8k"]
STYLES = ["clean", "rambling", "texting"]


def _synthetic(n_groups: int, start: int = 0, seed: int = 0) -> list[dict[str, Any]]:
    """``n_groups`` groups of 3 records each (clean + 2 casual variants)."""
    rng = random.Random(seed)
    out = []
    for g in range(start, start + n_groups):
        source = SOURCES[g % len(SOURCES)]
        category = CATEGORIES[g % len(CATEGORIES)]
        words = " ".join(f"t{rng.randrange(10**6)}" for _ in range(20))
        for style in STYLES:
            out.append(
                {
                    "id": f"{source}:{g}:{style}",
                    "source": source,
                    "source_id": f"{source}:{g}",
                    "category": category,
                    "style": style,
                    "prompt": f"{style} {words}",
                }
            )
    return out


def _group_splits(splits: dict[str, list[dict[str, Any]]]) -> dict[str, str]:
    return {group_id(r): name for name, recs in splits.items() for r in recs}


def test_assign_split_is_stable_and_validates_fractions() -> None:
    assert sp.assign_split("g1", seed=0) == sp.assign_split("g1", seed=0)
    assert {sp.assign_split(f"g{i}", seed=0) for i in range(200)} == set(sp.SPLIT_NAMES)
    with pytest.raises(ValueError):
        sp.assign_split("g", train_frac=0.9, val_frac=0.1)


def test_no_group_straddles_splits_and_all_records_kept() -> None:
    records = _synthetic(1000)
    splits = sp.split_records(records, seed=0)
    assert sum(len(v) for v in splits.values()) == 3000
    seen: dict[str, str] = {}
    for name, recs in splits.items():
        for r in recs:
            assert seen.setdefault(r["source_id"], name) == name
    report = sp.check_leakage(splits, near_duplicates=False)
    assert report.n_shared_groups == 0 and report.ok


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_proportions_within_tolerance_on_3000_records(seed: int) -> None:
    splits = sp.split_records(_synthetic(1000), seed=seed)  # 3,000 records, 1,000 groups
    summary = sp.summarise(splits, seed=seed)
    for name, target in summary.target.items():
        assert abs(summary.proportions[name] - target) < 0.04, (name, summary.proportions)
    assert set(summary.by_category) == set(CATEGORIES)
    for cat, row in summary.by_category.items():
        assert row["n_groups"] == pytest.approx(1000 / 6, abs=1)
        for name, target in summary.target.items():
            assert abs(row[name] - target) < 0.10, (cat, row)
    assert set(summary.by_source) == set(SOURCES)


def test_appending_groups_does_not_move_existing_groups() -> None:
    base = _synthetic(500)
    before = _group_splits(sp.split_records(base, seed=0))
    after = _group_splits(sp.split_records(base + _synthetic(500, start=500, seed=9), seed=0))
    assert all(after[g] == s for g, s in before.items())
    shuffled = base[:]
    random.Random(5).shuffle(shuffled)
    assert _group_splits(sp.split_records(shuffled, seed=0)) == before


def test_seed_changes_assignment() -> None:
    records = _synthetic(200)
    assert _group_splits(sp.split_records(records, 0)) != _group_splits(
        sp.split_records(records, 1)
    )


def test_records_without_source_id_group_by_id() -> None:
    records = [{"id": f"r{i}", "prompt": f"p{i}", "category": "fact"} for i in range(50)]
    splits = sp.split_records(records, seed=0)
    assert sum(len(v) for v in splits.values()) == 50


def test_freeze_and_verify(tmp_path: Path) -> None:
    splits = sp.split_records(_synthetic(100), seed=0)
    sp.write_splits(splits, tmp_path)
    lock_path = sp.freeze_test(tmp_path)
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    assert lock["group_ids"] == sorted({r["source_id"] for r in splits["test"]})
    assert sp.verify_frozen(tmp_path)["sha256"] == lock["sha256"]

    with pytest.raises(sp.FrozenTestSetError):
        sp.freeze_test(tmp_path)
    with pytest.raises(sp.FrozenTestSetError):
        sp.write_splits(splits, tmp_path)

    test_file = tmp_path / "test.jsonl"
    original = test_file.read_bytes()
    test_file.write_bytes(original + b'{"id": "sneaky", "prompt": "x"}\n')
    with pytest.raises(sp.FrozenTestSetError, match="changed"):
        sp.verify_frozen(tmp_path)
    test_file.write_bytes(original)
    sp.verify_frozen(tmp_path)


def test_verify_without_lock_raises(tmp_path: Path) -> None:
    with pytest.raises(sp.FrozenTestSetError):
        sp.verify_frozen(tmp_path)


def test_assert_no_test_access(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    splits = sp.split_records(_synthetic(50), seed=0)
    sp.write_splits(splits, tmp_path)
    sp.assert_no_test_access(tmp_path / "test.jsonl")  # not frozen yet: allowed
    sp.freeze_test(tmp_path)
    sp.assert_no_test_access(tmp_path / "train.jsonl")
    sp.assert_no_test_access(tmp_path)
    with pytest.raises(sp.TestAccessError):
        sp.assert_no_test_access(tmp_path / "test.jsonl")
    monkeypatch.setenv(sp.ALLOW_TEST_ENV, "1")
    sp.assert_no_test_access(tmp_path / "test.jsonl")


def test_leakage_detected_on_leaky_input() -> None:
    splits = sp.split_records(_synthetic(60), seed=0)
    leaked_group = splits["test"][0]["source_id"]
    leaky = {k: list(v) for k, v in splits.items()}
    leaky["train"].append(dict(splits["test"][0], id="copied-into-train"))
    near_dup = dict(splits["test"][1], id="paraphrase", source_id="new-group")
    leaky["val"].append(near_dup)

    report = sp.check_leakage(leaky)
    assert not report.ok
    assert report.shared_groups == {"test/train": [leaked_group]}
    assert report.near_duplicate_checked
    ids = {(p["a"]["id"], p["b"]["id"]) for p in report.near_duplicate_pairs}
    assert any("paraphrase" in pair for pair in ids)

    clean = sp.check_leakage(splits)
    assert clean.ok and clean.near_duplicate_checked


def test_cli_runs_and_freezes(tmp_path: Path) -> None:
    inp = tmp_path / "in.jsonl"
    write_jsonl(inp, _synthetic(100))
    out = tmp_path / "splits"
    assert sp.main(["--in", str(inp), "--out-dir", str(out), "--seed", "0", "--freeze"]) == 0
    manifest = json.loads((out / "splits.json").read_text(encoding="utf-8"))
    assert manifest["leakage"]["ok"] is True
    assert sum(manifest["summary"]["records"].values()) == 300
    sp.verify_frozen(out)
    counts = Counter(len(read_jsonl(out / f"{n}.jsonl")) for n in sp.SPLIT_NAMES)
    assert sum(k * v for k, v in counts.items()) == 300
