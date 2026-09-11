"""Tests for terser.judge.annotation: stratified sampling, blind sheets, import validation."""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from terser.data.categories import Category
from terser.judge.annotation import (
    SHEET_COLUMNS,
    AnnotationImportError,
    build_batch,
    human_label,
    import_sheet,
    load_jsonl,
    load_key,
    main,
    merge_annotations,
    sample_pairs,
    write_jsonl,
)


def make_records(n_per_cat: int = 60) -> list[dict[str, Any]]:
    return [
        {
            "id": f"{cat.value}{i:03d}:FHC",
            "category": cat.value,
            "prompt": f"request {cat.value} {i}",
            "original_output": f"ORIG {cat.value} {i}",
            "rewrite_output": f"REWR {cat.value} {i}",
            "stand_in": False,
        }
        for cat in Category
        for i in range(n_per_cat)
    ]


def test_sample_pairs_stratified_and_deterministic() -> None:
    recs = make_records()
    s1 = sample_pairs(recs, seed=7)
    s2 = sample_pairs(list(reversed(recs)), seed=7)
    assert len(s1) == 300
    assert Counter(r["category"] for r in s1) == {c.value: 50 for c in Category}
    assert [r["id"] for r in s1] == [r["id"] for r in s2]
    assert [r["id"] for r in sample_pairs(recs, seed=8)] != [r["id"] for r in s1]


def test_sample_pairs_errors() -> None:
    recs = make_records(10)
    with pytest.raises(ValueError, match="< 50"):
        sample_pairs(recs)
    assert len(sample_pairs(recs, allow_short=True)) == 60
    bad = make_records(50)
    bad[0]["stand_in"] = True
    with pytest.raises(ValueError, match="stand_in"):
        sample_pairs(bad)
    dup = make_records(50)
    dup.append(dict(dup[0]))
    with pytest.raises(ValueError, match="duplicate"):
        sample_pairs(dup)
    missing = make_records(50)
    del missing[3]["rewrite_output"]
    with pytest.raises(ValueError, match="missing"):
        sample_pairs(missing)


@pytest.fixture()
def batch(tmp_path: Path) -> tuple[Path, list[dict[str, Any]]]:
    pairs = sample_pairs(make_records(), seed=1)
    build_batch(pairs, tmp_path / "batch", annotators=("ann1", "ann2"), seed=1)
    return tmp_path / "batch", pairs


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_batch_is_blind_and_randomised(batch: tuple[Path, list[dict[str, Any]]]) -> None:
    out, pairs = batch
    key = load_key(out / "key.jsonl")
    assert len(key) == 300
    sheets = {a: read_csv(out / f"annotator_{a}.csv") for a in ("ann1", "ann2")}
    for name, rows in sheets.items():
        assert list(rows[0]) == list(SHEET_COLUMNS)
        assert rows == load_jsonl(out / f"annotator_{name}.jsonl")
        text = json.dumps(rows)
        # No source id, operator code, rewrite prompt or side name leaks into the sheet.
        assert ":FHC" not in text and "rewrite" not in text.lower() and "original" not in text
        assert all(r["verdict"] == "" for r in rows)
        # Response A really is the side the key says.
        by_source = {p["id"]: p for p in pairs}
        for r in rows:
            entry = key[r["item_id"]]
            side = entry["a_side"][name]
            assert r["response_a"] == by_source[entry["source_id"]][f"{side}_output"]
        a_rewrite = sum(key[r["item_id"]]["a_side"][name] == "rewrite" for r in rows)
        assert 110 < a_rewrite < 190  # roughly half
    # Independent orders and A/B assignments per annotator.
    assert [r["item_id"] for r in sheets["ann1"]] != [r["item_id"] for r in sheets["ann2"]]
    differ = sum(e["a_side"]["ann1"] != e["a_side"]["ann2"] for e in key.values())
    assert 100 < differ < 200
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["n_items"] == 300 and set(manifest["per_category"].values()) == {50}


def fill(out: Path, name: str, choose: Any, fmt: str = "csv") -> Path:
    """Fill a sheet; ``choose(side_of_A, side_of_B)`` returns the verdict string."""
    key = load_key(out / "key.jsonl")
    rows = read_csv(out / f"annotator_{name}.csv")
    for r in rows:
        a = key[r["item_id"]]["a_side"][name]
        b = "original" if a == "rewrite" else "rewrite"
        r["verdict"] = choose(a, b)
        r["confidence"] = "2"
    path = out / f"filled_{name}.{fmt}"
    if fmt == "csv":
        with path.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(SHEET_COLUMNS), lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
    else:
        write_jsonl(rows, path)
    return path


def prefer_rewrite(a: str, b: str) -> str:
    return "A" if a == "rewrite" else "B"


def prefer_original(a: str, b: str) -> str:
    return "A" if a == "original" else "B"


@pytest.mark.parametrize("fmt", ["csv", "jsonl"])
def test_import_maps_positions_back_to_sides(
    batch: tuple[Path, list[dict[str, Any]]], fmt: str
) -> None:
    out, _ = batch
    key = load_key(out / "key.jsonl")
    got = import_sheet(fill(out, "ann1", prefer_original, fmt), key, "ann1")
    assert len(got) == 300
    assert {a.side for a in got} == {"original"}
    assert {a.label for a in got} == {"degraded"}
    got2 = import_sheet(fill(out, "ann2", prefer_rewrite, fmt), key, "ann2")
    assert {a.label for a in got2} == {"preserved"}


def test_import_validation_collects_errors(batch: tuple[Path, list[dict[str, Any]]]) -> None:
    out, _ = batch
    key = load_key(out / "key.jsonl")
    path = fill(out, "ann1", lambda a, b: "tie")
    rows = read_csv(path)
    rows[0]["verdict"] = "maybe"
    rows[1]["confidence"] = "5"
    rows[2]["category"] = "fact" if rows[2]["category"] != "fact" else "code"
    rows[3]["item_id"] = "item999"
    rows.append(dict(rows[5]))
    rows.pop(10)
    rows[4]["verdict"] = "Cannot Judge"
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(SHEET_COLUMNS), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    with pytest.raises(AnnotationImportError) as info:
        import_sheet(path, key, "ann1")
    text = "\n".join(info.value.errors)
    assert "maybe" in text and "confidence" in text and "category changed" in text
    assert "unknown item_id" in text and "appears 2 times" in text and "missing from sheet" in text
    assert "Cannot Judge" not in text


def test_import_rejects_missing_columns_and_unknown_annotator(
    batch: tuple[Path, list[dict[str, Any]]],
) -> None:
    out, _ = batch
    key = load_key(out / "key.jsonl")
    bad = out / "bad.csv"
    bad.write_text("item_id,category\n", encoding="utf-8")
    with pytest.raises(AnnotationImportError, match="missing column"):
        import_sheet(bad, key, "ann1")
    with pytest.raises(AnnotationImportError, match="not in key"):
        import_sheet(fill(out, "ann1", prefer_rewrite), key, "ann3")


def test_merge_and_cannot_judge(batch: tuple[Path, list[dict[str, Any]]]) -> None:
    out, _ = batch
    key = load_key(out / "key.jsonl")
    ann1 = import_sheet(fill(out, "ann1", prefer_rewrite), key, "ann1")
    unsure = set(sorted(key)[:30])

    def ann2_choice(a: str, b: str) -> str:
        return "tie"

    ann2 = [
        replace(a, verdict="cannot_judge", side=None, label=None) if a.item_id in unsure else a
        for a in import_sheet(fill(out, "ann2", ann2_choice), key, "ann2")
    ]
    first_source = key[sorted(key)[0]]["source_id"]
    gold = merge_annotations({"ann1": ann1, "ann2": ann2}, adjudications={first_source: "degraded"})
    assert len(gold) == 300
    pending = [g for g in gold if g["gold_label"] is None]
    assert len(pending) == 29
    adj = next(g for g in gold if g["id"] == first_source)
    assert adj["adjudicated"] and adj["gold_label"] == "degraded"
    assert {g["gold_label"] for g in gold if not g["adjudicated"] and g["annotators_agree"]} == {
        "preserved"
    }
    with pytest.raises(ValueError, match="lacks"):
        merge_annotations({"ann1": ann1, "ann2": ann2[:-1]})


def test_human_label() -> None:
    assert human_label("original") == "degraded"
    assert human_label("rewrite") == "preserved"
    assert human_label("tie") == "preserved"
    assert human_label(None) is None


def test_cli_round_trip(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    records = tmp_path / "records.jsonl"
    write_jsonl(make_records(), records)
    out = tmp_path / "batch"
    assert main(["build", "--records", str(records), "--out", str(out), "--seed", "3"]) == 0
    s1 = fill(out, "ann1", prefer_rewrite)
    s2 = fill(out, "ann2", lambda a, b: "tie")
    gold_path = tmp_path / "human_300.jsonl"
    rc = main(
        [
            "import",
            "--batch",
            str(out),
            "--sheet",
            f"ann1={s1}",
            "--sheet",
            f"ann2={s2}",
            "--out",
            str(gold_path),
        ]
    )
    assert rc == 0
    gold = load_jsonl(gold_path)
    assert len(gold) == 300 and {g["gold_label"] for g in gold} == {"preserved"}
    assert main(["import", "--batch", str(out), "--sheet", "bad", "--out", str(gold_path)]) == 1
