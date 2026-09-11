"""Tests for MinHash LSH near-duplicate removal (plan row 2.1.5)."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from terser.data import dedupe as dd

_VOCAB = [f"w{i}" for i in range(400)]


def _text(rng: random.Random, n: int = 60) -> str:
    return " ".join(rng.choice(_VOCAB) for _ in range(n))


def _mutate(text: str, rng: random.Random, n_changes: int) -> str:
    words = text.split()
    for _ in range(n_changes):
        words[rng.randrange(len(words))] = rng.choice(_VOCAB)
    return " ".join(words)


def test_normalise_lowercases_strips_punctuation_and_collapses_whitespace() -> None:
    assert dd.normalise("  Hello,   WORLD!!\n\tIt's  fine. ") == "hello world it s fine"


def test_shingles_word_5grams_short_and_empty() -> None:
    assert dd.shingles("a b c d e f") == frozenset({"a b c d e", "b c d e f"})
    assert dd.shingles("Hi there!") == frozenset({"hi there"})
    assert dd.shingles(" ... ") == frozenset()


def test_exact_jaccard() -> None:
    assert dd.exact_jaccard(frozenset({"a", "b"}), frozenset({"b", "c"})) == pytest.approx(1 / 3)
    assert dd.exact_jaccard(frozenset(), frozenset()) == 0.0


def test_minhash_estimate_tracks_exact_jaccard() -> None:
    rng = random.Random(1)
    hasher = dd.MinHasher(num_perm=256, seed=0)
    errors = []
    for _ in range(40):
        base = _text(rng, 80)
        other = _mutate(base, rng, rng.randrange(0, 30))
        a, b = dd.shingles(base), dd.shingles(other)
        exact = dd.exact_jaccard(a, b)
        est = hasher.estimate(hasher.signature(a), hasher.signature(b))
        errors.append(abs(est - exact))
    assert max(errors) < 0.15
    assert sum(errors) / len(errors) < 0.05


def test_minhash_is_deterministic_given_seed() -> None:
    s = dd.shingles("the quick brown fox jumps over the lazy dog again")
    assert (dd.MinHasher(seed=3).signature(s) == dd.MinHasher(seed=3).signature(s)).all()
    assert not (dd.MinHasher(seed=3).signature(s) == dd.MinHasher(seed=4).signature(s)).all()


def test_optimal_bands_has_high_recall_at_threshold() -> None:
    bands, rows = dd.optimal_bands(0.8, 128)
    assert bands * rows <= 128
    assert 1 - (1 - 0.85**rows) ** bands > 0.98


def test_lsh_finds_the_same_pairs_as_brute_force() -> None:
    rng = random.Random(7)
    texts = []
    for _ in range(60):
        base = _text(rng, 100)
        texts.append(base)
        if rng.random() < 0.5:
            texts.append(_mutate(base, rng, rng.choice([1, 2, 30])))
    sets = [dd.shingles(t) for t in texts]
    brute = {
        (i, j)
        for i in range(len(texts))
        for j in range(i + 1, len(texts))
        if dd.exact_jaccard(sets[i], sets[j]) >= 0.8
    }
    found = {(i, j) for i, j, _ in dd.near_duplicate_pairs(texts, threshold=0.8)}
    assert brute  # the fixture really contains near-duplicates
    assert found <= brute  # every reported pair is verified
    assert len(found) >= 0.9 * len(brute)


def _rec(rid: str, prompt: str, source: str = "a", source_id: str | None = None) -> dict:
    rec = {"id": rid, "prompt": prompt, "source": source, "category": "fact"}
    if source_id is not None:
        rec["source_id"] = source_id
    return rec


BASE = (
    "please explain in simple terms how photosynthesis converts sunlight water and carbon "
    "dioxide into glucose and oxygen inside the chloroplasts of green plant leaves"
)


def test_dedupe_keeps_one_representative_by_source_priority_then_id() -> None:
    records = [
        _rec("z1", BASE, source="b"),
        _rec("a1", BASE + "!", source="c"),
        _rec("m1", BASE.upper(), source="b"),
        _rec("x9", "a completely unrelated request about sorting a list of integers in rust"),
    ]
    result = dd.dedupe(records, source_priority=["b", "c"])
    kept = [r["id"] for r in result.kept]
    assert kept == ["m1", "x9"]  # source b beats c; within b, lowest id m1 < z1
    report = result.report
    assert len(report.clusters) == 1
    assert report.clusters[0].kept_group == "m1"
    assert report.clusters[0].removed_groups == ["a1", "z1"]
    assert report.removed_record_ids == ["a1", "z1"]
    assert report.near_duplicate_rate_before == pytest.approx(0.5)
    assert report.near_duplicate_rate_after == 0.0


def test_paraphrase_groups_are_never_deduplicated_internally() -> None:
    records = [
        _rec("g1-clean", BASE, source_id="g1"),
        _rec("g1-casual1", BASE + " thanks", source_id="g1"),
        _rec("g1-casual2", BASE, source_id="g1"),
    ]
    result = dd.dedupe(records)
    assert len(result.kept) == 3
    assert result.report.clusters == []


def test_near_duplicate_groups_remove_the_whole_later_group() -> None:
    other = "write a haiku about autumn leaves falling slowly on a quiet mountain lake at dusk"
    records = [
        _rec("g2-clean", BASE, source_id="g2"),
        _rec("g2-casual", "hey so like " + other + " ok", source_id="g2"),
        _rec(
            "g1-clean",
            "totally different prompt about tax law in france and its history",
            source_id="g1",
        ),
        _rec("g1-casual", BASE + ".", source_id="g1"),
    ]
    result = dd.dedupe(records)
    assert [r["id"] for r in result.kept] == ["g1-clean", "g1-casual"]
    assert result.report.removed_group_ids == ["g2"]
    assert result.report.removed_record_ids == ["g2-casual", "g2-clean"]
    assert result.report.clusters[0].group_sizes == {"g1": 2, "g2": 2}


def test_transitive_clusters_and_threshold() -> None:
    rng = random.Random(3)
    a = _text(rng, 200)
    b = _mutate(a, rng, 2)
    c = _mutate(b, rng, 2)
    records = [_rec("r1", a), _rec("r2", b), _rec("r3", c)]
    assert len(dd.dedupe(records).kept) == 1
    assert len(dd.dedupe(records, threshold=1.0).kept) == 3


def test_source_schema_fields_are_supported() -> None:
    records = [
        {"source": "dolly", "source_id": "dolly:1", "instruction": BASE, "input": ""},
        {"source": "oasst", "source_id": "oasst:1", "instruction": BASE, "input": ""},
    ]
    result = dd.dedupe(records, source_priority=["oasst", "dolly"])
    assert [r["source_id"] for r in result.kept] == ["oasst:1"]


def test_duplicate_ids_rejected() -> None:
    with pytest.raises(ValueError):
        dd.dedupe([_rec("x", "a b c"), _rec("x", "d e f")])


def test_cli_writes_outputs(tmp_path: Path) -> None:
    inp = tmp_path / "in.jsonl"
    dd.write_jsonl(inp, [_rec("r1", BASE), _rec("r2", BASE + "?"), _rec("r3", "other text")])
    out, rep_json, rep_md = tmp_path / "o.jsonl", tmp_path / "r.json", tmp_path / "r.md"
    assert dd.main(["--in", str(inp), "--out", str(out), "--report", str(rep_json)]) == 0
    assert [r["id"] for r in dd.read_jsonl(out)] == ["r1", "r3"]
    report = json.loads(rep_json.read_text(encoding="utf-8"))
    assert report["removed_record_ids"] == ["r2"] and report["clusters"][0]["size"] == 2
    dd.main(["--in", str(inp), "--out", str(out), "--report", str(rep_md), "--threshold", "0.9"])
    assert "Duplicate report" in rep_md.read_text(encoding="utf-8")
    assert b"\r\n" not in out.read_bytes()


def test_pilot_fixture_runs() -> None:
    path = Path(__file__).resolve().parent.parent / "data" / "pilot" / "prompts.jsonl"
    if not path.is_file():
        pytest.skip("pilot data not present")
    result = dd.dedupe(dd.read_jsonl(path))
    assert result.report.near_duplicate_rate_after < 0.01
