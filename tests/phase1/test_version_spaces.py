"""Tests for the 1.3.7 version-space runner on synthetic labels (never the real report)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from terser.concept.attributes import DOMAINS, describe_prompt, instance
from terser.concept.candidate_elimination import candidate_elimination
from terser.concept.version_space import version_space
from terser.phase1.version_spaces import (
    AttributeCache,
    StandInLabelsError,
    VersionSpaceReport,
    build_report,
    load_labels,
    main,
    run,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "pilot_labels_synthetic.jsonl"


def identity(text: str) -> str:
    """Remover that drops nothing (filler_density is always low)."""
    return text


def _run(tmp_path: Path) -> tuple[str, VersionSpaceReport]:
    out = tmp_path / "reports" / "p1_version_spaces.md"
    report = run(FIXTURE, out, remover=identity)
    return out.read_text(encoding="utf-8"), report


def test_fixture_is_marked_synthetic() -> None:
    records = [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines()]
    assert records and all(r["synthetic"] is True and r["stand_in"] is False for r in records)
    assert {r["category"] for r in records} == {"fact", "creative"}


def test_nonempty_category_report(tmp_path: Path) -> None:
    text, _ = _run(tmp_path)
    fact = text.split("### fact")[1].split("###")[0]
    assert "Examples: **4** (3 preserved / 1 degraded); unlabelled records skipped: 1" in fact
    assert "Status: **non-empty**" in fact
    assert "Version-space size: **256**" in fact
    assert (
        "`<length=short, ?, has_payload=no, tone_cue=no, filler_density=low, ?, ?, op_D=no, "
        "op_I=no, op_L=no, op_C=no, op_S=no>`" in fact
    )
    assert "`<?, ?, ?, ?, ?, ?, ?, ?, ?, op_L=no, ?, ?>`" in fact


def test_nonempty_size_matches_library() -> None:
    records = [r for r in load_labels(FIXTURE) if r.category == "fact" and r.label]
    examples = [
        (instance(describe_prompt(r.original, identity), r.ops), r.label == "preserved")
        for r in records
    ]
    res = candidate_elimination(examples, DOMAINS)
    assert len(version_space(res, DOMAINS)) == 256


def test_collapse_report_names_conflicting_examples(tmp_path: Path) -> None:
    text, report = _run(tmp_path)
    creative = text.split("### creative")[1].split("###")[0]
    assert "Status: **collapsed**" in creative
    assert "on example 3 (`creative-syn-01:H`)" in creative
    assert "Minimal conflicting set: 3 examples" in creative
    assert (
        "| offending | 3 | `creative-syn-01:H` | `creative-syn-01` | `H` | degraded | short | no "
        "| no | yes | low | `0100000` |" in creative
    )
    assert "| conflicting | 0 | `creative-syn-01:F` |" in creative
    assert "| conflicting | 1 | `creative-syn-02:H` |" in creative
    assert "`creative-syn-02:F`" not in creative  # not part of the minimal set
    cats = {c.category: c for c in report.categories}
    result = cats["creative"].result
    assert result is not None and result.collapse is not None
    assert result.collapse.examples == (0, 1, 3)


def test_summary_and_stamps(tmp_path: Path) -> None:
    text, report = _run(tmp_path)
    assert "SYNTHETIC LABELS" in text
    assert "SMOKE RUN" not in text
    assert "**C1.3 Version space: FAIL (not valid: synthetic labels).**" in text
    assert "Non-empty for **1 of 6** categories" in text
    assert "| code | 0 | 0 | 0 | no examples |  |" in text
    assert "docs/derivations/inductive_bias.md" in text
    assert report.metrics["c13_pass"] == 0.0


def _stand_in_copy(tmp_path: Path) -> Path:
    path = tmp_path / "labels.jsonl"
    rows = [json.loads(line) for line in FIXTURE.read_text(encoding="utf-8").splitlines()]
    rows[0]["stand_in"] = True
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def test_refuses_stand_in_labels(tmp_path: Path) -> None:
    labels = _stand_in_copy(tmp_path)
    out = tmp_path / "out.md"
    with pytest.raises(StandInLabelsError, match="--allow-stand-in"):
        run(labels, out, remover=identity)
    assert not out.exists()
    assert main(["--labels", str(labels), "--out", str(out), "--no-track"]) == 2
    assert not out.exists()


def test_allow_stand_in_stamps_smoke_run(tmp_path: Path) -> None:
    labels = _stand_in_copy(tmp_path)
    out = tmp_path / "out.md"
    report = run(labels, out, allow_stand_in=True, remover=identity)
    text = out.read_text(encoding="utf-8")
    assert text.startswith("# Phase 1 version spaces per category\n\n> **SMOKE RUN.**")
    assert "not valid: stand-in labels" in text
    assert report.stand_in


def test_missing_stand_in_field_is_treated_as_stand_in(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    row = json.loads(FIXTURE.read_text(encoding="utf-8").splitlines()[0])
    del row["stand_in"]
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(StandInLabelsError):
        run(path, tmp_path / "out.md", remover=identity)


def test_attributes_computed_once_per_prompt() -> None:
    calls: list[str] = []

    def describe(text: str, remover: object) -> dict[str, str]:
        calls.append(text)
        return describe_prompt(text, identity)

    records = load_labels(FIXTURE)
    cache = AttributeCache(remover=identity, describe=describe)
    report = build_report(records, cache)
    n_prompts = len({r.prompt_id for r in records if r.label is not None})
    assert len(calls) == cache.calls == n_prompts == 5
    assert len(records) == 9
    assert report.metrics["attribute_evaluations"] == 5.0


def test_cache_rejects_conflicting_text() -> None:
    cache = AttributeCache(remover=identity)
    cache.get("p1", "Explain recursion.")
    with pytest.raises(ValueError, match="two different texts"):
        cache.get("p1", "Explain iteration.")


def test_cache_passes_remover() -> None:
    seen: list[object] = []

    def describe(text: str, remover: object) -> dict[str, str]:
        seen.append(remover)
        return describe_prompt(text, identity)

    AttributeCache(remover=identity, describe=describe).get("p", "Explain recursion.")
    assert seen == [identity]


def test_output_is_deterministic(tmp_path: Path) -> None:
    first, _ = _run(tmp_path / "a")
    second, _ = _run(tmp_path / "b")
    assert first == second
    raw = (tmp_path / "a" / "reports" / "p1_version_spaces.md").read_bytes()
    assert b"\r\n" not in raw


def test_cli_no_track_writes_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "vs.md"
    # The CLI uses the default remover (FHC operators); keep the test independent of them.
    monkeypatch.setattr("terser.concept.attributes.default_remover", identity)
    assert main(["--labels", str(FIXTURE), "--out", str(out), "--no-track"]) == 0
    assert "Version-space size: **256**" in out.read_text(encoding="utf-8")
    assert json.loads(capsys.readouterr().out)["n_nonempty"] == 1.0


def test_cli_missing_labels(tmp_path: Path) -> None:
    assert main(["--labels", str(tmp_path / "nope.jsonl"), "--no-track"]) == 1


def test_unknown_label_rejected(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    row = json.loads(FIXTURE.read_text(encoding="utf-8").splitlines()[0])
    row["label"] = "maybe"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown label"):
        load_labels(path)
