"""Tests for row 2.3.6: the judge validation report on synthetic data with known statistics.

Synthetic design (per category, 100 annotated pairs: 50 degraded + 50 preserved by annotator 1):

* annotator 2 flips ``hh_flips`` labels in each class, so both raters keep balanced marginals and
  human-human kappa = 1 - 4 * hh_flips / 100 (in every category and overall);
* the gold label is annotator 1's label (disagreements adjudicated to it);
* the judge flips ``fn`` degraded items to preserved and ``fp`` preserved items to degraded,
  so with ``fn == fp`` judge-human kappa = 1 - 4 * fn / 100 per category,
  sensitivity = (50 - fn) / 50 and specificity = (50 - fp) / 50;
* ``inconsistent`` judge records per category have the two A/B orders favour different sides
  without changing the label; extra non-annotated judge records add to the all-records rate.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from terser.judge.annotation import ImportedAnnotation, merge_annotations, write_jsonl
from terser.judge.labels import make_label_record
from terser.phase2.judge_validation import (
    StandInJudgeError,
    build_report,
    judge_sides,
    main,
    run,
)

N_PER_CLASS = 50


def _annotation(item: int, sid: str, cat: str, name: str, label: str) -> ImportedAnnotation:
    side = "original" if label == "degraded" else ("rewrite" if item % 2 else "tie")
    verdict = "tie" if side == "tie" else ("A" if item % 3 else "B")
    return ImportedAnnotation(
        item_id=f"item{item:04d}",
        source_id=sid,
        category=cat,
        annotator=name,
        verdict=verdict,
        side=side,  # type: ignore[arg-type]
        label=label,
        confidence=2,
        notes="",
    )


def _judge_record(
    sid: str,
    cat: str,
    label: str,
    *,
    inconsistent: bool,
    stand_in: bool = False,
    task_not_worse: bool | None = None,
) -> dict[str, Any]:
    if label == "degraded":
        ab, ba = ("original", "tie") if inconsistent else ("original", "original")
    else:
        ab, ba = ("tie", "rewrite") if inconsistent else ("rewrite", "rewrite")
    return make_label_record(
        id=sid,
        prompt_id=sid.split(":")[0],
        category=cat,
        target_model="synthetic-target",
        judge_model="synthetic-judge",
        judge_prompt_version="pairwise_v1",
        task_not_worse=task_not_worse,
        verdict_ab=ab,  # type: ignore[arg-type]
        verdict_ba=ba,  # type: ignore[arg-type]
        stand_in=stand_in,
    )


def make_data(
    spec: dict[str, dict[str, int]],
    *,
    extra_consistent: int = 50,
    extra_inconsistent: int = 0,
    stand_in: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Gold records (via :func:`merge_annotations`) and label-stage judge records.

    ``spec[category]`` has keys ``hh_flips``, ``fn``, ``fp``, ``inconsistent``.
    """
    ann1: list[ImportedAnnotation] = []
    ann2: list[ImportedAnnotation] = []
    adjudications: dict[str, str] = {}
    judge: list[dict[str, Any]] = []
    item = 0
    for cat, s in spec.items():
        for n in range(2 * N_PER_CLASS):
            item += 1
            sid = f"{cat}{n:03d}:FHC"
            gold = "degraded" if n < N_PER_CLASS else "preserved"
            k = n if n < N_PER_CLASS else n - N_PER_CLASS  # index within the class
            other = "preserved" if gold == "degraded" else "degraded"
            second = other if k < s["hh_flips"] else gold
            ann1.append(_annotation(item, sid, cat, "ann1", gold))
            ann2.append(_annotation(item, sid, cat, "ann2", second))
            if second != gold:
                adjudications[sid] = gold
            # Judge errors come from the end of each class so they overlap no human flips.
            flip = k >= N_PER_CLASS - (s["fn"] if gold == "degraded" else s["fp"])
            judge_label = other if flip else gold
            judge.append(
                _judge_record(
                    sid,
                    cat,
                    judge_label,
                    inconsistent=n % 10 == 5 and n // 10 < s["inconsistent"],
                    stand_in=stand_in,
                )
            )
    for n in range(extra_consistent + extra_inconsistent):
        judge.append(
            _judge_record(
                f"extra{n:03d}:F", "code", "preserved", inconsistent=n >= extra_consistent
            )
        )
    gold_records = merge_annotations({"ann1": ann1, "ann2": ann2}, adjudications=adjudications)
    return gold_records, judge


PASSING = {
    "code": {"hh_flips": 10, "fn": 5, "fp": 5, "inconsistent": 10},
    "fact": {"hh_flips": 10, "fn": 10, "fp": 10, "inconsistent": 10},
}
FAILING = {
    "code": {"hh_flips": 11, "fn": 5, "fp": 5, "inconsistent": 10},
    "fact": {"hh_flips": 11, "fn": 13, "fp": 13, "inconsistent": 10},
}


def test_synthetic_data_is_as_designed() -> None:
    gold, judge = make_data(PASSING)
    assert len(gold) == 200 and len(judge) == 250
    assert all(g["gold_label"] is not None for g in gold)
    assert sum(not g["annotators_agree"] for g in gold) == 40
    inconsistent = [r for r in judge if not r["position_consistent"]]
    assert len(inconsistent) == 20
    assert judge_sides(inconsistent[0]) in {("original", "tie"), ("tie", "rewrite")}


def test_report_known_statistics_all_pass() -> None:
    gold, judge = make_data(PASSING)
    report = build_report(gold, judge, n_boot=200, seed=1)
    m = report.metrics
    assert m["hh_kappa"] == pytest.approx(0.60)
    assert m["jh_kappa"] == pytest.approx(0.70)  # 1 - 4 * 15 / 200
    assert m["jh_kappa_code"] == pytest.approx(0.80)
    assert m["jh_kappa_fact"] == pytest.approx(0.60)
    assert m["sensitivity"] == pytest.approx(0.85)
    assert m["specificity"] == pytest.approx(0.85)
    assert m["position_disagreement"] == pytest.approx(20 / 250)
    assert m["hh_kappa_ci_low"] < 0.60 < m["hh_kappa_ci_high"]
    assert report.all_passed and not report.stand_in
    lines = [c.line() for c in report.checks]
    assert lines == [
        "C2.3 PASS - human-human kappa = 0.600 (>= 0.60)",
        "C2.4 PASS - judge-human kappa (overall) = 0.700 (>= 0.60)",
        "C2.4 PASS - judge-human kappa (code) = 0.800 (>= 0.50)",
        "C2.4 PASS - judge-human kappa (fact) = 0.600 (>= 0.50)",
        "C2.5 PASS - judge sensitivity (degraded) = 0.850 (>= 0.85)",
        "C2.5 PASS - judge specificity = 0.850 (>= 0.80)",
        "C2.6 PASS - A/B-order disagreement rate = 0.080 (<= 0.10)",
    ]
    md = report.markdown
    for line in lines:
        assert line in md
    for heading in (
        "## Checkpoints",
        "## Human-human agreement (C2.3)",
        "## Judge-human agreement (C2.4)",
        "## Judge as degradation detector (C2.5)",
        "## Position bias (C2.6)",
    ):
        assert heading in md
    assert "SMOKE RUN" not in md
    assert "annotators: ann1, ann2" in md
    assert "| 200 | 0.850 | 0.850 | 85 | 15 | 85 | 15 |" in md
    assert "| all judge records | 250 | 20 | 0.080 |" in md
    assert "| annotated subset | 200 | 20 | 0.100 |" in md
    assert "| ann1 \\ ann2 | preserved | degraded | total |" in md
    assert "| human \\ judge | preserved | degraded | total |" in md
    assert "| total | 100 | 100 | 200 |" in md


def test_report_known_statistics_failures() -> None:
    gold, judge = make_data(FAILING, extra_consistent=0, extra_inconsistent=10)
    report = build_report(gold, judge, n_boot=100)
    m = report.metrics
    assert m["hh_kappa"] == pytest.approx(0.56)  # 1 - 4 * 11 / 100
    assert m["jh_kappa"] == pytest.approx(0.64)  # 1 - 4 * 18 / 200
    assert m["jh_kappa_fact"] == pytest.approx(0.48)
    assert m["sensitivity"] == pytest.approx(0.82)
    assert m["specificity"] == pytest.approx(0.82)
    assert m["position_disagreement"] == pytest.approx(30 / 210)
    verdicts = [(c.check_id, c.description, c.passed) for c in report.checks]
    assert verdicts == [
        ("C2.3", "human-human kappa", False),
        ("C2.4", "judge-human kappa (overall)", True),
        ("C2.4", "judge-human kappa (code)", True),
        ("C2.4", "judge-human kappa (fact)", False),
        ("C2.5", "judge sensitivity (degraded)", False),
        ("C2.5", "judge specificity", True),
        ("C2.6", "A/B-order disagreement rate", False),
    ]
    assert not report.all_passed
    assert "C2.4 FAIL - judge-human kappa (fact) = 0.480 (>= 0.50)" in report.markdown
    assert "C2.6 FAIL - A/B-order disagreement rate = 0.143 (<= 0.10)" in report.markdown


def test_unadjudicated_and_cannot_judge_items_are_dropped() -> None:
    gold, judge = make_data(PASSING)
    gold[0]["gold_label"] = None  # degraded item the judge got right
    gold[1]["annotations"]["ann2"].update(verdict="cannot_judge", side=None, label=None)
    report = build_report(gold, judge, n_boot=50)
    assert "Items without a gold label (disagreement not adjudicated, or cannot judge): 1." in (
        report.markdown
    )
    assert "| binary label | 199 | 1 |" in report.markdown
    assert "| overall | 199 | 1 |" in report.markdown
    assert report.metrics["sensitivity"] == pytest.approx(84 / 99)


def test_label_source_final_uses_stage_label() -> None:
    gold, judge = make_data(PASSING)
    # Task metric worse on 10 human-preserved pairs the judge called preserved: the stage label
    # becomes degraded, the judge-only label does not.
    changed = 0
    by_id = {r["id"]: r for r in judge}
    for g in gold:
        rec = by_id[g["id"]]
        if g["gold_label"] == "preserved" and rec["label"] == "preserved" and changed < 10:
            replacement = _judge_record(
                rec["id"], rec["category"], "preserved", inconsistent=False, task_not_worse=False
            )
            rec.clear()
            rec.update(replacement)
            changed += 1
    judge_only = build_report(gold, judge, n_boot=20)
    final = build_report(gold, judge, label_source="final", n_boot=20)
    assert judge_only.metrics["specificity"] == pytest.approx(0.85)
    assert final.metrics["specificity"] == pytest.approx(0.75)
    assert final.metrics["sensitivity"] == pytest.approx(0.85)
    assert "judge label source: `final`" in final.markdown


def test_input_errors() -> None:
    gold, judge = make_data(PASSING)
    with pytest.raises(ValueError, match="no judge record"):
        build_report(gold, judge[1:], n_boot=10)
    with pytest.raises(ValueError, match="duplicate judge record"):
        build_report(gold, [*judge, judge[0]], n_boot=10)
    three = [
        dict(g, annotations={**g["annotations"], "ann3": g["annotations"]["ann1"]}) for g in gold
    ]
    with pytest.raises(ValueError, match="exactly 2 annotators"):
        build_report(three, judge, n_boot=10)
    bad = [dict(judge[0], verdicts={"original_first": "A", "rewrite_first": "tie"}), *judge[1:]]
    with pytest.raises(ValueError, match="invalid side"):
        build_report(gold, bad, n_boot=10)


def test_stand_in_judge_refused_unless_allowed() -> None:
    gold, judge = make_data(PASSING, stand_in=True)
    with pytest.raises(StandInJudgeError):
        build_report(gold, judge, n_boot=10)
    report = build_report(gold, judge, n_boot=10, allow_stand_in=True)
    assert report.stand_in
    assert "SMOKE RUN" in report.markdown


def test_run_writes_lf_report_with_provenance(tmp_path: Path) -> None:
    gold, judge = make_data(PASSING)
    gold_path, judge_path = tmp_path / "human_300.jsonl", tmp_path / "labels.jsonl"
    write_jsonl(gold, gold_path)
    write_jsonl(judge, judge_path)
    out = tmp_path / "reports" / "p2_judge_validation.md"
    report = run(gold_path, judge_path, out, n_boot=50)
    raw = out.read_bytes()
    assert b"\r\n" not in raw
    text = raw.decode("utf-8")
    assert text == report.markdown
    assert f"- gold: `{gold_path.as_posix()} (sha256 " in text
    assert f"- judge: `{judge_path.as_posix()} (sha256 " in text


def test_cli_on_tmp_files(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    gold, judge = make_data(FAILING, extra_consistent=0, extra_inconsistent=10)
    gold_path, judge_path = tmp_path / "gold.jsonl", tmp_path / "judge.jsonl"
    write_jsonl(gold, gold_path)
    write_jsonl(judge, judge_path)
    out = tmp_path / "report.md"
    argv = ["--gold", str(gold_path), "--judge", str(judge_path), "--out", str(out)]
    code = main([*argv, "--n-boot", "50", "--seed", "3", "--no-track"])
    assert code == 0  # checkpoint failures are reported, not an error exit
    stdout = capsys.readouterr().out.splitlines()
    assert stdout[0] == "C2.3 FAIL - human-human kappa = 0.560 (>= 0.60)"
    assert "C2.4 PASS - judge-human kappa (overall) = 0.640 (>= 0.60)" in stdout
    assert "C2.5 FAIL - judge sensitivity (degraded) = 0.820 (>= 0.85)" in stdout
    assert stdout[-1] == f'{{"out": "{out.as_posix()}", "all_passed": false}}'
    assert "C2.3 FAIL - human-human kappa = 0.560 (>= 0.60)" in out.read_text(encoding="utf-8")


def test_cli_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    gold, judge = make_data(PASSING, stand_in=True)
    gold_path, judge_path = tmp_path / "gold.jsonl", tmp_path / "judge.jsonl"
    write_jsonl(gold, gold_path)
    write_jsonl(judge, judge_path)
    out = tmp_path / "report.md"
    base = ["--gold", str(gold_path), "--judge", str(judge_path), "--out", str(out), "--no-track"]
    assert main([*base, "--n-boot", "10"]) == 2
    assert "stand_in" in capsys.readouterr().err
    assert not out.exists()
    assert main([*base, "--n-boot", "10", "--allow-stand-in"]) == 0
    assert "SMOKE RUN" in out.read_text(encoding="utf-8")
    missing = ["--gold", str(tmp_path / "nope.jsonl"), "--judge", str(judge_path), "--no-track"]
    assert main(missing) == 1
    write_jsonl(judge[1:], judge_path)
    assert main([*base, "--n-boot", "10", "--allow-stand-in"]) == 1
    assert "no judge record" in capsys.readouterr().err
