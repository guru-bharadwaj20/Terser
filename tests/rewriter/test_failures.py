"""Row 8.3.10: degraded-case sampling and the failure-analysis report."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from terser.models.rewriter.failures import allocate, degraded_cases, main, render, sample


def _w(pid: str, cat: str, fell_back: bool = False) -> dict[str, object]:
    return {"id": pid, "category": cat, "fell_back": fell_back, "savings": 0.2, "rewrite": "r"}


def _l(pid: str, target: str, label: str, tag: str = "r1") -> dict[str, str]:
    return {"candidate_id": f"{pid}:{tag}", "target": target, "label": label, "reason": "x"}


def test_degraded_cases_skip_fallbacks_other_rungs_and_preserved() -> None:
    rewrites = [_w("a", "code"), _w("b", "fact", fell_back=True), _w("c", "fact")]
    labels = [
        _l("a", "q", "degraded"),
        _l("a", "p", "preserved"),
        _l("b", "q", "degraded"),
        _l("c", "q", "degraded", tag="r2"),
        _l("c", "p", "degraded"),
        _l("c", "p", "degraded"),
    ]
    got = [(c["prompt_id"], c["target"]) for c in degraded_cases(rewrites, labels, "r1")]
    assert got == [("a", "q"), ("c", "p")]


def test_allocate_is_proportional_and_sums_to_n() -> None:
    sizes = {"code": 43, "transform": 29, "creative": 23, "fact": 22, "reason": 17, "explain": 10}
    alloc = allocate(sizes, 100)
    assert sum(alloc.values()) == 100
    assert alloc == {
        "code": 30,
        "transform": 20,
        "creative": 16,
        "fact": 15,
        "reason": 12,
        "explain": 7,
    }
    assert allocate({"a": 2}, 5) == {"a": 2}


def test_sample_is_seeded_and_stratified() -> None:
    cases = [{"prompt_id": f"{c}{i}", "target": "q", "category": c} for c in "xy" for i in range(9)]
    cases += [{"prompt_id": f"z{i}", "target": "q", "category": "z"} for i in range(20)]
    a, b = sample(cases, 10, seed=0), sample(cases, 10, seed=0)
    assert a == b
    assert [c["category"] for c in a].count("z") == 5
    assert sample(cases, 10, seed=1) != a


def _rec(failure: str, cat: str = "code", **kw: object) -> dict[str, object]:
    return {
        "prompt_id": f"p:{failure}:{cat}",
        "target": "org/T",
        "category": cat,
        "rung": "r1",
        "failure": failure,
        "note": "n",
        "annotator": "claude (LLM-assisted, not human)",
        **kw,
    }


def test_render_states_caveat_and_counts() -> None:
    recs = [
        _rec("other", savings=0.1, tags=["no_visible_loss"]),
        _rec("dropped constraint", "fact", savings=0.5, excerpt="orig / rewrite"),
    ]
    md = render(recs)
    assert "LLM assistant (Claude), not by a human annotator" in md
    assert "not been human-reviewed" in md
    assert "| dropped constraint | 1 | 50% | 50% |" in md
    assert "`no_visible_loss`" in md
    assert "orig / rewrite" in md
    with pytest.raises(ValueError):
        render([_rec("tone")])


def test_report_keeps_manual_section(tmp_path: Path) -> None:
    ann = tmp_path / "a.jsonl"
    ann.write_text(json.dumps(_rec("other")) + "\n", encoding="utf-8")
    out = tmp_path / "r.md"
    out.write_text("old\n<!-- manual -->\n## Discussion\nkeep me\n", encoding="utf-8")
    assert main(["report", "--annotations", str(ann), "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "keep me" in text and not text.startswith("old")
    assert text.startswith("# Phase 8 rewriter failure analysis")
