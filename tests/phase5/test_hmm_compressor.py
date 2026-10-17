"""Row 5.3.6 / C5.6: HMM compressor prep helpers and report on synthetic data."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from terser.models.classical.hmm import COST_BUCKETS, POS_CLASSES, DiscreteHMM, TokenClassEncoder
from terser.phase5 import hmm_compressor as hc


def _words(text: str) -> int:
    return len(text.split())


def _filler_hmm() -> DiscreteHMM:
    """KEEP emits non-lexicon classes, DROP emits filler-lexicon classes, REPLACE uniform."""
    enc = TokenClassEncoder()
    per_lex = len(POS_CLASSES) * len(COST_BUCKETS)
    is_lex = np.arange(enc.n_obs) >= per_lex
    keep = np.where(is_lex, 0.01 / is_lex.sum(), 0.99 / (~is_lex).sum())
    drop = np.where(is_lex, 0.99 / is_lex.sum(), 0.01 / (~is_lex).sum())
    rep = np.full(enc.n_obs, 1 / enc.n_obs)
    trans = np.array([[0.6, 0.3, 0.1], [0.3, 0.6, 0.1], [0.45, 0.45, 0.1]])
    return DiscreteHMM(np.array([0.5, 0.4, 0.1]), trans, np.stack([keep, drop, rep]))


# ------------------------------------------------------------------ selection


def test_select_prompts_stratified_deterministic() -> None:
    cats = ["code"] * 50 + ["fact"] * 30 + ["creative"] * 20
    recs: list[dict[str, Any]] = [
        {"id": f"p{i}", "category": c, "prompt": f"text {i}"} for i, c in enumerate(cats)
    ]
    recs += [{"id": "bad", "category": "code", "prompt": "x", "rejected": True}]
    a = hc.select_prompts(recs, n=10, seed=0)
    assert a == hc.select_prompts(recs, n=10, seed=0)
    assert Counter(r["category"] for r in a) == {"code": 5, "fact": 3, "creative": 2}
    assert "bad" not in {r["id"] for r in a}
    order = [int(r["id"][1:]) for r in a]
    assert order == sorted(order)
    assert hc.select_prompts(recs, n=10, seed=1) != a
    assert len(hc.select_prompts(recs, n=500)) == 100


def test_fit_span_hmm_from_table() -> None:
    sp = pd.DataFrame(
        {
            "prompt_id": ["a"] * 3 + ["b"] * 2,
            "position": [0, 1, 2, 1, 0],
            "token": ["um", "write", "code", "x", "so"],
            "label": ["DROP", "KEEP", "KEEP", "KEEP", "DROP"],
        }
    )
    hmm = hc.fit_span_hmm(sp)
    assert hmm.n_states == 3
    # both sequences start with DROP (b sorted by position starts with "so")
    assert hmm.start[1] > hmm.start[0] > 0


# ------------------------------------------------------------------ rewrite


def test_hmm_rewrite_drops_filler_and_keeps_code_lines() -> None:
    hmm = _filler_hmm()
    text = "Um, so please write a function.\n\n    def f(x):\n        return x"
    out, k = hc.hmm_rewrite(hmm, text)
    assert k == 3
    assert out == "Write a function.\n\n    def f(x):\n        return x"


def test_hmm_rewrite_protects_payload_and_identity() -> None:
    hmm = _filler_hmm()
    text = 'Say "um so please" now'
    assert hc.hmm_rewrite(hmm, text) == (text, 0)
    assert hc.hmm_rewrite(hmm, "") == ("", 0)
    # a prompt made only of droppable words falls back to the original
    assert hc.hmm_rewrite(hmm, "um so") == ("um so", 2)


def test_hmm_rewrite_removes_emptied_line() -> None:
    out, _ = hc.hmm_rewrite(_filler_hmm(), "Write code.\num, so\nThanks a lot")
    assert out.splitlines()[0] == "Write code."
    assert "um" not in out.split()


# ------------------------------------------------------------------ operator choice


def test_choose_operator_by_train_reward() -> None:
    prompts = [{"prompt": "um please write code now"}, {"prompt": "write code"}]

    def apply(text: str, code: str) -> str:
        if code == "F":
            return text.replace("um ", "")
        if code == "H":
            return text.replace("please ", "").replace("now", "")
        return text

    gate = {"F": 0.1, "H": 0.9}

    def g(orig: str, draft: str) -> float:
        return gate["F"] if "please" in draft else gate["H"]

    op, rows = hc.choose_operator(prompts, apply, _words, g, ["F", "H", "D"])
    assert op == "F"
    by = {r["operator"]: r for r in rows}
    assert by["F"]["changed"] == 1 and by["D"]["changed"] == 0
    assert by["F"]["mean_reward"] == pytest.approx(10.0)  # (20 + 0) / 2
    assert by["H"]["mean_reward"] < 0 and by["D"]["mean_reward"] == 0.0


# ------------------------------------------------------------------ report maths


def test_matched_savings_mixture() -> None:
    s = [0.2, 0.4, 0.6, 0.8]
    assert hc.matched_savings(s, 0.9, 0.8) == pytest.approx(0.5)  # already above: median
    assert hc.matched_savings(s, 0.5, 0.6) == pytest.approx(0.4)  # lam 0.2
    assert hc.matched_savings(s, 0.5, 0.75) == 0.0  # lam 0.5: half the mass at 0
    assert hc.weighted_median([3.0, 1.0, 2.0], [1, 1, 1]) == 2.0


def _fixture() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    prompts = {
        f"p{i}": {"id": f"p{i}", "category": c, "prompt": "a b c d e f g h i j"}
        for i, c in enumerate(["code", "code", "fact", "fact"])
    }
    cands = []
    for pid in prompts:
        cands.append({"prompt_id": pid, "candidate_id": f"{pid}:hmm", "method": "hmm"})
        cands[-1]["text"] = "a b c d e f"  # 40% savings
        cands.append({"prompt_id": pid, "candidate_id": f"{pid}:op_F", "method": "op_F"})
        cands[-1]["text"] = "a b c d e f g h i" if pid != "p3" else prompts[pid]["prompt"]
    labels = []
    # hmm: T1 all preserved, T2 p0 degraded; op: all preserved on both
    for pid in prompts:
        for t in ("org/T1", "org/T2"):
            for m in ("hmm", "op_F"):
                bad = m == "hmm" and t == "org/T2" and pid == "p0"
                labels.append(
                    {
                        "prompt_id": pid,
                        "candidate_id": f"{pid}:{m}",
                        "target_model": t,
                        "target": t,
                        "category": prompts[pid]["category"],
                        "label": "degraded" if bad else "preserved",
                    }
                )
    return cands, prompts, labels


def test_method_metrics_and_verdict() -> None:
    cands, prompts, labels = _fixture()
    hmm = hc.method_metrics(cands, prompts, labels, "hmm", _words)
    op = hc.method_metrics(cands, prompts, labels, "op_F", _words)
    assert hmm["median_savings"] == pytest.approx(0.4)
    assert hmm["preserved"] == pytest.approx(7 / 8)
    assert hmm["preserved_by_target"] == {"org/T1": 1.0, "org/T2": 0.75}
    assert hmm["preserved_by_category"] == {"code": 0.75, "fact": 1.0}
    assert hmm["missing_labels"] == 0 and hmm["identity"] == 0
    assert op["identity"] == 1 and op["not_shorter"] == 1
    assert op["median_savings"] == pytest.approx(0.1)
    v = hc.verdict(hmm, op)
    # T1: equal preservation, 40% > 10%; T2: HMM mixed up to 100% -> savings 0 < 10%
    assert v["per_target"]["org/T1"]["hmm_beats_op"]
    assert v["per_target"]["org/T2"]["mixed"] == "hmm"
    assert not v["per_target"]["org/T2"]["hmm_beats_op"]
    assert v["qualifying_targets"] == ["org/T1"]
    assert v["c5_6"]


def test_verdict_fails_on_missing_labels_or_low_savings() -> None:
    cands, prompts, labels = _fixture()
    hmm = hc.method_metrics(cands, prompts, labels[1:], "hmm", _words)
    op = hc.method_metrics(cands, prompts, labels, "op_F", _words)
    assert hmm["missing_labels"] == 1
    assert not hc.verdict(hmm, op)["c5_6"]
    for c in cands:
        if c["method"] == "hmm":
            c["text"] = "a b c d e f g h i"
    hmm = hc.method_metrics(cands, prompts, labels, "hmm", _words)
    v = hc.verdict(hmm, op)
    assert not v["median_savings_ok"] and not v["c5_6"]


def test_report_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import terser.tokenizers.count as count_mod

    monkeypatch.setattr(count_mod, "count_tokens", lambda text, target: _words(text))
    cands, prompts, labels = _fixture()

    def dump(name: str, rows: Any) -> Path:
        p = tmp_path / name
        p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        return p

    prep = tmp_path / "prep.json"
    prep.write_text(
        json.dumps(
            {
                "operator": "F",
                "operator_rule": "test rule",
                "operator_table": [
                    {
                        "operator": "F",
                        "n": 2,
                        "changed": 1,
                        "median_savings": 0.1,
                        "mean_savings": 0.1,
                        "gate_degraded": 0.0,
                        "mean_reward": 5.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "rep.md"
    argv = ["report", "--labels", str(dump("l.jsonl", labels))]
    argv += ["--prompts", str(dump("p.jsonl", prompts.values()))]
    argv += ["--candidates", str(dump("c.jsonl", cands)), "--prep", str(prep), "--out", str(out)]
    assert hc.main(argv) == 0
    md = out.read_text(encoding="utf-8")
    assert "**C5.6: pass**" in md and "`op_F`" in md and "test rule" in md
    js = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert js["verdict"]["c5_6"] and "savings" not in js["methods"]["hmm"]


def test_prep_refuses_test_split(tmp_path: Path) -> None:
    p = tmp_path / "test.jsonl"
    p.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="test split"):
        hc.main(["prep", "--val", str(p), "--operator", "F"])
