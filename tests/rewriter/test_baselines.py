"""Row 8.3.9 baselines and checkpoints C8.4 / C8.5."""

from __future__ import annotations

from pathlib import Path

import pytest

from terser.models.rewriter.baselines import (
    _markdown,
    build,
    c8_4,
    c8_5,
    candidate_rewrites,
    identity_metrics,
    parse_generation_logs,
    phase9_reward,
    select_best_operator,
)

PROMPTS = [
    {"id": "a", "category": "qa"},
    {"id": "b", "category": "qa"},
    {"id": "c", "category": "code"},
]


def _cand(pid: str, method: str, rejected: bool = False, text: str = "x") -> dict[str, object]:
    return {
        "prompt_id": pid,
        "candidate_id": f"{pid}:{method}",
        "method": method,
        "text": text,
        "rejected": rejected,
        "stand_in": False,
    }


def _l(pid: str, tag: str, target: str, label: str, split: str = "val") -> dict[str, str]:
    return {
        "prompt_id": pid,
        "candidate_id": f"{pid}:{tag}",
        "target": target,
        "label": label,
        "split": split,
        "category": "qa" if pid != "c" else "code",
    }


def test_candidate_rewrites_fallback_rules() -> None:
    cands = [_cand("a", "m"), _cand("b", "m", rejected=True), _cand("c", "m")]
    tokens = {"a:original": 10, "b:original": 10, "c:original": 10, "a:m": 6, "b:m": 2, "c:m": 10}
    rw = {w["id"]: w for w in candidate_rewrites(PROMPTS, cands, tokens, "m")}
    assert rw["a"]["savings"] == pytest.approx(0.4) and not rw["a"]["fell_back"]
    assert rw["b"]["fell_back"] and rw["b"]["savings"] == 0.0  # rejected
    assert rw["c"]["fell_back"] and rw["c"]["savings"] == 0.0  # not strictly shorter
    assert rw["c"]["tokens_out"] == 10


def test_identity_is_fully_preserved() -> None:
    m = identity_metrics(PROMPTS, ["p", "q"])
    assert m["median_savings"] == 0.0 and m["preserved"] == 1.0
    assert m["preserved_by_target"] == {"p": 1.0, "q": 1.0}
    assert m["preserved_by_category"] == {"code": 1.0, "qa": 1.0}


def test_reward_and_operator_choice() -> None:
    rw = [
        {"id": "a", "savings": 0.5, "fell_back": False},
        {"id": "b", "savings": 0.0, "fell_back": True},
    ]
    labels = [_l("a", "m", "p", "degraded"), _l("b", "m", "p", "degraded")]
    # a: 50 - 100; b: fallback is safe -> 0
    assert phase9_reward(rw, labels, "m") == pytest.approx(-25.0)
    assert select_best_operator({"ops_B": -3.0, "ops_A": -3.0, "ops_C": -9.0}) == "ops_A"


def test_c8_4_matching_and_latency() -> None:
    rung = {"preserved": 0.70, "median_savings": 0.30, "latency_s": 0.1}
    llm = {
        "lo": {"preserved": 0.705, "median_savings": 0.34, "latency_s": 2.0, "b1_est_s": 5.0},
        "hi": {"preserved": 0.50, "median_savings": 0.60, "latency_s": 0.5, "b1_est_s": 3.0},
    }
    c = c8_4(rung, llm)
    lv = {r["level"]: r for r in c["levels"]}
    assert c["matched"] == ["lo"] and c["closest"] == "lo"
    assert lv["lo"]["speedup"] == pytest.approx(20.0) and lv["lo"]["pass"]
    assert not lv["hi"]["matched"] and not lv["hi"]["latency_ok"]
    assert c["pass"]
    # interpolation between (0.50, 0.60) and (0.705, 0.34) at 0.60
    mid = c8_4({**rung, "preserved": 0.60}, llm)
    assert mid["interpolated_llm_savings"] == pytest.approx(0.60 - 0.26 * 0.1 / 0.205)
    assert not mid["pass"]
    slow = c8_4({**rung, "latency_s": 1.0}, llm)
    assert not slow["pass"]  # matched and close on savings, but only 2x faster


def test_c8_5() -> None:
    lingua = {"preserved": 0.3, "median_savings": 0.5}
    assert c8_5({"preserved": 0.7, "median_savings": 0.5}, lingua)["pass"]
    assert not c8_5({"preserved": 0.7, "median_savings": 0.1}, lingua)["pass"]
    assert not c8_5({"preserved": 0.3, "median_savings": 0.9}, lingua)["pass"]


def test_parse_generation_logs(tmp_path: Path) -> None:
    log = tmp_path / "gen.log"
    log.write_text(
        "noise {\n{\n"
        '  "n_prompts": 4,\n  "written_llm_x": 4,\n  "seconds_llm_x": 2.0,\n'
        '  "skipped_llm_y": 4\n}\nmore noise\n',
        encoding="utf-8",
    )
    runs = parse_generation_logs([log, tmp_path / "missing.log"])
    assert runs == {"llm_x": {"seconds": 2.0, "written": 4, "s_per_prompt": 0.5, "log": str(log)}}


def test_build_end_to_end_and_pending_rung() -> None:
    from terser.models.rewriter.baselines import LLM32_METHODS, OPERATOR_METHODS

    methods = (*OPERATOR_METHODS, "llmlingua2", *LLM32_METHODS)
    val = [{"id": "v1", "category": "qa"}, {"id": "v2", "category": "code"}]
    train = ["t1", "t2"]
    cands = [_cand(p, m) for p in ["v1", "v2", *train] for m in methods]
    tokens = {f"{p}:original": 10 for p in ["v1", "v2", *train]}
    tokens |= {f"{p}:{m}": 5 for p in ["v1", "v2", *train] for m in methods}
    labels = []
    for m in methods:
        for t in ("p", "q"):
            labels.append(_l("v1", m, t, "preserved"))
            labels.append(_l("v2", m, t, "degraded"))
            labels.append(_l("t1", m, t, "preserved", split="train"))
            ok = "preserved" if m == "ops_CHFIDSL" else "degraded"
            labels.append(_l("t2", m, t, ok, split="train"))
    rung_rw = [
        {
            "id": "v1",
            "category": "qa",
            "savings": 0.5,
            "oracle_savings": 0.5,
            "fell_back": False,
            "latency_s": 0.01,
        },
        {
            "id": "v2",
            "category": "code",
            "savings": 0.5,
            "oracle_savings": 0.0,
            "fell_back": False,
            "latency_s": 0.01,
        },
    ]
    rung_labels = [_l(p, "r1", t, "preserved") for p in ("v1", "v2") for t in ("p", "q")]
    res = build(
        val_prompts=val,
        candidates=cands,
        labels=labels,
        tokens=tokens,
        rungs={"r1": rung_rw, "r2": rung_rw},
        rung_labels=rung_labels,
        gen_runs={"llm_medium": {"s_per_prompt": 0.5, "log": "x"}},
        throughput={1: 10.0, 16: 100.0},
    )
    assert res["best_operator"]["chosen"] == "ops_CHFIDSL"
    assert res["rungs"] == ["r1"] and res["rungs_pending"] == ["r2"]
    sysm = res["systems"]
    assert sysm["Original"]["preserved"] == 1.0
    assert sysm["LLMLingua-2"]["preserved"] == 0.5
    assert sysm["SFT r1"]["preserved"] == 1.0
    assert res["llm_latency"]["llm_medium"]["latency_s"] == 0.5
    assert res["llm_latency"]["llm_low"]["latency_source"] == "b16_est"
    assert res["llm_latency"]["llm_low"]["latency_s"] == pytest.approx(0.05)
    # r1 preserves more than every level: no match, C8.4 fails; C8.5 savings equal, pres higher
    assert res["c8_4_verdict"] == "fail" and res["c8_5_verdict"] == "pass"
    md = _markdown(res)
    assert "HMM compressor" in md and "pending" in md and "`r2`" in md
