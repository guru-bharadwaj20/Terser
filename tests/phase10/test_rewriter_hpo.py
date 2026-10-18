"""Row 10.7 rewriter LoRA tuning: objective, proxy, stage-2 choice and a tiny CPU study."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from terser.hpo.audit import DataAccessLog
from terser.phase10.rewriter_hpo import (
    DEFAULT_PARAMS,
    SPACE,
    Calibration,
    HpoSettings,
    c10_3_verdict,
    candidate_records,
    choose_trial,
    estimate_preservation,
    finalize,
    fit_calibration,
    objective_value,
    proxy_agreement,
    run_stage1,
    select_topk,
    steps_for,
    trial_settings,
    trial_tag,
    write_candidates,
)

FIXTURE = Path(__file__).resolve().parents[2] / "data" / "samples" / "casual_prompts.jsonl"
TARGETS = ("Qwen/Qwen2.5-7B-Instruct", "microsoft/Phi-3.5-mini-instruct")


def test_space_and_step_mapping() -> None:
    assert set(SPACE.dims) == {"rank", "alpha_mult", "lr", "warmup_frac", "epochs"}
    assert [steps_for(e, 1262, 4) for e in (1, 2, 3)] == [316, 631, 947]
    st = trial_settings({**DEFAULT_PARAMS, "warmup_frac": 0.1, "rank": 32}, 1262, 4)
    assert st == {
        "rank": 32,
        "alpha": 64,
        "lr": 2e-4,
        "steps": 947,
        "warmup_steps": 95,
        "epochs": 3,
    }
    with pytest.raises(ValueError):
        steps_for(0, 10, 4)
    assert trial_tag(7) == "hpo07" and ":" not in trial_tag(123)


def test_objective_ranks_feasible_above_infeasible() -> None:
    assert objective_value(0.30, 0.93) == pytest.approx(0.30)
    near = objective_value(0.35, 0.62)  # untuned 1.5B v2: 30 pp short of 0.92
    lazy = objective_value(0.02, 0.95)  # copies its input: 23 pp short of 25% savings
    worse = objective_value(0.125, 0.71)
    assert near == pytest.approx(-1.30) and lazy == pytest.approx(-1.23)
    assert objective_value(0.26, 0.92) > lazy > near > worse
    assert all(v < 0 for v in (near, lazy, worse))


def test_calibration_and_estimate() -> None:
    rng = np.random.default_rng(0)
    scores = rng.uniform(0, 1, 400)
    preserved = rng.uniform(0, 1, 400) > scores  # higher P(degraded) -> less often preserved
    cal = fit_calibration(scores.tolist(), preserved.tolist())
    assert cal.b < 0
    assert cal([0.0])[0] > cal([1.0])[0]
    rows = [
        {"category": "code", "fell_back": True},
        {"category": "code", "fell_back": False},
        {"category": "fact", "fell_back": False},
    ]
    est = estimate_preservation(rows, [0.9, 0.0, 1.0], Calibration(0.0, -10.0))
    p0, p1 = 1 / (1 + np.exp(0.0)), 1 / (1 + np.exp(10.0))
    assert est["by_category"]["code"] == pytest.approx((1.0 + p0) / 2)
    assert est["preserved"] == pytest.approx((1.0 + p0 + p1) / 3)
    assert est["min_category"] == pytest.approx(p1)
    with pytest.raises(ValueError):
        fit_calibration([0.1, 0.2], [True, True])
    with pytest.raises(ValueError):
        estimate_preservation(rows, [0.1], cal)


def _rung(tag: str, n: int, rng: np.random.Generator, shift: float) -> tuple[list[dict], list]:
    rows, labels = [], []
    for i in range(n):
        fb = bool(rng.uniform() < 0.2)
        rows.append(
            {
                "id": f"p{i}",
                "category": "code" if i % 2 else "fact",
                "prompt": f"prompt {i}",
                "rewrite": f"prompt {i}" if fb else f"p {i}",
                "fell_back": fb,
                "savings": 0.0 if fb else float(rng.uniform(0.1, 0.5)),
                "oracle_savings": 0.2,
                "latency_s": 0.1,
            }
        )
    scores = [0.0 if r["fell_back"] else float(np.clip(rng.uniform() + shift, 0, 1)) for r in rows]
    for r, s in zip(rows, scores, strict=True):
        for t in TARGETS:
            ok = r["fell_back"] or bool(rng.uniform() > s)
            labels.append(
                {
                    "candidate_id": f"{r['id']}:{tag}",
                    "target": t,
                    "label": "preserved" if ok else "degraded",
                }
            )
    return [rows, scores], labels


def test_proxy_agreement_leave_one_rung_out() -> None:
    rng = np.random.default_rng(1)
    rungs, scores, labels = {}, {}, []
    for tag, shift in (("a", -0.3), ("b", 0.0), ("c", 0.3)):
        (rows, sc), lab = _rung(tag, 120, rng, shift)
        rungs[tag], scores[tag] = rows, sc
        labels += lab
    res = proxy_agreement(rungs, labels, scores, threshold=0.5)
    assert set(res["rungs"]) == {"a", "b", "c"}
    assert res["calibration"]["b"] < 0 and res["gate_auc_degraded"] > 0.6
    assert res["loro_mae"] < 0.1 and res["loro_spearman"] == pytest.approx(1.0)
    a = res["rungs"]["a"]
    assert a["judged_preserved"] > res["rungs"]["c"]["judged_preserved"]
    assert 0 <= a["gate_pass_share"] <= 1 and a["gated_preserved"] >= a["judged_preserved"]


def test_stage2_choice_and_verdict() -> None:
    res = {
        "hpo01": {"median_savings": 0.30, "preserved": 0.93, "c8_3": True},
        "hpo02": {"median_savings": 0.40, "preserved": 0.80, "c8_3": False},
        "hpo03": {"median_savings": 0.27, "preserved": 0.95, "c8_3": True},
    }
    assert choose_trial(res) == ("hpo01", True)
    low = {k: {**v, "preserved": v["preserved"] - 0.2} for k, v in res.items()}
    assert choose_trial(low) == ("hpo03", False)
    untuned = {"median_savings": 0.35, "preserved": 0.57, "c8_3": False}
    v = c10_3_verdict(res["hpo01"], untuned)
    assert v["verdict"] == "pass" and v["savings_gain_vs_untuned"] == pytest.approx(-0.05)
    passing = {**untuned, "median_savings": 0.295, "c8_3": True}
    assert c10_3_verdict(res["hpo01"], passing)["verdict"] == "plateau"  # +0.5 pp < 1 pp
    assert c10_3_verdict({**res["hpo01"], "median_savings": 0.31}, passing)["verdict"] == "pass"
    assert c10_3_verdict(res["hpo02"], untuned)["verdict"] == "plateau"
    trials = [
        {"trial": 0, "state": "COMPLETE", "value": -1.2, "median_savings": 0.3},
        {"trial": 1, "state": "PRUNED", "value": None},
        {"trial": 2, "state": "COMPLETE", "value": -1.1, "median_savings": 0.2},
    ]
    assert [t["trial"] for t in select_topk(trials, 3)] == [2, 0]
    recs = candidate_records([{"id": "x:y", "rewrite": "r"}], "hpo02")
    assert recs == [
        {"prompt_id": "x:y", "candidate_id": "x:y:hpo02", "method": "hpo02", "text": "r"}
    ]


def _fake_generator(calls: list[int]) -> Any:
    def gen(model_dir: Path, cfg: Any, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        calls.append(1)
        assert (model_dir / "loss.json").exists()
        out = []
        for r in rows:
            words = r["prompt"].split()
            text = " ".join(words[: max(1, len(words) // 2)])
            out.append(
                {
                    "id": r["id"],
                    "category": r["category"],
                    "prompt": r["prompt"],
                    "raw_rewrite": text,
                    "rewrite": text,
                    "fell_back": False,
                    "savings": 1 - len(text) / len(r["prompt"]),
                    "oracle_savings": 0.2,
                    "latency_s": 0.0,
                    "model": str(model_dir),
                }
            )
        return out

    return gen


def _labels(rows: list[dict[str, Any]], tag: str, label: str) -> list[dict[str, Any]]:
    return [
        {"candidate_id": f"{r['id']}:{tag}", "target": t, "label": label}
        for r in rows
        for t in TARGETS
    ]


def test_tiny_study_prunes_and_finalizes(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("optuna")
    data = tmp_path / "data"
    data.mkdir()
    train, val = data / "train.jsonl", data / "val.jsonl"
    shutil.copy(FIXTURE, train)
    shutil.copy(FIXTURE, val)
    log = DataAccessLog(tmp_path / "out" / "data_access.jsonl")
    val_rows = log.read_jsonl(val)
    s = HpoSettings(
        experiment="sft_tiny",
        base="tiny",
        hardware="cpu",
        train=train,
        val=val,
        n_train=len(log.read_jsonl(train)),
        batch_size=4,
        val_rows=val_rows,
        calibration=Calibration(2.0, -3.0),
        work=tmp_path / "work",
        eval_every=2,
        gate_threshold=0.5,
    )
    calls: list[int] = []
    good = {"rank": 8, "alpha_mult": 1, "lr": 5e-4, "warmup_frac": 0.0, "epochs": 2}
    slow = {**good, "lr": 5e-5}  # same start, 10x smaller steps: higher val loss at step 2
    summary = run_stage1(
        s,
        2,
        pruner_startup=1,
        pruner_warmup=0,
        enqueue=[good, slow],
        generator=_fake_generator(calls),
        scorer=lambda rows: [0.3] * len(rows),
        data_log=log,
    )
    t0, t1 = summary["trials"]
    assert (t0["state"], t1["state"]) == ("COMPLETE", "PRUNED")
    assert calls == [1]  # the pruned trial never reached generation
    assert t1["pruned_at_step"] == 2 and t1["value"] is None
    assert t0["steps"] == 10 and [s for s, _ in t0["val_loss_curve"]] == [2, 4, 6, 8, 10]
    assert t0["intermediate"]["2"] > t1["intermediate"]["2"]  # -loss: trial 0 learned faster
    assert t0["value"] == pytest.approx(objective_value(t0["median_savings"], t0["preserved_est"]))
    assert summary["data_access_ok"] is True and summary["best_value"] == t0["value"]
    assert (Path(t0["model_dir"]) / "trial.json").exists()

    out, rw = tmp_path / "out", tmp_path / "rewrites"
    (out / "stage1.json").write_text(json.dumps(summary), encoding="utf-8")
    topk = write_candidates(summary, 3, rw, out)
    assert topk["k"] == 1 and topk["trials"][0]["tag"] == "hpo00"
    cands = [json.loads(x) for x in (rw / "val_candidates_hpo.jsonl").read_text().splitlines()]
    assert len(cands) == len(val_rows) and cands[0]["candidate_id"].endswith(":hpo00")

    rows = [json.loads(x) for x in (rw / "hpo00.val.jsonl").read_text().splitlines()]
    labels = tmp_path / "val_labels_hpo.jsonl"
    for label, verdict, qualifies in (("preserved", "pass", True), ("degraded", "plateau", False)):
        labels.write_text(
            "\n".join(json.dumps(x) for x in _labels(rows, "hpo00", label)), encoding="utf-8"
        )
        res = finalize(
            out,
            labels,
            baseline_json=tmp_path / "missing.json",
            report=tmp_path / "p10_rewriter_hpo.md",
            register=False,
        )
        assert res["chosen"] == "hpo00" and res["chosen_meets_preservation"] is qualifies
        assert res["c10_3"]["verdict"] == verdict
        md = (tmp_path / "p10_rewriter_hpo.md").read_text(encoding="utf-8")
        assert f"C10.3: {verdict}" in md and "| 1 | PRUNED @ 2 |" in md
        assert json.loads((tmp_path / "p10_rewriter_hpo.json").read_text())["chosen"] == "hpo00"
