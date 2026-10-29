"""Row 9.2.4 judge verification and C9.1 / C9.2 helpers (synthetic labels, no LLM, no gate)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from terser.models.rl.env import RewriteEnv
from terser.models.rl.scratch_qlearning import QLearner
from terser.models.rl.verify import (
    FIXED_METHOD,
    RL_METHOD,
    best_fixed_set,
    build_candidates,
    c9_2_verdict,
    choose_prompts,
    convergence_md,
    fixed_set_returns,
    greedy_rollout,
    main,
    method_metrics,
    op_subsets,
    q_change,
)

FIXTURES = Path(__file__).resolve().parents[1] / "data" / "samples" / "casual_prompts.jsonl"
QWEN, PHI = "Qwen/Qwen2.5-7B-Instruct", "microsoft/Phi-3.5-mini-instruct"


def _words(t: str) -> int:
    return len(t.split())


def _episodes() -> list[tuple[str, str, list[str]]]:
    rows = [json.loads(x) for x in FIXTURES.read_text(encoding="utf-8").splitlines()][:5]
    return [(r["prompt"], r["category"], r["constraints"]) for r in rows]


def test_choose_prompts_deterministic_and_filtered() -> None:
    rows = [{"id": str(i), "stand_in": i % 7 == 0, "rejected": i % 11 == 0} for i in range(100)]
    a, b = choose_prompts(rows, 30, seed=0), choose_prompts(rows, 30, seed=0)
    assert a == b and len(a) == 30
    assert not any(r["stand_in"] or r["rejected"] for r in a)
    ids = [int(r["id"]) for r in a]
    assert ids == sorted(ids) and len(set(ids)) == 30
    assert choose_prompts(rows, 30, seed=1) != a
    assert len(choose_prompts(rows, 10_000)) == sum(
        not (r["stand_in"] or r["rejected"]) for r in rows
    )


def test_op_subsets() -> None:
    s = op_subsets()
    assert len(s) == 127 and len(set(s)) == 127
    assert all(x == "".join(sorted(x)) for x in s)
    assert "CDFHILS" in s and "F" in s


def test_fixed_set_ranking_uses_gate_reward() -> None:
    # Gate flags any draft under 70% of the original's characters as degraded.
    env = RewriteEnv(count=_words, gate=lambda o, d: float(len(d) < 0.7 * len(o)))
    eps = _episodes()
    ranking = fixed_set_returns(env, eps, op_subsets())
    assert len(ranking) == 127
    means = [r["mean_return"] for r in ranking]
    assert means == sorted(means, reverse=True)
    best = best_fixed_set(ranking)
    assert best == ranking[0]["ops"]
    # Tie-break: among equal returns the shorter set comes first.
    ties = [r for r in ranking if r["mean_return"] == ranking[0]["mean_return"]]
    assert [len(r["ops"]) for r in ties] == sorted(len(r["ops"]) for r in ties)


def test_greedy_rollout_matches_policy_rollout() -> None:
    env = RewriteEnv(count=_words, gate=lambda o, d: 0.0)
    eps = _episodes()
    ql = QLearner(alpha="visits", gamma=0.9, decay=0.99, seed=0).fit(env, eps, 500)
    for ep in eps:
        seq, ops, draft, unseen = greedy_rollout(ql, env, ep)
        assert (seq, draft) == ql.policy_rollout(env, ep)
        assert unseen == 0 and set(ops) <= set("FHDILCS")
    # A category never trained on: every decision is in an unseen state, F is picked first.
    seq, ops, _, unseen = greedy_rollout(QLearner(), env, (eps[0][0], "unseen", []))
    assert seq == "FHDILCS" and ops == "CDFHILS" and unseen == 7


def test_q_change() -> None:
    assert q_change({}, {}) == 0.0
    assert q_change({("s", "a"): 1.0}, {("s", "a"): 1.0005}) == pytest.approx(5e-4)
    assert q_change({("s", "a"): 1.0}, {("s", "b"): -2.0}) == pytest.approx(2.0)


def _synthetic() -> tuple[list[dict], list[dict], list[dict]]:
    prompts = [
        {"id": "p1", "category": "code", "prompt": "a b c d e f g h i j"},
        {"id": "p2", "category": "code", "prompt": "a b c d e f g h i j"},
        {"id": "p3", "category": "fact", "prompt": "a b c d e f g h i j"},
        {"id": "p4", "category": "fact", "prompt": "a b c d e f g h i j"},
    ]
    rl = {"p1": "a b c d e", "p2": "a b c d e f g h", "p3": "a b c d e f g h i j", "p4": "a b"}
    fx = {"p1": "a b c d e f g h i", "p2": "a b c d e f g h i", "p3": "a b c", "p4": "a b c d"}
    cands = build_candidates(prompts, {RL_METHOD: rl, FIXED_METHOD: fx})
    preserved = {
        # keys are (prompt id, method, target)
        ("p1", RL_METHOD, QWEN): True,
        ("p1", RL_METHOD, PHI): True,
        ("p2", RL_METHOD, QWEN): True,
        ("p2", RL_METHOD, PHI): False,
        # p3 rl is unchanged: no label written -> preserved by construction
        ("p4", RL_METHOD, QWEN): False,
        ("p4", RL_METHOD, PHI): False,
        ("p1", FIXED_METHOD, QWEN): True,
        ("p1", FIXED_METHOD, PHI): True,
        ("p2", FIXED_METHOD, QWEN): True,
        ("p2", FIXED_METHOD, PHI): True,
        ("p3", FIXED_METHOD, QWEN): False,
        ("p3", FIXED_METHOD, PHI): True,
        ("p4", FIXED_METHOD, QWEN): True,
        # p4 fixed on Phi missing -> degraded, reported as missing
    }
    cat = {p["id"]: p["category"] for p in prompts}
    labels = [
        {
            "schema_version": "label_v1",
            "prompt_id": pid,
            "candidate_id": f"{pid}:{m}",
            "target_model": t,
            "target": t,
            "category": cat[pid],
            "label": "preserved" if ok else "degraded",
        }
        for (pid, m, t), ok in preserved.items()
    ]
    labels.append({**labels[0], "candidate_id": "p1:other", "label": "degraded"})  # ignored
    return prompts, cands, labels


def test_build_candidates_format() -> None:
    _prompts, cands, _ = _synthetic()
    assert len(cands) == 8
    assert all(set(c) == {"prompt_id", "candidate_id", "method", "text"} for c in cands)
    assert {c["candidate_id"] for c in cands if c["prompt_id"] == "p1"} == {
        "p1:rl_greedy",
        "p1:fixed_order",
    }


def test_method_metrics_and_verdict() -> None:
    prompts, cands, labels = _synthetic()
    m = method_metrics(prompts, cands, labels, _words)
    rl, fx = m[RL_METHOD], m[FIXED_METHOD]
    assert rl["targets"] == sorted([QWEN, PHI])
    # RL savings: 0.5, 0.2, 0.0, 0.8 -> median 0.35; fixed: 0.1, 0.1, 0.7, 0.6 -> 0.35
    assert rl["median_savings"] == pytest.approx(0.35)
    assert fx["median_savings"] == pytest.approx(0.35)
    assert rl["unchanged"] == 1 and rl["missing_labels"] == 0
    assert fx["missing_labels"] == 1
    assert rl["preserved"] == pytest.approx(5 / 8)
    assert fx["preserved"] == pytest.approx(6 / 8)
    assert rl["preserved_by_target"][QWEN] == pytest.approx(3 / 4)
    assert rl["preserved_by_target"][PHI] == pytest.approx(2 / 4)
    assert rl["preserved_by_category"] == {"code": pytest.approx(3 / 4), "fact": 0.5}
    assert fx["preserved_by_category_target"]["fact"] == {QWEN: 0.5, PHI: 0.5}
    # RL "if preserved" savings: 0.5, 0, 0, 0 -> median 0
    assert rl["median_savings_if_preserved"] == pytest.approx(0.0)
    v = c9_2_verdict(rl, fx)
    assert v["preservation_diff"] == pytest.approx(-1 / 8)
    assert not v["equal_preservation"] and v["savings_ge_fixed"] and not v["pass"]
    ok = c9_2_verdict({"preserved": 0.90, "median_savings": 0.4}, fx | {"preserved": 0.91})
    assert ok["equal_preservation"] and ok["pass"]
    low = c9_2_verdict({"preserved": 0.95, "median_savings": 0.3}, fx)
    assert low["equal_preservation"] and low["pass"]  # more preservation, less savings


def _run(name: str, n: int, value: float, ok: bool) -> dict:
    return {
        "run": name,
        "alpha": "visits",
        "gamma": 0.99,
        "decay": 0.9995,
        "eps_min": 0.0,
        "episodes": n,
        "max_update_last_1000": value,
        "net_change_last_1000": value,
        "episodes_with_update_ge_target_last_1000": 0 if ok else 3,
        "q_entries": 10,
        "trailing": [{"episode": n, "max_dQ_window": value}],
        "pass": ok,
    }


def test_convergence_md_and_from_json(tmp_path: Path) -> None:
    runs = [_run("settled", 30000, 0.16, False), _run("best", 30000, 19.5, False)]
    runs.append(_run("settled, extended", 200000, 0.05, False))
    md = convergence_md(runs, 300)
    assert "fail**" in md and "Reading" in md and "| 200,000 | 0.05 |" in md
    assert "Reading" not in convergence_md([_run("s", 10, 1e-4, True)] * 3, 300)
    out = tmp_path / "conv.md"
    out.with_suffix(".json").write_text(json.dumps(runs), encoding="utf-8")
    assert main(["convergence", "--from-json", "--out", str(out)]) == 0
    assert out.read_text(encoding="utf-8") == md


def test_report_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import terser.tokenizers.count as count_mod

    monkeypatch.setattr(count_mod, "count_tokens", lambda text, target: _words(text))
    prompts, cands, labels = _synthetic()
    paths = {}
    for name, rows in (("p", prompts), ("c", cands), ("l", labels)):
        paths[name] = tmp_path / f"{name}.jsonl"
        paths[name].write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({"fixed_order": "FH", "summary": {}}), encoding="utf-8")
    out = tmp_path / "rep.md"
    rc = main(
        [
            "report",
            "--labels",
            str(paths["l"]),
            "--prompts",
            str(paths["p"]),
            "--candidates",
            str(paths["c"]),
            "--meta",
            str(meta),
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    text = out.read_text(encoding="utf-8")
    assert "C9.2: fail" in text and "fixed order `FH`" in text and "Missing labels" in text
    payload = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["c9_2"]["pass"] is False and payload["fixed_order"] == "FH"
    assert set(payload["methods"]) == {RL_METHOD, FIXED_METHOD}
