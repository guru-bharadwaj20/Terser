"""Judge verification of the tabular operator policy (row **9.2.4**; gates **C9.1**, **C9.2**).

Three subcommands:

``prep`` (CPU, no LLM calls)
    1. Retrains the 9.2.3 policy: the best learner of the 9.2.2 sweep (alpha = 1 / N(s, a),
       gamma 0.9, epsilon 1.0 * 0.999^t floored at 0.05, seed 0, 30,000 episodes) on the same
       300 training prompts and the same LightGBM quality-gate reward as
       :mod:`terser.phase9.qlearning_study`. Everything is deterministic and on CPU, so this
       reproduces the 9.2.3 Q-table; the greedy train-prompt sequences are stored in the meta
       file as a check against ``reports/p9_policy.md``.
    2. Chooses the **best fixed operator order** with a pre-judge rule that never sees val data:
       every non-empty operator subset (127; the environment applies a set in the canonical
       order ``C H F I D S L``, so a fixed order is a fixed subset) is run open-loop on the same
       300 training prompts and scored with the same reward the learner maximises (savings % -
       100 x gate-degraded, -0.01 per operator). The highest mean return wins (ties: fewer
       operators, then alphabetical).
    3. Draws 300 val prompts (seed 0, stand-in and rejected rows excluded; ``test.jsonl`` is
       never read) and writes them verbatim, the format :mod:`terser.phase2.respond` and
       :mod:`terser.phase2.label_stage` read. Writes two candidates per prompt: ``rl_greedy``
       (greedy rollout of the Q-table until ``STOP``) and ``fixed_order`` (the chosen subset).

``report`` (CPU)
    Joins judge labels (``label_v1`` from :mod:`terser.phase2.label_stage`) with the candidates:
    median savings on the Qwen2.5 tokenizer (``count_tokens(text, "qwen2.5")``, the counter the
    Phase 8 val report uses), preservation pooled over (prompt, target), per target and per
    category, and the C9.2 verdict.

    **C9.2 ("savings >= best fixed operator order at equal preservation").** Preservation is
    *equal* when the RL policy's pooled preservation is no more than
    :data:`PRESERVATION_TOLERANCE` (2 pp) below the fixed order's. C9.2 passes iff preservation
    is equal in that sense **and** the RL median savings are >= the fixed order's median
    savings. A candidate identical to its prompt is preserved by construction (identical
    outputs); any other candidate without a label for a target counts as degraded and is
    reported as missing.

``convergence`` (CPU)
    C9.1 ("Q-table change < 1e-3 over the last 1,000 episodes") on reruns of the sweep's
    settled learner and the 9.2.3 learner: the per-update statistic the study reports
    (max |dQ| of any single update in the last 1,000 episodes) and the net table change
    max_{s,a} |Q_T(s,a) - Q_{T-1000}(s,a)|, plus an extended run of the settled learner.

CLI::

    python -m terser.models.rl.verify prep
    python -m terser.models.rl.verify report --labels data/interim/rewrites/val_labels_rl.jsonl
    python -m terser.models.rl.verify convergence
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from terser.models.rl.env import OPERATOR_ACTIONS, STOP, RewriteEnv
from terser.models.rl.scratch_qlearning import QLearner

__all__ = [
    "BEST_HP",
    "FIXED_METHOD",
    "PRESERVATION_TOLERANCE",
    "RL_METHOD",
    "SETTLED_HP",
    "best_fixed_set",
    "build_candidates",
    "c9_2_verdict",
    "choose_prompts",
    "convergence_md",
    "fixed_set_returns",
    "greedy_rollout",
    "main",
    "method_metrics",
    "op_subsets",
    "q_change",
]

RL_METHOD = "rl_greedy"
FIXED_METHOD = "fixed_order"
METHODS = (RL_METHOD, FIXED_METHOD)
PRESERVATION_TOLERANCE = 0.02
CONVERGENCE_TARGET = 1e-3
WINDOW = 1000
# 9.2.2 sweep: highest return over the last 1,000 episodes (the 9.2.3 policy) ...
BEST_HP: dict[str, Any] = {"alpha": "visits", "gamma": 0.9, "decay": 0.999, "eps_min": 0.05}
# ... and the most settled table (smallest max |dQ| over the last 1,000 episodes).
SETTLED_HP: dict[str, Any] = {"alpha": "visits", "gamma": 0.99, "decay": 0.9995, "eps_min": 0.0}

REWRITES = Path("data/interim/rewrites")
Episode = tuple[str, str, Sequence[str]]


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def _usable(r: Mapping[str, Any]) -> bool:
    return r.get("stand_in") not in (True, "True", "true") and not r.get("rejected")


def choose_prompts(rows: Sequence[dict[str, Any]], n: int, seed: int = 0) -> list[dict[str, Any]]:
    """``n`` usable rows drawn without replacement (``default_rng(seed)``), in file order."""
    pool = [r for r in rows if _usable(r)]
    pick = np.random.default_rng(seed).choice(len(pool), min(n, len(pool)), replace=False)
    return [pool[i] for i in sorted(int(i) for i in pick)]


def train_prompts(rows: Sequence[dict[str, Any]], n: int, seed: int = 0) -> list[dict[str, Any]]:
    """The training draw of :mod:`terser.phase9.qlearning_study` (same rng call, same order)."""
    pick = np.random.default_rng(seed).choice(len(rows), min(n, len(rows)), replace=False)
    return [rows[i] for i in pick]


def episode_of(record: Mapping[str, Any]) -> Episode:
    """``(prompt, category, constraints)`` exactly as the 9.2 study builds it."""
    from terser.features.alignment import tokenize
    from terser.features.lexicons import is_constraint

    prompt = str(record["prompt"])
    cons = sorted({t for t in tokenize(prompt) if is_constraint(t)})
    return prompt, str(record["category"]), cons


def op_subsets() -> list[str]:
    """Every non-empty operator subset as a sorted code string (127 for 7 operators)."""
    out = []
    for k in range(1, len(OPERATOR_ACTIONS) + 1):
        out += ["".join(sorted(c)) for c in itertools.combinations(OPERATOR_ACTIONS, k)]
    return out


def run_fixed(env: RewriteEnv, episode: Episode, ops: str) -> tuple[float, str]:
    """Open-loop episode applying ``ops`` then ``STOP``: ``(return, draft)``."""
    env.reset(*episode)
    total, done = 0.0, False
    for a in ops:
        _, r, done = env.step(a)
        total += r
    if not done:
        _, r, _ = env.step(STOP)
        total += r
    return total, env.draft


def fixed_set_returns(
    env: RewriteEnv, episodes: Sequence[Episode], sets: Sequence[str]
) -> list[dict[str, Any]]:
    """Mean env return per fixed subset, best first (ties: fewer operators, alphabetical)."""
    rows = []
    for ops in sets:
        rets = [run_fixed(env, ep, ops)[0] for ep in episodes]
        rows.append({"ops": ops, "mean_return": float(np.mean(rets)), "n": len(rets)})
    return sorted(rows, key=lambda r: (-r["mean_return"], len(r["ops"]), r["ops"]))


def best_fixed_set(ranking: Sequence[Mapping[str, Any]]) -> str:
    return str(ranking[0]["ops"])


def greedy_rollout(ql: QLearner, env: RewriteEnv, episode: Episode) -> tuple[str, str, str, int]:
    """Greedy rollout identical to :meth:`QLearner.policy_rollout`.

    Returns ``(sequence, operator set, draft, unseen)`` where ``unseen`` counts decisions taken
    in a state none of whose legal actions was ever updated in training (all Q = 0, so the tie
    break picks the first legal operator).
    """
    s = env.reset(*episode)
    seq: list[str] = []
    unseen, done = 0, False
    while not done:
        legal = env.legal_actions()
        unseen += not any(ql.visits.get((s, a), 0) for a in legal)
        a = ql.greedy(s, legal)
        seq.append(a)
        s, _, done = env.step(a)
    return "".join(seq), str(s[4]), env.draft, unseen


def build_candidates(
    prompts: Sequence[Mapping[str, Any]], texts: Mapping[str, Mapping[str, str]]
) -> list[dict[str, str]]:
    """Candidate records ``{prompt_id, candidate_id, method, text}``; ``texts[method][pid]``."""
    out = []
    for method in METHODS:
        for p in prompts:
            pid = str(p["id"])
            out.append(
                {
                    "prompt_id": pid,
                    "candidate_id": f"{pid}:{method}",
                    "method": method,
                    "text": texts[method][pid],
                }
            )
    return out


def savings(original: str, text: str, count: Callable[[str], int]) -> float:
    n0 = count(original)
    return 1.0 - count(text) / n0 if n0 else 0.0


def method_metrics(
    prompts: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
    count: Callable[[str], int],
) -> dict[str, dict[str, Any]]:
    """Savings and judge preservation per method (pooled over the label targets)."""
    by_pid = {str(p["id"]): p for p in prompts}
    cand = {str(c["candidate_id"]): c for c in candidates if str(c["prompt_id"]) in by_pid}
    ok: dict[tuple[str, str], bool] = {}
    for r in labels:
        cid = str(r.get("candidate_id", ""))
        if cid in cand:
            target = str(r.get("target_model") or r["target"])
            ok[(cid, target)] = r["label"] == "preserved"
    targets = sorted({t for _, t in ok})
    out: dict[str, dict[str, Any]] = {}
    for method in sorted({str(c["method"]) for c in cand.values()}):
        rows = [c for c in cand.values() if c["method"] == method]
        sav, eff = [], []
        flat: list[bool] = []
        by_t: dict[str, list[bool]] = defaultdict(list)
        by_c: dict[str, list[bool]] = defaultdict(list)
        by_ct: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
        missing = unchanged = 0
        for c in rows:
            p = by_pid[str(c["prompt_id"])]
            same = str(c["text"]) == str(p["prompt"])
            unchanged += same
            s = savings(str(p["prompt"]), str(c["text"]), count)
            sav.append(s)
            all_ok = True
            for t in targets:
                key = (str(c["candidate_id"]), t)
                if key in ok:
                    v = ok[key]
                else:
                    v = same
                    missing += not same
                all_ok &= v
                flat.append(v)
                by_t[t].append(v)
                by_c[str(p["category"])].append(v)
                by_ct[str(p["category"])][t].append(v)
            eff.append(s if all_ok else 0.0)

        def rate(xs: Sequence[bool]) -> float:
            return sum(xs) / len(xs) if xs else float("nan")

        out[method] = {
            "n_prompts": len(rows),
            "targets": targets,
            "median_savings": statistics.median(sav) if sav else float("nan"),
            "mean_savings": statistics.fmean(sav) if sav else float("nan"),
            "median_savings_if_preserved": statistics.median(eff) if eff else float("nan"),
            "unchanged": unchanged,
            "missing_labels": missing,
            "preserved": rate(flat),
            "preserved_by_target": {t: rate(v) for t, v in sorted(by_t.items())},
            "preserved_by_category": {c: rate(v) for c, v in sorted(by_c.items())},
            "preserved_by_category_target": {
                c: {t: rate(v) for t, v in sorted(d.items())} for c, d in sorted(by_ct.items())
            },
            "n_by_category": dict(
                sorted(Counter(str(by_pid[str(c["prompt_id"])]["category"]) for c in rows).items())
            ),
        }
    return out


def c9_2_verdict(
    rl: Mapping[str, Any], fixed: Mapping[str, Any], tol: float = PRESERVATION_TOLERANCE
) -> dict[str, Any]:
    """C9.2: RL median savings >= fixed, with RL preservation >= fixed - ``tol``."""
    d_pres = float(rl["preserved"]) - float(fixed["preserved"])
    d_sav = float(rl["median_savings"]) - float(fixed["median_savings"])
    equal = d_pres >= -tol
    return {
        "tolerance": tol,
        "preservation_diff": d_pres,
        "median_savings_diff": d_sav,
        "equal_preservation": equal,
        "savings_ge_fixed": d_sav >= 0,
        "pass": bool(equal and d_sav >= 0),
    }


def q_change(
    before: Mapping[tuple[Hashable, str], float], after: Mapping[tuple[Hashable, str], float]
) -> float:
    """``max_{s,a} |after - before|`` over the union of keys (missing entries are 0)."""
    keys = set(before) | set(after)
    return max((abs(after.get(k, 0.0) - before.get(k, 0.0)) for k in keys), default=0.0)


# --------------------------------------------------------------------------------------------
# CLI


def _pct(x: float) -> str:
    return "n/a" if x != x else f"{x:.1%}"


def _short(target: str) -> str:
    return target.split("/")[-1]


def _gate_env(args: argparse.Namespace) -> tuple[RewriteEnv, Callable[[str], int], dict[str, str]]:
    from terser.operators.registry import apply_ops
    from terser.phase9.qlearning_study import build_gate

    gate, cost, category_of = build_gate(args.table, args.target)
    env = RewriteEnv(
        count=cost, gate=gate, target=args.target, apply=lru_cache(maxsize=None)(apply_ops)
    )
    return env, cost, category_of


def _train_episodes(args: argparse.Namespace, category_of: dict[str, str]) -> list[Episode]:
    rows = train_prompts(_jsonl(args.train), args.n_train, args.seed)
    for r in rows:
        category_of[str(r["prompt"])] = str(r["category"])
    return [episode_of(r) for r in rows]


def _learner(hp: Mapping[str, Any], seed: int) -> QLearner:
    return QLearner(
        alpha=hp["alpha"], gamma=hp["gamma"], decay=hp["decay"], eps_min=hp["eps_min"], seed=seed
    )


def cmd_prep(args: argparse.Namespace) -> int:
    if "test" in args.val.name:
        raise SystemExit("refusing to read a test split")
    env, cost, category_of = _gate_env(args)
    episodes = _train_episodes(args, category_of)
    ql = _learner(BEST_HP, args.seed).fit(env, episodes, args.episodes)
    train_seqs: dict[str, Counter[str]] = defaultdict(Counter)
    for ep in episodes:
        train_seqs[ep[1]][greedy_rollout(ql, env, ep)[0]] += 1
    print(json.dumps({"trained": args.episodes, "q_entries": len(ql.q)}), flush=True)

    ranking = fixed_set_returns(env, episodes, op_subsets())
    fixed = best_fixed_set(ranking)
    print(json.dumps({"fixed_order": fixed, "top5": ranking[:5]}), flush=True)

    val = choose_prompts(_jsonl(args.val), args.n_val, args.seed)
    texts: dict[str, dict[str, str]] = {m: {} for m in METHODS}
    per_prompt = []
    for p in val:
        pid, prompt = str(p["id"]), str(p["prompt"])
        category_of[prompt] = str(p["category"])
        ep = episode_of(p)
        seq, ops, draft, unseen = greedy_rollout(ql, env, ep)
        _, fdraft = run_fixed(env, ep, fixed)
        texts[RL_METHOD][pid], texts[FIXED_METHOD][pid] = draft, fdraft
        per_prompt.append(
            {
                "prompt_id": pid,
                "category": p["category"],
                "rl_sequence": seq,
                "rl_ops": ops,
                "rl_unseen_decisions": unseen,
                "rl_savings": savings(prompt, draft, cost),
                "fixed_savings": savings(prompt, fdraft, cost),
                "rl_gate_p": env.gate(prompt, draft),
                "fixed_gate_p": env.gate(prompt, fdraft),
            }
        )
    n_p = _write_jsonl(args.prompts_out, val)
    n_c = _write_jsonl(args.candidates_out, build_candidates(val, texts))

    def summ(prefix: str) -> dict[str, Any]:
        sv = [r[f"{prefix}_savings"] for r in per_prompt]
        gp = [r[f"{prefix}_gate_p"] for r in per_prompt]
        unchanged = sum(
            texts[RL_METHOD if prefix == "rl" else FIXED_METHOD][str(p["id"])] == p["prompt"]
            for p in val
        )
        return {
            "median_savings": statistics.median(sv),
            "mean_savings": statistics.fmean(sv),
            "gate_degraded": float(np.mean([g >= env.threshold for g in gp])),
            "unchanged": unchanged,
        }

    summary = {
        "val_prompts": n_p,
        "candidates": n_c,
        "by_category": dict(sorted(Counter(str(p["category"]) for p in val).items())),
        RL_METHOD: summ("rl"),
        FIXED_METHOD: summ("fixed"),
        "rl_prompts_with_unseen_state": sum(r["rl_unseen_decisions"] > 0 for r in per_prompt),
        "rl_sequences": Counter(r["rl_sequence"] for r in per_prompt).most_common(10),
    }
    meta = {
        "row": "9.2.4",
        "learner": {**BEST_HP, "eps0": 1.0, "seed": args.seed, "episodes": args.episodes},
        "train": {"split": str(args.train), "n": len(episodes), "seed": args.seed},
        "val": {"split": str(args.val), "n": n_p, "seed": args.seed},
        "target_tokenizer": args.target,
        "fixed_rule": "max mean env return (gate reward) on the train prompts over all "
        "non-empty operator subsets; ties: fewer operators, then alphabetical",
        "fixed_order": fixed,
        "fixed_ranking_top10": ranking[:10],
        "train_greedy_sequences": {c: v.most_common(3) for c, v in sorted(train_seqs.items())},
        "summary": summary,
        "per_prompt": per_prompt,
    }
    args.meta_out.parent.mkdir(parents=True, exist_ok=True)
    args.meta_out.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


def _report_md(
    results: Mapping[str, Mapping[str, Any]],
    verdict: Mapping[str, Any] | None,
    meta: Mapping[str, Any] | None,
    labels_path: Path,
) -> str:
    fixed_ops = meta.get("fixed_order", "?") if meta else "?"
    names = {RL_METHOD: "RL greedy (9.2.3 Q-table)", FIXED_METHOD: f"fixed order `{fixed_ops}`"}
    targets = sorted({t for m in results.values() for t in m["targets"]})
    lines = [
        "# Judge verification of the tabular policy (row 9.2.4, gate C9.2)",
        "",
        f"300 val prompts (seed 0), labels `{labels_path.as_posix()}` "
        "(Qwen2.5-32B-Instruct-AWQ pairwise judge, label_v1). Savings = 1 - tokens(rewrite) / "
        "tokens(original) on the Qwen2.5 tokenizer; preservation pooled over (prompt, target) "
        f"for {', '.join(_short(t) for t in targets) or 'no targets'}.",
        "",
        "Methods: `rl_greedy` is the greedy rollout of the 9.2.3 learner (alpha = 1 / N, gamma "
        "0.9, retrained deterministically on CPU) until `STOP`; `fixed_order` is the single "
        "operator subset with the best mean gate-reward return on the 300 training prompts "
        "(chosen before judging, no val labels).",
        "",
        "| method | n | median savings | mean savings | median savings if preserved | unchanged "
        "| preserved | " + " | ".join(f"preserved {_short(t)}" for t in targets) + " |",
        "|---" * (7 + len(targets)) + "|",
    ]
    for method, m in results.items():
        lines.append(
            f"| {names.get(method, method)} | {m['n_prompts']} | {_pct(m['median_savings'])} "
            f"| {_pct(m['mean_savings'])} | {_pct(m['median_savings_if_preserved'])} "
            f"| {m['unchanged']} | {_pct(m['preserved'])} | "
            + " | ".join(_pct(m["preserved_by_target"].get(t, float("nan"))) for t in targets)
            + " |"
        )
    cats = sorted({c for m in results.values() for c in m["preserved_by_category"]})
    lines += ["", "Preservation by category (pooled over targets; n prompts in brackets):", ""]
    lines += ["| method | " + " | ".join(cats) + " |", "|---" * (len(cats) + 1) + "|"]
    for method, m in results.items():
        pc, nc = m["preserved_by_category"], m["n_by_category"]
        lines.append(
            f"| {names.get(method, method)} | "
            + " | ".join(f"{_pct(pc.get(c, float('nan')))} ({nc.get(c, 0)})" for c in cats)
            + " |"
        )
    for t in targets:
        lines += ["", f"Preservation by category on {_short(t)}:", ""]
        lines += ["| method | " + " | ".join(cats) + " |", "|---" * (len(cats) + 1) + "|"]
        for method, m in results.items():
            pct = m["preserved_by_category_target"]
            lines.append(
                f"| {names.get(method, method)} | "
                + " | ".join(_pct(pct.get(c, {}).get(t, float("nan"))) for c in cats)
                + " |"
            )
    missing = {k: m["missing_labels"] for k, m in results.items() if m["missing_labels"]}
    if missing:
        lines += ["", f"Missing labels (counted as degraded): {missing}."]
    lines += ["", "## C9.2 Tabular policy value", ""]
    if verdict is None:
        lines.append("Both methods are needed for the verdict.")
    else:
        lines += [
            "Definition: *equal preservation* means the RL policy's pooled preservation is at "
            f"most {verdict['tolerance']:.0%} (absolute) below the fixed order's. C9.2 passes "
            "iff preservation is equal and the RL median savings are >= the fixed order's.",
            "",
            f"* Preservation RL - fixed: {verdict['preservation_diff'] * 100:+.1f} pp "
            f"(equal: **{'yes' if verdict['equal_preservation'] else 'no'}**)",
            f"* Median savings RL - fixed: {verdict['median_savings_diff'] * 100:+.1f} pp "
            f"(>= 0: **{'yes' if verdict['savings_ge_fixed'] else 'no'}**)",
            f"* **C9.2: {'pass' if verdict['pass'] else 'fail'}**",
        ]
    if meta:
        s = meta.get("summary", {})
        lines += [
            "",
            "Pre-judge context (from `prep`): gate-estimated degraded share RL "
            f"{_pct(s.get(RL_METHOD, {}).get('gate_degraded', float('nan')))}, fixed "
            f"{_pct(s.get(FIXED_METHOD, {}).get('gate_degraded', float('nan')))}; RL prompts "
            f"that reached a state unseen in training: {s.get('rl_prompts_with_unseen_state')} "
            "(all Q = 0 there, so the greedy tie break applies the next operator in `F H D I L "
            "C S` order). Fixed-order train mean return "
            f"{(meta.get('fixed_ranking_top10') or [{}])[0].get('mean_return', float('nan')):.3f}"
            " (rule: best mean gate-reward return over all 127 non-empty operator subsets).",
        ]
    return "\n".join(lines) + "\n"


def cmd_report(args: argparse.Namespace) -> int:
    from terser.tokenizers.count import count_tokens

    prompts = _jsonl(args.prompts)
    candidates = _jsonl(args.candidates)
    labels = _jsonl(args.labels)
    results = method_metrics(prompts, candidates, labels, lambda t: count_tokens(t, args.target))
    verdict = (
        c9_2_verdict(results[RL_METHOD], results[FIXED_METHOD])
        if all(m in results for m in METHODS)
        else None
    )
    meta = json.loads(args.meta.read_text(encoding="utf-8")) if args.meta.exists() else None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(_report_md(results, verdict, meta, args.labels), encoding="utf-8")
    payload = {
        "row": "9.2.4",
        "labels": str(args.labels),
        "fixed_order": meta.get("fixed_order") if meta else None,
        "methods": results,
        "c9_2": verdict,
    }
    args.out.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n", "utf-8")
    print(json.dumps({"c9_2": verdict, **{k: v["preserved"] for k, v in results.items()}}))
    return 0


def _convergence_run(
    env: RewriteEnv, episodes: Sequence[Episode], hp: Mapping[str, Any], n: int, seed: int
) -> dict[str, Any]:
    ql = _learner(hp, seed)
    snaps: dict[int, dict[tuple[Hashable, str], float]] = {}
    marks = {n - WINDOW - 1}

    def cb(t: int, _: float) -> None:
        if t in marks:
            snaps[t] = dict(ql.q)

    ql.fit(env, episodes, n, callback=cb)
    per_update = float(np.max(ql.max_delta_[-WINDOW:]))
    net = q_change(snaps[n - WINDOW - 1], ql.q)
    # trailing-window per-update statistic every 10% of the run
    trail = [
        {"episode": e, "max_dQ_window": float(np.max(ql.max_delta_[e - WINDOW : e]))}
        for e in range(max(WINDOW, n // 10), n + 1, max(1, n // 10))
    ]
    return {
        **hp,
        "episodes": n,
        "max_update_last_1000": per_update,
        "net_change_last_1000": net,
        "episodes_with_update_ge_target_last_1000": int(
            sum(d >= CONVERGENCE_TARGET for d in ql.max_delta_[-WINDOW:])
        ),
        "q_entries": len(ql.visits),
        "trailing": trail,
        "pass": bool(max(per_update, net) < CONVERGENCE_TARGET),
    }


def convergence_md(runs: Sequence[Mapping[str, Any]], n_train: int) -> str:
    """Markdown for ``reports/p9_convergence.md``; ``runs[0]`` is the plan-budget settled run."""
    main_run, best_run, ext = runs[0], runs[1], runs[-1]
    lines = [
        "# C9.1 Tabular convergence (rerun)",
        "",
        "Generated by `python -m terser.models.rl.verify convergence`: CPU reruns of "
        f"`terser.phase9.qlearning_study` learners on the same {n_train} train prompts, "
        "gate reward and seed. Two statistics over the last 1,000 episodes: **max update** = "
        "the largest |dQ| of any single update (what `reports/p9_qlearning.md` reports) and "
        "**net change** = max over (s, a) of |Q_T - Q_{T-1000}|. C9.1 asks for < 1e-3.",
        "",
        "| run | alpha | gamma | eps decay | eps min | episodes | max update | net change "
        "| episodes with update >= 1e-3 | C9.1 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in runs:
        lines.append(
            f"| {r['run']} | {r['alpha']} | {r['gamma']} | {r['decay']} | {r['eps_min']} "
            f"| {r['episodes']:,} | {r['max_update_last_1000']:.3g} "
            f"| {r['net_change_last_1000']:.3g} "
            f"| {r['episodes_with_update_ge_target_last_1000']} "
            f"| {'pass' if r['pass'] else 'fail'} |"
        )
    lines += [
        "",
        "Extended run, max update over the trailing 1,000 episodes:",
        "",
        "| episode | max update |",
        "|---|---|",
        *[f"| {t['episode']:,} | {t['max_dQ_window']:.3g} |" for t in ext["trailing"]],
        "",
        f"**C9.1 at the plan budget ({main_run['episodes']:,} episodes): "
        f"{'pass' if main_run['pass'] else 'fail'}** (most settled learner: max update "
        f"{main_run['max_update_last_1000']:.3g}, net change "
        f"{main_run['net_change_last_1000']:.3g}). With {ext['episodes']:,} episodes: "
        f"{'pass' if ext['pass'] else 'fail'}.",
    ]
    if main_run["pass"]:
        return "\n".join(lines) + "\n"
    lines += [
        "",
        "**Reading.** The Q-learner does log the table change (`QLearner.max_delta_`, the largest "
        "|dQ| per episode), and the reruns reproduce the 9.2.2 sweep exactly (the plan-budget "
        "rows equal `reports/p9_qlearning.md`), so the rerun learners are the same tables as the "
        "study's. The criterion is not met on either statistic. The 9.2.3 policy learner keeps "
        "epsilon >= 0.05, so it keeps exploring and moves by up to "
        f"{best_run['net_change_last_1000']:.3g} per window. The settled learner (epsilon "
        "annealed to 0, step 1 / N(s, a)) shrinks the change about as 1 / N, but the state is "
        "aliased: one (state, action) pair covers many prompts whose STOP rewards range from "
        "about +40 to about -100, so each update moves Q by (reward spread) / N. Even at "
        f"{ext['episodes']:,} episodes the window maximum is {ext['max_update_last_1000']:.3g}, "
        "and it is not monotone (it jumps whenever a rarely visited pair is updated). Reaching "
        "1e-3 against a spread of about 140 needs on the order of 10^5 visits per pair; a richer "
        "state (less aliasing) or a far longer run is needed. C9.1 is recorded as **fail**.",
    ]
    return "\n".join(lines) + "\n"


def cmd_convergence(args: argparse.Namespace) -> int:
    json_out = args.out.with_suffix(".json")
    if args.from_json:
        runs = json.loads(json_out.read_text(encoding="utf-8"))
    else:
        env, _, category_of = _gate_env(args)
        episodes = _train_episodes(args, category_of)
        runs = []
        for name, hp, n in (
            ("settled", SETTLED_HP, args.episodes),
            ("best (9.2.3 policy)", BEST_HP, args.episodes),
            ("settled, extended", SETTLED_HP, args.long_episodes),
        ):
            r = {"run": name, **_convergence_run(env, episodes, hp, n, args.seed)}
            runs.append(r)
            print(json.dumps({k: v for k, v in r.items() if k != "trailing"}), flush=True)
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(runs, indent=2) + "\n", "utf-8")
    args.out.write_text(convergence_md(runs, args.n_train), encoding="utf-8")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m terser.models.rl.verify", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--table", type=Path, default=Path("data/processed/p4_pairs.parquet"))
        p.add_argument("--train", type=Path, default=Path("data/processed/splits/train.jsonl"))
        p.add_argument("--n-train", type=int, default=300)
        p.add_argument("--episodes", type=int, default=30000)
        p.add_argument("--target", default="qwen2.5")
        p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("prep", help="retrain policy, pick fixed order, write val candidates")
    common(p)
    p.add_argument("--val", type=Path, default=Path("data/processed/splits/val.jsonl"))
    p.add_argument("--n-val", type=int, default=300)
    p.add_argument("--prompts-out", type=Path, default=REWRITES / "rl_val300_prompts.jsonl")
    p.add_argument("--candidates-out", type=Path, default=REWRITES / "val_candidates_rl.jsonl")
    p.add_argument("--meta-out", type=Path, default=REWRITES / "val_candidates_rl.meta.json")

    r = sub.add_parser("report", help="judge-verified savings / preservation and C9.2")
    r.add_argument("--labels", type=Path, required=True)
    r.add_argument("--prompts", type=Path, default=REWRITES / "rl_val300_prompts.jsonl")
    r.add_argument("--candidates", type=Path, default=REWRITES / "val_candidates_rl.jsonl")
    r.add_argument("--meta", type=Path, default=REWRITES / "val_candidates_rl.meta.json")
    r.add_argument("--target", default="qwen2.5", help="tokenizer for savings")
    r.add_argument("--out", type=Path, default=Path("reports/p9_judge_verification.md"))

    c = sub.add_parser("convergence", help="C9.1 rerun")
    common(c)
    c.add_argument("--long-episodes", type=int, default=200000)
    c.add_argument("--out", type=Path, default=Path("reports/p9_convergence.md"))
    c.add_argument(
        "--from-json", action="store_true", help="re-render the .md from the saved .json runs"
    )

    args = ap.parse_args(argv)
    cmds: dict[str, Callable[[argparse.Namespace], int]] = {
        "prep": cmd_prep,
        "report": cmd_report,
        "convergence": cmd_convergence,
    }
    return cmds[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
