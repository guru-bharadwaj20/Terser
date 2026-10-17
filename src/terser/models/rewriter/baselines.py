"""Row **8.3.9** baselines on the rewriter val table, and checkpoints **C8.4** / **C8.5**.

Every baseline is scored on the same 285 val prompts and with the same metric definitions as the
SFT val report (:func:`terser.models.rewriter.val_report.rung_metrics`): median savings on the
Qwen2.5 tokenizer over all prompts (fallbacks count 0%), preservation pooled over (prompt,
target) pairs for both targets with fallbacks counted as preserved, and preservation per category.

Baselines are built from the Phase 2 candidates that were already judged (no new generation):

* **Original** -- the identity: 0% savings, preserved by definition.
* **Best operator** -- the plan's "best single operator" would need judged single-operator
  candidates; Phase 2 judged only the two fixed stacks (``ops_FHC``, ``ops_CHFIDSL``), so the
  baseline is the stack with the highest mean Phase 9 reward ``100 * savings - 100 *
  1[degraded]`` per (prompt, target) on the **train** split, then applied to val.
* **LLMLingua-2** -- the ``llmlingua2`` candidates (rate 0.5).
* **Zero-shot Qwen2.5-32B-Instruct-AWQ** -- the ``llm_<level>`` candidates. They come from the
  ``rewrite_v1`` instruction (rules plus an aggressiveness level), not a plain "rewrite
  concisely" prompt, so every level is reported and none is claimed to be that prompt.
* **HMM compressor** -- row 5.3.6 has not been judged; the row stays pending.

A candidate is served only if it is not rejected (row 2.2.1 validation), non-empty and strictly
shorter than the original on the Qwen2.5 tokenizer; otherwise the system falls back to the
original, exactly like the 8.3.7 safeguard of the SFT rungs.

C8.4: at equal preservation (+-1 pp) the rung's savings are within 5 pp of zero-shot 32B and
its rewrite latency is >= 10x lower on the same GPU. C8.5: the rung has higher preservation than
LLMLingua-2 at equal or better median savings.

Entry point (later runs add v2 rungs by name once their labels exist)::

    python -m terser.models.rewriter.baselines --rungs sft05,sft15,sft3b,sft15v2,sft3bv2
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from terser.models.rewriter.val_report import rung_metrics

__all__ = [
    "LLM32_METHODS",
    "OPERATOR_METHODS",
    "RUNG_PATHS",
    "c8_4",
    "c8_5",
    "candidate_rewrites",
    "identity_metrics",
    "main",
    "parse_generation_logs",
    "phase9_reward",
    "select_best_operator",
    "system_metrics",
]

R = "data/interim/rewrites"
RUNG_PATHS: dict[str, str] = {
    "sft05": f"{R}/sft-qwen0.5b-full.val.jsonl",
    "sft15": f"{R}/sft-qwen1.5b-lora.val.jsonl",
    "sft3b": f"{R}/sft-qwen3b-lora.val.jsonl",
    "sft15v2": f"{R}/sft15v2.val.jsonl",
    "sft3bv2": f"{R}/sft3bv2.val.jsonl",
}
"""Known SFT rung tags and their generate outputs; ``--rungs`` picks from these."""

OPERATOR_METHODS: tuple[str, ...] = ("ops_FHC", "ops_CHFIDSL")
LINGUA_METHOD = "llmlingua2"
LLM32_METHODS: tuple[str, ...] = ("llm_low", "llm_medium", "llm_high", "llm_max", "llm_min")
TARGET_SHORT = {"Qwen/Qwen2.5-7B-Instruct": "Qwen7B", "microsoft/Phi-3.5-mini-instruct": "Phi3.5"}

PRES_TOL = 0.01
MAX_SAVINGS_GAP = 0.05
MIN_SPEEDUP = 10.0


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


# ---------------------------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------------------------


def candidate_rewrites(
    prompts: Sequence[Mapping[str, Any]],
    candidates: Iterable[Mapping[str, Any]],
    tokens: Mapping[str, int],
    method: str,
) -> list[dict[str, Any]]:
    """One val_report-style rewrite record per prompt for a Phase 2 candidate ``method``.

    ``prompts`` need ``id`` and ``category``; ``tokens`` maps candidate ids (and
    ``<prompt_id>:original``) to Qwen2.5 token counts. A missing, rejected, stand-in, empty or
    not strictly shorter candidate falls back to the original (savings 0).
    """
    by_pid = {c["prompt_id"]: c for c in candidates if c["method"] == method}
    out = []
    for p in prompts:
        pid = str(p["id"])
        n0 = tokens[f"{pid}:original"]
        c = by_pid.get(pid)
        n1 = tokens.get(str(c["candidate_id"]), n0) if c is not None else n0
        fell_back = (
            c is None
            or bool(c.get("rejected"))
            or bool(c.get("stand_in"))
            or not str(c["text"]).strip()
            or n1 >= n0
        )
        out.append(
            {
                "id": pid,
                "category": p["category"],
                "savings": 0.0 if fell_back else 1 - n1 / n0,
                "oracle_savings": 0.0,
                "fell_back": fell_back,
                "latency_s": 0.0,
                "tokens_out": n1 if c is not None else None,
            }
        )
    return out


def system_metrics(
    rewrites: Sequence[dict[str, Any]], labels: Sequence[dict[str, Any]], tag: str
) -> dict[str, Any]:
    """:func:`rung_metrics` plus preservation per target (same pooling rule, one target)."""
    m = rung_metrics(rewrites, labels, tag)
    m.pop("oracle_median_savings", None)
    m["preserved_by_target"] = {
        t: rung_metrics(rewrites, [r for r in labels if r["target"] == t], tag)["preserved"]
        for t in m["targets"]
    }
    return m


def identity_metrics(
    prompts: Sequence[Mapping[str, Any]], targets: Sequence[str], tag: str = "original"
) -> dict[str, Any]:
    """The original prompt as a system: every (prompt, target) pair preserved, 0% savings."""
    rewrites = [
        {
            "id": str(p["id"]),
            "category": p["category"],
            "savings": 0.0,
            "oracle_savings": 0.0,
            "fell_back": True,
            "latency_s": 0.0,
        }
        for p in prompts
    ]
    labels = [
        {"candidate_id": f"{p['id']}:{tag}", "target": t, "label": "preserved"}
        for p in prompts
        for t in targets
    ]
    return system_metrics(rewrites, labels, tag)


def phase9_reward(
    rewrites: Sequence[dict[str, Any]], labels: Sequence[dict[str, Any]], tag: str
) -> float:
    """Mean ``100 * savings - 100 * 1[degraded]`` over (prompt, target); fallbacks are safe."""
    ok: dict[tuple[str, str], bool] = {}
    for r in labels:
        pid, _, t = r["candidate_id"].rpartition(":")
        if t == tag:
            ok[(pid, r["target"])] = r["label"] == "preserved"
    targets = sorted({t for _, t in ok})
    vals = [
        100 * w["savings"] - (0.0 if w["fell_back"] or ok.get((w["id"], t), False) else 100.0)
        for w in rewrites
        for t in targets
    ]
    return sum(vals) / len(vals) if vals else 0.0


def select_best_operator(rewards: Mapping[str, float]) -> str:
    """Highest train reward; ties go to the earlier name alphabetically."""
    return min(rewards, key=lambda k: (-rewards[k], k))


# ---------------------------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------------------------


def parse_generation_logs(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    """Per-method generation wall time from the JSON summaries of ``terser.data.candidates`` logs.

    Returns ``{method: {seconds, written, s_per_prompt, log}}`` for every ``seconds_<method>``
    key of a summary block (a top-level ``{ ... }`` object spread over lines).
    """
    out: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.exists():
            continue
        block: list[str] | None = None
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.rstrip() == "{":
                block = [line]
            elif block is not None:
                block.append(line)
                if line.rstrip() == "}":
                    try:
                        summary = json.loads("\n".join(block))
                    except json.JSONDecodeError:
                        summary = {}
                    block = None
                    for key, sec in summary.items():
                        if key.startswith("seconds_"):
                            m = key[len("seconds_") :]
                            n = int(summary.get(f"written_{m}", 0))
                            if n:
                                out[m] = {
                                    "seconds": float(sec),
                                    "written": n,
                                    "s_per_prompt": float(sec) / n,
                                    "log": str(path),
                                }
    return out


def serving_throughput(path: Path) -> dict[int, float]:
    """``{batch_size: aggregate decode tok/s}`` from a row 7.6 serving-bench JSON."""
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {int(b["batch_size"]): float(b["tokens_per_s"]) for b in data.get("batches", [])}


def llm_latency(
    rewrites: Sequence[dict[str, Any]],
    run: Mapping[str, Any] | None,
    throughput: Mapping[int, float],
) -> dict[str, Any]:
    """Recorded and estimated per-prompt rewrite latency of one zero-shot 32B level.

    ``run_s`` is the measured generation wall time per prompt (concurrent requests, vLLM
    continuous batching). ``b16_est_s`` / ``b1_est_s`` divide the mean output length (Qwen2.5
    tokens) by the measured aggregate decode throughput at batch 16 / 1; they ignore prefill,
    so they are lower bounds. The primary value is ``run_s`` when recorded, else ``b16_est_s``.
    """
    outs = [w["tokens_out"] for w in rewrites if w.get("tokens_out") is not None]
    mean_out = statistics.mean(outs) if outs else None
    b16 = mean_out / throughput[16] if mean_out is not None and 16 in throughput else None
    b1 = mean_out / throughput[1] if mean_out is not None and 1 in throughput else None
    run_s = float(run["s_per_prompt"]) if run else None
    primary, source = (run_s, "run") if run_s is not None else (b16, "b16_est")
    return {
        "mean_tokens_out": mean_out,
        "run_s": run_s,
        "b16_est_s": b16,
        "b1_est_s": b1,
        "latency_s": primary,
        "latency_source": source if primary is not None else None,
    }


# ---------------------------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------------------------


def _interpolate(pres: float, ref: Mapping[str, Mapping[str, Any]]) -> float | None:
    """Savings of the reference curve at preservation ``pres`` (linear between levels)."""
    pts = sorted((m["preserved"], m["median_savings"]) for m in ref.values())
    for (p0, s0), (p1, s1) in zip(pts, pts[1:], strict=False):
        if p0 <= pres <= p1:
            return s0 if p1 == p0 else s0 + (s1 - s0) * (pres - p0) / (p1 - p0)
    return None


def c8_4(
    rung: Mapping[str, Any],
    llm: Mapping[str, Mapping[str, Any]],
    tol: float = PRES_TOL,
    max_gap: float = MAX_SAVINGS_GAP,
    min_speedup: float = MIN_SPEEDUP,
) -> dict[str, Any]:
    """C8.4 for one rung against every zero-shot 32B level (``llm`` metrics with latency).

    A level is *matched* when preservation differs by at most ``tol``. The rung passes if some
    matched level has a savings gap (level minus rung) of at most ``max_gap`` and a latency
    ratio (level / rung) of at least ``min_speedup``.
    """
    rows = []
    for name, m in llm.items():
        lat = m.get("latency_s")
        speedup = lat / rung["latency_s"] if lat is not None and rung["latency_s"] > 0 else None
        b1 = m.get("b1_est_s")
        rows.append(
            {
                "level": name,
                "pres_diff": rung["preserved"] - m["preserved"],
                "savings_gap": m["median_savings"] - rung["median_savings"],
                "speedup": speedup,
                "speedup_b1_est": b1 / rung["latency_s"] if b1 and rung["latency_s"] > 0 else None,
                "matched": abs(rung["preserved"] - m["preserved"]) <= tol + 1e-12,
            }
        )
    for r in rows:
        r["savings_ok"] = r["savings_gap"] <= max_gap + 1e-12
        r["latency_ok"] = r["speedup"] is not None and r["speedup"] >= min_speedup
        r["pass"] = r["matched"] and r["savings_ok"] and r["latency_ok"]
    closest = min(rows, key=lambda r: (abs(r["pres_diff"]), r["level"])) if rows else None
    interp = _interpolate(rung["preserved"], llm)
    return {
        "levels": rows,
        "matched": [r["level"] for r in rows if r["matched"]],
        "closest": closest["level"] if closest else None,
        "interpolated_llm_savings": interp,
        "interpolated_gap": interp - rung["median_savings"] if interp is not None else None,
        "latency_ok_any": any(r["latency_ok"] for r in rows),
        "pass": any(r["pass"] for r in rows),
    }


def c8_5(rung: Mapping[str, Any], lingua: Mapping[str, Any]) -> dict[str, Any]:
    """C8.5: higher preservation than LLMLingua-2 at equal or better median savings."""
    dp = rung["preserved"] - lingua["preserved"]
    ds = rung["median_savings"] - lingua["median_savings"]
    return {
        "pres_diff": dp,
        "savings_diff": ds,
        "pres_ok": dp > 0,
        "savings_ok": ds >= -1e-12,
        "pass": dp > 0 and ds >= -1e-12,
    }


# ---------------------------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------------------------


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.1%}"


def _pp(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:+.1f} pp"


def _sec(x: float | None) -> str:
    return "n/r" if x is None else f"{x:.2f}"


def _markdown(res: dict[str, Any]) -> str:
    systems: dict[str, dict[str, Any]] = res["systems"]
    targets = sorted({t for m in systems.values() for t in m.get("targets", [])})
    best = res["best_operator"]
    lines = [
        "# Phase 8 baselines on the rewriter val table (row 8.3.9, checkpoints C8.4 / C8.5)",
        "",
        f"Same {res['n_prompts']} val prompts and metric definitions as"
        " [p8_rewriter_val.md](p8_rewriter_val.md) (`val_report.rung_metrics`): median savings on"
        " the Qwen2.5 tokenizer over all prompts (fallbacks count 0%); preservation pooled over"
        " (prompt, target) for both targets, judged by Qwen2.5-32B-Instruct-AWQ (`pairwise_avg`),"
        " fallbacks counted as preserved. Baselines reuse the judged Phase 2 candidates; nothing"
        " was regenerated. A candidate that was rejected in row 2.2.1 or is not strictly shorter"
        " on Qwen2.5 falls back to the original, like the 8.3.7 safeguard of the SFT rungs.",
        "",
        "| System | Source | n | Median savings | Fell back | Preserved | "
        + " | ".join(f"Preserved {TARGET_SHORT.get(t, t)}" for t in targets)
        + " | Latency (s/prompt) | C8.3 |",
        "|---" * (8 + len(targets)) + "|",
    ]
    for name, m in systems.items():
        if m.get("pending"):
            cells = ["pending"] * (5 + len(targets))
            lines.append(f"| {name} | {m['source']} | " + " | ".join(cells) + " | — |")
            continue
        pt = m["preserved_by_target"]
        lines.append(
            f"| {name} | {m['source']} | {m['n']} | {m['median_savings']:.1%} | {m['fell_back']} "
            f"| {m['preserved']:.1%} | "
            + " | ".join(_pct(pt.get(t)) for t in targets)
            + f" | {_sec(m.get('latency_s'))} | {'pass' if m['c8_3'] else 'fail'} |"
        )
    if res["rungs_pending"]:
        lines += [
            "",
            "SFT rungs requested but not in the labels file yet (pending judge): "
            + ", ".join(f"`{t}`" for t in res["rungs_pending"])
            + ".",
        ]
    lines += [
        "",
        "Latency: SFT rungs are the mean generation time per prompt in batches of 16"
        " (HF `generate`, RTX 4500 Ada). Zero-shot 32B levels are the measured wall time of the"
        " vLLM generation run divided by the prompts it wrote (concurrent requests, continuous"
        " batching, same RTX 4500 Ada); `llm_low` has no recorded GPU time (its run was"
        " client-bound, [p2_candidates.md](p2_candidates.md)), so it shows the batch-16 decode"
        " estimate below. Operators and LLMLingua-2 timings were not recorded (n/r).",
        "",
        "Preservation by category:",
        "",
    ]
    cats = sorted({c for m in systems.values() for c in m.get("preserved_by_category", {})})
    lines += ["| System | " + " | ".join(cats) + " |", "|---" * (len(cats) + 1) + "|"]
    for name, m in systems.items():
        if m.get("pending"):
            lines.append(f"| {name} | " + " | ".join("pending" for _ in cats) + " |")
            continue
        pc = m["preserved_by_category"]
        lines.append(f"| {name} | " + " | ".join(_pct(pc.get(c)) for c in cats) + " |")

    lines += [
        "",
        "## Best operator (chosen on train)",
        "",
        "Phase 2 judged two fixed operator stacks, not single operators, so the plan's"
        ' "best single operator" is the better stack on the train split'
        f" ({best['n_train']} prompts) by the Phase 9 reward `100 * savings - 100 * 1[degraded]`"
        " per (prompt, target), fallbacks safe.",
        "",
        "| Stack | Train median savings | Train preserved | Train reward | Chosen |",
        "|---|---|---|---|---|",
    ]
    for k, t in best["train"].items():
        lines.append(
            f"| `{k}` | {t['median_savings']:.1%} | {t['preserved']:.1%} | {t['reward']:.2f} "
            f"| {'yes' if k == best['chosen'] else ''} |"
        )

    llm = res["llm_latency"]
    lines += [
        "",
        "## Zero-shot Qwen2.5-32B-Instruct-AWQ",
        "",
        'No existing level is a plain "rewrite concisely" instruction. All `llm_*` candidates'
        " come from `src/terser/data/prompts/rewrite_v1.txt` (temperature 0, no examples): six"
        " preservation rules (keep the task, every constraint, tone cues, payload verbatim; same"
        " language; must be shorter) plus one aggressiveness paragraph per level. `low` removes"
        " filler and politeness; `medium` adds direct imperatives and merged repeats; `high`"
        " telegram style; `max` drops background and examples; `min` keyword style. A bare"
        ' "rewrite concisely" prompt is closest in spirit to `medium` / `high` but would lack the'
        " preservation rules, so it would likely save more and preserve less; all five levels are"
        " reported and C8.4 is checked against each of them.",
        "",
        "| Level | Mean output tokens | Run wall / prompt (s) | Batch-16 decode est. (s) |"
        " Batch-1 decode est. (s) |",
        "|---|---|---|---|---|",
    ]
    for k, v in llm.items():
        mo = v["mean_tokens_out"]
        lines.append(
            f"| `{k}` | {'n/a' if mo is None else f'{mo:.1f}'} | {_sec(v['run_s'])} "
            f"| {_sec(v['b16_est_s'])} | {_sec(v['b1_est_s'])} |"
        )
    tp = res["throughput"]
    lines += [
        "",
        "Run wall times come from the summaries in "
        + (", ".join(f"`{p}`" for p in res.get("generation_logs", [])) or "no log")
        + ". Estimates divide the mean output length by the measured decode throughput of the"
        f" 32B AWQ on the RTX 4500 Ada (row 7.6: {tp.get('1', 'n/a')} tok/s at batch 1,"
        f" {tp.get('16', 'n/a')} tok/s aggregate at batch 16); prefill is ignored, so they are"
        " lower bounds.",
    ]

    lines += [
        "",
        "## C8.4: vs zero-shot Qwen2.5-32B-AWQ",
        "",
        f"Pass needs a level within ±{100 * PRES_TOL:.0f} pp preservation, a savings gap"
        f" (level − rung) ≤ {100 * MAX_SAVINGS_GAP:.0f} pp and a latency ratio (level / rung)"
        f" ≥ {MIN_SPEEDUP:.0f}×. Latency ratio uses the primary 32B latency (run wall per prompt,"
        " else batch-16 estimate) against the rung's batch-16 latency; the batch-1 column divides"
        " the 32B batch-1 estimate by the same rung latency (different batching, indicative only).",
        "",
        "| Rung | Level | Δ preservation (rung − level) | Savings gap (level − rung)"
        " | Latency ratio"
        " | Batch-1 ratio | Matched ±1 pp | Pass |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for tag, c in res["c8_4"].items():
        for r in c["levels"]:
            ratio = "n/a" if r["speedup"] is None else f"{r['speedup']:.2f}×"
            b1 = "n/a" if r["speedup_b1_est"] is None else f"{r['speedup_b1_est']:.1f}×"
            lines.append(
                f"| {tag} | `{r['level']}` | {_pp(r['pres_diff'])} | {_pp(r['savings_gap'])} "
                f"| {ratio} | {b1} | {'yes' if r['matched'] else 'no'} "
                f"| {'pass' if r['pass'] else 'fail'} |"
            )
    lines += ["", "| Rung | Closest level | 32B savings at rung's preservation (interp.) | C8.4 |"]
    lines.append("|---|---|---|---|")
    for tag, c in res["c8_4"].items():
        ip = c["interpolated_llm_savings"]
        ipt = (
            "outside the 32B range"
            if ip is None
            else f"{ip:.1%} (gap {_pp(c['interpolated_gap'])})"
        )
        lines.append(f"| {tag} | `{c['closest']}` | {ipt} | {'pass' if c['pass'] else 'fail'} |")
    lines += ["", f"**C8.4: {res['c8_4_verdict']}.** {res['c8_4_reading']}"]

    lines += [
        "",
        "## C8.5: vs LLMLingua-2",
        "",
        "Pass needs higher preservation than LLMLingua-2 and equal or better median savings.",
        "",
        "| Rung | Δ preservation (rung − LLMLingua-2) | Δ median savings | C8.5 |",
        "|---|---|---|---|",
    ]
    for tag, c in res["c8_5"].items():
        lines.append(
            f"| {tag} | {_pp(c['pres_diff'])} | {_pp(c['savings_diff'])} "
            f"| {'pass' if c['pass'] else 'fail'} |"
        )
    lines += ["", f"**C8.5: {res['c8_5_verdict']}.** {res['c8_5_reading']}", ""]
    return "\n".join(lines)


def _verdicts(res: dict[str, Any]) -> None:
    c4, c5 = res["c8_4"], res["c8_5"]
    rungs = res["rungs"]
    if not rungs:
        res["c8_4_verdict"] = res["c8_5_verdict"] = "not evaluated"
        res["c8_4_reading"] = res["c8_5_reading"] = "No judged SFT rung."
        return
    res["c8_4_verdict"] = "pass" if any(c["pass"] for c in c4.values()) else "fail"
    parts = []
    for tag in rungs:
        c = c4[tag]
        lv = {r["level"]: r for r in c["levels"]}
        best_ratio = max((r["speedup"] for r in lv.values() if r["speedup"]), default=None)
        parts.append(
            f"{tag}: matched levels {', '.join(c['matched']) or 'none'}, closest `{c['closest']}`"
            f" ({_pp(lv[c['closest']]['pres_diff'])} preservation,"
            f" {_pp(lv[c['closest']]['savings_gap'])} savings gap), best latency ratio"
            f" {'n/a' if best_ratio is None else f'{best_ratio:.2f}×'}"
        )
    res["c8_4_reading"] = (
        "; ".join(parts)
        + '. No level of the provisional 32B baseline is a plain "rewrite concisely" prompt,'
        " and SFT and 32B latencies come from different batching set-ups, so a batch-1 latency"
        " benchmark of both on the same GPU is still needed for a final latency call."
    )
    res["c8_5_verdict"] = "pass" if any(c["pass"] for c in c5.values()) else "fail"
    res["c8_5_reading"] = (
        "; ".join(
            f"{tag}: preservation {_pp(c5[tag]['pres_diff'])}, median savings"
            f" {_pp(c5[tag]['savings_diff'])}"
            for tag in rungs
        )
        + "."
    )


def build(
    *,
    val_prompts: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    labels: Sequence[dict[str, Any]],
    tokens: Mapping[str, int],
    rungs: Mapping[str, Sequence[dict[str, Any]]],
    rung_labels: Sequence[dict[str, Any]],
    gen_runs: Mapping[str, Mapping[str, Any]],
    throughput: Mapping[int, float],
) -> dict[str, Any]:
    """All baseline metrics, the train choice of operator stack, C8.4 and C8.5."""
    val_ids = {str(p["id"]) for p in val_prompts}
    val_labels = [r for r in labels if r["prompt_id"] in val_ids]
    train_cat: dict[str, str] = {}
    for r in labels:
        if r.get("split") == "train":
            train_cat[r["prompt_id"]] = r["category"]
    train_prompts = [{"id": k, "category": v} for k, v in sorted(train_cat.items())]
    train_labels = [r for r in labels if r["prompt_id"] in train_cat]

    train: dict[str, dict[str, Any]] = {}
    for m in OPERATOR_METHODS:
        rw = candidate_rewrites(train_prompts, candidates, tokens, m)
        tm = rung_metrics(rw, train_labels, m)
        train[m] = {
            "median_savings": tm["median_savings"],
            "preserved": tm["preserved"],
            "reward": phase9_reward(rw, train_labels, m),
        }
    chosen = select_best_operator({k: v["reward"] for k, v in train.items()})

    systems: dict[str, dict[str, Any]] = {}
    built: dict[str, tuple[list[dict[str, Any]], dict[str, Any]]] = {}
    for m in (*OPERATOR_METHODS, LINGUA_METHOD, *LLM32_METHODS):
        rw = candidate_rewrites(val_prompts, candidates, tokens, m)
        built[m] = (rw, system_metrics(rw, val_labels, m))
    targets = sorted({t for _, mm in built.values() for t in mm["targets"]})

    orig = identity_metrics(val_prompts, targets)
    systems["Original"] = {**orig, "source": "identity", "latency_s": 0.0}
    for m in (chosen, *(o for o in OPERATOR_METHODS if o != chosen)):
        name = "Best operator" if m == chosen else "Other operator stack"
        systems[name] = {**built[m][1], "source": f"`{m}`", "latency_s": None}
    systems["HMM compressor"] = {
        "pending": True,
        "source": "row 5.3.6 not judged",
        "preserved_by_category": {},
    }
    systems["LLMLingua-2"] = {
        **built[LINGUA_METHOD][1],
        "source": "`llmlingua2`",
        "latency_s": None,
    }
    llm_lat: dict[str, dict[str, Any]] = {}
    llm_systems: dict[str, dict[str, Any]] = {}
    for m in LLM32_METHODS:
        rw, mm = built[m]
        llm_lat[m] = llm_latency(rw, gen_runs.get(m), throughput)
        llm_systems[m] = {**mm, **llm_lat[m]}
        systems[f"Zero-shot 32B {m[4:]}"] = {
            **mm,
            "source": f"`{m}`",
            "latency_s": llm_lat[m]["latency_s"],
        }

    rung_tags: list[str] = []
    pending: list[str] = []
    for tag, rung_rw in rungs.items():
        if not any(r["candidate_id"].rpartition(":")[2] == tag for r in rung_labels):
            pending.append(tag)
            continue
        rm = system_metrics(rung_rw, rung_labels, tag)
        systems[f"SFT {tag}"] = {**rm, "source": f"`{tag}`"}
        rung_tags.append(tag)

    res: dict[str, Any] = {
        "n_prompts": len(val_prompts),
        "systems": systems,
        "best_operator": {"chosen": chosen, "train": train, "n_train": len(train_prompts)},
        "llm_latency": llm_lat,
        "throughput": {str(k): v for k, v in sorted(throughput.items())},
        "rungs": rung_tags,
        "rungs_pending": pending,
        "c8_4": {t: c8_4(systems[f"SFT {t}"], llm_systems) for t in rung_tags},
        "c8_5": {t: c8_5(systems[f"SFT {t}"], systems["LLMLingua-2"]) for t in rung_tags},
    }
    _verdicts(res)
    return res


def _tokens(paths: Sequence[Path], target: str) -> dict[str, int]:
    import pandas as pd

    tc = pd.concat([pd.read_parquet(p) for p in paths]).drop_duplicates(["candidate_id", "target"])
    tc = tc[tc.target == target]
    return {str(k): int(v) for k, v in zip(tc.candidate_id, tc.tokens, strict=True)}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Row 8.3.9 baselines, C8.4 and C8.5.")
    ap.add_argument(
        "--rungs",
        default="sft05,sft15,sft3b",
        help=f"comma-separated SFT rung tags from {sorted(RUNG_PATHS)}",
    )
    ap.add_argument("--rung", action="append", default=[], help="extra rung as tag=path")
    ap.add_argument("--rung-labels", type=Path, default=Path(f"{R}/val_labels_all.jsonl"))
    ap.add_argument("--val-prompts", type=Path, default=Path(f"{R}/val_prompts.jsonl"))
    ap.add_argument(
        "--candidates",
        type=Path,
        nargs="+",
        default=[Path("data/interim/candidates.jsonl"), Path("data/interim/candidates_aug.jsonl")],
    )
    ap.add_argument(
        "--labels",
        type=Path,
        nargs="+",
        default=[Path("data/processed/labels.jsonl"), Path("data/processed/labels_aug.jsonl")],
    )
    ap.add_argument(
        "--tokens",
        type=Path,
        nargs="+",
        default=[
            Path("data/interim/token_counts.parquet"),
            Path("data/interim/token_counts_aug.parquet"),
        ],
    )
    ap.add_argument("--token-target", default="qwen2.5")
    ap.add_argument(
        "--gen-logs",
        type=Path,
        nargs="+",
        default=[
            Path("logs/p2_candidates.log"),
            Path("logs/p2_candidates_2.log"),
            Path("logs/aug_cand.log"),
        ],
    )
    ap.add_argument(
        "--serving-bench",
        type=Path,
        default=Path("docs/checkpoints/C7/serving_qwen2.5-32b_awq.json"),
    )
    ap.add_argument("--out", type=Path, default=Path("reports/p8_baselines.md"))
    args = ap.parse_args(argv)

    specs = [t.strip() for t in args.rungs.split(",") if t.strip()]
    unknown = [t for t in specs if t not in RUNG_PATHS]
    if unknown:
        ap.error(f"unknown rung(s) {unknown}; known: {sorted(RUNG_PATHS)} (or use --rung tag=path)")
    paths = {t: Path(RUNG_PATHS[t]) for t in specs}
    for spec in args.rung:
        tag, _, p = spec.partition("=")
        paths[tag] = Path(p)
    rungs = {t: _jsonl(p) for t, p in paths.items() if p.exists()}
    missing = [t for t, p in paths.items() if not p.exists()]

    gen_runs = parse_generation_logs(args.gen_logs)
    res = build(
        val_prompts=_jsonl(args.val_prompts),
        candidates=[c for p in args.candidates for c in _jsonl(p)],
        labels=[r for p in args.labels for r in _jsonl(p)],
        tokens=_tokens(args.tokens, args.token_target),
        rungs=rungs,
        rung_labels=_jsonl(args.rung_labels) if args.rung_labels.exists() else [],
        gen_runs=gen_runs,
        throughput=serving_throughput(args.serving_bench),
    )
    res["rungs_pending"] += missing
    res["generation_logs"] = sorted({str(v["log"]) for v in gen_runs.values()})
    _verdicts(res)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(_markdown(res), encoding="utf-8")
    args.out.with_suffix(".json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "best_operator": res["best_operator"]["chosen"],
                "c8_4": res["c8_4_verdict"],
                "c8_5": res["c8_5_verdict"],
                "rungs": res["rungs"],
                "pending": res["rungs_pending"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
