"""End-to-end HMM compressor on 500 val prompts (plan row **5.3.6**, checkpoint **C5.6**).

Two CLI stages around the GPU ``respond`` + ``label`` stages of Phase 2:

* ``prep`` (CPU): pick 500 val prompts (seed 0, stratified by category; ``test.jsonl`` is never
  read), rewrite each with Viterbi tags of the supervised HMM (method ``hmm``) and with the best
  single operator chosen on the train split (method ``op_<code>``), and write the prompt file
  ``terser.phase2.respond --prompts`` reads plus the candidate file it reads with
  ``--candidates``.
* ``report`` (CPU): from the ``label_v1`` records of those candidates, median savings on the
  Qwen2.5 tokenizer, preservation overall / per target / per category, and the C5.6 verdict.

**HMM.** The supervised MLE HMM of row 5.3.2 (Laplace smoothing 1) fitted on every labelled
train prompt in ``span_features.parquet`` (row 3.1.1 KEEP / DROP / REPLACE alignment labels,
train split only), with the row 5.3.1 :class:`~terser.models.classical.hmm.TokenClassEncoder`.
Row 5.3.5 measured Viterbi token macro-F1 0.438 for this model vs 0.346 after Baum-Welch: EM
raises the likelihood of the token classes, not the agreement with the tags, so the supervised
model is the better tagger and the one used here.

**Rewrite.** Tokens are the row 3.1.1 tokens (:data:`~terser.features.alignment.TOKEN_RE`), the
same units the HMM was trained on. A ``DROP`` token is deleted with the horizontal space before
it; ``KEEP`` and ``REPLACE`` tokens are kept (no replacement text is known, the conservative
reading of :func:`~terser.models.classical.hmm.apply_tags`). Tokens inside the operators'
protected payload spans (:data:`~terser.operators.masking.PROTECTED`: fenced code, inline code,
double-quoted text) are forced to ``KEEP``, the same payload rule every operator obeys. Lines
with no deleted token are copied byte for byte (code indentation survives); lines with a
deletion get the operator spacing / punctuation repairs, and a line left with no word is removed.

**Best single operator.** Phase 2 never judged single operators, and C5.6 must not be decided by
val labels, so the operator is chosen before judging by the Phase 9 reward
(:func:`terser.models.rl.env.stop_reward`: ``100 * savings - 100 * 1[gate >= 0.5]``) averaged
over 300 train prompts (seed 0), with the row 9.2 LightGBM quality gate trained on the train
pair table (:func:`terser.phase9.qlearning_study.build_gate`) as the degradation model. An
operator that leaves a prompt unchanged scores 0 on it. One operator only, to keep judge cost
small.

**Equal preservation (reported, not gated since ADR-0012).** HMM and operator are single
operating points. The one with the lower preservation ``p_lo`` is mixed with the identity
rewrite (send the original prompt, preserved by construction) on a fraction ``lam = (p_hi -
p_lo) / (1 - p_lo)`` of prompts, which moves it to exactly the other's preservation ``p_hi``;
its median savings at that point is the weighted median of its savings (weight ``1 - lam``) and
a point mass of 0 (weight ``lam``). The HMM beats the operator on a target when its median
savings at the matched preservation is strictly higher. C5.6 (as revised by ADR-0012) passes
when the HMM median savings is >= 15% and some target has HMM preservation >= 55%; the
equal-preservation comparison is reported but no longer gates.

CLI::

    python -m terser.phase5.hmm_compressor prep
    python -m terser.phase5.hmm_compressor report \\
        --labels data/interim/rewrites/hmm_val500_labels.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from terser.features.alignment import TOKEN_RE
from terser.models.classical.hmm import TAGS, DiscreteHMM, TokenClassEncoder, fit_supervised
from terser.operators.masking import PROTECTED

__all__ = [
    "HMM_METHOD",
    "choose_operator",
    "fit_span_hmm",
    "hmm_rewrite",
    "main",
    "matched_savings",
    "method_metrics",
    "select_prompts",
    "verdict",
]

HMM_METHOD = "hmm"
N_PROMPTS = 500
SEED = 0
TOKEN_TARGET = "qwen2.5"
MIN_SAVINGS = 0.15
MIN_PRESERVED = 0.55  # C5.6 as revised by ADR-0012 (was 0.90)
GATE_PROMPTS = 300
GATE_THRESHOLD = 0.5
_DROP = TAGS.index("DROP")

VAL = Path("data/processed/splits/val.jsonl")
TRAIN = Path("data/processed/splits/train.jsonl")
SPANS = Path("data/processed/span_features.parquet")
PAIRS = Path("data/processed/p4_pairs.parquet")
REWRITES = Path("data/interim/rewrites")
PROMPTS_OUT = REWRITES / "hmm_val500_prompts.jsonl"
CANDS_OUT = REWRITES / "val_candidates_hmm.jsonl"
PREP_OUT = REWRITES / "hmm_val500_prep.json"
REPORT_OUT = Path("reports/p5_hmm_compressor.md")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------------------------
# Prompt selection
# --------------------------------------------------------------------------------------------


def select_prompts(
    records: Sequence[Mapping[str, Any]], n: int = N_PROMPTS, seed: int = SEED
) -> list[dict[str, Any]]:
    """``n`` usable records stratified by category (proportional, largest remainder), file order.

    Rejected and stand-in records are skipped. Within a category the records are a seeded random
    permutation, so the pick is deterministic in (file, ``n``, ``seed``).
    """
    usable = [dict(r) for r in records if not r.get("rejected") and not r.get("stand_in")]
    if n >= len(usable):
        return usable
    by_cat: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(usable):
        by_cat[str(r["category"])].append(i)
    cats = sorted(by_cat)
    quota = {c: n * len(by_cat[c]) / len(usable) for c in cats}
    take = {c: int(quota[c]) for c in cats}
    rest = sorted(cats, key=lambda c: (-(quota[c] - take[c]), c))
    for c in rest[: n - sum(take.values())]:
        take[c] += 1
    rng = np.random.default_rng(seed)
    keep: set[int] = set()
    for c in cats:
        idx = by_cat[c]
        keep.update(idx[j] for j in rng.permutation(len(idx))[: take[c]])
    return [usable[i] for i in sorted(keep)]


# --------------------------------------------------------------------------------------------
# HMM rewrite
# --------------------------------------------------------------------------------------------


def fit_span_hmm(
    spans: Any, encoder: TokenClassEncoder | None = None, smoothing: float = 1.0
) -> DiscreteHMM:
    """Supervised HMM (row 5.3.2) on a span table with ``prompt_id``, ``position``, ``token``,
    ``label`` columns (one row per row-3.1.1 token)."""
    enc = encoder or TokenClassEncoder()
    tag_id = {t: i for i, t in enumerate(TAGS)}
    obs, tags = [], []
    for _, g in spans.sort_values(["prompt_id", "position"]).groupby("prompt_id", sort=True):
        obs.append(enc.encode(g.token.tolist()))
        tags.append(np.array([tag_id[t] for t in g.label], dtype=np.int64))
    return fit_supervised(obs, tags, len(TAGS), enc.n_obs, smoothing=smoothing)


def _protected(text: str) -> list[tuple[int, int]]:
    return [m.span() for m in PROTECTED.finditer(text)]


def _repair(body: str) -> str:
    """Operator-style spacing and punctuation repairs for a line body that lost tokens."""
    body = re.sub(r"[ \t]+", " ", body)
    body = re.sub(r" +([,.;:!?])", r"\1", body)
    body = re.sub(r"([,;:])(?: *[,;:])+", r"\1", body)
    body = re.sub(r", *([.!?])", r"\1", body)
    body = re.sub(r"^[ ,;:.!?]+", "", body).strip()
    return body if any(ch.isalnum() for ch in body) else ""


def hmm_rewrite(
    hmm: DiscreteHMM, text: str, encoder: TokenClassEncoder | None = None
) -> tuple[str, int]:
    """Viterbi-tag ``text`` and delete its ``DROP`` tokens; returns ``(rewrite, n_dropped)``.

    See the module docstring for the rules (payload spans kept, untouched lines copied verbatim).
    """
    enc = encoder or TokenClassEncoder()
    matches = list(TOKEN_RE.finditer(text))
    if not matches:
        return text, 0
    tags, _ = hmm.viterbi(enc.encode([m.group(0) for m in matches]))
    guard = _protected(text)
    drop = [
        m.span()
        for m, t in zip(matches, tags.tolist(), strict=True)
        if t == _DROP and not any(s < m.end() and m.start() < e for s, e in guard)
    ]
    if not drop:
        return text, 0
    out_lines: list[str] = []
    offset, k = 0, 0
    for line in text.split("\n"):
        end = offset + len(line)
        cut: list[tuple[int, int]] = []
        while k < len(drop) and drop[k][0] < end:
            cut.append((drop[k][0] - offset, drop[k][1] - offset))
            k += 1
        if not cut:
            out_lines.append(line)
        else:
            pieces, pos = [], 0
            for s, e in cut:
                pieces += [line[pos:s], " "]
                pos = e
            pieces.append(line[pos:])
            indent = line[: len(line) - len(line.lstrip(" \t"))]
            body = _repair("".join(pieces)[len(indent) :])
            if line[len(indent) : len(indent) + 1].isupper() and body[:1].islower():
                body = body[0].upper() + body[1:]
            if body:
                out_lines.append(indent + body)
        offset = end + 1
    out = re.sub(r"\n{3,}", "\n\n", "\n".join(out_lines)).strip("\n")
    return (out if out.strip() else text), len(drop)


# --------------------------------------------------------------------------------------------
# Best single operator (pre-judge rule on train)
# --------------------------------------------------------------------------------------------


def choose_operator(
    prompts: Sequence[Mapping[str, Any]],
    apply: Callable[[str, str], str],
    count: Callable[[str], int],
    gate: Callable[[str, str], float],
    codes: Sequence[str],
    threshold: float = GATE_THRESHOLD,
) -> tuple[str, list[dict[str, Any]]]:
    """Operator with the highest mean Phase 9 stop reward on ``prompts`` (ties: earlier code).

    ``apply(text, code)`` rewrites with one operator; an unchanged prompt scores 0 (no savings,
    identity is preserved). Returns the chosen code and one summary row per code.
    """
    from terser.models.rl.env import stop_reward

    rows = []
    for code in codes:
        rewards, savings, degraded, changed = [], [], [], 0
        for p in prompts:
            text = str(p["prompt"])
            draft = apply(text, code)
            n0 = count(text)
            if draft == text or n0 <= 0:
                rewards.append(0.0)
                savings.append(0.0)
                degraded.append(False)
                continue
            changed += 1
            deg = gate(text, draft) >= threshold
            rewards.append(stop_reward(n0, count(draft), deg))
            savings.append(1 - count(draft) / n0)
            degraded.append(deg)
        rows.append(
            {
                "operator": code,
                "n": len(prompts),
                "changed": changed,
                "median_savings": statistics.median(savings) if savings else 0.0,
                "mean_savings": float(np.mean(savings)) if savings else 0.0,
                "gate_degraded": float(np.mean(degraded)) if degraded else 0.0,
                "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            }
        )
    best = max(rows, key=lambda r: (r["mean_reward"], -codes.index(r["operator"])))
    return str(best["operator"]), rows


# --------------------------------------------------------------------------------------------
# prep
# --------------------------------------------------------------------------------------------


def _savings(count: Callable[[str], int], prompt: str, text: str) -> float:
    n0 = count(prompt)
    return 1 - count(text) / n0 if n0 else 0.0


def candidate_rows(
    prompts: Sequence[Mapping[str, Any]], method: str, rewrite: Callable[[str], str]
) -> list[dict[str, Any]]:
    """Candidates in the ``val_candidates_v2.jsonl`` format for one method."""
    return [
        {
            "prompt_id": str(p["id"]),
            "candidate_id": f"{p['id']}:{method}",
            "method": method,
            "text": rewrite(str(p["prompt"])),
        }
        for p in prompts
    ]


def run_prep(args: argparse.Namespace) -> dict[str, Any]:
    import pandas as pd

    from terser.operators.registry import OPERATORS, apply_ops
    from terser.tokenizers.count import count_tokens

    if "test" in Path(args.val).name:
        raise ValueError("prep must not read the test split")
    prompts = select_prompts(_jsonl(args.val), args.n, args.seed)
    picked = {str(p["id"]) for p in prompts}

    spans = pd.read_parquet(args.spans)
    leak = picked & set(spans.prompt_id.astype(str))
    if leak:
        raise ValueError(f"{len(leak)} selected val prompts are in the HMM training spans")
    enc = TokenClassEncoder()
    hmm = fit_span_hmm(spans, enc)

    def count(text: str) -> int:
        return count_tokens(text, TOKEN_TARGET)

    op_rows: list[dict[str, Any]] = []
    if args.operator:
        op = str(args.operator)
    else:
        from terser.phase9.qlearning_study import build_gate

        train = [r for r in _jsonl(args.train) if not r.get("rejected")]
        rng = np.random.default_rng(args.seed)
        sub = [train[i] for i in rng.choice(len(train), min(GATE_PROMPTS, len(train)), False)]
        gate, cost, category_of = build_gate(args.pairs, TOKEN_TARGET)
        for p in sub:
            category_of[str(p["prompt"])] = str(p["category"])
        op, op_rows = choose_operator(
            sub,
            lambda t, c: apply_ops(t, c, TOKEN_TARGET),
            cost,
            gate,
            [c for c in "FHDILCS" if c in OPERATORS],
        )

    op_method = f"op_{op}"
    cands = candidate_rows(prompts, HMM_METHOD, lambda t: hmm_rewrite(hmm, t, enc)[0])
    cands += candidate_rows(prompts, op_method, lambda t: apply_ops(t, op, TOKEN_TARGET))
    _write_jsonl(args.prompts_out, prompts)
    _write_jsonl(args.candidates_out, cands)

    text_of = {str(p["id"]): str(p["prompt"]) for p in prompts}
    per_method: dict[str, dict[str, Any]] = {}
    for m in (HMM_METHOD, op_method):
        rows = [c for c in cands if c["method"] == m]
        sav = [_savings(count, text_of[c["prompt_id"]], c["text"]) for c in rows]
        per_method[m] = {
            "n": len(rows),
            "identity": sum(c["text"] == text_of[c["prompt_id"]] for c in rows),
            "not_shorter": sum(s <= 0 for s in sav),
            "median_savings": statistics.median(sav),
            "mean_savings": float(np.mean(sav)),
        }
    info = {
        "n_prompts": len(prompts),
        "categories": dict(sorted(Counter(str(p["category"]) for p in prompts).items())),
        "seed": args.seed,
        "val": str(args.val),
        "hmm": {
            "model": "supervised MLE, Laplace 1 (row 5.3.2)",
            "train_prompts": int(spans.prompt_id.nunique()),
            "train_tokens": len(spans),
        },
        "operator": op,
        "operator_method": op_method,
        "operator_rule": (
            "max mean Phase 9 stop reward (LightGBM gate >= 0.5) on "
            f"{GATE_PROMPTS} train prompts, seed {args.seed}"
            if op_rows
            else "given on the command line"
        ),
        "operator_table": op_rows,
        "pre_judge": per_method,
    }
    args.prep_out.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    return info


# --------------------------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------------------------


def weighted_median(values: Sequence[float], weights: Sequence[float]) -> float:
    """Smallest value whose cumulative weight reaches half the total."""
    order = np.argsort(values, kind="stable")
    v, w = np.asarray(values, float)[order], np.asarray(weights, float)[order]
    cum = np.cumsum(w)
    return float(v[int(np.searchsorted(cum, 0.5 * cum[-1] - 1e-12))])


def matched_savings(savings: Sequence[float], preserved: float, target: float) -> float:
    """Median savings after mixing with the identity rewrite up to preservation ``target``."""
    if preserved >= target or not savings:
        return statistics.median(savings) if savings else 0.0
    lam = (target - preserved) / (1 - preserved)
    n = len(savings)
    return weighted_median([*savings, 0.0], [(1 - lam) / n] * n + [lam])


def method_metrics(
    candidates: Sequence[Mapping[str, Any]],
    prompts: Mapping[str, Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
    method: str,
    count: Callable[[str], int],
) -> dict[str, Any]:
    """Savings and preservation of one method; ``labels`` may hold other methods too."""
    rows = [c for c in candidates if c["method"] == method and c["prompt_id"] in prompts]
    ids = {str(c["candidate_id"]) for c in rows}
    ok: dict[tuple[str, str], bool] = {}
    for r in labels:
        if str(r["candidate_id"]) in ids:
            t = str(r.get("target") or r["target_model"])
            ok[(str(r["prompt_id"]), t)] = r["label"] == "preserved"
    targets = sorted({t for _, t in ok})
    savings: list[float] = []
    by_cat_sav: dict[str, list[float]] = defaultdict(list)
    pooled: list[bool] = []
    by_t: dict[str, list[bool]] = defaultdict(list)
    by_cat: dict[str, list[bool]] = defaultdict(list)
    by_t_cat: dict[str, dict[str, list[bool]]] = defaultdict(lambda: defaultdict(list))
    missing = 0
    identity = not_shorter = 0
    for c in rows:
        p = prompts[c["prompt_id"]]
        cat = str(p["category"])
        s = _savings(count, str(p["prompt"]), str(c["text"]))
        savings.append(s)
        by_cat_sav[cat].append(s)
        identity += c["text"] == p["prompt"]
        not_shorter += s <= 0
        for t in targets:
            key = (str(c["prompt_id"]), t)
            if key not in ok:
                missing += 1
                continue
            pooled.append(ok[key])
            by_t[t].append(ok[key])
            by_cat[cat].append(ok[key])
            by_t_cat[t][cat].append(ok[key])

    def rate(xs: Sequence[bool]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    return {
        "n": len(rows),
        "targets": targets,
        "labelled_pairs": len(pooled),
        "missing_labels": missing,
        "identity": identity,
        "not_shorter": not_shorter,
        "median_savings": statistics.median(savings) if savings else 0.0,
        "mean_savings": float(np.mean(savings)) if savings else 0.0,
        "savings": savings,
        "median_savings_by_category": {
            c: statistics.median(v) for c, v in sorted(by_cat_sav.items())
        },
        "preserved": rate(pooled),
        "preserved_by_target": {t: rate(v) for t, v in sorted(by_t.items())},
        "preserved_by_category": {c: rate(v) for c, v in sorted(by_cat.items())},
        "preserved_by_target_category": {
            t: {c: rate(v) for c, v in sorted(d.items())} for t, d in sorted(by_t_cat.items())
        },
    }


def compare(
    hmm: Mapping[str, Any], op: Mapping[str, Any], p_hmm: float, p_op: float
) -> dict[str, Any]:
    """HMM vs operator median savings at equal preservation (module docstring)."""
    level = max(p_hmm, p_op)
    s_hmm = matched_savings(hmm["savings"], p_hmm, level)
    s_op = matched_savings(op["savings"], p_op, level)
    return {
        "preservation_hmm": p_hmm,
        "preservation_op": p_op,
        "matched_preservation": level,
        "mixed": "hmm" if p_hmm < p_op else ("op" if p_op < p_hmm else "none"),
        "savings_hmm": s_hmm,
        "savings_op": s_op,
        "hmm_beats_op": s_hmm > s_op,
    }


def verdict(hmm: Mapping[str, Any], op: Mapping[str, Any]) -> dict[str, Any]:
    """C5.6 clauses and verdict."""
    per_target = {
        t: compare(hmm, op, hmm["preserved_by_target"][t], op["preserved_by_target"].get(t, 0.0))
        for t in hmm["targets"]
    }
    pooled = compare(hmm, op, hmm["preserved"], op["preserved"])
    savings_ok = hmm["median_savings"] >= MIN_SAVINGS
    qualifying = [t for t in hmm["targets"] if hmm["preserved_by_target"][t] >= MIN_PRESERVED]
    return {
        "median_savings_ok": savings_ok,
        "targets_preserved_ok": [
            t for t in hmm["targets"] if hmm["preserved_by_target"][t] >= MIN_PRESERVED
        ],
        "targets_beats_op": [t for t, v in per_target.items() if v["hmm_beats_op"]],
        "qualifying_targets": qualifying,
        "per_target": per_target,
        "pooled": pooled,
        "c5_6": bool(savings_ok and qualifying and hmm["missing_labels"] == 0),
    }


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _short(t: str) -> str:
    return t.split("/")[-1]


def _markdown(
    results: Mapping[str, Mapping[str, Any]],
    v: Mapping[str, Any],
    prep: Mapping[str, Any] | None,
    op_method: str,
) -> str:
    hmm = results[HMM_METHOD]
    targets = hmm["targets"]
    lines = [
        "# p5_hmm_compressor",
        "",
        "Generated by `python -m terser.phase5.hmm_compressor report` (row 5.3.6, gate C5.6).",
        "",
        f"{hmm['n']} val prompts (seed 0, stratified by category). Rewrites: Viterbi tags of the"
        " supervised row 5.3.2 HMM (DROP tokens deleted, KEEP / REPLACE kept, payload spans"
        f" protected) and the best single operator `{op_method}` chosen on train before judging."
        " Savings on the Qwen2.5 tokenizer (`1 - tokens(rewrite) / tokens(prompt)`, every"
        " prompt counted, identity rewrites 0%). Preservation: Phase 2 `label_v1` labels"
        " (`pairwise_avg` judge + task metrics) pooled over (prompt, target).",
        "",
        "| Method | n | Median savings | Mean savings | Identity | Not shorter | Preserved | "
        + " | ".join(f"Preserved {_short(t)}" for t in targets)
        + " | Missing labels |",
        "|---" * (8 + len(targets)) + "|",
    ]
    for m, r in results.items():
        lines.append(
            f"| `{m}` | {r['n']} | {_pct(r['median_savings'])} | {_pct(r['mean_savings'])} | "
            f"{r['identity']} | {r['not_shorter']} | {_pct(r['preserved'])} | "
            + " | ".join(_pct(r["preserved_by_target"].get(t, 0.0)) for t in targets)
            + f" | {r['missing_labels']} |"
        )
    cats = sorted({c for r in results.values() for c in r["preserved_by_category"]})
    lines += ["", "Preservation by category (pooled over targets):", ""]
    lines += ["| Method | " + " | ".join(cats) + " |", "|---" * (len(cats) + 1) + "|"]
    for m, r in results.items():
        pc = r["preserved_by_category"]
        lines.append(f"| `{m}` | " + " | ".join(_pct(pc.get(c, 0.0)) for c in cats) + " |")
    lines += ["", "Preservation by target and category:", ""]
    lines += ["| Method | Target | " + " | ".join(cats) + " |", "|---" * (len(cats) + 2) + "|"]
    for m, r in results.items():
        for t, pc in r["preserved_by_target_category"].items():
            lines.append(
                f"| `{m}` | {_short(t)} | " + " | ".join(_pct(pc.get(c, 0.0)) for c in cats) + " |"
            )
    lines += ["", "Median savings by category:", ""]
    lines += ["| Method | " + " | ".join(cats) + " |", "|---" * (len(cats) + 1) + "|"]
    for m, r in results.items():
        sc = r["median_savings_by_category"]
        lines.append(f"| `{m}` | " + " | ".join(_pct(sc.get(c, 0.0)) for c in cats) + " |")

    lines += [
        "",
        "## Equal-preservation comparison",
        "",
        "The method with lower preservation is mixed with the identity rewrite (original prompt,"
        " preserved by construction) on a fraction `(p_hi - p_lo) / (1 - p_lo)` of prompts, which"
        " lifts it exactly to the other's preservation; its median savings there is the weighted"
        " median of its savings and a point mass at 0. The HMM wins when its savings at the"
        " matched preservation is strictly higher.",
        "",
        "| Scope | HMM preserved | Op preserved | Matched at | Mixed | HMM savings | Op savings |"
        " HMM wins |",
        "|---|---|---|---|---|---|---|---|",
    ]
    scopes = [(_short(t), x) for t, x in v["per_target"].items()] + [("pooled", v["pooled"])]
    for scope, c in scopes:
        lines.append(
            f"| {scope} | {_pct(c['preservation_hmm'])} | {_pct(c['preservation_op'])} | "
            f"{_pct(c['matched_preservation'])} | {c['mixed']} | {_pct(c['savings_hmm'])} | "
            f"{_pct(c['savings_op'])} | {'yes' if c['hmm_beats_op'] else 'no'} |"
        )

    if prep and prep.get("operator_table"):
        lines += [
            "",
            "## Best single operator (chosen on train, before judging)",
            "",
            f"Rule: {prep['operator_rule']}. Reward per prompt"
            " `100 * savings - 100 * 1[degraded]` (`terser.models.rl.env.stop_reward`), unchanged"
            " prompts 0; savings on Qwen2.5.",
            "",
            "| Operator | Changed | Median savings | Mean savings | Gate degraded | Mean reward |"
            " Chosen |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in prep["operator_table"]:
            lines.append(
                f"| {r['operator']} | {r['changed']} / {r['n']} | {_pct(r['median_savings'])} | "
                f"{_pct(r['mean_savings'])} | {_pct(r['gate_degraded'])} | "
                f"{r['mean_reward']:.2f} | {'yes' if r['operator'] == prep['operator'] else ''} |"
            )

    hmm_t = ", ".join(_short(t) for t in v["targets_preserved_ok"]) or "none"
    win_t = ", ".join(_short(t) for t in v["targets_beats_op"]) or "none"
    lines += [
        "",
        "## C5.6",
        "",
        f"* Median savings >= 15%: **{_pct(hmm['median_savings'])}**, "
        f"{'pass' if v['median_savings_ok'] else 'fail'}.",
        f"* Preservation >= 55% on >= 1 target: targets at >= 55%: {hmm_t}.",
        f"* Beats `{op_method}` at equal preservation (reported, not gated): targets won: {win_t}.",
        f"* Missing labels: {hmm['missing_labels']}.",
        "",
        f"**C5.6: {'pass' if v['c5_6'] else 'fail'}** (needs the savings clause and a target"
        " meeting the preservation clause; targets as revised by ADR-0012).",
    ]
    return "\n".join(lines) + "\n"


def run_report(args: argparse.Namespace) -> dict[str, Any]:
    from terser.tokenizers.count import count_tokens

    prompts = {str(p["id"]): p for p in _jsonl(args.prompts)}
    cands = _jsonl(args.candidates)
    labels = [r for path in args.labels for r in _jsonl(path)]
    prep = json.loads(args.prep.read_text(encoding="utf-8")) if args.prep.exists() else None
    methods = sorted({str(c["method"]) for c in cands}, key=lambda m: (m != HMM_METHOD, m))
    ops = [m for m in methods if m != HMM_METHOD]
    if HMM_METHOD not in methods or len(ops) != 1:
        raise ValueError(f"need method {HMM_METHOD!r} and exactly one operator, got {methods}")

    def count(text: str) -> int:
        return count_tokens(text, TOKEN_TARGET)

    results = {m: method_metrics(cands, prompts, labels, m, count) for m in methods}
    v = verdict(results[HMM_METHOD], results[ops[0]])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(_markdown(results, v, prep, ops[0]), encoding="utf-8")
    payload = {
        "methods": {m: {k: x for k, x in r.items() if k != "savings"} for m, r in results.items()},
        "operator_method": ops[0],
        "verdict": v,
        "prep": prep,
    }
    args.out.with_suffix(".json").write_text(json.dumps(payload, indent=2) + "\n", "utf-8")
    return {
        m: {"median_savings": r["median_savings"], "preserved": r["preserved"]}
        for m, r in results.items()
    } | {"c5_6": v["c5_6"]}


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(prog="python -m terser.phase5.hmm_compressor", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep", help="select val prompts and write HMM / operator candidates")
    p.add_argument("--val", type=Path, default=VAL)
    p.add_argument("--train", type=Path, default=TRAIN)
    p.add_argument("--spans", type=Path, default=SPANS)
    p.add_argument("--pairs", type=Path, default=PAIRS)
    p.add_argument("--n", type=int, default=N_PROMPTS)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--operator", default=None, help="skip the train rule and use this code")
    p.add_argument("--prompts-out", type=Path, default=PROMPTS_OUT)
    p.add_argument("--candidates-out", type=Path, default=CANDS_OUT)
    p.add_argument("--prep-out", type=Path, default=PREP_OUT)
    r = sub.add_parser("report", help="C5.6 report from judge labels")
    r.add_argument("--labels", type=Path, nargs="+", required=True)
    r.add_argument("--prompts", type=Path, default=PROMPTS_OUT)
    r.add_argument("--candidates", type=Path, default=CANDS_OUT)
    r.add_argument("--prep", type=Path, default=PREP_OUT)
    r.add_argument("--out", type=Path, default=REPORT_OUT)
    args = ap.parse_args(argv)
    res = run_prep(args) if args.cmd == "prep" else run_report(args)
    print(json.dumps(res, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
