"""Train and export the CPU fast-path models to ONNX (row **11.3.2**).

* **Gate**: scikit-learn gradient boosting on :func:`terser.features.shared.gate_vector` (the
  serving feature code, row 11.1.3) of every labelled train pair (original, candidate) ->
  ``degraded``. Scored with grouped 5-fold CV, then refit on all pairs and exported to
  ``models/fast/gate.onnx``.
* **Gate operating point** (row **3.2.3**): on the same grouped out-of-fold scores, the threshold
  that maximises specificity subject to sensitivity >= 0.95
  (:func:`terser.eval.threshold.choose_threshold`), translated to the service's rejection rule
  ``P(degraded) > gate_threshold`` (:func:`service_threshold`). Sensitivity, specificity and the
  share of rewrites passed are reported at that threshold and at the old default 0.5, overall
  and per category, together with the fast path's median savings after gating.
* **Span tagger**: decision tree (depth 10) on :func:`terser.serving.fast_models.span_matrix`
  over the row 3.1.2 span table (droppable = DROP / REPLACE). Grouped CV token macro-F1, then
  ``models/fast/tagger.onnx``.

Both are checked for sklearn / ONNX Runtime parity (max |dP| <= 1e-4), and per-call CPU latency
is measured. Report: ``reports/p11_fast_path.md``.

CLI::

    python -m terser.phase11.train_fast_path
    # operating point only: no tagger training, no export, latency kept from the JSON
    python -m terser.phase11.train_fast_path --operating-point-only
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from terser.eval.curves import average_precision, roc_auc
from terser.eval.cv import OofResult, out_of_fold_scores
from terser.eval.threshold import MIN_SENSITIVITY, choose_threshold
from terser.features.shared import GATE_FEATURE_NAMES, extract_features, gate_vector
from terser.serving.fast_models import SPAN_FAST_FEATURES, OnnxSpanRewriter, span_matrix
from terser.serving.onnx_export import check_parity, export_sklearn
from terser.serving.paths import OnnxGate
from terser.tokenizers.count import count_tokens

__all__ = [
    "DEFAULT_THRESHOLD",
    "GatePairs",
    "fast_path_savings",
    "gate_oof",
    "gate_pairs",
    "main",
    "make_gate",
    "operating_point",
    "render_report",
    "service_threshold",
]

DEFAULT_THRESHOLD = 0.5
"""The ``gate_threshold`` the serving configs used before the row 3.2.3 rule was applied."""

N_LATENCY_PROMPTS = 300


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def make_gate() -> Any:
    """The deployed gate's model spec (``terser-gate``)."""
    from sklearn.ensemble import GradientBoostingClassifier

    return GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=0)


@dataclass(frozen=True)
class GatePairs:
    """Gate training matrix: one row per labelled (train prompt, candidate, target) record."""

    X: npt.NDArray[np.float32]
    y: npt.NDArray[np.int64]
    categories: list[str]
    groups: list[str]


def gate_pairs(
    prompts: Mapping[str, Mapping[str, Any]],
    cands: Mapping[str, Mapping[str, Any]],
    labels: Iterable[Mapping[str, Any]],
    feats: Callable[[str], dict[str, float]],
) -> GatePairs:
    """Gate vectors of every labelled pair whose prompt is in ``prompts`` (stand-in and rejected
    candidates skipped); group = source prompt id."""
    X, y, cats, groups = [], [], [], []
    for r in labels:
        p, c = prompts.get(r["prompt_id"]), cands.get(r["candidate_id"])
        if p is None or c is None or r.get("stand_in") or c.get("rejected"):
            continue
        X.append(gate_vector(feats(p["prompt"]), feats(c["text"])))
        y.append(int(r["label"] == "degraded"))
        cats.append(p["category"])
        groups.append(r["prompt_id"])
    return GatePairs(np.asarray(X, np.float32), np.asarray(y, np.int64), cats, groups)


def gate_oof(pairs: GatePairs, seed: int = 0) -> tuple[OofResult, list[Any]]:
    """Grouped 5-fold out-of-fold scores of :func:`make_gate`, and the model fitted in each fold
    (``models[k]`` never saw the groups with ``fold == k``)."""
    models: list[Any] = []

    def fp(Xt: np.ndarray, yt: np.ndarray, Xv: np.ndarray) -> np.ndarray:
        m = make_gate().fit(Xt, yt)
        models.append(m)
        return np.asarray(m.predict_proba(Xv)[:, 1], dtype=np.float64)

    oof = out_of_fold_scores(fp, pairs.X, pairs.y, pairs.categories, pairs.groups, seed=seed)
    return oof, models


def service_threshold(
    y: npt.ArrayLike, scores: npt.ArrayLike, min_sensitivity: float = MIN_SENSITIVITY
) -> float:
    """``gate_threshold`` value for the row 3.2.3 rule under the service's rejection rule.

    :func:`~terser.eval.threshold.choose_threshold` flags ``score >= t``; the service rejects
    ``score > gate_threshold``. The returned value ``v`` is the shortest decimal with
    ``lo <= v < t`` (``lo`` = the next lower distinct score), so ``scores > v`` flags exactly
    the examples ``scores >= t`` does.
    """
    s = np.asarray(scores, dtype=np.float64)
    t = choose_threshold(np.asarray(y), s, min_sensitivity).threshold
    below = s[s < t]
    lo = float(below.max()) if below.size else t - 1.0
    v = lo
    for digits in range(2, 17):
        cand = round((lo + t) / 2, digits)
        if lo <= cand < t:
            v = cand
            break
    if not np.array_equal(s > v, s >= t):
        raise RuntimeError(f"threshold translation failed: t={t!r}, v={v!r}")
    return v


def _rates(y: np.ndarray, rejected: np.ndarray) -> dict[str, Any]:
    pos, passed = y == 1, ~rejected
    return {
        "n": int(len(y)),
        "sensitivity": float(rejected[pos].mean()) if pos.any() else None,
        "specificity": float(passed[~pos].mean()) if (~pos).any() else None,
        "pass_rate": float(passed.mean()),
        "degraded_share_passed": float(y[passed].mean()) if passed.any() else None,
    }


def operating_point(
    y: npt.ArrayLike,
    scores: npt.ArrayLike,
    threshold: float,
    categories: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Rates when the service rejects ``score > threshold`` (degraded = positive).

    ``sensitivity`` = share of degraded rewrites rejected, ``specificity`` = share of preserved
    rewrites passed, ``pass_rate`` = share of all rewrites passed, ``degraded_share_passed`` =
    degraded share among the passed ones. With ``categories``, the same per category plus
    ROC-AUC and AP.
    """
    ya, s = np.asarray(y, dtype=np.int64), np.asarray(scores, dtype=np.float64)
    out: dict[str, Any] = {"threshold": float(threshold), **_rates(ya, s > threshold)}
    if categories is not None:
        cats = np.asarray(categories)
        per: dict[str, Any] = {}
        for c in sorted(set(categories)):
            m = cats == c
            per[c] = _rates(ya[m], s[m] > threshold)
            if len(np.unique(ya[m])) == 2:
                per[c]["auc"] = roc_auc(ya[m], s[m])
                per[c]["ap"] = average_precision(ya[m], s[m])
        out["per_category"] = per
    return out


def fast_path_savings(
    prompts: Sequence[str],
    rewrite: Callable[[str], str],
    cost: Callable[[str], int],
    score: Callable[[int, str, str], float],
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    """Fast-path token savings per prompt before and after the gate.

    ``score(i, prompt, rewrite)`` is the gate's ``P(degraded)``; as in the service, an unchanged
    prompt is not scored (score 0) and a rejected rewrite saves nothing.
    """
    sav, sc, changed = [], [], []
    for i, t in enumerate(prompts):
        out = rewrite(t)
        sav.append(1 - cost(out) / max(1, cost(t)))
        sc.append(score(i, t, out) if out != t else 0.0)
        changed.append(out != t)
    sav_a, sc_a, ch = np.asarray(sav), np.asarray(sc), np.asarray(changed, dtype=bool)
    res: dict[str, Any] = {
        "n_prompts": len(prompts),
        "n_changed": int(ch.sum()),
        "median_savings_before_gate": float(np.median(sav_a)),
        "mean_savings_before_gate": float(np.mean(sav_a)),
    }
    for name, thr in thresholds.items():
        rej = sc_a > thr
        after = np.where(rej, 0.0, sav_a)
        res[name] = {
            "threshold": float(thr),
            "rejected": int(rej.sum()),
            "rejected_share_of_changed": float(rej[ch].mean()) if ch.any() else None,
            "median_savings_after_gate": float(np.median(after)),
            "mean_savings_after_gate": float(np.mean(after)),
        }
    return res


def _op_row(name: str, op: Mapping[str, Any]) -> str:
    return (
        f"| {name} | {op['threshold']:g} | {op['sensitivity']:.3f} | {op['specificity']:.3f} "
        f"| {op['pass_rate']:.1%} | {op['degraded_share_passed']:.1%} |"
    )


def _savings_row(name: str, s: Mapping[str, Any], n_changed: int) -> str:
    return (
        f"| {name} | {s['threshold']:g} | {s['rejected']} of {n_changed} "
        f"({s['rejected_share_of_changed']:.1%}) | {s['median_savings_after_gate']:.1%} "
        f"| {s['mean_savings_after_gate']:.1%} |"
    )


def render_report(res: Mapping[str, Any]) -> str:
    """``reports/p11_fast_path.md`` from the metrics JSON."""
    op = res["operating_point"]
    chosen, default = op["chosen"], op["default_0_5"]
    e2e, dep = op["savings_oof"], op["savings_deployed"]
    n_ch = e2e["n_changed"]
    cat_rows = "\n".join(
        f"| {c} | {v['n']:,} | {v['auc']:.3f} | {v['ap']:.3f} | {v['sensitivity']:.3f} "
        f"| {v['specificity']:.3f} | {v['pass_rate']:.1%} | "
        f"{default['per_category'][c]['sensitivity']:.3f} | "
        f"{default['per_category'][c]['specificity']:.3f} |"
        for c, v in chosen["per_category"].items()
    )
    below = [
        f"{c} ({v['sensitivity']:.3f})"
        for c, v in chosen["per_category"].items()
        if v["sensitivity"] < MIN_SENSITIVITY
    ]
    below_txt = (
        "Categories below the 0.95 floor at the pooled threshold: " + ", ".join(below) + "."
        if below
        else "Every category meets the 0.95 floor at the pooled threshold."
    )
    savings_head = (
        "| threshold | value | rewrites rejected | median savings after gate | mean savings "
        "after gate |\n|---|---:|---:|---:|---:|\n"
    )
    return (
        "# CPU fast path exported to ONNX (row 11.3.2)\n\n"
        "Generated by `python -m terser.phase11.train_fast_path`. No GPU and no language model "
        "on this path: features are the serving feature code (`terser.features.shared`, "
        "`terser.serving.fast_models.span_matrix`), inference is ONNX Runtime "
        "`CPUExecutionProvider`.\n\n"
        "| component | model | training data | grouped CV | ONNX vs sklearn max dP | file |\n"
        "|---|---|---|---|---|---|\n"
        f"| gate | GradientBoosting (200 x depth 3) on {res['n_gate_features']} shared features "
        f"| {res['n_gate_pairs']:,} labelled train pairs | AUC {res['gate_auc']:.3f} | "
        f"{res['gate_parity']:.1e} | `models/fast/gate.onnx` |\n"
        f"| span tagger | DecisionTree (depth 10) on {res['n_tagger_features']} cheap word "
        f"features | {res['n_tagger_words']:,} train words | droppable macro-F1 "
        f"{res['tagger_macro_f1']:.3f} | {res['tagger_parity']:.1e} | "
        "`models/fast/tagger.onnx` |\n\n"
        f"Single-request CPU latency (tagger rewrite + gate, {N_LATENCY_PROMPTS} train prompts, "
        f"warm): p50 **{res['p50_ms']:.1f} ms**, p95 **{res['p95_ms']:.1f} ms**. Median savings "
        f"of the fast-path rewrite before the gate: {res['median_savings']:.1%}.\n\n"
        "The gate is weaker than the Phase 5 LightGBM (AUC 0.778) because it sees only "
        "surface features (no GPT-2 surprisal, no embeddings); that is the price of a "
        "GPU-free, millisecond path. The quality path (11.3.3) carries the generative "
        "rewriter.\n\n"
        "## Gate operating point (row 3.2.3)\n\n"
        "Scores: the grouped 5-fold out-of-fold `P(degraded)` behind the AUC above (same "
        f"{res['n_gate_features']} features, same model spec, groups = source prompt, seed 0; "
        f"{res['n_gate_pairs']:,} pairs, degraded = positive). The service rejects a rewrite "
        "when `P(degraded) > gate_threshold` and returns the original prompt, so sensitivity "
        "is the share of degraded rewrites rejected and specificity the share of preserved "
        "rewrites passed. Rule: maximise specificity subject to sensitivity >= 0.95 "
        "(`terser.eval.threshold.choose_threshold`, which flags `score >= "
        f"{op['choose_threshold']:.6f}`), written as the shortest decimal that flags the same "
        "pairs under the service's strict `>` (`service_threshold`).\n\n"
        "| threshold | value | sensitivity | specificity | rewrites passed | degraded among "
        "passed |\n"
        "|---|---:|---:|---:|---:|---:|\n"
        f"{_op_row('old default', default)}\n"
        f"{_op_row('**row 3.2.3 rule (deployed)**', chosen)}\n\n"
        f"Overall degraded share: {op['degraded_share']:.1%}. Average precision (pooled OOF): "
        f"{op['ap']:.3f}. At 0.5 the gate passes {1 - default['sensitivity']:.1%} of degraded "
        "rewrites, so it does not meet the 0.95 sensitivity floor. The chosen value is set as "
        f"`gate_threshold: {chosen['threshold']!r}` in `configs/serving/fast_onnx.yaml`.\n\n"
        "Per category at the chosen threshold (fitted on the pooled scores, not per "
        "category):\n\n"
        "| category | pairs | ROC-AUC | AP | sensitivity | specificity | passed | "
        "sensitivity at 0.5 | specificity at 0.5 |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"
        f"{cat_rows}\n\n"
        f"{below_txt}\n\n"
        "### Fast-path savings after the gate\n\n"
        f"Same {e2e['n_prompts']} train prompts as the latency run. The deployed tagger "
        f"(`models/fast/tagger.onnx`) changed {n_ch} of them. A rejected rewrite saves 0. "
        f"Median savings before the gate: {e2e['median_savings_before_gate']:.1%} (mean "
        f"{e2e['mean_savings_before_gate']:.1%}).\n\n"
        "Gate scores from the fold model that did not see the prompt's pairs (grouped OOF; "
        f"{op['savings_oof_refit_prompts']} prompts with no labelled pairs use the refit "
        "gate):\n\n"
        f"{savings_head}"
        f"{_savings_row('old default', e2e['default_0_5'], n_ch)}\n"
        f"{_savings_row('row 3.2.3 rule', e2e['chosen'], n_ch)}\n\n"
        "Median savings after the gate: "
        f"{e2e['default_0_5']['median_savings_after_gate']:.1%} at 0.5 and "
        f"{e2e['chosen']['median_savings_after_gate']:.1%} at the row 3.2.3 threshold, against "
        f"{e2e['median_savings_before_gate']:.1%} before it.\n\n"
        "With the deployed `gate.onnx` (refit on all train pairs, so these prompts' own "
        "candidate pairs were in its training data):\n\n"
        f"{savings_head}"
        f"{_savings_row('old default', dep['default_0_5'], n_ch)}\n"
        f"{_savings_row('row 3.2.3 rule', dep['chosen'], n_ch)}\n\n"
        "Caveats: the gate was trained on Phase 2 candidates, not on tagger rewrites, and no "
        "judge labels exist for tagger rewrites (checkpoint C5.6), so the gate's sensitivity on "
        "what the fast path actually produces is not measured. The tagger was trained on these "
        "train prompts. The threshold is chosen on the same pooled OOF scores its sensitivity / "
        "specificity are read from, so sensitivity 0.95 holds by construction (in-sample). "
        "The threshold is fitted on fold-model scores and applied to the refit "
        "`gate.onnx`. Refit sklearn gate vs the exported `gate.onnx`: max dP "
        f"{op['deployed_parity']:.1e} on 2,000 training rows.\n"
    )


def _gate_operating_point(
    target: str,
    pairs: GatePairs,
    oof: OofResult,
    fold_models: Sequence[Any],
    gate: Any,
    sample: Sequence[Mapping[str, Any]],
    cost: Callable[[str], int],
    feats: Callable[[str], dict[str, float]],
    gpath: Path,
    tpath: Path,
) -> dict[str, Any]:
    thr = service_threshold(pairs.y, oof.scores)
    thresholds = {"default_0_5": DEFAULT_THRESHOLD, "chosen": thr}
    op: dict[str, Any] = {
        "rule": "maximise specificity subject to sensitivity >= 0.95 (row 3.2.3)",
        "scores": "grouped 5-fold OOF, seed 0",
        # The threshold is chosen on these same pooled OOF scores and the rates below are read
        # off them, so sensitivity >= 0.95 holds by construction (an in-sample operating point).
        "threshold_selection": "in-sample on the pooled OOF scores",
        "reject_rule": "P(degraded) > gate_threshold",
        "degraded_share": float(pairs.y.mean()),
        "ap": average_precision(pairs.y, oof.scores),
        "choose_threshold": choose_threshold(pairs.y, oof.scores).threshold,
    }
    for name, t in thresholds.items():
        op[name] = operating_point(pairs.y, oof.scores, t, pairs.categories)

    fold_of = {g: int(k) for g, k in zip(pairs.groups, oof.fold.tolist(), strict=True)}
    texts = [p["prompt"] for p in sample]
    ids = [p["id"] for p in sample]
    rw = OnnxSpanRewriter(tpath, cost)

    def rewrite(t: str) -> str:
        return rw(t, target).text

    def oof_score(i: int, t: str, out: str) -> float:
        k = fold_of.get(ids[i])
        model = gate if k is None else fold_models[k]
        x = np.asarray([gate_vector(feats(t), feats(out))], dtype=np.float32)
        return float(model.predict_proba(x)[0, 1])

    og = OnnxGate(gpath)
    op["savings_oof"] = fast_path_savings(texts, rewrite, cost, oof_score, thresholds)
    op["savings_oof_refit_prompts"] = sum(1 for i in ids if i not in fold_of)
    op["savings_deployed"] = fast_path_savings(
        texts, rewrite, cost, lambda i, t, out: og.score(feats(t), feats(out)), thresholds
    )
    op["deployed_parity"] = check_parity(gate, gpath, pairs.X[:2000])
    return op


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train + export the fast-path gate and tagger.")
    ap.add_argument("--prompts", type=Path, default=Path("data/interim/label_prompts.jsonl"))
    ap.add_argument("--candidates", type=Path, default=Path("data/interim/candidates.jsonl"))
    ap.add_argument("--labels", type=Path, default=Path("data/processed/labels.jsonl"))
    ap.add_argument("--spans", type=Path, default=Path("data/processed/span_features.parquet"))
    ap.add_argument("--out", type=Path, default=Path("models/fast"))
    ap.add_argument("--report", type=Path, default=Path("reports/p11_fast_path.md"))
    ap.add_argument("--metrics", type=Path, default=Path("reports/p11_fast_path.json"))
    ap.add_argument("--target", default="qwen2.5")
    ap.add_argument(
        "--operating-point-only",
        action="store_true",
        help="score the gate operating point of the models already in --out: no tagger "
        "training, no export; tagger and latency metrics are kept from --metrics",
    )
    args = ap.parse_args(argv)
    from sklearn.tree import DecisionTreeClassifier

    @cache
    def cost(t: str) -> int:
        return count_tokens(t, args.target)

    @cache
    def feats(t: str) -> dict[str, float]:
        return extract_features(t, args.target, cost)

    # ---- gate
    prompts = {p["id"]: p for p in _jsonl(args.prompts) if p.get("split") == "train"}
    cands = {c["candidate_id"]: c for c in _jsonl(args.candidates)}
    pairs = gate_pairs(prompts, cands, _jsonl(args.labels), feats)
    Xg, yg = pairs.X, pairs.y
    oof, fold_models = gate_oof(pairs)
    gate_auc = roc_auc(yg, oof.scores)
    gate = make_gate().fit(Xg, yg)
    gpath, tpath = args.out / "gate.onnx", args.out / "tagger.onnx"
    sample = list(prompts.values())[:N_LATENCY_PROMPTS]

    def finish(res: dict[str, Any]) -> None:
        res["operating_point"] = _gate_operating_point(
            args.target, pairs, oof, fold_models, gate, sample, cost, feats, gpath, tpath
        )
        args.report.write_text(render_report(res), encoding="utf-8")
        args.metrics.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(res))

    if args.operating_point_only:
        res = json.loads(args.metrics.read_text(encoding="utf-8"))
        if abs(res["gate_auc"] - gate_auc) > 1e-9:
            raise SystemExit(
                f"OOF AUC {gate_auc!r} != stored gate_auc {res['gate_auc']!r}: the inputs or "
                "the model spec changed, rerun without --operating-point-only"
            )
        res.update(
            n_gate_features=len(GATE_FEATURE_NAMES),
            n_gate_pairs=len(yg),
            n_tagger_features=len(SPAN_FAST_FEATURES),
            n_tagger_words=len(pd.read_parquet(args.spans, columns=["prompt_id"])),
        )
        finish(res)
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    gpath = export_sklearn(gate, len(GATE_FEATURE_NAMES), gpath)
    gate_diff = check_parity(gate, gpath, Xg[:2000])

    # ---- span tagger
    sp = pd.read_parquet(args.spans).sort_values(["prompt_id", "position"])
    mats, labs, grp_tok = [], [], []
    for pid, g in sp.groupby("prompt_id"):
        mats.append(span_matrix(g.token.tolist(), cost))
        labs.append((g.label != "KEEP").to_numpy().astype(int))
        grp_tok.extend([pid] * len(g))
    Xs, ys = np.concatenate(mats), np.concatenate(labs)
    from sklearn.model_selection import GroupKFold

    pred = np.zeros_like(ys)
    for tr, va in GroupKFold(n_splits=5).split(Xs, ys, grp_tok):
        m = DecisionTreeClassifier(max_depth=10, min_samples_leaf=50, random_state=0)
        pred[va] = m.fit(Xs[tr], ys[tr]).predict(Xs[va])
    f1 = []
    for k in (0, 1):
        tp = np.sum((ys == k) & (pred == k))
        f1.append(
            2
            * tp
            / max(1, 2 * tp + np.sum((ys != k) & (pred == k)) + np.sum((ys == k) & (pred != k)))
        )
    tagger = DecisionTreeClassifier(max_depth=10, min_samples_leaf=50, random_state=0).fit(Xs, ys)
    tpath = export_sklearn(tagger, len(SPAN_FAST_FEATURES), tpath)
    tag_diff = check_parity(tagger, tpath, Xs[:5000])

    # ---- latency (single request, CPU, warm): explicit warm-up calls first, and the original
    # prompt's features are extracted uncached inside the timer, as the server does per request.
    og, rw = OnnxGate(gpath), OnnxSpanRewriter(tpath, cost)
    for t in [p["prompt"] for p in sample][:5]:
        og.score(extract_features(t, args.target, cost), extract_features(t, args.target, cost))
        rw(t, args.target)
    lat, sav = [], []
    for t in (p["prompt"] for p in sample):
        t0 = time.perf_counter()
        out = rw(t, args.target).text
        og.score(extract_features(t, args.target, cost), extract_features(out, args.target, cost))
        lat.append((time.perf_counter() - t0) * 1000)
        sav.append(1 - cost(out) / max(1, cost(t)))
    lat_a = np.asarray(lat)
    finish(
        {
            "gate_auc": gate_auc,
            "gate_parity": gate_diff,
            "tagger_macro_f1": float(np.mean(f1)),
            "tagger_parity": tag_diff,
            "p50_ms": float(np.percentile(lat_a, 50)),
            "p95_ms": float(np.percentile(lat_a, 95)),
            "median_savings": float(np.median(sav)),
            "n_gate_features": len(GATE_FEATURE_NAMES),
            "n_gate_pairs": len(yg),
            "n_tagger_features": len(SPAN_FAST_FEATURES),
            "n_tagger_words": len(ys),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
