"""Train and export the CPU fast-path models to ONNX (row **11.3.2**).

* **Gate**: scikit-learn gradient boosting on :func:`terser.features.shared.gate_vector` (the
  serving feature code, row 11.1.3) of every labelled train pair (original, candidate) ->
  ``degraded``. Scored with grouped 5-fold CV, then refit on all pairs and exported to
  ``models/fast/gate.onnx``.
* **Span tagger**: decision tree (depth 10) on :func:`terser.serving.fast_models.span_matrix`
  over the row 3.1.2 span table (droppable = DROP / REPLACE). Grouped CV token macro-F1, then
  ``models/fast/tagger.onnx``.

Both are checked for sklearn / ONNX Runtime parity (max |dP| <= 1e-4), and per-call CPU latency
is measured. Report: ``reports/p11_fast_path.md``.

CLI::

    python -m terser.phase11.train_fast_path
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from terser.eval.curves import roc_auc
from terser.eval.cv import out_of_fold_scores
from terser.features.shared import GATE_FEATURE_NAMES, extract_features, gate_vector
from terser.serving.fast_models import SPAN_FAST_FEATURES, OnnxSpanRewriter, span_matrix
from terser.serving.onnx_export import check_parity, export_sklearn
from terser.serving.paths import OnnxGate
from terser.tokenizers.count import count_tokens

__all__ = ["main"]


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


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
    args = ap.parse_args(argv)
    from sklearn.ensemble import GradientBoostingClassifier
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
    X, y, cats, groups = [], [], [], []
    for r in _jsonl(args.labels):
        p, c = prompts.get(r["prompt_id"]), cands.get(r["candidate_id"])
        if p is None or c is None or r.get("stand_in") or c.get("rejected"):
            continue
        X.append(gate_vector(feats(p["prompt"]), feats(c["text"])))
        y.append(int(r["label"] == "degraded"))
        cats.append(p["category"])
        groups.append(r["prompt_id"])
    Xg, yg = np.asarray(X, np.float32), np.asarray(y)

    def make_gate() -> GradientBoostingClassifier:
        return GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=0)

    def fp(Xt: np.ndarray, yt: np.ndarray, Xv: np.ndarray) -> np.ndarray:
        return make_gate().fit(Xt, yt).predict_proba(Xv)[:, 1]

    gate_auc = roc_auc(yg, out_of_fold_scores(fp, Xg, yg, cats, groups, seed=0).scores)
    gate = make_gate().fit(Xg, yg)
    args.out.mkdir(parents=True, exist_ok=True)
    gpath = export_sklearn(gate, len(GATE_FEATURE_NAMES), args.out / "gate.onnx")
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
    tpath = export_sklearn(tagger, len(SPAN_FAST_FEATURES), args.out / "tagger.onnx")
    tag_diff = check_parity(tagger, tpath, Xs[:5000])

    # ---- latency (single request, CPU, warm)
    og, rw = OnnxGate(gpath), OnnxSpanRewriter(tpath, cost)
    sample = [p["prompt"] for p in list(prompts.values())[:300]]
    lat, sav = [], []
    for t in sample:
        t0 = time.perf_counter()
        out = rw(t, args.target).text
        og.score(feats(t), extract_features(out, args.target, cost))
        lat.append((time.perf_counter() - t0) * 1000)
        sav.append(1 - cost(out) / max(1, cost(t)))
    lat_a = np.asarray(lat)
    res = {
        "gate_auc": gate_auc,
        "gate_parity": gate_diff,
        "tagger_macro_f1": float(np.mean(f1)),
        "tagger_parity": tag_diff,
        "p50_ms": float(np.percentile(lat_a, 50)),
        "p95_ms": float(np.percentile(lat_a, 95)),
        "median_savings": float(np.median(sav)),
    }
    args.report.write_text(
        "# CPU fast path exported to ONNX (row 11.3.2)\n\n"
        "Generated by `python -m terser.phase11.train_fast_path`. No GPU and no language model "
        "on this path: features are the serving feature code (`terser.features.shared`, "
        "`terser.serving.fast_models.span_matrix`), inference is ONNX Runtime "
        "`CPUExecutionProvider`.\n\n"
        "| component | model | training data | grouped CV | ONNX vs sklearn max dP | file |\n"
        "|---|---|---|---|---|---|\n"
        f"| gate | GradientBoosting (200 x depth 3) on {len(GATE_FEATURE_NAMES)} shared features "
        f"| {len(yg):,} labelled train pairs | AUC {gate_auc:.3f} | {gate_diff:.1e} | "
        "`models/fast/gate.onnx` |\n"
        f"| span tagger | DecisionTree (depth 10) on {len(SPAN_FAST_FEATURES)} cheap word "
        f"features | {len(ys):,} train words | droppable macro-F1 {np.mean(f1):.3f} | "
        f"{tag_diff:.1e} | `models/fast/tagger.onnx` |\n\n"
        f"Single-request CPU latency (tagger rewrite + gate, 300 train prompts, warm): p50 "
        f"**{res['p50_ms']:.1f} ms**, p95 **{res['p95_ms']:.1f} ms**. Median savings of the "
        f"fast-path rewrite before the gate: {res['median_savings']:.1%}.\n\n"
        "The gate is weaker than the Phase 5 LightGBM (AUC 0.778) because it sees only "
        "surface features (no GPT-2 surprisal, no embeddings); that is the price of a "
        "GPU-free, millisecond path. The quality path (11.3.3) carries the generative "
        "rewriter.\n",
        encoding="utf-8",
    )
    args.metrics.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
