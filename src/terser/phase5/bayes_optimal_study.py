"""Bayes optimal classifier vs the MAP hypothesis (row **5.2.4**; checkpoint **C5.4**).

``H`` is the Phase 1 hypothesis language (:mod:`terser.concept`): every conjunctive hypothesis
over the 12 attributes (5 prompt attributes + 7 operator flags), 708,589 hypotheses including
the empty one. The Phase 1 version spaces are empty in every category (C1.3), so no hypothesis is
consistent with the data; :class:`terser.models.classical.bayes_optimal.BayesOptimalClassifier`
uses a label-noise likelihood, so every hypothesis keeps a non-zero posterior.

A hypothesis predicts ``preserved`` (1) on the instances it covers. Per category and seed the
pilot labels are split 70 / 30 by ``prompt_id`` (no prompt on both sides); ``P(h | D)`` comes
from the 70% part and both rules are scored on the 30% part. C5.4 asks that the posterior-
weighted vote is no worse than the MAP hypothesis on validation accuracy.

CLI::

    python -m terser.phase5.bayes_optimal_study
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from itertools import product
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from terser.concept.attributes import DOMAINS, instance
from terser.phase1.version_spaces import POSITIVE_LABEL, AttributeCache, load_labels
from terser.phase5.bayes_studies import bayes_optimal_vs_map

__all__ = ["coverage_matrix", "hypothesis_codes", "main"]

ANY = -1
BOTTOM = -2


def hypothesis_codes(domains: Sequence[Sequence[str]] = DOMAINS) -> npt.NDArray[np.int8]:
    """Every conjunctive hypothesis as value indices (``-1`` = ``?``); row 0 is the empty one."""
    rows = np.array(
        list(product(*[range(-1, len(values)) for values in domains])), dtype=np.int8
    ).reshape(-1, len(domains))
    return np.vstack([np.full((1, len(domains)), BOTTOM, np.int8), rows])


def coverage_matrix(H: npt.NDArray[np.int8], X: npt.NDArray[np.int8]) -> npt.NDArray[np.int8]:
    """``P[h, i] = 1`` if hypothesis ``h`` covers instance ``i`` (predicts ``preserved``)."""
    cov = np.ones((len(H), len(X)), dtype=bool)
    for a in range(H.shape[1]):
        cov &= (H[:, a, None] == ANY) | (H[:, a, None] == X[None, :, a])
    return cov.astype(np.int8)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Bayes optimal vs MAP over the Phase 1 hypotheses.")
    ap.add_argument("--labels", type=Path, default=Path("data/pilot/labels.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("reports/p5_bayes_optimal.md"))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--noise", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.3])
    args = ap.parse_args(argv)

    cache = AttributeCache()
    recs = [r for r in load_labels(args.labels) if r.label is not None]
    index = [{v: i for i, v in enumerate(values)} for values in DOMAINS]
    X = np.array(
        [
            [index[a][v] for a, v in enumerate(instance(cache.get(r.prompt_id, r.original), r.ops))]
            for r in recs
        ],
        dtype=np.int8,
    )
    y = np.array([int(r.label == POSITIVE_LABEL) for r in recs])
    cats = np.array([r.category for r in recs])
    pids = np.array([r.prompt_id for r in recs])
    H = hypothesis_codes()

    rows = []
    for cat in sorted(set(cats)):
        m = cats == cat
        Xc, yc, pc = X[m], y[m], pids[m]
        P = coverage_matrix(H, Xc)
        prompts = np.array(sorted(set(pc)))
        for seed in range(args.seeds):
            rng = np.random.default_rng(seed)
            val_p = set(rng.choice(prompts, size=max(1, round(0.3 * len(prompts))), replace=False))
            va = np.array([p in val_p for p in pc])
            tr = ~va
            maj = int(yc[tr].mean() >= 0.5)
            for noise in args.noise:
                res = bayes_optimal_vs_map(P[:, tr], yc[tr], P[:, va], yc[va], noise=noise)
                rows.append(
                    {
                        "category": cat,
                        "seed": seed,
                        "noise": noise,
                        "n_val": int(va.sum()),
                        "map": res["acc_map"],
                        "bayes_optimal": res["acc_bayes_optimal"],
                        "majority": float(np.mean(yc[va] == maj)),
                        "map_posterior": res["map_posterior"],
                    }
                )
        print(json.dumps({"category": cat, "n": int(m.sum())}), flush=True)

    df = pd.DataFrame(rows)
    df["n_correct_map"] = df["map"] * df.n_val
    df["n_correct_bo"] = df.bayes_optimal * df.n_val
    by_noise = df.groupby("noise").agg(
        n_val=("n_val", "sum"), map_c=("n_correct_map", "sum"), bo_c=("n_correct_bo", "sum")
    )
    by_noise["MAP accuracy"] = by_noise.map_c / by_noise.n_val
    by_noise["Bayes optimal accuracy"] = by_noise.bo_c / by_noise.n_val
    by_noise["difference"] = by_noise["Bayes optimal accuracy"] - by_noise["MAP accuracy"]
    main_noise = 0.1 if 0.1 in args.noise else args.noise[0]
    per_cat = (
        df[df.noise == main_noise]
        .groupby("category")
        .agg(
            n_val=("n_val", "sum"),
            majority=("majority", "mean"),
            map=("map", "mean"),
            bayes_optimal=("bayes_optimal", "mean"),
            map_posterior=("map_posterior", "mean"),
        )
        .reset_index()
    )
    d_main = float(by_noise.loc[main_noise, "difference"])
    worst = float(by_noise["difference"].min())
    ok = d_main >= 0

    def md(t: pd.DataFrame) -> str:
        cols = list(t.columns)
        out = ["| " + " | ".join(cols) + " |", "|---" * len(cols) + "|"]
        for r in t.itertuples(index=False):
            out.append(
                "| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |"
            )
        return "\n".join(out)

    noise_tab = by_noise.reset_index()[
        ["noise", "n_val", "MAP accuracy", "Bayes optimal accuracy", "difference"]
    ]
    text = (
        "# Bayes optimal classifier vs MAP (row 5.2.4; C5.4)\n\n"
        "Generated by `python -m terser.phase5.bayes_optimal_study`. "
        f"H = all {len(H):,} conjunctive hypotheses of the Phase 1 language (5 prompt "
        "attributes + 7 operator flags); a hypothesis predicts *preserved* on what it covers. "
        f"{len(recs)} labelled pilot pairs (`{args.labels.as_posix()}`), per category a 70 / 30 "
        f"split by prompt, {args.seeds} seeds. Likelihood: each label flipped with probability "
        "`noise`; uniform prior.\n\n"
        "## Pooled over categories and seeds\n\n"
        + md(noise_tab)
        + f"\n\n## Per category (noise {main_noise}, mean over seeds)\n\n"
        + md(per_cat)
        + "\n\n"
        f"**C5.4** (Bayes optimal no worse than MAP on validation accuracy, noise {main_noise}): "
        f"difference {d_main:+.3f}, **{'pass' if ok else 'fail'}** (smallest difference over "
        f"the noise sweep {worst:+.3f}).\n"
    )
    args.out.write_text(text, encoding="utf-8")
    print(json.dumps({"diff": d_main, "worst": worst, "ok": ok}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
