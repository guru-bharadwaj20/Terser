"""Phase 4 / 5 studies on the real span and label tables (rows **4.1.1**, **5.2.3**, **5.2.6**,
**5.3.5**; checkpoints **C3.4**, **C5.3**, **C5.5**).

* ``perceptron`` (4.1.1): from-scratch perceptron on span features. Filler-lexicon tokens the
  oracle rewrite dropped vs kept non-stopword content tokens (separable, must converge), then
  DROP vs KEEP over all tokens (not separable, must fail to converge within 100 epochs).
* ``span_tagger`` (C3.4): KEEP / DROP / REPLACE tagger from span features, decision tree and
  multinomial LR, grouped 5-fold by prompt; token macro-F1.
* ``naive_bayes`` (5.2.3 / C5.3): :func:`~terser.phase5.bayes_studies.nb_vs_lr` on a 5-word window
  around each token; label = droppable (DROP or REPLACE) in the oracle rewrite.
* ``gmm`` (5.2.6 / C5.5): 2-component GMM on the judge's ``P(rewrite) - P(original)`` per
  labelled pair; ARI against the judge labels.
* ``baum_welch`` (5.3.5): supervised HMM from alignment labels on 80% of prompts, Baum-Welch on
  unlabelled casual train prompts, token macro-F1 on the other 20% before / after.

Train split only. ``python -m terser.phase5.real_studies {study ...} --out reports``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from terser.features.alignment import tokenize
from terser.features.span import SPAN_FEATURES
from terser.models.classical.hmm import TAGS, TokenClassEncoder
from terser.models.neural.scratch_perceptron import Perceptron
from terser.phase5.bayes_studies import gmm_on_deltas, nb_vs_lr
from terser.phase5.hmm_tagger import baum_welch_f1, token_macro_f1

__all__ = ["main"]

STUDIES = ("perceptron", "span_tagger", "naive_bayes", "gmm", "baum_welch")
FIGS = Path("reports/figures")


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|---" * len(cols) + "|"]
    for r in df.astype(object).itertuples(index=False):
        out.append(
            "| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |"
        )
    return "\n".join(out)


def study_perceptron(sp: pd.DataFrame, rng: np.random.Generator) -> tuple[str, dict[str, Any]]:
    X = sp[list(SPAN_FEATURES)].to_numpy(np.float64)
    X = (X - X.mean(0)) / (X.std(0) + 1e-12)
    filler = (sp.is_filler == 1) & (sp.label == "DROP")
    content = (sp.is_filler == 0) & (sp.label == "KEEP") & (sp.stopword_ratio < 1.0)
    content &= sp.pos_NOUN.eq(1) | sp.pos_VERB.eq(1) | sp.pos_NUM.eq(1)
    sub = np.flatnonzero(filler | content)
    p_sep = Perceptron(max_epochs=100, shuffle=True, seed=0).fit(X[sub], filler.to_numpy()[sub])
    allx = rng.choice(len(sp), min(20000, len(sp)), replace=False)
    y_all = (sp.label != "KEEP").to_numpy()[allx].astype(int)
    p_all = Perceptron(max_epochs=100, shuffle=True, seed=0).fit(X[allx], y_all)
    res = {
        "separable_n": len(sub),
        "separable_converged": p_sep.converged_,
        "separable_epochs": len(p_sep.errors_),
        "full_n": len(allx),
        "full_converged": p_all.converged_,
        "full_final_errors": p_all.errors_[-1],
        "full_min_errors": min(p_all.errors_),
    }
    text = (
        "## 4.1.1 Perceptron from scratch\n\n"
        f"* **Separable subset**: {len(sub):,} tokens, filler-lexicon words the oracle rewrite "
        "dropped vs kept NOUN / VERB / NUM content words, standardised span features. "
        f"Converged: **{p_sep.converged_}** after {len(p_sep.errors_)} epochs "
        f"(mistakes per epoch {p_sep.errors_[:8]}{'...' if len(p_sep.errors_) > 8 else ''}).\n"
        f"* **Full set**: {len(allx):,} sampled tokens, DROP / REPLACE vs KEEP. Converged: "
        f"**{p_all.converged_}** within 100 epochs; mistakes per epoch stay at "
        f"{min(p_all.errors_):,}-{max(p_all.errors_):,} (last {p_all.errors_[-1]:,}). The classes "
        "overlap, so the perceptron rule keeps cycling and never stops, as the convergence "
        "theorem predicts for non-separable data.\n"
    )
    return text, res


def study_span_tagger(sp: pd.DataFrame) -> tuple[str, dict[str, Any]]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier

    X = sp[list(SPAN_FEATURES)].to_numpy(np.float64)
    y = sp.label.map({t: i for i, t in enumerate(TAGS)}).to_numpy()
    groups = sp.prompt_id.astype(str).tolist()
    cats = sp.category.astype(str).tolist()
    models = {
        "tree (depth 8, leaf 50)": lambda: DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=50, class_weight="balanced", random_state=0
        ),
        "LR (multinomial, balanced)": lambda: LogisticRegression(
            max_iter=2000, class_weight="balanced"
        ),
    }
    from sklearn.model_selection import StratifiedGroupKFold

    # grouped_folds stratifies a binary label; the tagger has three, so stratify on category.
    folds = list(
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0).split(X, cats, groups)
    )
    rows = []
    for name, make in models.items():
        pred = np.zeros_like(y)
        for tr, va in folds:
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
            m = make().fit((X[tr] - mu) / sd, y[tr])
            pred[va] = m.predict((X[va] - mu) / sd)
        per = {
            t: float(
                2 * np.sum((y == k) & (pred == k)) / max(1, np.sum(y == k) + np.sum(pred == k))
            )
            for k, t in enumerate(TAGS)
        }
        rows.append({"model": name, "macro_F1": token_macro_f1([y], [pred]), **per})
    df = pd.DataFrame(rows)
    best = float(df.macro_F1.max())
    text = (
        "## C3.4 Span tagger (best of tree / LR)\n\n"
        f"{len(sp):,} train tokens, {sp.prompt_id.nunique():,} prompts, grouped 5-fold by prompt, "
        "labels from aligning each prompt with its Qwen2.5 oracle rewrite (row 3.1.1).\n\n"
        + _md(df)
        + f"\n\nBest token macro-F1 **{best:.3f}**; C3.4 (>= 0.65): "
        f"**{'pass' if best >= 0.65 else 'fail'}**.\n"
    )
    return text, {"span_macro_f1": best, "c3_4": best >= 0.65}


def study_naive_bayes(sp: pd.DataFrame, rng: np.random.Generator) -> tuple[str, dict[str, Any]]:
    sp = sp.sort_values(["prompt_id", "position"])
    toks = sp.groupby("prompt_id").token.apply(list).to_dict()
    pick = rng.choice(len(sp), min(40000, len(sp)), replace=False)
    sub = sp.iloc[np.sort(pick)]
    texts = []
    for pid, pos, tok in zip(sub.prompt_id, sub.position, sub.token, strict=True):
        w = toks[pid]
        left, right = " ".join(w[max(0, pos - 2) : pos]), " ".join(w[pos + 1 : pos + 3])
        texts.append(f"{left} __{tok.lower()}__ {right}")
    # droppable = DROP or REPLACE, the same positive class as every other tagger here
    y = (sub.label != "KEEP").to_numpy().astype(int)
    table, curves = nb_vs_lr(texts, y, sub.category.tolist(), sub.prompt_id.tolist(), seed=0)
    p = float(table.loc[table.model == "naive_bayes", "precision_at_recall_0.70"].iloc[0])
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4))
    for name, c in curves.items():
        ax.plot(c.x, c.y, label=name)
    ax.axvline(0.70, ls="--", c="grey")
    ax.axhline(0.85, ls=":", c="grey")
    ax.set(xlabel="recall", ylabel="precision", title="DROP-token detector, grouped OOF")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS / "p5_nb_pr.png", dpi=150)
    plt.close(fig)
    text = (
        "## 5.2.3 Naive Bayes filler detector (C5.3)\n\n"
        f"{len(sub):,} sampled train tokens; text = 5-word window with the centre token marked; "
        "label = droppable in the oracle rewrite (DROP or REPLACE, "
        f"{y.mean():.1%} positive). Multinomial NB with Laplace smoothing (alpha = 1, the MAP "
        "estimate under a symmetric Dirichlet(2) prior, so no unseen n-gram gets zero "
        "probability) vs logistic regression on the same 1-3-gram counts, grouped 5-fold.\n\n"
        "![PR curves](figures/p5_nb_pr.png)\n\n"
        + _md(table)
        + f"\n\nNB precision at recall >= 0.70: **{p:.3f}**; C5.3 (>= 0.85): "
        f"**{'pass' if p >= 0.85 else 'fail'}**.\n"
    )
    return text, {"nb_precision_at_r70": p, "c5_3": p >= 0.85}


def study_gmm(labels: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    d, y = [], []
    for r in labels:
        agg = r.get("aggregate") or {}
        probs = agg.get("probs")
        if r.get("split") != "train" or r.get("stand_in") or not probs:
            continue
        d.append(probs["rewrite"] - probs["original"])
        y.append(int(r["label"] == "degraded"))
    res = gmm_on_deltas(np.asarray(d)[:, None], np.asarray(y), seed=0)
    ari = float(res["ari"])
    means = sorted((float(res["mean_degraded"]), float(res["mean_preserved"])))
    text = (
        "## 5.2.6 GMM on judge score deltas (C5.5)\n\n"
        f"{len(d):,} train label records; delta = P(rewrite better) - P(original better) from the "
        "pairwise judge's averaged verdict. 2-component GMM (from-scratch EM); the lower-mean "
        f"component is called degraded. Component means {means[0]:.3f} / {means[1]:.3f}.\n\n"
        f"ARI vs the judge labels: **{ari:.3f}**; C5.5 (>= 0.4): "
        f"**{'pass' if ari >= 0.4 else 'fail - judge-threshold review needed'}**.\n"
    )
    return text, {"gmm_ari": ari, "c5_5": ari >= 0.4}


def study_baum_welch(
    sp: pd.DataFrame, casual: list[dict[str, Any]], rng: np.random.Generator
) -> tuple[str, dict[str, Any]]:
    enc = TokenClassEncoder()
    sp = sp.sort_values(["prompt_id", "position"])
    seqs = {
        pid: (enc.encode(g.token.tolist()), g.label.map({t: i for i, t in enumerate(TAGS)}).values)
        for pid, g in sp.groupby("prompt_id")
    }
    pids = sorted(seqs)
    rng.shuffle(pids)
    cut = int(0.8 * len(pids))
    tr, va = pids[:cut], pids[cut:]
    unl = [enc.encode(tokenize(p["prompt"])) for p in casual if p["id"] not in seqs]
    unl = [u for u in unl if len(u)][:3000]
    res = baum_welch_f1(
        [seqs[p][0] for p in tr],
        [seqs[p][1] for p in tr],
        unl,
        [seqs[p][0] for p in va],
        [seqs[p][1] for p in va],
        enc.n_obs,
        n_iter=20,
    )
    ll = [float(x) for x in res["log_likelihood"]]
    text = (
        "## 5.3.5 Baum-Welch\n\n"
        f"Supervised MLE (Laplace smoothing) on {len(tr):,} labelled prompts, then Baum-Welch on "
        f"{len(unl):,} unlabelled casual train prompts (20 iterations), Viterbi token macro-F1 on "
        f"{len(va):,} held-out labelled prompts.\n\n"
        f"| model | val token macro-F1 |\n|---|---|\n| supervised | {res['f1_supervised']:.3f} |\n"
        f"| + Baum-Welch | {res['f1_baum_welch']:.3f} |\n\n"
        f"Log-likelihood {ll[0]:.0f} -> {ll[-1]:.0f} (non-decreasing: "
        f"{all(b >= a - 1e-6 for a, b in zip(ll, ll[1:], strict=False))}). "
        "EM maximises the likelihood of the *observations*, not the agreement with KEEP / DROP / "
        "REPLACE, so the hidden states can drift toward clusters of token classes that explain the "
        "text better without matching the tags.\n"
    )
    return text, {
        "f1_supervised": float(res["f1_supervised"]),
        "f1_baum_welch": float(res["f1_baum_welch"]),
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("studies", nargs="+", choices=STUDIES)
    ap.add_argument("--spans", type=Path, default=Path("data/processed/span_features.parquet"))
    ap.add_argument("--labels", type=Path, default=Path("data/processed/labels.jsonl"))
    ap.add_argument("--casual", type=Path, default=Path("data/processed/splits/train.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("reports"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    sp = pd.read_parquet(args.spans)
    names = {
        "perceptron": "p4_perceptron.md",
        "span_tagger": "p3_span_tagger.md",
        "naive_bayes": "p5_naive_bayes.md",
        "gmm": "p5_gmm_deltas.md",
        "baum_welch": "p5_baum_welch.md",
    }
    for s in args.studies:
        if s == "perceptron":
            text, res = study_perceptron(sp, rng)
        elif s == "span_tagger":
            text, res = study_span_tagger(sp)
        elif s == "naive_bayes":
            text, res = study_naive_bayes(sp, rng)
        elif s == "gmm":
            text, res = study_gmm(_jsonl(args.labels))
        else:
            text, res = study_baum_welch(sp, _jsonl(args.casual), rng)
        head = f"# {names[s][:-3]}\n\nGenerated by `python -m terser.phase5.real_studies {s}`.\n\n"
        (args.out / names[s]).write_text(head + text, encoding="utf-8")
        print(json.dumps({s: res}, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
