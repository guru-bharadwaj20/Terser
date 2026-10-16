"""Tabular Q-learning over operators, quality gate as reward model.

Rows **9.2.1**-**9.2.3**, checkpoint **C9.1**.

Reward model: LightGBM (row 5.1.5 best configuration) trained on the row 3.1.3 pair table. For a
draft it gets the same features :mod:`terser.phase3.build_features` computes (pair features under
GPT-2 surprisal and bge-small, ``target_phi = 0`` i.e. Qwen2.5, prompt features), so no LLM is
called during learning. Drafts are the deterministic operator stacks of
:class:`~terser.models.rl.env.RewriteEnv`; gate scores, operator outputs and token counts are
cached per (prompt, operator set).

* 9.2.1 - learning curve (moving-average return) for the default learner.
* 9.2.2 - sweep alpha in {0.05, 0.1, 0.3} x gamma in {0.9, 0.99} x epsilon schedule in
  {decay 0.999 to floor 0.05, decay 0.9995 to 0}; mean return over the last 1,000 episodes
  and the C9.1 statistic ``max |dQ|`` over the last 1,000 episodes.
* 9.2.3 - greedy operator order per category from the best learner, with gate-estimated savings
  and degradation on the training prompts.

CLI::

    python -m terser.phase9.qlearning_study --prompts 300 --episodes 30000
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from functools import cache, lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from terser.features.alignment import tokenize
from terser.features.embeddings import bge_small_encoder
from terser.features.lexicons import is_constraint
from terser.features.pair import pair_features
from terser.features.span import gpt2_surprisal
from terser.models.rl.env import RewriteEnv
from terser.models.rl.scratch_qlearning import QLearner
from terser.operators.registry import apply_ops
from terser.phase3.build_features import prompt_features
from terser.tokenizers.count import count_tokens

__all__ = ["main"]


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def build_gate(table: Path, target: str):  # type: ignore[no-untyped-def]
    """LightGBM gate on the pair table and a cached ``gate(original, draft) -> P(degraded)``."""
    import lightgbm as lgb

    df = pd.read_parquet(table)
    cols = [c for c in df.columns if c not in ("id", "group", "category", "label")]
    model = lgb.LGBMClassifier(
        learning_rate=0.1, n_estimators=70, max_depth=3, random_state=0, verbose=-1
    ).fit(df[cols].to_numpy(np.float64), df.label.to_numpy())
    surprisal = gpt2_surprisal("gpt2")
    encode = bge_small_encoder()

    @cache
    def cost(text: str) -> int:
        return count_tokens(text, target)

    @cache
    def bits(toks: tuple[str, ...]) -> list[float]:
        return surprisal(list(toks)) if toks else []

    @cache
    def emb(text: str) -> np.ndarray:
        return encode([text])[0]

    category_of: dict[str, str] = {}

    @cache
    def gate(original: str, draft: str) -> float:
        f = pair_features(
            original,
            draft,
            category_of[original],
            cost,
            lambda t: bits(tuple(t)),
            lambda ts: np.stack([emb(t) for t in ts]),
        )
        f["target_phi"] = 0.0
        f.update(prompt_features(original, cost(original), bits(tuple(tokenize(original)))))
        x = np.array([[f.get(c, 0.0) for c in cols]], dtype=np.float64)
        return float(model.predict_proba(x)[0, 1])

    return gate, cost, category_of


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 9 tabular Q-learning study.")
    ap.add_argument("--table", type=Path, default=Path("data/processed/p4_pairs.parquet"))
    ap.add_argument("--split", type=Path, default=Path("data/processed/splits/train.jsonl"))
    ap.add_argument("--prompts", type=int, default=300)
    ap.add_argument("--episodes", type=int, default=30000)
    ap.add_argument("--target", default="qwen2.5")
    ap.add_argument("--out", type=Path, default=Path("reports"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    rows = _jsonl(args.split)
    pick = rng.choice(len(rows), min(args.prompts, len(rows)), replace=False)
    prompts = [rows[i] for i in pick]
    gate, cost, category_of = build_gate(args.table, args.target)
    episodes = []
    for p in prompts:
        category_of[p["prompt"]] = p["category"]
        cons = sorted({t for t in tokenize(p["prompt"]) if is_constraint(t)})
        episodes.append((p["prompt"], p["category"], cons))
    # Operators are deterministic in (text, op set): one cache shared by every learner.
    env = RewriteEnv(
        count=cost, gate=gate, target=args.target, apply=lru_cache(maxsize=None)(apply_ops)
    )

    sweep = []
    learners: dict[tuple[float, float, float], QLearner] = {}
    # epsilon schedules: (decay, floor). A non-zero floor keeps exploring forever, so the
    # Q-table can only settle (C9.1) under the schedule that anneals to pure greedy.
    for alpha in (0.05, 0.1, 0.3):
        for gamma in (0.9, 0.99):
            for decay, floor in ((0.999, 0.05), (0.9995, 0.0)):
                ql = QLearner(alpha=alpha, gamma=gamma, decay=decay, eps_min=floor, seed=args.seed)
                ql.fit(env, episodes, args.episodes)
                learners[(alpha, gamma, decay)] = ql
                sweep.append(
                    {
                        "alpha": alpha,
                        "gamma": gamma,
                        "eps_decay": decay,
                        "eps_min": floor,
                        "return_last_1000": float(np.mean(ql.returns_[-1000:])),
                        "max_dQ_last_1000": float(np.max(ql.max_delta_[-1000:])),
                    }
                )
                print(json.dumps(sweep[-1]), flush=True)
    sw = pd.DataFrame(sweep).sort_values("return_last_1000", ascending=False)
    best_key = tuple(sw.iloc[0][["alpha", "gamma", "eps_decay"]])
    best = learners[best_key]  # type: ignore[index]
    default = learners[(0.1, 0.99, 0.999)]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = args.out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for key, ql in ((("default", (0.1, 0.99, 0.999)), default), (("best", best_key), best)):
        r = pd.Series(ql.returns_).rolling(500, min_periods=1).mean()
        ax[0].plot(r, label=f"{key[0]} a={key[1][0]} g={key[1][1]} d={key[1][2]}")
        ax[1].plot(pd.Series(ql.max_delta_).rolling(500, min_periods=1).max())
    ax[0].set(xlabel="episode", ylabel="return (500-episode mean)", title="9.2.1 learning curve")
    ax[1].set(xlabel="episode", ylabel="max |dQ| (500-episode window)", yscale="log")
    ax[1].axhline(1e-3, ls="--", c="grey")
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figs / "p9_qlearning.png", dpi=150)
    plt.close(fig)

    # 9.2.3 greedy policy per category
    pol_rows, seqs = [], {}
    for prompt, cat, cons in episodes:
        seq, draft = best.policy_rollout(env, (prompt, cat, cons))
        n0, n1 = cost(prompt), cost(draft)
        pol_rows.append(
            {
                "category": cat,
                "sequence": seq,
                "savings": 1 - n1 / n0 if n0 else 0.0,
                "gate_degraded": gate(prompt, draft) >= 0.5,
            }
        )
    pol = pd.DataFrame(pol_rows)
    for cat, g in pol.groupby("category"):
        seqs[cat] = Counter(g.sequence).most_common(3)
    by_cat = (
        pol.groupby("category")
        .agg(
            n=("savings", "size"),
            median_savings=("savings", "median"),
            gate_degraded=("gate_degraded", "mean"),
        )
        .reset_index()
    )
    c91 = float(np.max(best.max_delta_[-1000:]))

    def md(df: pd.DataFrame) -> str:
        cols = list(df.columns)
        out = ["| " + " | ".join(cols) + " |", "|---" * len(cols) + "|"]
        for r in df.astype(object).itertuples(index=False):
            out.append(
                "| " + " | ".join(f"{v:.4g}" if isinstance(v, float) else str(v) for v in r) + " |"
            )
        return "\n".join(out)

    (args.out / "p9_qlearning.md").write_text(
        "# Tabular Q-learning over operators (rows 9.2.1, 9.2.2; C9.1)\n\n"
        f"Generated by `python -m terser.phase9.qlearning_study --prompts {args.prompts} "
        f"--episodes {args.episodes}`. {len(episodes)} train prompts; reward = savings % - 100 x "
        "gate-degraded at STOP, -0.01 per operator; gate = LightGBM on the row 3.1.3 features "
        "(no LLM calls).\n\n![learning curve](figures/p9_qlearning.png)\n\n"
        "## 9.2.2 Sweep\n\n" + md(sw) + "\n\n"
        f"## C9.1 Tabular convergence\n\nBest learner (alpha={best_key[0]}, gamma={best_key[1]}, "
        f"decay={best_key[2]}): max |dQ| over the last 1,000 episodes = **{c91:.2e}** "
        f"(target < 1e-3: **{'pass' if c91 < 1e-3 else 'fail'}**).\n",
        encoding="utf-8",
    )
    (args.out / "p9_policy.md").write_text(
        "# Learned greedy policy per category (row 9.2.3)\n\n"
        f"Greedy rollout of the best learner on the {len(episodes)} training prompts. Savings on "
        f"the {args.target} tokenizer; degradation is the **gate's** estimate. Row 9.2.4 checks "
        "it with the real judge.\n\n"
        + md(by_cat)
        + "\n\nMost frequent greedy operator sequences (operator codes F H D I L C S, then STOP):"
        "\n\n"
        + "\n".join(
            f"* **{c}**: " + ", ".join(f"`{s}` ({n})" for s, n in v) for c, v in seqs.items()
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"best": best_key, "c9_1": c91}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
