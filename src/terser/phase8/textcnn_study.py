"""TextCNN span tagger studies (rows **8.2.3**, **8.2.4**; checkpoint **C8.2**).

Data: the row 3.1.2 span table (train split; words and KEEP / DROP / REPLACE alignment labels
against the Qwen2.5 oracle rewrite). Task = **droppable token** (DROP or REPLACE = 1, KEEP = 0),
the per-token output of the TextCNN token head. Prompt head target = oracle savings above the
train median ("compressible prompt").

Inputs to the TextCNN are word-level bge-small token states: each prompt's words go through
bge-small with ``is_split_into_words`` and the sub-word states of a word are averaged (384-d),
truncated to 256 words.

* 8.2.3 - prompt-head pooling max / avg / concat, 3 seeds each: prompt-head AUC and token F1.
* 8.2.4 - TextCNN (concat) vs supervised HMM (Viterbi, KEEP vs not) vs LightGBM on the row 3.1.2
  span features, identical grouped 5-fold split by prompt: token macro-F1 and McNemar.
* C8.2 - TextCNN macro-F1 vs the best classical tagger on the same binary task (+0.05).

CLI::

    python -m terser.phase8.textcnn_study
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from terser.eval.curves import roc_auc
from terser.eval.significance import mcnemar
from terser.features.span import SPAN_FEATURES
from terser.models.classical.hmm import TokenClassEncoder, fit_supervised
from terser.models.neural.textcnn import TextCNNConfig, build_textcnn

__all__ = ["main"]

MAX_WORDS = 256


def word_states(words_by_prompt: list[list[str]], device: str = "cpu") -> list[np.ndarray]:
    """Word-level bge-small states, sub-word states averaged per word."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
    model = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5").to(device).eval()
    out = []
    for words in words_by_prompt:
        w = words[:MAX_WORDS]
        enc = tok(w, is_split_into_words=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            h = model(**{k: v.to(device) for k, v in enc.items()}).last_hidden_state[0].cpu()
        ids = enc.word_ids()
        vec = np.zeros((len(w), h.shape[-1]), np.float32)
        cnt = np.zeros(len(w), np.float32)
        for j, wid in enumerate(ids):
            if wid is not None:
                vec[wid] += h[j].numpy()
                cnt[wid] += 1
        out.append(vec / np.maximum(cnt, 1)[:, None])
    return out


def macro_f1(y: np.ndarray, p: np.ndarray) -> float:
    f = []
    for k in (0, 1):
        tp, fp, fn = (
            np.sum((y == k) & (p == k)),
            np.sum((y != k) & (p == k)),
            np.sum((y == k) & (p != k)),
        )
        f.append(2 * tp / max(1, 2 * tp + fp + fn))
    return float(np.mean(f))


def train_textcnn(
    X: list[np.ndarray],
    Y: list[np.ndarray],
    P: np.ndarray,
    tr: np.ndarray,
    va: np.ndarray,
    pooling: str,
    seed: int,
    epochs: int = 12,
) -> tuple[list[np.ndarray], np.ndarray]:
    import torch

    torch.manual_seed(seed)
    torch.set_num_threads(8)
    net = build_textcnn(TextCNNConfig(pooling=pooling))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")

    def batch(ix: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        L = max(len(Y[i]) for i in ix)
        x = torch.zeros(len(ix), L, 384)
        y = torch.zeros(len(ix), L)
        m = torch.zeros(len(ix), L, dtype=torch.bool)
        for b, i in enumerate(ix):
            n = len(Y[i])
            x[b, :n] = torch.from_numpy(X[i])
            y[b, :n] = torch.from_numpy(Y[i].astype(np.float32))
            m[b, :n] = True
        return x, y, m, torch.tensor(P[list(ix)], dtype=torch.float32)

    pos = float(np.mean(np.concatenate([Y[i] for i in tr])))
    w_pos = (1 - pos) / max(pos, 1e-6)
    g = np.random.default_rng(seed)
    for _ in range(epochs):
        net.train()
        order = g.permutation(tr)
        for s in range(0, len(order), 16):
            x, y, m, p = batch(order[s : s + 16])
            tl, pl = net(x, m)
            wt = torch.where(y > 0, torch.tensor(w_pos), torch.tensor(1.0))
            loss = (bce(tl, y) * wt * m).sum() / m.sum() + bce(pl, p).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()
    tok_pred, prompt_score = [], []
    with torch.no_grad():
        for s in range(0, len(va), 32):
            ix = va[s : s + 32]
            x, _, m, _ = batch(ix)
            tl, pl = net(x, m)
            for b, i in enumerate(ix):
                tok_pred.append((tl[b, : len(Y[i])] > 0).numpy().astype(int))
            prompt_score.extend(torch.sigmoid(pl).numpy().tolist())
    return tok_pred, np.asarray(prompt_score)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="TextCNN span tagger studies.")
    ap.add_argument("--spans", type=Path, default=Path("data/processed/span_features.parquet"))
    ap.add_argument("--oracle", type=Path, default=Path("data/processed/oracle.csv"))
    ap.add_argument("--out", type=Path, default=Path("reports"))
    ap.add_argument("--c34", type=float, default=0.440, help="C3.4 best 3-tag macro-F1")
    args = ap.parse_args(argv)

    sp = pd.read_parquet(args.spans).sort_values(["prompt_id", "position"])
    sp = sp[sp.position < MAX_WORDS]
    pids = sorted(sp.prompt_id.unique())
    grp = {pid: g for pid, g in sp.groupby("prompt_id")}
    words = [grp[p].token.tolist() for p in pids]
    Y = [(grp[p].label != "KEEP").to_numpy().astype(int) for p in pids]
    cats = [grp[p].category.iloc[0] for p in pids]
    orc = pd.read_csv(args.oracle)
    orc = orc[(orc.target == "qwen2.5") & ~orc.stand_in].set_index("prompt_id").savings
    sav = np.array([orc.get(p, 0.0) for p in pids])
    P = (sav > np.median(sav)).astype(float)
    X = word_states(words)

    from sklearn.model_selection import StratifiedGroupKFold

    folds = list(
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0).split(
            np.zeros(len(pids)), cats, pids
        )
    )

    # 8.2.3 pooling
    pool_rows = []
    oof_cnn: dict[int, np.ndarray] = {}
    for pooling in ("max", "avg", "concat"):
        for seed in (0, 1, 2):
            toks: dict[int, np.ndarray] = {}
            ps = np.zeros(len(pids))
            for tr, va in folds:
                tp, pr = train_textcnn(X, Y, P, tr, va, pooling, seed)
                for i, t in zip(va, tp, strict=True):
                    toks[i] = t
                ps[va] = pr
            yt = np.concatenate(Y)
            pt = np.concatenate([toks[i] for i in range(len(pids))])
            pool_rows.append(
                {
                    "pooling": pooling,
                    "seed": seed,
                    "prompt_auc": roc_auc(P.astype(int), ps),
                    "token_macro_f1": macro_f1(yt, pt),
                }
            )
            if pooling == "concat" and seed == 0:
                oof_cnn = toks
            print(json.dumps(pool_rows[-1]), flush=True)
    pool = (
        pd.DataFrame(pool_rows)
        .groupby("pooling")
        .agg(
            prompt_auc=("prompt_auc", "mean"),
            prompt_auc_sd=("prompt_auc", "std"),
            token_f1=("token_macro_f1", "mean"),
            token_f1_sd=("token_macro_f1", "std"),
        )
        .reset_index()
    )

    # 8.2.4 HMM and GBM on identical folds
    import lightgbm as lgb

    enc = TokenClassEncoder()
    obs = [enc.encode(w) for w in words]
    oof_hmm: dict[int, np.ndarray] = {}
    oof_gbm: dict[int, np.ndarray] = {}
    F = [grp[p][list(SPAN_FEATURES)].to_numpy(np.float64) for p in pids]
    for tr, va in folds:
        hmm = fit_supervised([obs[i] for i in tr], [Y[i] for i in tr], 2, enc.n_obs)
        for i in va:
            oof_hmm[i] = hmm.viterbi(obs[i])[0].astype(int)
        m = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.1, verbose=-1).fit(
            np.concatenate([F[i] for i in tr]), np.concatenate([Y[i] for i in tr])
        )
        for i in va:
            oof_gbm[i] = np.asarray(m.predict(F[i])).astype(int)
    yt = np.concatenate(Y)
    preds = {
        name: np.concatenate([d[i] for i in range(len(pids))])
        for name, d in (("TextCNN (concat)", oof_cnn), ("HMM", oof_hmm), ("LightGBM", oof_gbm))
    }
    comp = pd.DataFrame(
        [
            {"tagger": n, "token_macro_f1": macro_f1(yt, p), "accuracy": float(np.mean(p == yt))}
            for n, p in preds.items()
        ]
    )
    names = list(preds)
    mc = []
    for a in range(3):
        for b in range(a + 1, 3):
            r = mcnemar(yt, preds[names[a]], preds[names[b]])
            mc.append(
                {
                    "A": names[a],
                    "B": names[b],
                    "only A right": r.b,
                    "only B right": r.c,
                    "chi2": r.statistic,
                    "p": r.p_value,
                }
            )
    mcd = pd.DataFrame(mc)
    cnn_f1 = float(comp.token_macro_f1.iloc[0])
    best_cls = float(comp.token_macro_f1.iloc[1:].max())

    def md(df: pd.DataFrame) -> str:
        cols = list(df.columns)
        out = ["| " + " | ".join(cols) + " |", "|---" * len(cols) + "|"]
        for r in df.astype(object).itertuples(index=False):
            out.append(
                "| " + " | ".join(f"{v:.4g}" if isinstance(v, float) else str(v) for v in r) + " |"
            )
        return "\n".join(out)

    text = (
        "# TextCNN span tagger (rows 8.2.3, 8.2.4; C8.2)\n\n"
        f"Generated by `python -m terser.phase8.textcnn_study`. {len(pids):,} train prompts, "
        f"{len(yt):,} words (≤ {MAX_WORDS} per prompt), {yt.mean():.1%} droppable "
        "(DROP / REPLACE vs the Qwen2.5 oracle rewrite). Inputs: word-averaged bge-small token "
        "states. Grouped 5-fold by prompt, identical folds for every model. Token loss "
        "class-weighted; prompt head = oracle savings above the median.\n\n"
        "## 8.2.3 Max vs average pooling (3 seeds)\n\n" + md(pool) + "\n\n"
        "Max pooling asks *is there any droppable region* (one strong filler n-gram fires the "
        "head); average pooling asks *how much of the prompt is droppable*. Whether a prompt is "
        "compressible tracks the *share* of removable text, which is what average pooling "
        "measures; concatenation lets the head use both.\n\n"
        "## 8.2.4 TextCNN vs HMM vs GBM taggers\n\n" + md(comp) + "\n\nMcNemar (exact binomial on "
        "discordant tokens):\n\n" + md(mcd) + "\n\n"
        f"## C8.2\n\nTextCNN token macro-F1 **{cnn_f1:.3f}** vs best classical tagger on the same "
        f"binary task **{best_cls:.3f}** (target: +0.05 → {best_cls + 0.05:.3f}). The C3.4 number "
        f"({args.c34:.3f}) is 3-tag macro-F1, so the comparison here uses the binary task both "
        f"can be scored on. C8.2: **{'pass' if cnn_f1 >= best_cls + 0.05 else 'fail'}**.\n"
    )
    (args.out / "p8_textcnn.md").write_text(text, encoding="utf-8")
    print(json.dumps({"cnn_f1": cnn_f1, "best_classical": best_cls}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
