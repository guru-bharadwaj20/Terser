"""Span-tagger compressors for the final table: LightGBM and TextCNN taggers (row 12.2).

Both taggers are the row 8.2.4 models, refit on the whole row 3.1.2 train span table
(``data/processed/span_features.parquet``; droppable = DROP or REPLACE against the Qwen2.5 oracle
rewrite):

* **GBM tagger** -- ``LGBMClassifier(n_estimators=200, learning_rate=0.1)`` on
  :data:`terser.features.span.SPAN_FEATURES` (Qwen2.5 token cost, GPT-2 surprisal, spaCy POS,
  lexicon flags). A new prompt gets the same features from
  :func:`terser.features.span.span_features`.
* **TextCNN tagger** -- the concat-pooling TextCNN of row 8.2.3 on word-averaged bge-small states,
  trained by :func:`terser.phase8.textcnn_study.train_textcnn` (same code, same 12 epochs) with the
  new prompts as its prediction set. Words past :data:`~terser.phase8.textcnn_study.MAX_WORDS` keep.

**Rewrite rule** (:func:`delete_tokens`). Tokens are the row 3.1.1 units
(:data:`terser.features.alignment.TOKEN_RE`). A token predicted droppable (``P >= 0.5``) is
deleted unless it is inside an operator payload span (:data:`terser.operators.masking.PROTECTED`)
or is a negation, number or constraint keyword (the fast path's hard-keep rule,
:class:`terser.serving.fast_models.OnnxSpanRewriter`). Lines without a deletion are copied byte
for byte; edited lines get the operator spacing / punctuation repairs. This mirrors the row 5.3.6
HMM rewrite (:func:`terser.phase5.hmm_compressor.hmm_rewrite`) plus the hard-keep rule.
The gate is applied later, in :mod:`terser.phase12.results`, so one judged candidate serves both
the gated system and the gate-off ablation.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from terser.features.alignment import TOKEN_RE
from terser.features.lexicons import is_constraint, is_negation, is_number
from terser.operators.masking import PROTECTED

__all__ = [
    "GBM_PARAMS",
    "delete_tokens",
    "fit_gbm_tagger",
    "gbm_drop_mask",
    "hard_keep",
    "textcnn_drop_masks",
]

GBM_PARAMS: dict[str, Any] = {"n_estimators": 200, "learning_rate": 0.1, "verbose": -1}
"""Row 8.2.4 LightGBM tagger settings."""


def hard_keep(token: str) -> bool:
    """Never delete negations, numbers or constraint keywords (fast-path rule)."""
    return bool(is_negation(token) or is_number(token) or is_constraint(token))


def _repair(body: str) -> str:
    body = re.sub(r"[ \t]+", " ", body)
    body = re.sub(r" +([,.;:!?])", r"\1", body)
    body = re.sub(r"([,;:])(?: *[,;:])+", r"\1", body)
    body = re.sub(r", *([.!?])", r"\1", body)
    body = re.sub(r"^[ ,;:.!?]+", "", body).strip()
    return body if any(ch.isalnum() for ch in body) else ""


def delete_tokens(text: str, drop: Sequence[bool], *, protect_lexicon: bool = True) -> str:
    """Delete the ``TOKEN_RE`` tokens of ``text`` flagged in ``drop`` (see the module docstring).

    Raises:
        ValueError: If ``drop`` does not have one flag per token.
    """
    matches = list(TOKEN_RE.finditer(text))
    if len(drop) != len(matches):
        raise ValueError(f"expected {len(matches)} drop flags, got {len(drop)}")
    guard = [m.span() for m in PROTECTED.finditer(text)]
    cuts = [
        m.span()
        for m, d in zip(matches, drop, strict=True)
        if d
        and not any(s < m.end() and m.start() < e for s, e in guard)
        and not (protect_lexicon and hard_keep(m.group(0)))
    ]
    if not cuts:
        return text
    out_lines: list[str] = []
    offset, k = 0, 0
    for line in text.split("\n"):
        end = offset + len(line)
        cut: list[tuple[int, int]] = []
        while k < len(cuts) and cuts[k][0] < end:
            cut.append((cuts[k][0] - offset, cuts[k][1] - offset))
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
    return out if out.strip() else text


# --------------------------------------------------------------------------------------------
# LightGBM tagger
# --------------------------------------------------------------------------------------------


def fit_gbm_tagger(spans: Any, seed: int = 0) -> Any:
    """Row 8.2.4 LightGBM tagger on a span table (``label`` KEEP / DROP / REPLACE)."""
    import lightgbm as lgb

    from terser.features.span import SPAN_FEATURES

    X = spans[list(SPAN_FEATURES)].to_numpy(np.float64)
    y = (spans.label != "KEEP").to_numpy().astype(int)
    return lgb.LGBMClassifier(**GBM_PARAMS, random_state=seed).fit(X, y)


def gbm_drop_mask(
    model: Any,
    text: str,
    cost: Callable[[str], int],
    surprisal: Callable[[Sequence[str]], Sequence[float]],
    pos: Callable[[Sequence[str]], Sequence[str]] | None,
    threshold: float = 0.5,
) -> list[bool]:
    """Per-token droppable flags of ``text`` from the LightGBM tagger."""
    from terser.features.span import SPAN_FEATURES, span_features

    tokens = [m.group(0) for m in TOKEN_RE.finditer(text)]
    if not tokens:
        return []
    rows = span_features(tokens, cost, surprisal, pos)
    X = np.asarray([[r[f] for f in SPAN_FEATURES] for r in rows], dtype=np.float64)
    p = np.asarray(model.predict_proba(X))[:, 1]
    return [bool(x >= threshold) for x in p]


# --------------------------------------------------------------------------------------------
# TextCNN tagger
# --------------------------------------------------------------------------------------------


def textcnn_drop_masks(
    spans: Any,
    oracle: Any,
    texts: Sequence[str],
    *,
    seed: int = 0,
    pooling: str = "concat",
    epochs: int = 12,
    word_states: Callable[[list[list[str]]], tuple[list[np.ndarray], int]] | None = None,
    train: Callable[..., tuple[list[np.ndarray], np.ndarray]] | None = None,
) -> list[list[bool]]:
    """Per-token droppable flags for ``texts`` from a TextCNN fitted on the train span table.

    ``oracle`` is the oracle table (``prompt_id``, ``target``, ``stand_in``, ``savings``) for the
    prompt-head target, as in :func:`terser.phase8.textcnn_study.main`. ``word_states`` and
    ``train`` default to the row 8.2 implementations (injectable for tests).
    """
    from terser.phase8 import textcnn_study as tc

    states = word_states or tc.word_states
    fit = train or tc.train_textcnn
    sp = spans.sort_values(["prompt_id", "position"])
    sp = sp[sp.position < tc.MAX_WORDS]
    pids = sorted(sp.prompt_id.unique())
    grp = {pid: g for pid, g in sp.groupby("prompt_id")}
    words = [grp[p].token.tolist() for p in pids]
    Y = [(grp[p].label != "KEEP").to_numpy().astype(int) for p in pids]
    orc = oracle[(oracle.target == "qwen2.5") & ~oracle.stand_in].set_index("prompt_id").savings
    sav = np.array([orc.get(p, 0.0) for p in pids])
    P = (sav > np.median(sav)).astype(float)

    new_tokens = [[m.group(0) for m in TOKEN_RE.finditer(t)] for t in texts]
    new_words = [w[: tc.MAX_WORDS] for w in new_tokens]
    keep_new = [i for i, w in enumerate(new_words) if w]
    X, _ = states(words + [new_words[i] for i in keep_new])
    Y_all = Y + [np.zeros(len(new_words[i]), dtype=int) for i in keep_new]
    P_all = np.concatenate([P, np.zeros(len(keep_new))])
    tr = np.arange(len(pids))
    va = np.arange(len(pids), len(pids) + len(keep_new))
    preds, _ = fit(X, Y_all, P_all, tr, va, pooling, seed, epochs)
    out: list[list[bool]] = [[False] * len(t) for t in new_tokens]
    for i, pr in zip(keep_new, preds, strict=True):
        flags = [bool(x) for x in np.asarray(pr).tolist()]
        out[i] = flags + [False] * (len(new_tokens[i]) - len(flags))
    return out
