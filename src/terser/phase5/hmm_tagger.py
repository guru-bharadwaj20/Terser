"""HMM span-tagger experiments (plan rows **5.3.5** and **5.3.6**).

* :func:`baum_welch_f1` (5.3.5): supervised MLE on labelled sequences (row 5.3.2), then
  Baum-Welch on unlabelled casual prompts from that initialisation; token macro-F1 of Viterbi tags
  on a labelled validation set before vs after.
* :func:`compress` (5.3.6): Viterbi tags -> rewritten prompt for each input. The end-to-end row
  then judges these rewrites with the Phase 2 judge on 500 validation prompts (GPU-served model),
  which needs the Phase 2 dataset and is not run here.

Labelled sequences come from row 3.1.1 alignment labels (KEEP / DROP / REPLACE per word).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import numpy.typing as npt

from terser.models.classical.hmm import (
    TAGS,
    DiscreteHMM,
    TokenClassEncoder,
    apply_tags,
    baum_welch,
    fit_supervised,
)

__all__ = ["baum_welch_f1", "compress", "token_macro_f1"]

IntArray = npt.NDArray[np.int64]


def token_macro_f1(
    gold: Sequence[npt.ArrayLike], pred: Sequence[npt.ArrayLike], n_tags: int = len(TAGS)
) -> float:
    """Macro-F1 over tag classes, pooling all tokens (classes absent from gold and pred skipped)."""
    g = np.concatenate([np.asarray(x) for x in gold])
    p = np.concatenate([np.asarray(x) for x in pred])
    f1s = []
    for k in range(n_tags):
        tp = int(np.sum((g == k) & (p == k)))
        fp = int(np.sum((g != k) & (p == k)))
        fn = int(np.sum((g == k) & (p != k)))
        if tp + fp + fn == 0:
            continue
        f1s.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s)) if f1s else 0.0


def _decode(hmm: DiscreteHMM, seqs: Sequence[npt.ArrayLike]) -> list[IntArray]:
    return [hmm.viterbi(o)[0] for o in seqs]


def baum_welch_f1(
    train_obs: Sequence[npt.ArrayLike],
    train_tags: Sequence[npt.ArrayLike],
    unlabelled_obs: Sequence[npt.ArrayLike],
    val_obs: Sequence[npt.ArrayLike],
    val_tags: Sequence[npt.ArrayLike],
    n_obs: int,
    smoothing: float = 1.0,
    n_iter: int = 20,
) -> dict[str, float | list[float]]:
    """Row 5.3.5: validation token macro-F1 before and after Baum-Welch."""
    sup = fit_supervised(train_obs, train_tags, len(TAGS), n_obs, smoothing=smoothing)
    em, hist = baum_welch(sup, unlabelled_obs, n_iter=n_iter)
    return {
        "f1_supervised": token_macro_f1(val_tags, _decode(sup, val_obs)),
        "f1_baum_welch": token_macro_f1(val_tags, _decode(em, val_obs)),
        "log_likelihood": hist,
    }


def compress(
    hmm: DiscreteHMM,
    prompts: Sequence[str],
    encoder: TokenClassEncoder | None = None,
    replacements: Mapping[str, str] | None = None,
) -> list[str]:
    """Row 5.3.6: whitespace-tokenise, encode, Viterbi-decode and rebuild each prompt."""
    enc = encoder or TokenClassEncoder()
    out = []
    for p in prompts:
        words = p.split()
        if not words:
            out.append(p)
            continue
        tags, _ = hmm.viterbi(enc.encode(words))
        out.append(apply_tags(words, tags.tolist(), replacements) or p)
    return out
