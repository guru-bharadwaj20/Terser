"""Bayesian-learning experiments (plan rows **5.2.3**, **5.2.4**, **5.2.6**).

* :func:`nb_vs_lr` (5.2.3): multinomial NB (from scratch) vs logistic regression on the same
  1-3-gram
  counts of filler / non-filler spans, under the grouped CV protocol; returns the comparison
  table (AP, ROC-AUC, best precision at recall >= 0.70 for checkpoint C5.3) and both PR curves.
* :func:`bayes_optimal_vs_map` (5.2.4): posterior over a finite hypothesis space from training
  likelihood; validation accuracy of the posterior-weighted vote vs the MAP hypothesis (C5.4).
* :func:`gmm_on_deltas` (5.2.6): 2-component GMM (from scratch EM) on judge score deltas
  ``rewrite - original``; the component with the lower mean is "degraded"; ARI against the judge
  threshold labels (C5.5).

Each takes in-memory arrays; the real inputs (span texts, Phase 1 hypotheses' predictions, judge
scores) come from the Phase 2 / 3 artefacts and are wired up in :mod:`terser.phase5.real_studies`.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from terser.eval.curves import average_precision, pr_curve, roc_auc
from terser.eval.cv import out_of_fold_scores
from terser.models.classical.bayes_optimal import BayesOptimalClassifier
from terser.models.classical.gmm import GaussianMixtureEM
from terser.models.classical.scratch_naive_bayes import MultinomialNaiveBayes

__all__ = ["bayes_optimal_vs_map", "gmm_on_deltas", "nb_vs_lr", "precision_at_recall"]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def precision_at_recall(y: IntArray, scores: FloatArray, min_recall: float = 0.70) -> float:
    """Highest precision over thresholds whose recall is at least ``min_recall``."""
    c = pr_curve(y, scores)
    ok = np.asarray(c.x) >= min_recall
    return float(np.max(np.asarray(c.y)[ok])) if ok.any() else 0.0


def nb_vs_lr(
    texts: Sequence[str],
    y: IntArray,
    categories: Sequence[Hashable],
    groups: Sequence[Hashable],
    alpha: float = 1.0,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Row 5.2.3: grouped-CV comparison table and PR curves."""
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression

    idx = np.arange(len(texts), dtype=np.float64)[:, None]
    texts_arr = np.asarray(texts, dtype=object)

    def _vec(tr: FloatArray, q: FloatArray) -> tuple[Any, Any]:
        v = CountVectorizer(ngram_range=(1, 3))
        a = v.fit_transform(texts_arr[tr[:, 0].astype(int)])
        return a, v.transform(texts_arr[q[:, 0].astype(int)])

    def nb(tr: FloatArray, yt: IntArray, q: FloatArray) -> FloatArray:
        a, b = _vec(tr, q)
        return MultinomialNaiveBayes(alpha=alpha).fit(a, yt).predict_proba(b)[:, 1]

    def lr(tr: FloatArray, yt: IntArray, q: FloatArray) -> FloatArray:
        a, b = _vec(tr, q)
        m = LogisticRegression(max_iter=2000).fit(a, yt)
        return np.asarray(m.predict_proba(b)[:, 1])

    rows, curves = [], {}
    for name, fp in (("naive_bayes", nb), ("logistic_regression", lr)):
        s = out_of_fold_scores(fp, idx, y, categories, groups, seed=seed).scores
        rows.append(
            {
                "model": name,
                "roc_auc": roc_auc(y, s),
                "average_precision": average_precision(y, s),
                "precision_at_recall_0.70": precision_at_recall(y, s),
            }
        )
        curves[name] = pr_curve(y, s)
    return pd.DataFrame(rows), curves


def bayes_optimal_vs_map(
    preds_train: npt.ArrayLike,
    y_train: npt.ArrayLike,
    preds_val: npt.ArrayLike,
    y_val: npt.ArrayLike,
    noise: float = 0.1,
) -> dict[str, float]:
    """Row 5.2.4: posterior from training likelihood; val accuracy of the vote vs MAP."""
    m = BayesOptimalClassifier(noise=noise).fit(preds_train, y_train)
    yv = np.asarray(y_val)
    post = m.posterior_
    return {
        "acc_bayes_optimal": float(np.mean(m.predict(preds_val) == yv)),
        "acc_map": float(np.mean(m.predict_map(preds_val) == yv)),
        "map_posterior": float(post[m.map_index_]),
        "posterior_entropy": float(-(post * np.log(np.clip(post, 1e-300, None))).sum()),
    }


def gmm_on_deltas(
    deltas: npt.ArrayLike, labels: npt.ArrayLike, seed: int = 0
) -> dict[str, float | GaussianMixtureEM]:
    """Row 5.2.6: fit a 2-component GMM; ARI of the induced split vs the threshold labels.

    ``labels`` are 1 = degraded. The component with the lower mean delta is mapped to degraded.
    """
    from sklearn.metrics import adjusted_rand_score

    d = np.asarray(deltas, dtype=np.float64)
    gmm = GaussianMixtureEM(n_components=2, seed=seed).fit(d)
    comp = gmm.predict(d)
    degraded = int(np.argmin(gmm.means_[:, 0]))
    pred = (comp == degraded).astype(int)
    return {
        "ari": float(adjusted_rand_score(np.asarray(labels), pred)),
        "agreement": float(np.mean(pred == np.asarray(labels))),
        "mean_degraded": float(gmm.means_[degraded, 0]),
        "mean_preserved": float(gmm.means_[1 - degraded, 0]),
        "weight_degraded": float(gmm.weights_[degraded]),
        "model": gmm,
    }
