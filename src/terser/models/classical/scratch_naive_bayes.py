"""Multinomial naive Bayes with Laplace smoothing, from scratch (plan row **5.2.3**).

Derivation: ``docs/derivations/bayes.md`` §3. For class ``c`` and n-gram vocabulary ``V``:

* prior ``P(c) = N_c / N`` (MLE);
* likelihood ``theta_{c,j} = (count_{c,j} + alpha) / (sum_j' count_{c,j'} + alpha |V|)``.

``alpha = 0`` is the MLE; ``alpha = 1`` (Laplace) is the **MAP** estimate under a symmetric
Dirichlet(alpha + 1) prior on ``theta_c``, whose mode is exactly the formula above. Without it, one
n-gram never seen with a class sets that class's posterior to zero for any document containing it.

Prediction: ``log P(c | x) = log P(c) + sum_j x_j log theta_{c,j} - log Z``. Matches
``sklearn.naive_bayes.MultinomialNB`` (tested).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from scipy.special import logsumexp

__all__ = ["MultinomialNaiveBayes"]

FloatArray = npt.NDArray[np.float64]


@dataclass
class MultinomialNaiveBayes:
    """Multinomial NB on a non-negative count matrix (e.g. 1-3-gram counts)."""

    alpha: float = 1.0
    classes_: npt.NDArray[np.generic] = field(init=False, repr=False)
    class_log_prior_: FloatArray = field(init=False, repr=False)
    feature_log_prob_: FloatArray = field(init=False, repr=False)

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> MultinomialNaiveBayes:
        Xa = np.asarray(X.toarray() if hasattr(X, "toarray") else X, dtype=np.float64)
        ya = np.asarray(y)
        if (Xa < 0).any():
            raise ValueError("counts must be non-negative")
        if self.alpha < 0:
            raise ValueError("alpha must be >= 0")
        self.classes_ = np.unique(ya)
        counts = np.stack([Xa[ya == c].sum(axis=0) for c in self.classes_]) + self.alpha
        n_c = np.array([(ya == c).sum() for c in self.classes_], dtype=np.float64)
        self.class_log_prior_ = np.log(n_c / n_c.sum())
        with np.errstate(divide="ignore"):
            self.feature_log_prob_ = np.log(counts) - np.log(counts.sum(axis=1, keepdims=True))
        return self

    def joint_log_likelihood(self, X: npt.ArrayLike) -> FloatArray:
        Xa = np.asarray(X.toarray() if hasattr(X, "toarray") else X, dtype=np.float64)
        flp = np.where(np.isneginf(self.feature_log_prob_), -1e300, self.feature_log_prob_)
        return np.asarray(Xa @ flp.T + self.class_log_prior_)

    def predict_log_proba(self, X: npt.ArrayLike) -> FloatArray:
        jll = self.joint_log_likelihood(X)
        return np.asarray(jll - logsumexp(jll, axis=1, keepdims=True))

    def predict_proba(self, X: npt.ArrayLike) -> FloatArray:
        return np.exp(self.predict_log_proba(X))

    def predict(self, X: npt.ArrayLike) -> npt.NDArray[np.generic]:
        return np.asarray(self.classes_[np.argmax(self.joint_log_likelihood(X), axis=1)])
