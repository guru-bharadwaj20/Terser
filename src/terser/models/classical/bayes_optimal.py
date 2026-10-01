"""Bayes optimal classifier over a finite hypothesis space (plan row **5.2.4**).

Derivation: ``docs/derivations/bayes.md`` §4. ``H`` is a finite set of hypotheses (the Phase 1
version-space members); each maps an instance to ``{0, 1}``. Assuming each validation label is
flipped independently with probability ``noise`` (so a consistent-only posterior is not forced),

    P(D | h) = (1 - noise)^{n - m_h} * noise^{m_h},     m_h = # mistakes of h on D,

and ``P(h | D) ∝ P(D | h) P(h)``. The **MAP hypothesis** is ``argmax_h P(h | D)``; the **Bayes
optimal classifier** predicts ``argmax_v sum_h P(v | h, x) P(h | D)``, a posterior-weighted vote.
No classifier using the same ``H`` and prior can have lower expected error, which is the claim
checkpoint C5.4 tests on validation data.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from scipy.special import logsumexp

__all__ = ["BayesOptimalClassifier"]

FloatArray = npt.NDArray[np.float64]


@dataclass
class BayesOptimalClassifier:
    """Posterior over hypotheses from their 0/1 predictions on labelled data.

    Fit and predict take a prediction matrix ``P`` of shape ``(n_hypotheses, n_samples)`` with
    entries in ``{0, 1}``: ``P[h, i] = h(x_i)``. This keeps the class independent of how a
    hypothesis is represented.
    """

    noise: float = 0.1
    prior: Sequence[float] | None = None
    log_posterior_: FloatArray = field(init=False, repr=False)

    def fit(self, preds: npt.ArrayLike, y: npt.ArrayLike) -> BayesOptimalClassifier:
        P = np.asarray(preds)
        ya = np.asarray(y)
        if P.ndim != 2 or P.shape[1] != len(ya):
            raise ValueError("preds must have shape (n_hypotheses, n_samples)")
        if not 0 < self.noise < 0.5:
            raise ValueError("noise must be in (0, 0.5)")
        mistakes = (P != ya[None, :]).sum(axis=1)
        n = len(ya)
        log_lik = (n - mistakes) * np.log1p(-self.noise) + mistakes * np.log(self.noise)
        log_prior = (
            np.full(len(P), -np.log(len(P)))
            if self.prior is None
            else np.log(np.asarray(self.prior, dtype=np.float64))
        )
        if len(log_prior) != len(P):
            raise ValueError("prior must have one entry per hypothesis")
        joint = log_lik + log_prior
        self.log_posterior_ = np.asarray(joint - logsumexp(joint))
        return self

    @property
    def posterior_(self) -> FloatArray:
        return np.exp(self.log_posterior_)

    @property
    def map_index_(self) -> int:
        """Index of the MAP hypothesis (lowest index on ties)."""
        return int(np.argmax(self.log_posterior_))

    def predict_proba(self, preds: npt.ArrayLike) -> FloatArray:
        """``P(y=1 | x, D) = sum_h P(h | D) [h(x) = 1]``."""
        return np.asarray(self.posterior_ @ np.asarray(preds, dtype=np.float64))

    def predict(self, preds: npt.ArrayLike) -> npt.NDArray[np.int64]:
        """Bayes optimal vote (ties at 0.5 go to 0)."""
        return (self.predict_proba(preds) > 0.5).astype(np.int64)

    def predict_map(self, preds: npt.ArrayLike) -> npt.NDArray[np.int64]:
        """Prediction of the single MAP hypothesis."""
        return np.asarray(preds, dtype=np.int64)[self.map_index_]
