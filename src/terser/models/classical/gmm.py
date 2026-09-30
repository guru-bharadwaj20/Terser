"""Gaussian mixture model fitted by expectation maximisation, from scratch (plan row **5.2.5**).

Model: ``p(x) = sum_k pi_k N(x | mu_k, Sigma_k)`` with full covariances.

* **E-step** - responsibilities ``r_ik = pi_k N(x_i | mu_k, Sigma_k) / p(x_i)``, computed in log
  space with log-sum-exp.
* **M-step** - with ``N_k = sum_i r_ik``: ``pi_k = N_k / n``, ``mu_k = sum_i r_ik x_i / N_k``,
  ``Sigma_k = sum_i r_ik (x_i - mu_k)(x_i - mu_k)^T / N_k + reg * I``.

Each iteration cannot decrease the log-likelihood (EM maximises a lower bound that is tight after
the E-step); :attr:`GaussianMixtureEM.log_likelihood_history_` records it for the monotonicity
test. With the same initialisation the result matches ``sklearn.mixture.GaussianMixture``.
Used on judge score deltas in row 5.2.6.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from scipy.special import logsumexp

__all__ = ["GaussianMixtureEM"]

FloatArray = npt.NDArray[np.float64]


@dataclass
class GaussianMixtureEM:
    """Full-covariance GMM.

    Attributes:
        n_components: Number of mixture components ``K``.
        max_iter: Iteration cap.
        tol: Stop when the mean per-sample log-likelihood improves by less than this.
        reg_covar: Added to covariance diagonals to keep them positive definite.
        seed: Seed for initialisation (means at ``K`` distinct random data points).
    """

    n_components: int = 2
    max_iter: int = 500
    tol: float = 1e-6
    reg_covar: float = 1e-6
    seed: int = 0
    weights_: FloatArray = field(init=False, repr=False)
    means_: FloatArray = field(init=False, repr=False)
    covariances_: FloatArray = field(init=False, repr=False)
    log_likelihood_history_: list[float] = field(init=False, repr=False, default_factory=list)
    converged_: bool = field(init=False, default=False)

    def _log_gauss(self, X: FloatArray) -> FloatArray:
        """``log N(x_i | mu_k, Sigma_k)``, shape (n, K), via Cholesky factors."""
        n, d = X.shape
        out = np.empty((n, self.n_components))
        for k in range(self.n_components):
            L = np.linalg.cholesky(self.covariances_[k])
            sol = np.linalg.solve(L, (X - self.means_[k]).T)
            maha = np.sum(sol**2, axis=0)
            log_det = 2.0 * np.sum(np.log(np.diag(L)))
            out[:, k] = -0.5 * (d * np.log(2 * np.pi) + log_det + maha)
        return out

    def _e_step(self, X: FloatArray) -> tuple[FloatArray, float]:
        """Log responsibilities (n, K) and the mean per-sample log-likelihood."""
        weighted = self._log_gauss(X) + np.log(self.weights_)
        log_px = logsumexp(weighted, axis=1)
        return weighted - log_px[:, None], float(log_px.mean())

    def _m_step(self, X: FloatArray, log_resp: FloatArray) -> None:
        resp = np.exp(log_resp)
        nk = resp.sum(axis=0) + 10 * np.finfo(np.float64).eps
        self.weights_ = nk / len(X)
        self.means_ = resp.T @ X / nk[:, None]
        d = X.shape[1]
        covs = np.empty((self.n_components, d, d))
        for k in range(self.n_components):
            diff = X - self.means_[k]
            covs[k] = (resp[:, k, None] * diff).T @ diff / nk[k] + self.reg_covar * np.eye(d)
        self.covariances_ = covs

    def fit(
        self,
        X: npt.ArrayLike,
        means_init: npt.ArrayLike | None = None,
        weights_init: npt.ArrayLike | None = None,
        covariances_init: npt.ArrayLike | None = None,
    ) -> GaussianMixtureEM:
        """Run EM on ``X`` (n, d) or (n,) for 1-D data.

        Without explicit initial values, means start at random distinct data points, weights are
        uniform and every covariance is the data covariance.
        """
        Xa = np.asarray(X, dtype=np.float64)
        if Xa.ndim == 1:
            Xa = Xa[:, None]
        n, d = Xa.shape
        if n < self.n_components:
            raise ValueError(f"need at least {self.n_components} samples, got {n}")
        rng = np.random.default_rng(self.seed)
        if means_init is None:
            self.means_ = Xa[rng.choice(n, self.n_components, replace=False)].copy()
        else:
            self.means_ = np.asarray(means_init, dtype=np.float64).reshape(self.n_components, d)
        self.weights_ = (
            np.full(self.n_components, 1.0 / self.n_components)
            if weights_init is None
            else np.asarray(weights_init, dtype=np.float64)
        )
        if covariances_init is None:
            base = np.atleast_2d(np.cov(Xa, rowvar=False)) + self.reg_covar * np.eye(d)
            self.covariances_ = np.repeat(base[None], self.n_components, axis=0)
        else:
            self.covariances_ = np.asarray(covariances_init, dtype=np.float64).reshape(
                self.n_components, d, d
            )
        self.log_likelihood_history_ = []
        self.converged_ = False
        log_resp, ll = self._e_step(Xa)
        self.log_likelihood_history_.append(ll)
        for _ in range(self.max_iter):
            self._m_step(Xa, log_resp)
            log_resp, ll = self._e_step(Xa)
            self.log_likelihood_history_.append(ll)
            if ll - self.log_likelihood_history_[-2] < self.tol:
                self.converged_ = True
                break
        return self

    def _as_2d(self, X: npt.ArrayLike) -> FloatArray:
        Xa = np.asarray(X, dtype=np.float64)
        return Xa[:, None] if Xa.ndim == 1 else Xa

    def predict_proba(self, X: npt.ArrayLike) -> FloatArray:
        """Responsibilities, shape (n, K)."""
        return np.exp(self._e_step(self._as_2d(X))[0])

    def predict(self, X: npt.ArrayLike) -> npt.NDArray[np.int64]:
        """Most responsible component per sample."""
        return self.predict_proba(X).argmax(axis=1).astype(np.int64)

    def score(self, X: npt.ArrayLike) -> float:
        """Mean per-sample log-likelihood."""
        return self._e_step(self._as_2d(X))[1]
