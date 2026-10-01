"""Rosenblatt perceptron with a step activation, from scratch (plan row **4.1.1**).

Prediction ``y_hat = step(w.x + b)`` with ``step(z) = 1 if z > 0 else 0``. With ``s = 2y - 1``
the update on a mistake (``s (w.x + b) <= 0``) is ``w <- w + eta s x``, ``b <- b + eta s``, which
equals ``eta (y - y_hat)`` whenever the point is strictly misclassified. By the perceptron
convergence theorem the number of mistakes on linearly separable data with margin ``gamma`` and
radius ``R`` is at most ``(R / gamma)^2``, so an epoch without mistakes is eventually reached;
on non-separable data (the full filler-span set) no such epoch exists and training stops at
``max_epochs`` with :attr:`Perceptron.converged_` False. The per-epoch mistake counts are kept in
:attr:`Perceptron.errors_` so both outcomes can be logged.

With ``shuffle=False`` the updates are identical to ``sklearn.linear_model.Perceptron`` with
``eta0=eta``, ``alpha=0``, ``shuffle=False`` (tested).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

__all__ = ["Perceptron"]

FloatArray = npt.NDArray[np.float64]


@dataclass
class Perceptron:
    """Binary perceptron on labels ``{0, 1}``."""

    eta: float = 1.0
    max_epochs: int = 100
    shuffle: bool = False
    seed: int = 0
    coef_: FloatArray = field(init=False, repr=False)
    intercept_: float = field(init=False, default=0.0)
    errors_: list[int] = field(init=False, default_factory=list)
    converged_: bool = field(init=False, default=False)

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> Perceptron:
        Xa = np.asarray(X, dtype=np.float64)
        ya = np.asarray(y)
        if not np.isin(ya, (0, 1)).all():
            raise ValueError("labels must be 0/1")
        rng = np.random.default_rng(self.seed)
        self.coef_ = np.zeros(Xa.shape[1])
        self.intercept_ = 0.0
        self.errors_ = []
        self.converged_ = False
        for _ in range(self.max_epochs):
            order = rng.permutation(len(ya)) if self.shuffle else np.arange(len(ya))
            mistakes = 0
            for i in order:
                # A point on the boundary (z = 0) counts as a mistake for either class, so
                # training ends only with every point strictly on its correct side.
                sign = 1.0 if ya[i] == 1 else -1.0
                if sign * (Xa[i] @ self.coef_ + self.intercept_) <= 0:
                    delta = self.eta * sign
                    self.coef_ += delta * Xa[i]
                    self.intercept_ += delta
                    mistakes += 1
            self.errors_.append(mistakes)
            if mistakes == 0:
                self.converged_ = True
                break
        return self

    def decision_function(self, X: npt.ArrayLike) -> FloatArray:
        return np.asarray(np.asarray(X, dtype=np.float64) @ self.coef_ + self.intercept_)

    def predict(self, X: npt.ArrayLike) -> npt.NDArray[np.int64]:
        return (self.decision_function(X) > 0).astype(np.int64)
