"""Trivial baselines for every Phase 3+ table (plan row **3.6.5**).

Each baseline is a ``fit_predict(X_train, y_train, X_val) -> scores`` callable, the same
contract as :data:`terser.eval.cv.FitPredict`, so it runs through the identical grouped-CV
protocol as the real models:

* ``majority``: constant score equal to the training majority class (0 or 1);
* ``stratified``: random 0/1 draws with the training positive rate (seeded);
* ``always_preserved``: constant 0 (never flag ``degraded``), the "original prompt" baseline of
  the experiment standards: 0% savings, 100% preservation.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

from terser.eval.cv import FitPredict

__all__ = ["BASELINES", "always_preserved", "majority", "stratified"]

FloatArray = npt.NDArray[np.float64]


def majority(X_train: FloatArray, y_train: npt.NDArray[np.int64], X_val: FloatArray) -> FloatArray:
    """Training majority class for every validation example (ties go to ``preserved``)."""
    value = 1.0 if np.mean(y_train) > 0.5 else 0.0
    return np.full(len(X_val), value)


def stratified(seed: int = 0) -> FitPredict:
    """Random labels with the training positive rate."""

    def fit_predict(
        X_train: FloatArray, y_train: npt.NDArray[np.int64], X_val: FloatArray
    ) -> FloatArray:
        rng = np.random.default_rng(seed)
        return (rng.random(len(X_val)) < np.mean(y_train)).astype(np.float64)

    return fit_predict


def always_preserved(
    X_train: FloatArray, y_train: npt.NDArray[np.int64], X_val: FloatArray
) -> FloatArray:
    """Never predicts ``degraded``."""
    return np.zeros(len(X_val))


BASELINES: dict[str, Callable[[], FitPredict]] = {
    "majority": lambda: majority,
    "stratified": lambda: stratified(0),
    "always_preserved": lambda: always_preserved,
}
