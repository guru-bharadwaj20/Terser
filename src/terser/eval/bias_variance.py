"""Bootstrap bias-variance decomposition (plan row **3.6.1**).

Train the same learner on ``B`` bootstrap resamples of the training set and predict a fixed
evaluation set. For evaluation point ``x`` with target ``y`` and predictions ``f_1..f_B``:

**Squared loss** (regression, e.g. KNN savings predictor)::

    E[(f - y)^2] = (fbar - y)^2 + E[(f - fbar)^2]
                 =     bias^2   +     variance

(the noise term is folded into bias^2 because only one noisy ``y`` per ``x`` is observed).

**0-1 loss** (Domingos 2000, two classes, noise folded into bias). The *main prediction* ``m`` is
the mode of ``f_1..f_B``; ``bias(x) = 1[m != y]``; ``variance(x) = P(f != m)``. Then::

    E[1(f != y)] = bias + (1 - 2 bias) * variance

so variance *adds* to the error where the main prediction is right (unbiased points) and
*subtracts* where it is wrong (biased points): ``loss = bias + V_unbiased - V_biased``, where
``V_unbiased`` / ``V_biased`` average ``variance(x)`` over all points with the indicator applied.
All quantities are averaged over the evaluation points.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = ["BiasVariance", "bootstrap_predictions", "decompose_squared", "decompose_zero_one"]

FloatArray = npt.NDArray[np.float64]
FitPredictAny = Callable[[FloatArray, FloatArray, FloatArray], FloatArray]


def bootstrap_predictions(
    fit_predict: FitPredictAny,
    X_train: npt.ArrayLike,
    y_train: npt.ArrayLike,
    X_eval: npt.ArrayLike,
    n_boot: int = 200,
    seed: int = 0,
) -> FloatArray:
    """Predictions of shape ``(n_boot, len(X_eval))`` from models fit on bootstrap resamples."""
    Xa = np.asarray(X_train, dtype=np.float64)
    ya = np.asarray(y_train, dtype=np.float64)
    Xe = np.asarray(X_eval, dtype=np.float64)
    rng = np.random.default_rng(seed)
    out = np.empty((n_boot, len(Xe)))
    for b in range(n_boot):
        idx = rng.integers(0, len(ya), len(ya))
        out[b] = fit_predict(Xa[idx], ya[idx], Xe)
    return out


@dataclass(frozen=True)
class BiasVariance:
    """Mean loss and its parts over the evaluation points.

    For squared loss ``loss = bias + variance`` (``bias`` is the squared bias). For 0-1 loss
    ``loss = bias + variance_unbiased - variance_biased`` and ``variance`` is their sum.
    """

    loss: float
    bias: float
    variance: float
    variance_unbiased: float = float("nan")
    variance_biased: float = float("nan")


def decompose_squared(preds: npt.ArrayLike, y: npt.ArrayLike) -> BiasVariance:
    """Squared-loss decomposition of ``preds`` (B, n) against targets ``y`` (n,)."""
    P = np.asarray(preds, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    mean = P.mean(axis=0)
    loss = float(np.mean((P - ya) ** 2))
    bias2 = float(np.mean((mean - ya) ** 2))
    var = float(np.mean(P.var(axis=0)))
    return BiasVariance(loss=loss, bias=bias2, variance=var)


def decompose_zero_one(preds: npt.ArrayLike, y: npt.ArrayLike) -> BiasVariance:
    """Domingos 0-1 decomposition of 0/1 ``preds`` (B, n) against labels ``y`` (n,).

    The main prediction on an exact 50/50 split is class 0.
    """
    P = np.asarray(preds, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if not np.isin(P, (0.0, 1.0)).all():
        raise ValueError("0-1 decomposition needs hard 0/1 predictions")
    share1 = P.mean(axis=0)
    main = (share1 > 0.5).astype(np.float64)
    bias = (main != ya).astype(np.float64)
    var = np.where(main == 1.0, 1.0 - share1, share1)
    loss = float(np.mean(P != ya))
    v_unb = float(np.mean((1 - bias) * var))
    v_b = float(np.mean(bias * var))
    return BiasVariance(
        loss=loss,
        bias=float(bias.mean()),
        variance=v_unb + v_b,
        variance_unbiased=v_unb,
        variance_biased=v_b,
    )
