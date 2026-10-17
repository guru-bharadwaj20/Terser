"""The skip-short-prompts rule from predicted savings (plan row **3.5.4**).

Rewriting a prompt costs the rewriter's own overhead (tokens, or token-equivalent cost of latency).
With a savings predictor (KNN regression, row 3.5.2) the rule is: rewrite only when the predicted
saving exceeds a threshold ``tau``; otherwise send the prompt unchanged. For a threshold the
expected net saving per prompt is

    net(tau) = mean_i [ 1{pred_i > tau} * (actual_i - overhead_i) ]

where ``actual_i`` is the realised saving (tokens on T) when rewriting. Skipped prompts contribute
0 (no saving, no overhead). ``tau = -inf`` is "always rewrite", ``tau = +inf`` "never rewrite".
The curve over ``tau`` is the row's deliverable; :func:`best_threshold` picks its maximiser, which
must be chosen on validation folds, never on test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = ["NetSavingsCurve", "best_threshold", "net_savings_curve", "should_rewrite"]

FloatArray = npt.NDArray[np.float64]


def should_rewrite(predicted_saving: npt.ArrayLike, threshold: float) -> npt.NDArray[np.bool_]:
    """True where the predicted saving is strictly above ``threshold``."""
    return np.asarray(predicted_saving, dtype=np.float64) > threshold


@dataclass(frozen=True)
class NetSavingsCurve:
    """Mean net saving per prompt and rewrite rate at each threshold."""

    thresholds: FloatArray
    net_savings: FloatArray
    rewrite_rate: FloatArray


def net_savings_curve(
    predicted: npt.ArrayLike,
    actual: npt.ArrayLike,
    overhead: npt.ArrayLike | float,
    thresholds: npt.ArrayLike | None = None,
) -> NetSavingsCurve:
    """Expected net saving per prompt for each threshold.

    Args:
        predicted: Predicted saving per prompt.
        actual: Realised saving per prompt if rewritten.
        overhead: Rewriter cost per prompt (scalar or per prompt), in the same unit.
        thresholds: Thresholds to evaluate; default ``-inf``, every distinct prediction, ``+inf``.

    Raises:
        ValueError: If lengths differ.
    """
    p = np.asarray(predicted, dtype=np.float64)
    a = np.asarray(actual, dtype=np.float64)
    o = np.broadcast_to(np.asarray(overhead, dtype=np.float64), p.shape)
    if a.shape != p.shape:
        raise ValueError("predicted and actual lengths differ")
    if thresholds is None:
        t = np.concatenate([[-np.inf], np.unique(p), [np.inf]])
    else:
        t = np.sort(np.asarray(thresholds, dtype=np.float64))  # best_threshold relies on order
    gain = a - o
    take = p[None, :] > t[:, None]
    return NetSavingsCurve(
        thresholds=t,
        net_savings=(take * gain[None, :]).mean(axis=1),
        rewrite_rate=take.mean(axis=1),
    )


def best_threshold(curve: NetSavingsCurve) -> float:
    """Threshold with the largest net saving (the smallest such threshold on ties)."""
    return float(curve.thresholds[int(np.argmax(curve.net_savings))])
