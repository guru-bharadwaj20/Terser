"""Operating-threshold rule for the quality gate (plan row **3.2.3**).

A missed degraded rewrite (false negative) reaches the user silently, while a false alarm only
forfeits savings (:mod:`terser.eval.labels`). The gate therefore flags ``score >= threshold`` as
degraded with the threshold chosen to **maximise specificity subject to sensitivity >= 0.95**.

Pick the threshold on validation (or out-of-fold) scores only, never on test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from terser.eval.curves import roc_curve
from terser.eval.labels import ArrayLike

__all__ = ["MIN_SENSITIVITY", "ThresholdChoice", "apply_threshold", "choose_threshold"]

MIN_SENSITIVITY: Final[float] = 0.95
"""Sensitivity floor from plan row 3.2.3."""


@dataclass(frozen=True)
class ThresholdChoice:
    """The chosen threshold and the sensitivity/specificity it achieves on the fitting data."""

    threshold: float
    sensitivity: float
    specificity: float


def choose_threshold(
    y_true: ArrayLike, scores: npt.ArrayLike, min_sensitivity: float = MIN_SENSITIVITY
) -> ThresholdChoice:
    """Maximise specificity subject to ``sensitivity >= min_sensitivity``.

    Candidate thresholds are the distinct scores. Among those meeting the floor, the one with the
    highest specificity wins; ties go to the highest sensitivity, then the highest threshold.

    Args:
        y_true: True labels (``degraded`` positive).
        scores: Degradation scores, higher = more likely degraded.
        min_sensitivity: The sensitivity floor, in ``(0, 1]``.

    Returns:
        A :class:`ThresholdChoice`. A floor of 1.0 is always reachable (flag everything at or
        above the lowest-scored degraded example).

    Raises:
        ValueError: For a floor outside ``(0, 1]`` or inputs :func:`terser.eval.curves.roc_curve`
            rejects.
    """
    if not 0.0 < min_sensitivity <= 1.0:
        raise ValueError(f"min_sensitivity must be in (0, 1], got {min_sensitivity}")
    c = roc_curve(y_true, scores)
    # Drop the (+inf, 0, 0) point: it flags nothing.
    fpr, tpr, thr = c.x[1:], c.y[1:], c.thresholds[1:]
    ok = np.flatnonzero(tpr >= min_sensitivity - 1e-12)
    # Along the curve fpr and tpr are non-decreasing, so the first feasible point has the lowest
    # fpr; among points sharing that fpr, the last one has the highest tpr.
    best_fpr = fpr[ok[0]]
    idx = int(ok[fpr[ok] == best_fpr][-1])
    return ThresholdChoice(
        threshold=float(thr[idx]), sensitivity=float(tpr[idx]), specificity=float(1.0 - fpr[idx])
    )


def apply_threshold(scores: npt.ArrayLike, threshold: float) -> npt.NDArray[np.int64]:
    """Predicted labels: ``1`` (degraded) where ``score >= threshold``, else ``0``."""
    return (np.asarray(scores, dtype=np.float64) >= threshold).astype(np.int64)
