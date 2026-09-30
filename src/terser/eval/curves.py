"""ROC and precision-recall curves for the quality gate (plan row **3.2.2**).

Scores are "probability the rewrite is degraded"; the positive class is ``degraded``
(:mod:`terser.eval.labels`). Degraded rewrites are the minority, so PR curves and average precision
are reported next to ROC/AUC. Values match scikit-learn to 1e-9 (tested).

* :func:`roc_curve` / :func:`roc_auc` - ROC points over every distinct threshold and the
  trapezoidal area under them.
* :func:`pr_curve` / :func:`average_precision` - PR points and AP as the step-wise sum
  ``sum_k (R_k - R_{k-1}) P_k`` (no interpolation, as scikit-learn).
* :func:`plot_curves` - the report figure template: ROC and PR side by side, one line per model.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from terser.eval.labels import ArrayLike, _as_binary_array

if TYPE_CHECKING:
    from matplotlib.figure import Figure

__all__ = [
    "Curve",
    "average_precision",
    "plot_curves",
    "pr_curve",
    "roc_auc",
    "roc_curve",
]

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Curve:
    """One curve: ``x`` and ``y`` coordinates and the score threshold at each point."""

    x: FloatArray
    y: FloatArray
    thresholds: FloatArray


def _counts_by_threshold(
    y_true: ArrayLike, scores: npt.ArrayLike
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Cumulative (tp, fp) when predicting positive for ``score >= threshold``, per distinct score.

    Thresholds are returned in decreasing order.
    """
    y = _as_binary_array(y_true)
    s = np.asarray(scores, dtype=np.float64)
    if s.ndim != 1 or s.shape != y.shape:
        raise ValueError(f"scores must be 1-D and match y_true: {s.shape} vs {y.shape}")
    if not np.all(np.isfinite(s)):
        raise ValueError("scores must be finite")
    if y.sum() == 0 or y.sum() == len(y):
        raise ValueError("curves need both classes in y_true")
    order = np.argsort(-s, kind="mergesort")
    s, y = s[order], y[order]
    last = np.r_[np.flatnonzero(np.diff(s)), len(s) - 1]
    tp = np.cumsum(y)[last].astype(np.float64)
    fp = (last + 1 - tp).astype(np.float64)
    return tp, fp, s[last]


def roc_curve(y_true: ArrayLike, scores: npt.ArrayLike) -> Curve:
    """ROC curve: ``x`` = false-positive rate (1 - specificity), ``y`` = sensitivity.

    The first point is ``(0, 0)`` at threshold ``+inf``.
    """
    tp, fp, thr = _counts_by_threshold(y_true, scores)
    tpr = np.r_[0.0, tp / tp[-1]]
    fpr = np.r_[0.0, fp / fp[-1]]
    return Curve(x=fpr, y=tpr, thresholds=np.r_[np.inf, thr])


def roc_auc(y_true: ArrayLike, scores: npt.ArrayLike) -> float:
    """Area under the ROC curve (trapezoidal rule, ties count one half)."""
    c = roc_curve(y_true, scores)
    return float(np.trapezoid(c.y, c.x))


def pr_curve(y_true: ArrayLike, scores: npt.ArrayLike) -> Curve:
    """Precision-recall curve: ``x`` = recall, ``y`` = precision, recall increasing.

    The first point is ``(0, 1)`` at threshold ``+inf``, by the usual convention.
    """
    tp, fp, thr = _counts_by_threshold(y_true, scores)
    precision = tp / (tp + fp)
    recall = tp / tp[-1]
    return Curve(x=np.r_[0.0, recall], y=np.r_[1.0, precision], thresholds=np.r_[np.inf, thr])


def average_precision(y_true: ArrayLike, scores: npt.ArrayLike) -> float:
    """Average precision: ``sum_k (R_k - R_{k-1}) * P_k`` over thresholds, no interpolation."""
    c = pr_curve(y_true, scores)
    return float(np.sum(np.diff(c.x) * c.y[1:]))


def plot_curves(
    y_true: ArrayLike,
    scores_by_model: Mapping[str, npt.ArrayLike],
    title: str = "",
    path: str | Path | None = None,
) -> Figure:
    """Report figure template: ROC (left) and PR (right), one labelled line per model.

    The ROC panel shows the chance diagonal; the PR panel shows the positive-rate baseline.
    Legends carry AUC and AP. Saved to ``path`` (PNG at 150 dpi) when given.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y = _as_binary_array(y_true)
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(10, 4.5))
    for name, scores in scores_by_model.items():
        roc = roc_curve(y, scores)
        pr = pr_curve(y, scores)
        auc = roc_auc(y, scores)
        ap = average_precision(y, scores)
        ax_roc.plot(roc.x, roc.y, label=f"{name} (AUC {auc:.3f})")
        ax_pr.step(pr.x, pr.y, where="post", label=f"{name} (AP {ap:.3f})")
    ax_roc.plot([0, 1], [0, 1], ls="--", c="grey", lw=1)
    ax_pr.axhline(float(y.mean()), ls="--", c="grey", lw=1)
    ax_roc.set(xlabel="False-positive rate (1 - specificity)", ylabel="Sensitivity", title="ROC")
    ax_pr.set(xlabel="Recall (degraded)", ylabel="Precision (degraded)", title="Precision-recall")
    for ax in (ax_roc, ax_pr):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.legend(loc="lower right" if ax is ax_roc else "lower left", fontsize=8)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=150)
    return fig
