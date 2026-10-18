"""Probability calibration: Brier score, ECE and reliability diagrams (plan row **3.4.3**).

* Brier score: ``mean (p_i - y_i)^2``, a proper scoring rule (lower is better).
* Reliability curve: split ``[0, 1]`` into ``n_bins`` equal-width bins; for each non-empty bin,
  the mean predicted probability against the observed positive rate. Bins match scikit-learn's
  ``calibration_curve(strategy="uniform")``.
* Expected calibration error: ``ECE = sum_b (n_b / n) |acc_b - conf_b|`` over the same bins, the
  bin-weighted gap between the diagonal and the reliability curve.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from terser.eval.labels import ArrayLike, _as_binary_array

__all__ = [
    "ReliabilityCurve",
    "brier_score",
    "expected_calibration_error",
    "plot_reliability",
    "reliability_curve",
]

FloatArray = npt.NDArray[np.float64]


def _probs(p: npt.ArrayLike, n: int) -> FloatArray:
    pa = np.asarray(p, dtype=np.float64)
    if pa.shape != (n,):
        raise ValueError(f"expected {n} probabilities, got shape {pa.shape}")
    if np.any((pa < 0) | (pa > 1)):
        raise ValueError("probabilities must lie in [0, 1]")
    return pa


def brier_score(y_true: ArrayLike, p: npt.ArrayLike) -> float:
    """Mean squared error between probabilities and 0/1 outcomes."""
    y = _as_binary_array(y_true)
    if len(y) == 0:
        raise ValueError("brier_score needs at least one example")
    return float(np.mean((_probs(p, len(y)) - y) ** 2))


@dataclass(frozen=True)
class ReliabilityCurve:
    """Per non-empty bin: observed positive rate, mean predicted probability, count."""

    prob_true: FloatArray
    prob_pred: FloatArray
    counts: npt.NDArray[np.int64]


def reliability_curve(y_true: ArrayLike, p: npt.ArrayLike, n_bins: int = 10) -> ReliabilityCurve:
    """Equal-width reliability curve; empty bins are omitted."""
    y = _as_binary_array(y_true)
    pa = _probs(p, len(y))
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ids = np.searchsorted(edges[1:-1], pa)
    counts = np.bincount(ids, minlength=n_bins)
    sums_p = np.bincount(ids, weights=pa, minlength=n_bins)
    sums_y = np.bincount(ids, weights=y.astype(np.float64), minlength=n_bins)
    nz = counts > 0
    return ReliabilityCurve(
        prob_true=sums_y[nz] / counts[nz],
        prob_pred=sums_p[nz] / counts[nz],
        counts=counts[nz].astype(np.int64),
    )


def expected_calibration_error(y_true: ArrayLike, p: npt.ArrayLike, n_bins: int = 10) -> float:
    """Bin-weighted mean of ``|observed rate - mean probability|``."""
    c = reliability_curve(y_true, p, n_bins)
    if c.counts.sum() == 0:
        raise ValueError("expected_calibration_error needs at least one example")
    return float(np.sum(c.counts * np.abs(c.prob_true - c.prob_pred)) / c.counts.sum())


def plot_reliability(
    curves: dict[str, tuple[ArrayLike, npt.ArrayLike]],
    path: str | Path,
    n_bins: int = 10,
    title: str = "Reliability diagram",
) -> Path:
    """Reliability diagram for several models; legend carries Brier and ECE. Returns the path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1, label="perfect")
    for name, (y, p) in curves.items():
        c = reliability_curve(y, p, n_bins)
        label = (
            f"{name} (Brier {brier_score(y, p):.3f}, "
            f"ECE {expected_calibration_error(y, p, n_bins):.3f})"
        )
        ax.plot(c.prob_pred, c.prob_true, marker="o", label=label)
    ax.set_xlabel("mean predicted P(degraded)")
    ax.set_ylabel("observed degraded rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=8)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
