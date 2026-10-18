"""Statistical comparison of models on shared out-of-fold predictions (plan row **3.2.8**).

* :func:`mcnemar` - paired test on two models' correctness over the same examples: exact binomial
  p-value on the discordant pairs, plus the continuity-corrected chi-square statistic.
* :func:`bootstrap_ci` - percentile bootstrap CI of any metric, optionally resampling whole groups
  (source prompts) so the interval respects the CV grouping.
* :func:`paired_bootstrap_diff` - CI of ``metric(model A) - metric(model B)`` on the same resamples.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import stats

from terser.eval.labels import ArrayLike, _as_binary_array

__all__ = [
    "BootstrapCI",
    "McNemarResult",
    "bootstrap_ci",
    "mcnemar",
    "paired_bootstrap_diff",
]

Metric = Callable[[npt.NDArray[np.int64], npt.NDArray[np.float64]], float]
"""``metric(y_true, prediction_or_score) -> float``."""


@dataclass(frozen=True)
class McNemarResult:
    """``b`` = only model A correct, ``c`` = only model B correct."""

    b: int
    c: int
    statistic: float
    p_value: float


def mcnemar(y_true: ArrayLike, pred_a: ArrayLike, pred_b: ArrayLike) -> McNemarResult:
    """McNemar's test for two classifiers on the same examples.

    ``p_value`` is the exact two-sided binomial test on the discordant pairs (valid for small
    counts); ``statistic`` is the continuity-corrected ``(|b - c| - 1)^2 / (b + c)``.
    With no discordant pairs the models are indistinguishable: statistic 0, p-value 1.
    """
    y = _as_binary_array(y_true)
    pa, pb = _as_binary_array(pred_a), _as_binary_array(pred_b)
    # Check before comparing: ``==`` would broadcast a length-1 prediction over every example.
    if not (len(y) == len(pa) == len(pb)):
        raise ValueError("y_true, pred_a and pred_b must have equal length")
    a, bb = pa == y, pb == y
    b = int(np.sum(a & ~bb))
    c = int(np.sum(~a & bb))
    if b + c == 0:
        return McNemarResult(b=0, c=0, statistic=0.0, p_value=1.0)
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    p = float(stats.binomtest(b, b + c, 0.5).pvalue)
    return McNemarResult(b=b, c=c, statistic=float(stat), p_value=min(p, 1.0))


@dataclass(frozen=True)
class BootstrapCI:
    """Point estimate on the full data and a percentile interval."""

    estimate: float
    low: float
    high: float
    n_resamples: int
    """Resamples on which the metric was defined (the ones the interval is built from)."""


def _resample_indices(
    n: int, groups: Sequence[Hashable] | None, n_resamples: int, seed: int
) -> list[npt.NDArray[np.int64]]:
    rng = np.random.default_rng(seed)
    if groups is None:
        return [rng.integers(0, n, n) for _ in range(n_resamples)]
    if len(groups) != n:
        raise ValueError(f"groups has length {len(groups)}, expected {n}")
    _, gid = np.unique(np.asarray([str(g) for g in groups]), return_inverse=True)
    members = [np.flatnonzero(gid == k) for k in range(gid.max() + 1)]
    out = []
    for _ in range(n_resamples):
        pick = rng.integers(0, len(members), len(members))
        out.append(np.concatenate([members[k] for k in pick]).astype(np.int64))
    return out


def _interval(values: list[float], alpha: float) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:  # the metric was undefined on every resample
        return float("nan"), float("nan")
    lo, hi = np.quantile(arr, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def _as_values(values: npt.ArrayLike) -> npt.NDArray[Any]:
    """Predictions or scores as an array, keeping their dtype (labels stay labels, ints stay ints).

    Casting to float would make metrics over labels (``accuracy`` and friends) raise.
    """
    return np.asarray(values)


def _finite_count(values: list[float]) -> int:
    return int(np.isfinite(np.asarray(values, dtype=np.float64)).sum())


def bootstrap_ci(
    metric: Metric,
    y_true: ArrayLike,
    values: npt.ArrayLike,
    groups: Sequence[Hashable] | None = None,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> BootstrapCI:
    """Percentile bootstrap CI of ``metric(y_true, values)``.

    Args:
        metric: e.g. ``terser.eval.curves.roc_auc`` or a lambda over :mod:`terser.eval.metrics`.
        y_true: True labels.
        values: Predictions or scores passed to ``metric``.
        groups: If given, whole groups are resampled (cluster bootstrap).
        n_resamples: Number of resamples. Resamples where the metric is undefined (raises
            ``ValueError`` or returns ``nan``, e.g. a single class) are skipped.
        alpha: ``1 - confidence``.
        seed: RNG seed.
    """
    y = _as_binary_array(y_true)
    v = _as_values(values)
    reps: list[float] = []
    for idx in _resample_indices(len(y), groups, n_resamples, seed):
        try:
            reps.append(metric(y[idx], v[idx]))
        except ValueError:
            continue
    lo, hi = _interval(reps, alpha)
    return BootstrapCI(estimate=metric(y, v), low=lo, high=hi, n_resamples=_finite_count(reps))


def paired_bootstrap_diff(
    metric: Metric,
    y_true: ArrayLike,
    values_a: npt.ArrayLike,
    values_b: npt.ArrayLike,
    groups: Sequence[Hashable] | None = None,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> BootstrapCI:
    """CI of ``metric(A) - metric(B)`` with both models evaluated on the same resamples."""
    y = _as_binary_array(y_true)
    a = _as_values(values_a)
    b = _as_values(values_b)
    reps: list[float] = []
    for idx in _resample_indices(len(y), groups, n_resamples, seed):
        try:
            reps.append(metric(y[idx], a[idx]) - metric(y[idx], b[idx]))
        except ValueError:
            continue
    lo, hi = _interval(reps, alpha)
    return BootstrapCI(
        estimate=metric(y, a) - metric(y, b), low=lo, high=hi, n_resamples=_finite_count(reps)
    )
