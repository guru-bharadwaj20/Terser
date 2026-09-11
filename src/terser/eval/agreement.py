"""Agreement statistics for judge validation (plan row **2.3.6**, checkpoints C2.3-C2.6).

* :func:`cohen_kappa` - Cohen's kappa for two raters (matches
  ``sklearn.metrics.cohen_kappa_score``), with :func:`kappa_with_ci` adding a percentile
  bootstrap 95% CI over items and :func:`per_category_kappa` a per-category breakdown.
* :func:`confusion_table` - a rater-by-rater contingency table.
* :func:`detector_metrics` - the judge as a degradation detector against human labels, with
  ``degraded`` as the positive class (:mod:`terser.eval.labels`).
* :func:`position_disagreement_rate` - share of pairs whose two A/B orders favour different sides.
* :func:`checkpoint_checks` - explicit PASS/FAIL lines for C2.3-C2.6.

Items where either rater has no label (``None``, e.g. "cannot judge" or an unparseable verdict) are
dropped pairwise and counted in ``n_dropped``.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from terser.eval.labels import LABEL_NAMES, confusion_counts, sensitivity, specificity

__all__ = [
    "THRESHOLDS",
    "CheckResult",
    "ConfusionTable",
    "DetectorMetrics",
    "KappaResult",
    "checkpoint_checks",
    "cohen_kappa",
    "confusion_table",
    "detector_metrics",
    "kappa_with_ci",
    "per_category_kappa",
    "position_disagreement_rate",
]

THRESHOLDS: Final[Mapping[str, float]] = {
    "human_human_kappa": 0.60,
    "judge_human_kappa": 0.60,
    "judge_human_kappa_per_category": 0.50,
    "sensitivity": 0.85,
    "specificity": 0.80,
    "position_disagreement": 0.10,
}
"""Checkpoint thresholds from CONTRIBUTING.md (C2.3-C2.6)."""


def _paired(
    y1: Sequence[Hashable | None], y2: Sequence[Hashable | None]
) -> tuple[list[Hashable], list[Hashable], int]:
    if len(y1) != len(y2):
        raise ValueError(f"rater label sequences differ in length: {len(y1)} vs {len(y2)}")
    a: list[Hashable] = []
    b: list[Hashable] = []
    for u, v in zip(y1, y2, strict=True):
        if u is None or v is None:
            continue
        a.append(u)
        b.append(v)
    return a, b, len(y1) - len(a)


def _kappa_from_codes(i: npt.NDArray[np.intp], j: npt.NDArray[np.intp], k: int) -> float:
    n = i.shape[0]
    if n == 0:
        return float("nan")
    m = np.zeros((k, k), dtype=np.float64)
    np.add.at(m, (i, j), 1.0)
    po = np.trace(m) / n
    pe = float(m.sum(axis=1) @ m.sum(axis=0)) / (n * n)
    if math.isclose(pe, 1.0):
        return float("nan")
    return float((po - pe) / (1.0 - pe))


def _encode(
    a: Sequence[Hashable], b: Sequence[Hashable], labels: Sequence[Hashable] | None
) -> tuple[npt.NDArray[np.intp], npt.NDArray[np.intp], int]:
    cats = list(labels) if labels is not None else sorted(set(a) | set(b), key=str)
    index = {c: n for n, c in enumerate(cats)}
    unknown = (set(a) | set(b)) - set(index)
    if unknown:
        raise ValueError(f"labels not in the label set: {sorted(map(str, unknown))}")
    return (
        np.fromiter((index[x] for x in a), dtype=np.intp, count=len(a)),
        np.fromiter((index[x] for x in b), dtype=np.intp, count=len(b)),
        len(cats),
    )


def cohen_kappa(
    y1: Sequence[Hashable | None],
    y2: Sequence[Hashable | None],
    *,
    labels: Sequence[Hashable] | None = None,
) -> float:
    """Unweighted Cohen's kappa, ``(p_o - p_e) / (1 - p_e)``.

    ``None`` entries are dropped pairwise. Returns ``nan`` when no items remain or when chance
    agreement is 1 (both raters used one and the same label), where kappa is undefined.
    """
    a, b, _ = _paired(y1, y2)
    i, j, k = _encode(a, b, labels)
    return _kappa_from_codes(i, j, k)


@dataclass(frozen=True)
class KappaResult:
    """Kappa with a percentile bootstrap confidence interval."""

    kappa: float
    ci_low: float
    ci_high: float
    n: int
    n_dropped: int
    observed_agreement: float

    def format(self) -> str:
        """``0.712 [0.601, 0.810] (n=300)``."""
        return f"{_fmt(self.kappa)} [{_fmt(self.ci_low)}, {_fmt(self.ci_high)}] (n={self.n})"


def kappa_with_ci(
    y1: Sequence[Hashable | None],
    y2: Sequence[Hashable | None],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    labels: Sequence[Hashable] | None = None,
) -> KappaResult:
    """Cohen's kappa with a ``1 - alpha`` percentile bootstrap CI (items resampled, replacement).

    Resamples where kappa is undefined are skipped; the CI is ``nan`` if all are undefined.
    """
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    a, b, dropped = _paired(y1, y2)
    i, j, k = _encode(a, b, labels)
    n = i.shape[0]
    kappa = _kappa_from_codes(i, j, k)
    po = float(np.mean(i == j)) if n else float("nan")
    lo = hi = float("nan")
    if n:
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, n, size=(n_boot, n))
        boots = np.array([_kappa_from_codes(i[r], j[r], k) for r in idx])
        boots = boots[~np.isnan(boots)]
        if boots.size:
            lo, hi = (float(x) for x in np.quantile(boots, [alpha / 2, 1 - alpha / 2]))
    return KappaResult(kappa, lo, hi, n, dropped, po)


def per_category_kappa(
    categories: Sequence[str],
    y1: Sequence[Hashable | None],
    y2: Sequence[Hashable | None],
    *,
    n_boot: int = 2000,
    seed: int = 0,
    labels: Sequence[Hashable] | None = None,
) -> dict[str, KappaResult]:
    """:func:`kappa_with_ci` within each category (sorted by category name)."""
    if not len(categories) == len(y1) == len(y2):
        raise ValueError("categories, y1 and y2 must have the same length")
    out: dict[str, KappaResult] = {}
    for cat in sorted(set(categories)):
        rows = [n for n, c in enumerate(categories) if c == cat]
        out[cat] = kappa_with_ci(
            [y1[n] for n in rows],
            [y2[n] for n in rows],
            n_boot=n_boot,
            seed=seed,
            labels=labels,
        )
    return out


@dataclass(frozen=True)
class ConfusionTable:
    """Counts ``matrix[r][c]`` of items rater 1 labelled ``labels[r]`` and rater 2 ``labels[c]``."""

    labels: tuple[str, ...]
    matrix: tuple[tuple[int, ...], ...]
    row_name: str
    col_name: str

    def markdown(self) -> str:
        """Render as a Markdown table with totals."""
        head = f"| {self.row_name} \\ {self.col_name} | " + " | ".join(self.labels) + " | total |"
        sep = "| :--- |" + " ---: |" * (len(self.labels) + 1)
        lines = [head, sep]
        for name, row in zip(self.labels, self.matrix, strict=True):
            lines.append(f"| {name} | " + " | ".join(map(str, row)) + f" | {sum(row)} |")
        cols = [sum(r[c] for r in self.matrix) for c in range(len(self.labels))]
        lines.append("| total | " + " | ".join(map(str, cols)) + f" | {sum(cols)} |")
        return "\n".join(lines)


def confusion_table(
    y1: Sequence[Hashable | None],
    y2: Sequence[Hashable | None],
    *,
    labels: Sequence[Hashable] = tuple(LABEL_NAMES.values()),
    row_name: str = "rater 1",
    col_name: str = "rater 2",
) -> ConfusionTable:
    """Contingency table over items where both raters gave a label."""
    a, b, _ = _paired(y1, y2)
    i, j, k = _encode(a, b, labels)
    m = np.zeros((k, k), dtype=np.int64)
    np.add.at(m, (i, j), 1)
    return ConfusionTable(
        tuple(str(x) for x in labels),
        tuple(tuple(int(v) for v in row) for row in m),
        row_name,
        col_name,
    )


@dataclass(frozen=True)
class DetectorMetrics:
    """Judge labels scored against human labels, positive class = degraded."""

    sensitivity: float
    specificity: float
    tn: int
    fp: int
    fn: int
    tp: int
    n_dropped: int


def detector_metrics(
    human: Sequence[str | int | None], judge: Sequence[str | int | None]
) -> DetectorMetrics:
    """Sensitivity and specificity of the judge as a degradation detector (humans = truth)."""
    h, j, dropped = _paired(human, judge)
    y_true: list[str | int] = [x for x in h if isinstance(x, str | int)]
    y_pred: list[str | int] = [x for x in j if isinstance(x, str | int)]
    if len(y_true) != len(h) or len(y_pred) != len(j):
        raise TypeError("labels must be str or int")
    tn, fp, fn, tp = confusion_counts(y_true, y_pred)
    return DetectorMetrics(
        sensitivity(y_true, y_pred), specificity(y_true, y_pred), tn, fp, fn, tp, dropped
    )


def position_disagreement_rate(
    sides_ab: Sequence[str | None], sides_ba: Sequence[str | None]
) -> tuple[float, int, int]:
    """Share of pairs whose ``original_first`` and ``rewrite_first`` sides differ.

    Sides must already be mapped from positions (``original`` / ``rewrite`` / ``tie``). Pairs with
    a missing verdict are excluded.

    Returns:
        ``(rate, n_disagree, n_parsed)``; rate is ``nan`` if no pair parsed.
    """
    a, b, _ = _paired(sides_ab, sides_ba)
    n = len(a)
    disagree = sum(1 for u, v in zip(a, b, strict=True) if u != v)
    return (disagree / n if n else float("nan")), disagree, n


@dataclass(frozen=True)
class CheckResult:
    """One checkpoint line."""

    check_id: str
    description: str
    value: float
    threshold: float
    passed: bool
    comparator: str

    def line(self) -> str:
        """``C2.3 PASS - human-human kappa = 0.712 (>= 0.60)``."""
        verdict = "PASS" if self.passed else "FAIL"
        return (
            f"{self.check_id} {verdict} - {self.description} = {_fmt(self.value)} "
            f"({self.comparator} {self.threshold:.2f})"
        )


def _ge(value: float, threshold: float) -> bool:
    return not math.isnan(value) and value >= threshold


def checkpoint_checks(
    *,
    human_human_kappa: float,
    judge_human_kappa: float,
    judge_human_kappa_per_category: Mapping[str, float],
    sensitivity: float,
    specificity: float,
    position_disagreement: float,
    thresholds: Mapping[str, float] = THRESHOLDS,
) -> list[CheckResult]:
    """PASS/FAIL results for C2.3 (one line), C2.4 (overall + per category), C2.5 (two), C2.6.

    ``nan`` values always fail. C2.4 fails if any category is below its threshold.
    """
    t = thresholds
    out = [
        CheckResult(
            "C2.3",
            "human-human kappa",
            human_human_kappa,
            t["human_human_kappa"],
            _ge(human_human_kappa, t["human_human_kappa"]),
            ">=",
        ),
        CheckResult(
            "C2.4",
            "judge-human kappa (overall)",
            judge_human_kappa,
            t["judge_human_kappa"],
            _ge(judge_human_kappa, t["judge_human_kappa"]),
            ">=",
        ),
    ]
    per_cat = t["judge_human_kappa_per_category"]
    for cat, value in sorted(judge_human_kappa_per_category.items()):
        out.append(
            CheckResult(
                "C2.4", f"judge-human kappa ({cat})", value, per_cat, _ge(value, per_cat), ">="
            )
        )
    out.append(
        CheckResult(
            "C2.5",
            "judge sensitivity (degraded)",
            sensitivity,
            t["sensitivity"],
            _ge(sensitivity, t["sensitivity"]),
            ">=",
        )
    )
    out.append(
        CheckResult(
            "C2.5",
            "judge specificity",
            specificity,
            t["specificity"],
            _ge(specificity, t["specificity"]),
            ">=",
        )
    )
    pd = position_disagreement
    out.append(
        CheckResult(
            "C2.6",
            "A/B-order disagreement rate",
            pd,
            t["position_disagreement"],
            not math.isnan(pd) and pd <= t["position_disagreement"],
            "<=",
        )
    )
    return out


def _fmt(x: float) -> str:
    return "n/a" if math.isnan(x) else f"{x:.3f}"
