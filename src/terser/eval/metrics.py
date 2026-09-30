"""Classification metric suite for the quality gate (plan row **3.2.1**).

Every metric uses ``degraded`` as the positive class (:mod:`terser.eval.labels`), so
*sensitivity* is the share of degraded rewrites caught and *specificity* the share of preserved
rewrites accepted. Values match scikit-learn to 1e-9 with ``pos_label=POSITIVE_LABEL`` and
``zero_division=0`` (tested).

* :func:`classification_metrics` - accuracy, precision, recall, sensitivity, specificity, F1 and
  the confusion counts for one set of predictions.
* :func:`confusion_matrix` - the 2x2 matrix in scikit-learn's ``[[tn, fp], [fn, tp]]`` layout.
* :func:`per_category_metrics` - the same suite per task category, plus an ``"overall"`` row.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import asdict, dataclass

import numpy as np
import numpy.typing as npt

from terser.eval.labels import ArrayLike, _as_binary_array, confusion_counts

__all__ = [
    "ClassificationMetrics",
    "classification_metrics",
    "confusion_matrix",
    "per_category_metrics",
]


def _ratio(num: int, denom: int) -> float:
    """``num / denom``, or ``0.0`` for an empty denominator (scikit-learn ``zero_division=0``)."""
    return float(num / denom) if denom else 0.0


@dataclass(frozen=True)
class ClassificationMetrics:
    """The row 3.2.1 metric suite for one set of predictions (positive class ``degraded``)."""

    n: int
    tn: int
    fp: int
    fn: int
    tp: int
    accuracy: float
    precision: float
    recall: float
    sensitivity: float
    specificity: float
    f1: float

    def as_dict(self) -> dict[str, float | int]:
        """Flat mapping, convenient for MLflow ``log_metrics`` and report tables."""
        return asdict(self)


def classification_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> ClassificationMetrics:
    """Compute the full metric suite.

    Args:
        y_true: True labels (anything :func:`terser.eval.labels.to_binary` accepts).
        y_pred: Predicted labels, same length.

    Returns:
        A :class:`ClassificationMetrics`. Undefined ratios (empty denominators) are ``0.0``, as
        scikit-learn returns with ``zero_division=0``; recall and sensitivity are the same number.

    Raises:
        ValueError: If lengths differ or the input is empty.
    """
    tn, fp, fn, tp = confusion_counts(y_true, y_pred)
    n = tn + fp + fn + tp
    if n == 0:
        raise ValueError("cannot compute metrics on zero examples")
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    return ClassificationMetrics(
        n=n,
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
        accuracy=_ratio(tp + tn, n),
        precision=precision,
        recall=recall,
        sensitivity=recall,
        specificity=_ratio(tn, tn + fp),
        f1=_ratio(2 * tp, 2 * tp + fp + fn),
    )


def confusion_matrix(y_true: ArrayLike, y_pred: ArrayLike) -> npt.NDArray[np.int64]:
    """2x2 confusion matrix, rows = truth, columns = prediction, order ``[preserved, degraded]``."""
    tn, fp, fn, tp = confusion_counts(y_true, y_pred)
    return np.array([[tn, fp], [fn, tp]], dtype=np.int64)


def per_category_metrics(
    y_true: ArrayLike, y_pred: ArrayLike, categories: Sequence[Hashable]
) -> dict[Hashable, ClassificationMetrics]:
    """The metric suite per task category, plus the pooled ``"overall"`` entry.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        categories: Category of each example, same length.

    Returns:
        Mapping from category (sorted by ``str``) to its metrics, ending with ``"overall"``.

    Raises:
        ValueError: If lengths differ.
    """
    t = _as_binary_array(y_true)
    p = _as_binary_array(y_pred)
    if not (len(t) == len(p) == len(categories)):
        raise ValueError(
            f"lengths differ: y_true {len(t)}, y_pred {len(p)}, categories {len(categories)}"
        )
    cats = np.asarray(list(categories), dtype=object)
    out: dict[Hashable, ClassificationMetrics] = {}
    for cat in sorted(set(categories), key=str):
        mask = cats == cat
        out[cat] = classification_metrics(t[mask], p[mask])
    out["overall"] = classification_metrics(t, p)
    return out
