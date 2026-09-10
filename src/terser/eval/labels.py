"""Positive-class convention for the quality gate.

Plan row **1.1.3**, recorded in ``docs/decisions/ADR-0002-problem-framing.md`` (section
"Positive-class convention").

The quality gate is a binary classifier on (original, rewrite, task category, target). Its
**positive class is ``degraded``**:

* ``PRESERVED = 0`` - the rewrite keeps the answer's quality (row 2.3.4 label ``preserved``).
* ``DEGRADED = 1`` - the rewrite makes the answer worse (label ``degraded``).
* ``POSITIVE_LABEL = DEGRADED``.

Consequences for every metric in :mod:`terser.eval`:

* **Sensitivity** (recall of the positive class, TP / (TP + FN)) is the share of truly degraded
  rewrites the gate *catches*. A miss (FN) is silent over-compression reaching the user, the
  costliest error, so thresholds are chosen at sensitivity >= 0.95 (row 3.2.3).
* **Specificity** (TN / (TN + FP)) is the share of truly preserved rewrites the gate *accepts*.
  A false alarm (FP) only forfeits savings: the system falls back to a lighter rewrite or the
  original prompt.
* Precision, F1, ROC and PR curves are all computed with ``pos_label=DEGRADED``; with
  scikit-learn pass ``pos_label=POSITIVE_LABEL`` explicitly.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

__all__ = [
    "DEGRADED",
    "LABEL_NAMES",
    "POSITIVE_LABEL",
    "PRESERVED",
    "confusion_counts",
    "sensitivity",
    "specificity",
    "to_binary",
]

PRESERVED: int = 0
DEGRADED: int = 1
POSITIVE_LABEL: int = DEGRADED

LABEL_NAMES: dict[int, str] = {PRESERVED: "preserved", DEGRADED: "degraded"}
_NAME_TO_LABEL: dict[str, int] = {name: label for label, name in LABEL_NAMES.items()}

ArrayLike: TypeAlias = Iterable[str | bool | int] | npt.NDArray[np.generic]


def to_binary(label: str | bool | int) -> int:
    """Map one label to the integer convention (``PRESERVED = 0``, ``DEGRADED = 1``).

    Args:
        label: ``"preserved"`` / ``"degraded"`` (case and surrounding whitespace ignored), a
            ``bool`` (``True`` means degraded, i.e. "is positive"), or the integer ``0`` / ``1``
            (numpy integer and bool scalars are accepted too).

    Returns:
        ``0`` or ``1``.

    Raises:
        ValueError: For an unknown string or an integer other than 0 and 1.
        TypeError: For any other type (e.g. ``float``), to avoid silently thresholding scores.
    """
    if isinstance(label, bool | np.bool_):
        return DEGRADED if bool(label) else PRESERVED
    if isinstance(label, str):
        key = label.strip().lower()
        if key in _NAME_TO_LABEL:
            return _NAME_TO_LABEL[key]
        raise ValueError(f"unknown label {label!r}; expected 'preserved' or 'degraded'")
    if isinstance(label, int | np.integer):
        value = int(label)
        if value in (PRESERVED, DEGRADED):
            return value
        raise ValueError(f"integer label must be 0 (preserved) or 1 (degraded), got {value}")
    raise TypeError(
        f"label must be str, bool or int, got {type(label).__name__}; "
        "threshold scores before converting"
    )


def _as_binary_array(y: ArrayLike) -> npt.NDArray[np.int64]:
    """Convert a sequence of labels to a 1-D ``int64`` array of 0/1 via :func:`to_binary`."""
    if isinstance(y, np.ndarray):
        if y.ndim != 1:
            raise ValueError(f"labels must be 1-D, got shape {y.shape}")
        items: list[str | bool | int] = y.tolist()
    else:
        items = list(y)
    return np.fromiter((to_binary(v) for v in items), dtype=np.int64, count=len(items))


def confusion_counts(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[int, int, int, int]:
    """Confusion counts with ``degraded`` as the positive class.

    Args:
        y_true: True labels (any form accepted by :func:`to_binary`).
        y_pred: Predicted labels, same length as ``y_true``.

    Returns:
        ``(tn, fp, fn, tp)``, in scikit-learn's ``confusion_matrix(...).ravel()`` order.

    Raises:
        ValueError: If lengths differ.
    """
    t = _as_binary_array(y_true)
    p = _as_binary_array(y_pred)
    if t.shape != p.shape:
        raise ValueError(f"y_true and y_pred differ in length: {t.shape[0]} vs {p.shape[0]}")
    tp = int(np.sum((t == DEGRADED) & (p == DEGRADED)))
    fn = int(np.sum((t == DEGRADED) & (p == PRESERVED)))
    tn = int(np.sum((t == PRESERVED) & (p == PRESERVED)))
    fp = int(np.sum((t == PRESERVED) & (p == DEGRADED)))
    return tn, fp, fn, tp


def sensitivity(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Share of truly degraded rewrites the gate flags: TP / (TP + FN).

    Returns ``nan`` when ``y_true`` contains no degraded examples (the ratio is undefined).
    """
    _, _, fn, tp = confusion_counts(y_true, y_pred)
    denom = tp + fn
    return float(tp / denom) if denom else float("nan")


def specificity(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Share of truly preserved rewrites the gate accepts: TN / (TN + FP).

    Returns ``nan`` when ``y_true`` contains no preserved examples (the ratio is undefined).
    """
    tn, fp, _, _ = confusion_counts(y_true, y_pred)
    denom = tn + fp
    return float(tn / denom) if denom else float("nan")
