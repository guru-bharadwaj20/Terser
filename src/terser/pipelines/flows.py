"""Alias of :mod:`terser.flows` (plan rows **11.6.1**, **11.6.2**).

The Prefect flows moved to the :mod:`terser.flows` package (flows, retried and cached tasks,
failure notifications, deployment registration, CLI ``python -m terser.flows``). This module
re-exports the public names so older references keep working; import from :mod:`terser.flows`.
"""

from __future__ import annotations

from terser.flows import (
    FLOWS,
    StepError,
    StepResult,
    data_flow,
    drift_flow,
    evaluate_flow,
    promote_flow,
    train_classical_flow,
    train_rewriter_flow,
)

__all__ = [
    "FLOWS",
    "StepError",
    "StepResult",
    "data_flow",
    "drift_flow",
    "evaluate_flow",
    "promote_flow",
    "train_classical_flow",
    "train_rewriter_flow",
]
