"""Alias of :mod:`terser.flows` (plan rows **11.6.1**, **11.6.2**).

The Prefect flows moved to the :mod:`terser.flows` package (flows, retried and cached tasks,
failure notifications, deployment registration, CLI ``python -m terser.flows``). This module
re-exports every public name of :mod:`terser.flows` so older references keep working; import
from :mod:`terser.flows`.
"""

from __future__ import annotations

from terser.flows import (
    FLOWS,
    PLAN_PARAMS,
    StepError,
    StepResult,
    canary_flow,
    ct_flow,
    data_flow,
    drift_flow,
    evaluate_flow,
    llm_step,
    notify_failure,
    notify_flow_failure,
    plan,
    promote_flow,
    rewriter_job,
    shell_step,
    train_classical_flow,
    train_rewriter_flow,
)

__all__ = [
    "FLOWS",
    "PLAN_PARAMS",
    "StepError",
    "StepResult",
    "canary_flow",
    "ct_flow",
    "data_flow",
    "drift_flow",
    "evaluate_flow",
    "llm_step",
    "notify_failure",
    "notify_flow_failure",
    "plan",
    "promote_flow",
    "rewriter_job",
    "shell_step",
    "train_classical_flow",
    "train_rewriter_flow",
]
