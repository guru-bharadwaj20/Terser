"""Prefect orchestration (plan rows **11.6.1** flows, **11.6.2** reliability, **11.4.3** CT).

Eight flows over the project's existing entry points (``dvc repro`` stages, ``python -m`` modules,
the :mod:`terser.gpu_queue` API, :mod:`terser.phase11.register`, :mod:`terser.monitoring`):
``data_flow``, ``train_classical_flow``, ``train_rewriter_flow`` (submits to ``gpu_queue/``),
``evaluate_flow``, ``promote_flow``, ``drift_flow``, ``ct_flow`` (continuous training, which
composes the others) and ``canary_flow`` (nightly quality canary, row 11.5.4;
:mod:`terser.flows.flows`). Tasks retry, LLM steps are cached on an input
hash, and every failure is notified (:mod:`terser.flows.reliability`). Every flow has a
``dry_run`` mode that returns its command plan without running anything.

CLI (``python -m terser.flows --help``)::

    python -m terser.flows list                      # the registered flows and their policies
    python -m terser.flows plan [FLOW ...]           # dry-run command plans
    python -m terser.flows deploy                    # register deployments with the Prefect API
    python -m terser.flows run FLOW [--dry-run] [--param k=JSON ...]
    python -m terser.flows serve                     # long-running: run deployments on schedule
    python -m terser.flows demo --out reports/p11_flows.md

Requires the ``orchestrate`` extra (``pip install -e ".[orchestrate]"``, Prefect 3).
"""

from __future__ import annotations

from terser.flows.flows import (
    FLOWS,
    PLAN_PARAMS,
    canary_flow,
    ct_flow,
    data_flow,
    drift_flow,
    evaluate_flow,
    plan,
    promote_flow,
    rewriter_job,
    train_classical_flow,
    train_rewriter_flow,
)
from terser.flows.reliability import notify_failure, notify_flow_failure
from terser.flows.steps import StepError, StepResult, llm_step, shell_step

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
