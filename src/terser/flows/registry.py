"""Register the flows as Prefect deployments (plan rows **11.6.1** "Flows registered", **11.4.3**).

:func:`build_deployments` turns every flow in :data:`terser.flows.flows.FLOWS` into a
:class:`prefect.deployments.runner.RunnerDeployment` (module entrypoint
``terser.flows.flows:<flow>``, tag ``terser``, ``drift_flow`` on a daily cron, the rewriter flow
also tagged ``gpu_queue``). ``ct_flow`` (continuous training, row 11.4.3) has two starts:

* an **event trigger** :data:`CT_DRIFT_TRIGGER`: every ``prefect.flow-run.Failed`` event of a
  ``drift_flow`` run (a drift alert fails that flow) starts ``ct_flow`` with ``trigger="drift"``;
* a daily cron (04:00, after the 03:00 drift check) with ``check_drift=False``: it only looks for
  a new data batch (``dvc status`` of the data stages), since drift has its own trigger.

:func:`register_deployments` applies them to the configured Prefect API (the trigger becomes a
Prefect automation) and reads them back with :func:`registered_deployments` and
:func:`registered_triggers`.

No server daemon is needed: with ``PREFECT_API_URL`` unset, Prefect starts a temporary local
server for the duration of the command and stops it on exit; the deployments persist in the
local Prefect database (``~/.prefect/prefect.db``), so ``prefect deployment ls`` lists them
afterwards. ``python -m terser.flows serve`` is the long-running process that executes them on
their schedules (event triggers need a long-running server: ``prefect server start``).
"""

from __future__ import annotations

import json
from typing import Any

from prefect.deployments.runner import EntrypointType, RunnerDeployment
from prefect.events import DeploymentEventTrigger

from terser.flows.flows import FLOWS, SCHEDULES

__all__ = [
    "CT_DRIFT_TRIGGER",
    "DEPLOYMENT_PARAMETERS",
    "TAGS",
    "TRIGGERS",
    "build_deployments",
    "register_deployments",
    "registered_deployments",
    "registered_triggers",
]

TAGS: dict[str, list[str]] = {name: ["terser", "row-11.6"] for name in FLOWS}
TAGS["train_rewriter_flow"].append("gpu_queue")
TAGS["drift_flow"].append("monitoring")
TAGS["ct_flow"] += ["row-11.4.3", "gpu_queue"]

CT_DRIFT_TRIGGER = DeploymentEventTrigger(
    name="ct-on-drift-alert",
    description="A failed drift_flow run (drift alert) starts continuous training.",
    expect={"prefect.flow-run.Failed"},
    match_related={"prefect.resource.role": "flow", "prefect.resource.name": "drift_flow"},
    parameters={"trigger": "drift"},
)
TRIGGERS: dict[str, list[DeploymentEventTrigger]] = {"ct_flow": [CT_DRIFT_TRIGGER]}
# Deployment defaults: the scheduled CT run checks for new data only (drift has its trigger).
DEPLOYMENT_PARAMETERS: dict[str, dict[str, Any]] = {"ct_flow": {"check_drift": False}}


def build_deployments() -> list[RunnerDeployment]:
    """One deployment per flow, named after the flow."""
    return [
        f.to_deployment(
            name=name,
            cron=SCHEDULES.get(name),
            tags=TAGS[name],
            description=(f.description or "").strip().splitlines()[0],
            entrypoint_type=EntrypointType.MODULE_PATH,
            parameters=DEPLOYMENT_PARAMETERS.get(name),
            triggers=TRIGGERS.get(name),
        )
        for name, f in FLOWS.items()
    ]


def registered_deployments() -> list[dict[str, Any]]:
    """Every deployment the Prefect API knows of: ``flow/deployment``, id, schedule, tags."""
    from prefect import get_client

    rows = []
    with get_client(sync_client=True) as client:
        for d in client.read_deployments():
            flow = client.read_flow(d.flow_id)
            crons = [getattr(s.schedule, "cron", str(s.schedule)) for s in d.schedules]
            rows.append(
                {
                    "deployment": f"{flow.name}/{d.name}",
                    "id": str(d.id),
                    "entrypoint": d.entrypoint or "",
                    "schedule": ", ".join(crons) or "on demand",
                    "tags": ", ".join(d.tags),
                }
            )
    order = {name: i for i, name in enumerate(FLOWS)}
    return sorted(
        rows, key=lambda r: (order.get(r["deployment"].split("/")[0], 99), r["deployment"])
    )


def _related(spec: Any) -> str:
    specs = spec if isinstance(spec, list) else [spec]
    parts = []
    for s in specs:
        items = getattr(s, "root", s) or {}
        parts.append(", ".join(f"{k}={v}" for k, v in sorted(dict(items).items())))
    return "; ".join(p for p in parts if p)


def registered_triggers() -> list[dict[str, Any]]:
    """Every automation that runs a deployment: name, expected events, match, target, params."""
    from prefect import get_client

    rows = []
    with get_client(sync_client=True) as client:
        names = {
            str(d.id): f"{client.read_flow(d.flow_id).name}/{d.name}"
            for d in client.read_deployments()
        }
        for a in client.read_automations():
            trig = a.trigger
            for action in a.actions:
                dep = str(getattr(action, "deployment_id", "") or "")
                if dep not in names:
                    continue
                rows.append(
                    {
                        "automation": a.name,
                        "expect": ", ".join(sorted(getattr(trig, "expect", None) or [])),
                        "match_related": _related(getattr(trig, "match_related", None)),
                        "deployment": names[dep],
                        "parameters": json.dumps(getattr(action, "parameters", None) or {}),
                        "enabled": a.enabled,
                    }
                )
    return sorted(rows, key=lambda r: (r["deployment"], r["automation"]))


def register_deployments() -> list[dict[str, Any]]:
    """Apply every deployment to the Prefect API; return what the API now lists."""
    for d in build_deployments():
        d.apply()
    return [r for r in registered_deployments() if r["deployment"].split("/")[0] in FLOWS]
