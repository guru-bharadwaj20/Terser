"""Register the six flows as Prefect deployments (plan row **11.6.1**, "Flows registered").

:func:`build_deployments` turns every flow in :data:`terser.flows.flows.FLOWS` into a
:class:`prefect.deployments.runner.RunnerDeployment` (module entrypoint
``terser.flows.flows:<flow>``, tag ``terser``, ``drift_flow`` on a daily cron, the rewriter flow
also tagged ``gpu_queue``). :func:`register_deployments` applies them to the configured Prefect
API and reads them back with :func:`registered_deployments`.

No server daemon is needed: with ``PREFECT_API_URL`` unset, Prefect starts a temporary local
server for the duration of the command and stops it on exit; the deployments persist in the
local Prefect database (``~/.prefect/prefect.db``), so ``prefect deployment ls`` lists them
afterwards. ``python -m terser.flows serve`` is the long-running process that executes them on
their schedules.
"""

from __future__ import annotations

from typing import Any

from prefect.deployments.runner import EntrypointType, RunnerDeployment

from terser.flows.flows import FLOWS, SCHEDULES

__all__ = ["TAGS", "build_deployments", "register_deployments", "registered_deployments"]

TAGS: dict[str, list[str]] = {name: ["terser", "row-11.6"] for name in FLOWS}
TAGS["train_rewriter_flow"].append("gpu_queue")
TAGS["drift_flow"].append("monitoring")


def build_deployments() -> list[RunnerDeployment]:
    """One deployment per flow, named after the flow."""
    return [
        f.to_deployment(
            name=name,
            cron=SCHEDULES.get(name),
            tags=TAGS[name],
            description=(f.description or "").strip().splitlines()[0],
            entrypoint_type=EntrypointType.MODULE_PATH,
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


def register_deployments() -> list[dict[str, Any]]:
    """Apply every deployment to the Prefect API; return what the API now lists."""
    for d in build_deployments():
        d.apply()
    return [r for r in registered_deployments() if r["deployment"].split("/")[0] in FLOWS]
