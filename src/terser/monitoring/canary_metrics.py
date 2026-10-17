"""Quality-canary metrics contract for the dashboard and alert (rows 11.5.4, 11.5.5).

The nightly canary (row 11.5.4) is a batch job, so it does not expose ``/metrics`` itself: after
each run it pushes its result to the Prometheus Pushgateway (``pushgateway:9091`` in
``compose.yaml``) under job :data:`CANARY_JOB`, and Prometheus scrapes the gateway. The Grafana
dashboard (``docker/monitoring/grafana/dashboards/terser-serving.json``) and the alert rule
(``docker/monitoring/grafana/provisioning/alerting/terser-alerts.yaml``) read these gauges:

=============================================  ===============================================
:data:`PRESERVATION`                           preserved / judged pairs in this run
:data:`ROLLING_PRESERVATION`                   pooled over the last :data:`ROLLING_WINDOW` runs;
                                               **alert when < 0.92** (:data:`ALERT_THRESHOLD`)
:data:`PROMPTS`                                judged (prompt, target) pairs in this run
:data:`ROLLING_RUNS`                           runs inside the rolling window (<= 7)
:data:`LAST_RUN`                               Unix time of the push (staleness panel)
=============================================  ===============================================

"Rolling" pools counts rather than averaging ratios, so a short run cannot outweigh a full one.

CLI (also used for the row 11.5.5 alert drill)::

    python -m terser.monitoring.canary_metrics --gateway http://localhost:9091 \\
        --preserved 186 --total 200 --history 1300/1400
    python -m terser.monitoring.canary_metrics --gateway http://localhost:9091 --delete
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Sequence
from typing import Any

__all__ = [
    "ALERT_THRESHOLD",
    "CANARY_JOB",
    "LAST_RUN",
    "METRIC_NAMES",
    "PRESERVATION",
    "PROMPTS",
    "ROLLING_PRESERVATION",
    "ROLLING_RUNS",
    "ROLLING_WINDOW",
    "canary_registry",
    "delete_canary",
    "push_canary",
    "rolling_preservation",
]

CANARY_JOB = "terser_canary"
PRESERVATION = "terser_canary_preservation_ratio"
ROLLING_PRESERVATION = "terser_canary_rolling_preservation_ratio"
PROMPTS = "terser_canary_prompts"
ROLLING_RUNS = "terser_canary_rolling_runs"
LAST_RUN = "terser_canary_last_run_timestamp_seconds"
METRIC_NAMES: tuple[str, ...] = (
    PRESERVATION,
    ROLLING_PRESERVATION,
    PROMPTS,
    ROLLING_RUNS,
    LAST_RUN,
)
ROLLING_WINDOW = 7  # nightly runs, i.e. one week
ALERT_THRESHOLD = 0.92  # C8.3 overall preservation bar; the Grafana rule fires below it


def rolling_preservation(
    runs: Sequence[tuple[int, int]], window: int = ROLLING_WINDOW
) -> tuple[float, int]:
    """Pooled preservation over the last ``window`` ``(preserved, total)`` runs, oldest first.

    Returns ``(ratio, runs_used)``. Runs with ``total == 0`` are skipped.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    used = [(p, t) for p, t in runs if t > 0][-window:]
    for p, t in used:
        if not 0 <= p <= t:
            raise ValueError(f"preserved must be in [0, total], got {p}/{t}")
    if not used:
        raise ValueError("no non-empty canary runs")
    return sum(p for p, _ in used) / sum(t for _, t in used), len(used)


def canary_registry(
    preserved: int,
    total: int,
    history: Sequence[tuple[int, int]] = (),
    *,
    window: int = ROLLING_WINDOW,
    timestamp: float | None = None,
) -> Any:
    """A fresh :class:`~prometheus_client.CollectorRegistry` holding one run's gauges.

    ``history`` holds the earlier runs (oldest first); the rolling value pools them with this run.
    """
    from prometheus_client import CollectorRegistry, Gauge

    runs = [*history, (preserved, total)]
    rolling, used = rolling_preservation(runs, window)
    reg = CollectorRegistry()
    Gauge(PRESERVATION, "Canary preservation, this run", registry=reg).set(preserved / total)
    Gauge(
        ROLLING_PRESERVATION,
        f"Canary preservation pooled over the last {window} runs",
        registry=reg,
    ).set(rolling)
    Gauge(PROMPTS, "Judged canary (prompt, target) pairs, this run", registry=reg).set(total)
    Gauge(ROLLING_RUNS, "Canary runs in the rolling window", registry=reg).set(used)
    Gauge(LAST_RUN, "Unix time of the canary push", registry=reg).set(
        time.time() if timestamp is None else timestamp
    )
    return reg


def push_canary(
    gateway: str,
    preserved: int,
    total: int,
    history: Sequence[tuple[int, int]] = (),
    *,
    window: int = ROLLING_WINDOW,
    handler: Callable[..., Any] | None = None,
) -> float:
    """Push one canary run to the Pushgateway (replacing the job's group); returns the rolling
    value."""
    from prometheus_client import push_to_gateway
    from prometheus_client.exposition import default_handler

    reg = canary_registry(preserved, total, history, window=window)
    push_to_gateway(gateway, job=CANARY_JOB, registry=reg, handler=handler or default_handler)
    rolling, _ = rolling_preservation([*history, (preserved, total)], window)
    return rolling


def delete_canary(gateway: str, handler: Callable[..., Any] | None = None) -> None:
    """Remove the canary group from the Pushgateway (after a drill)."""
    from prometheus_client import delete_from_gateway
    from prometheus_client.exposition import default_handler

    delete_from_gateway(gateway, job=CANARY_JOB, handler=handler or default_handler)


def _run(text: str) -> tuple[int, int]:
    p, t = text.split("/")
    return int(p), int(t)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gateway", required=True, help="Pushgateway address, e.g. localhost:9091")
    ap.add_argument("--preserved", type=int)
    ap.add_argument("--total", type=int)
    ap.add_argument(
        "--history", nargs="*", type=_run, default=[], help="earlier runs as preserved/total"
    )
    ap.add_argument("--delete", action="store_true", help="delete the canary group and exit")
    args = ap.parse_args(argv)
    if args.delete:
        delete_canary(args.gateway)
        print(f"deleted job={CANARY_JOB} from {args.gateway}")
        return 0
    if args.preserved is None or args.total is None:
        ap.error("--preserved and --total are required unless --delete")
    rolling = push_canary(args.gateway, args.preserved, args.total, args.history)
    print(
        f"pushed job={CANARY_JOB}: run {args.preserved}/{args.total} "
        f"= {args.preserved / args.total:.4f}, rolling {rolling:.4f} "
        f"(alert < {ALERT_THRESHOLD})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
