"""Retry policy, failure notifications and failure injection for the flows (plan row **11.6.2**).

* **Retries.** Shell steps retry :data:`SHELL_RETRIES` times after :data:`SHELL_RETRY_DELAY`
  seconds; LLM steps (vLLM / hosted calls, which fail on timeouts and server restarts) retry
  :data:`LLM_RETRIES` times with the growing delays :data:`LLM_RETRY_DELAYS`. Deterministic
  checks (GPU queue validation, the promotion guard) do not retry: a second attempt cannot pass.
* **Failure notifications.** :func:`notify_failure` is the ``on_failure`` hook of every task
  (:func:`notify_flow_failure` of every flow, same behaviour). It appends one JSON record to
  the alert log (``logs/flow_alerts.jsonl``, or the path in ``TERSER_ALERT_LOG``), logs it as
  ``CRITICAL`` on the ``terser.alerts`` logger (the logger the drift alerts of rows 11.5.2 and
  11.5.3 use), and, only when ``TERSER_ALERT_WEBHOOK`` is set,
  POSTs the record to that URL. With the variable unset no network call is made. A hook never
  raises, so a broken webhook cannot mask the original failure.
* **Failure injection.** :func:`flaky_command` builds a command that fails its first ``n`` runs
  and then succeeds; :func:`counter_command` counts executions. Tests and
  ``python -m terser.flows demo`` use them to show a step recovering, a permanent failure
  notifying, and a cached step not re-running.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from terser.config import project_root

__all__ = [
    "ALERT_LOG_ENV",
    "LLM_RETRIES",
    "LLM_RETRY_DELAYS",
    "SHELL_RETRIES",
    "SHELL_RETRY_DELAY",
    "WEBHOOK_ENV",
    "alert_log_path",
    "counter_command",
    "flaky_command",
    "notify_failure",
    "notify_flow_failure",
    "read_alerts",
]

REPO_ROOT = project_root()
ALERT_LOGGER = "terser.alerts"
ALERT_LOG_ENV = "TERSER_ALERT_LOG"
WEBHOOK_ENV = "TERSER_ALERT_WEBHOOK"
DEFAULT_ALERT_LOG = REPO_ROOT / "logs" / "flow_alerts.jsonl"

SHELL_RETRIES = 2
SHELL_RETRY_DELAY = 10.0
LLM_RETRIES = 3
LLM_RETRY_DELAYS: list[float] = [10.0, 30.0, 90.0]
WEBHOOK_TIMEOUT_S = 5.0

_log = logging.getLogger(ALERT_LOGGER)


def alert_log_path() -> Path:
    """The alert log: ``$TERSER_ALERT_LOG`` if set, else ``logs/flow_alerts.jsonl``."""
    return Path(os.environ.get(ALERT_LOG_ENV) or DEFAULT_ALERT_LOG)


def _record(obj: Any, run: Any, state: Any) -> dict[str, Any]:
    is_task = hasattr(run, "task_key")
    return {
        "time": datetime.now(UTC).isoformat(timespec="seconds"),
        "kind": "task" if is_task else "flow",
        "name": str(getattr(obj, "name", obj)),
        "run_name": str(getattr(run, "name", "")),
        "run_id": str(getattr(run, "id", "")),
        "attempts": int(getattr(run, "run_count", 0) or 0),
        "state": str(getattr(state, "name", state)),
        "message": str(getattr(state, "message", "") or "")[:2000],
    }


def notify_failure(obj: Any, run: Any, state: Any) -> None:
    """``on_failure`` hook for tasks ``(task, task_run, state)``.

    Writes the alert record, logs it, and posts it to ``$TERSER_ALERT_WEBHOOK`` when set.
    """
    _notify(_record(obj, run, state))


def notify_flow_failure(flow: Any, flow_run: Any, state: Any) -> None:
    """``on_failure`` hook for flows (Prefect passes these three by name)."""
    _notify(_record(flow, flow_run, state))


def _notify(record: dict[str, Any]) -> None:
    try:
        path = alert_log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except OSError as exc:  # pragma: no cover - read-only checkout
        _log.error("cannot write alert log: %s", exc)
    _log.critical(
        "%s %s failed after %d attempt(s): %s",
        record["kind"],
        record["name"],
        record["attempts"],
        record["message"],
    )
    url = os.environ.get(WEBHOOK_ENV)
    if not url:
        return
    try:
        import httpx

        httpx.post(url, json=record, timeout=WEBHOOK_TIMEOUT_S).raise_for_status()
    except Exception as exc:  # noqa: BLE001 - a notification failure must not mask the failure
        _log.error("alert webhook %s failed: %s", url, exc)


def read_alerts(path: Path | None = None) -> list[dict[str, Any]]:
    """Every record in the alert log (empty if it does not exist)."""
    path = path or alert_log_path()
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(x) for x in lines if x.strip()]


_COUNT = (
    "import pathlib,sys; p=pathlib.Path(sys.argv[1]); "
    "n=int(p.read_text() or 0) if p.exists() else 0; n+=1; p.write_text(str(n)); "
)


def counter_command(counter: Path) -> list[str]:
    """A command that increments the integer in ``counter`` and prints it (execution count)."""
    return ["python", "-c", _COUNT + "print(n)", str(counter)]


def flaky_command(counter: Path, fail_times: int) -> list[str]:
    """A command that exits 1 on its first ``fail_times`` runs and succeeds afterwards.

    The run count lives in ``counter``, so the failure is transient across retries of the same
    step, exactly like a vLLM server that is still loading.
    """
    code = (
        _COUNT + "k=int(sys.argv[2]); "
        "print(f'injected failure {n}/{k}', file=sys.stderr) if n <= k else None; "
        "print(f'attempt {n}: ok') if n > k else None; sys.exit(1 if n <= k else 0)"
    )
    return ["python", "-c", code, str(counter), str(fail_times)]
