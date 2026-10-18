"""Evidence report for plan rows **11.6.1** and **11.6.2** (``reports/p11_flows.md``).

``python -m terser.flows demo --out reports/p11_flows.md`` runs, against an isolated temporary
Prefect API (:func:`prefect.testing.utilities.prefect_test_harness`, so ``~/.prefect`` is not
touched and no server outlives the command):

1. **Registration**: every flow applied as a deployment and read back from the API.
2. **Task plans**: every flow in dry-run mode, recording its commands.
3. **Injected failure recovers**: the real ``evaluate_flow`` with a step that fails its first two
   attempts; the production retry policy (2 retries, 10 s apart) recovers it.
4. **Permanent failure notifies**: a step that never succeeds; the ``on_failure`` hooks write the
   alert log and POST to a webhook (a receiver on ``127.0.0.1``, no external network).
5. **Cached LLM step**: the same ``llm_step`` twice (second ``Cached``, executed once), then again
   after an input changed (re-executed).
"""

from __future__ import annotations

import contextlib
import http.server
import json
import logging
import os
import tempfile
import threading
import time
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import prefect
from prefect import flow, get_client
from prefect.client.schemas.filters import FlowRunFilter, FlowRunFilterId
from prefect.settings import PREFECT_LOCAL_STORAGE_PATH, temporary_settings
from prefect.testing.utilities import prefect_test_harness

from terser.flows import steps
from terser.flows.flows import FLOWS, evaluate_flow, module_path, plan
from terser.flows.registry import register_deployments
from terser.flows.reliability import (
    ALERT_LOG_ENV,
    WEBHOOK_ENV,
    counter_command,
    flaky_command,
    read_alerts,
)
from terser.flows.steps import StepResult, llm_step

__all__ = ["isolated_prefect", "llm_cache_demo", "run_demo", "task_states"]

POLICY_TASKS = (
    steps.shell_step,
    steps.llm_step,
    steps.submit_gpu_job,
    steps.drift_check,
    steps.input_drift_check,
    steps.promotion_guard,
    steps.registry_promote,
)


@contextlib.contextmanager
def _env(**values: str | None) -> Iterator[None]:
    old = {k: os.environ.get(k) for k in values}
    try:
        for k, v in values.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@contextlib.contextmanager
def isolated_prefect(tmp: Path) -> Iterator[None]:
    """Temporary Prefect API + database, result storage and alert log under ``tmp``."""
    with (
        _env(
            PREFECT_SERVER_ANALYTICS_ENABLED="false",
            **{ALERT_LOG_ENV: str(tmp / "flow_alerts.jsonl"), WEBHOOK_ENV: None},
        ),
        prefect_test_harness(),
        temporary_settings({PREFECT_LOCAL_STORAGE_PATH: tmp / "storage"}),
    ):
        yield


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.INFO)
        self.lines: list[tuple[str, str, str]] = []

    def emit(self, record: logging.LogRecord) -> None:
        run = str(getattr(record, "task_run_name", "") or getattr(record, "flow_run_name", ""))
        self.lines.append((record.levelname, run, record.getMessage().splitlines()[0]))


@contextlib.contextmanager
def _capture_logs() -> Iterator[_Capture]:
    handler = _Capture()
    logger = logging.getLogger("prefect")
    logger.addHandler(handler)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)


@contextlib.contextmanager
def _webhook_receiver() -> Iterator[tuple[str, list[dict[str, Any]]]]:
    received: list[dict[str, Any]] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # http.server API
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            received.append(json.loads(body))
            self.send_response(204)
            self.end_headers()

        def log_message(self, format: str, *args: Any) -> None:
            del format, args

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/alert", received
    finally:
        server.shutdown()
        server.server_close()


def task_states(flow_run_id: UUID | None) -> dict[str, list[str]]:
    """``{task run name: [state names in order]}`` for one flow run, read from the API."""
    out: dict[str, list[str]] = {}
    if flow_run_id is None:
        return out
    with get_client(sync_client=True) as client:
        runs = client.read_task_runs(
            flow_run_filter=FlowRunFilter(id=FlowRunFilterId(any_=[flow_run_id]))
        )
        for tr in sorted(runs, key=lambda r: str(r.start_time or r.expected_start_time)):
            states = sorted(client.read_task_run_states(tr.id), key=lambda s: s.timestamp)
            out[tr.name] = [str(s.name) for s in states]
    return out


@flow(name="llm_cache_demo", log_prints=True)
def llm_cache_demo(cmd: Sequence[str], inputs: Sequence[str]) -> list[str]:
    """The same LLM step twice: the second call must come back ``Cached``."""
    states = [llm_step("judge_demo", list(cmd), list(inputs), return_state=True) for _ in "ab"]
    return [str(s.name) for s in states]


def _md_table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    def cell(v: Any) -> str:
        return str(v).replace("|", "\\|").replace("\n", " ")

    return [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
        *("| " + " | ".join(cell(v) for v in r) + " |" for r in rows),
    ]


def _short(text: str, n: int = 110) -> str:
    return text if len(text) <= n else text[: n - 3] + "..."


def _middle(text: str, n: int = 170) -> str:
    """Keep both ends (exception head, ``Retry i/n will start ...`` tail) of a long line."""
    return text if len(text) <= n else text[: n // 2 - 3] + " ... " + text[-(n // 2) :]


def _policy_rows() -> list[list[Any]]:
    rows = []
    for t in POLICY_TASKS:
        delay = t.retry_delay_seconds
        cache = "input-hash key, persisted" if t.cache_key_fn else "none"
        rows.append([f"`{t.name}`", t.retries, delay if t.retries else "-", cache])
    return rows


def run_demo(out: Path) -> str:
    """Run every demo section and write the markdown report to ``out``; returns the text."""
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp_s:
        tmp = Path(tmp_s)
        with isolated_prefect(tmp):
            registered = register_deployments()
            plans = {name: plan(name) for name in FLOWS}

            counter = tmp / "flaky.count"
            t_retry = time.perf_counter()
            with _capture_logs() as cap:
                state = evaluate_flow(
                    steps={"flaky_eval": flaky_command(counter, 2)}, return_state=True
                )
            retry_s = time.perf_counter() - t_retry
            retry_result: list[StepResult] = state.result()
            retry_states = task_states(state.state_details.flow_run_id)
            attempts = int(counter.read_text())
            retry_logs = [x for x in cap.lines if x[2] and not x[2].startswith("View at")]

            with _webhook_receiver() as (url, received), _env(**{WEBHOOK_ENV: url}):
                broken = evaluate_flow(
                    steps={"broken_eval": flaky_command(tmp / "broken.count", 99)},
                    return_state=True,
                )
            broken_states = task_states(broken.state_details.flow_run_id)
            alerts = read_alerts(tmp / "flow_alerts.jsonl")

            data = tmp / "judge_input.jsonl"
            data.write_text('{"prompt": "hi"}\n', encoding="utf-8")
            execs = tmp / "llm.count"
            cmd = counter_command(execs)
            first = llm_cache_demo(cmd, [str(data)])
            n_first = int(execs.read_text())
            second = llm_cache_demo(cmd, [str(data)])
            n_second = int(execs.read_text())
            data.write_text('{"prompt": "hi there"}\n', encoding="utf-8")
            third = llm_cache_demo(cmd, [str(data)])
            n_third = int(execs.read_text())
    total = time.perf_counter() - t0

    lines = [
        "# Prefect flows and reliability (rows 11.6.1, 11.6.2)",
        "",
        "Generated by `python -m terser.flows demo --out reports/p11_flows.md` on "
        f"{datetime.now(UTC):%Y-%m-%d %H:%M} UTC with Prefect {prefect.__version__}, in "
        f"{total:.0f} s. Everything ran against an isolated temporary Prefect API and database "
        "(`prefect_test_harness`), so no server is left running and `~/.prefect` is untouched. "
        "Retry delays are the production values. Code: `src/terser/flows/`; tests: "
        "`tests/test_flows.py`; how to run: `docs/orchestration.md`.",
        "",
        "## Registered flows (11.6.1)",
        "",
        f"`register_deployments()` applied {len(registered)} deployments; the Prefect API lists:",
        "",
        *_md_table(
            ["deployment", "entrypoint", "schedule", "tags"],
            [
                [f"`{r['deployment']}`", f"`{r['entrypoint']}`", r["schedule"], r["tags"]]
                for r in registered
            ],
        ),
        "",
        "## Task plans (dry run)",
        "",
        "Every flow run end to end through Prefect with `dry_run=True`: each task records the "
        "command it would run. The GPU-queue submission still validates the job against the "
        "whole queue (in a scratch copy) and the promotion guard is evaluated for real on the "
        "committed validation report, since both are cheap and read-only.",
        "",
    ]
    for name, results in plans.items():
        lines += [f"### `{name}` (`{module_path(name)}`)", ""]
        rows = []
        for i, r in enumerate(results, 1):
            note = "planned" if not r.executed else ("ok" if r.ok else "**refused**")
            if r.executed and r.output:
                note += f": {_short(r.output.removeprefix('refused: '), 160)}"
            rows.append([i, f"`{r.name}`", r.kind, f"`{r.command}`", note])
        lines += _md_table(["#", "step", "kind", "command", "result"], rows)
        lines.append("")

    final = retry_result[0]
    lines += [
        "## Injected failure recovers (11.6.2)",
        "",
        "`evaluate_flow(steps={'flaky_eval': flaky_command(counter, 2)})`: the step exits 1 on "
        "its first two runs (a server that is still loading) and succeeds on the third.",
        "",
        f"- flow run final state: **{state.name}** after {retry_s:.1f} s",
        f"- attempts (executions of the step): **{attempts}** (1 + {steps.shell_step.retries}"
        " retries allowed)",
        f"- step output: `{final.output}`",
        f"- task state history (from the API): {' -> '.join(retry_states.get('flaky_eval', []))}",
        "",
        "Prefect log lines of the run:",
        "",
        "```text",
        *(f"{lvl:8} {run or '-':14} {_middle(msg)}" for lvl, run, msg in retry_logs),
        "```",
        "",
        "## Permanent failure notifies",
        "",
        "`evaluate_flow(steps={'broken_eval': flaky_command(counter, 99)})` with "
        "`TERSER_ALERT_WEBHOOK` pointing at a receiver on 127.0.0.1.",
        "",
        f"- flow run final state: **{broken.name}**",
        f"- task state history: {' -> '.join(broken_states.get('broken_eval', []))}",
        f"- alert log records: **{len(alerts)}**; webhook POSTs received: **{len(received)}**",
        "",
        *_md_table(
            ["kind", "name", "run", "attempts", "state", "message"],
            [
                [
                    a["kind"],
                    a["name"],
                    a["run_name"],
                    a["attempts"],
                    a["state"],
                    _short(a["message"]),
                ]
                for a in alerts
            ],
        ),
        "",
        "With `TERSER_ALERT_WEBHOOK` unset the hook only writes the log (no network call).",
        "",
        "## Cached LLM step",
        "",
        "`llm_step('judge_demo', counter_command(...), inputs=[judge_input.jsonl])` twice per "
        "flow run; the cache key hashes the step name, command and input content.",
        "",
        *_md_table(
            ["flow run", "input", "task states", "step executions so far"],
            [
                [1, "original", ", ".join(first), n_first],
                [2, "unchanged", ", ".join(second), n_second],
                [3, "changed", ", ".join(third), n_third],
            ],
        ),
        "",
        "## Retry and cache policy per task",
        "",
        *_md_table(["task", "retries", "retry delay (s)", "cache"], _policy_rows()),
        "",
        "Every task has `on_failure=[notify_failure]` and every flow "
        "`on_failure=[notify_flow_failure]` (same alert record, log and webhook).",
        "",
    ]
    text = "\n".join(lines)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    return text
