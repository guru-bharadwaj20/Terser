"""Prefect flows (row 11.6.1) and their reliability (row 11.6.2).

Runs against an isolated temporary Prefect API (``prefect_test_harness``); no GPU, data or LLM
server. Covers: (a) every flow's dry-run command plan and the six registered deployments, (b) an
injected transient failure recovering through the retry policy, (c) the failure hook (alert log,
optional webhook) on a permanent failure, (d) the LLM-step cache.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("prefect")

from terser.flows import flows as fl  # noqa: E402
from terser.flows import steps  # noqa: E402
from terser.flows.__main__ import main as cli  # noqa: E402
from terser.flows.demo import isolated_prefect, llm_cache_demo, task_states  # noqa: E402
from terser.flows.registry import build_deployments, register_deployments  # noqa: E402
from terser.flows.reliability import (  # noqa: E402
    ALERT_LOG_ENV,
    WEBHOOK_ENV,
    counter_command,
    flaky_command,
    read_alerts,
)
from terser.flows.steps import StepResult, llm_cache_key  # noqa: E402

NAMES = [
    "data_flow",
    "train_classical_flow",
    "train_rewriter_flow",
    "evaluate_flow",
    "promote_flow",
    "drift_flow",
]


@pytest.fixture(scope="module", autouse=True)
def _prefect_api(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    with isolated_prefect(tmp_path_factory.mktemp("prefect")):
        yield


@pytest.fixture
def alert_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "alerts.jsonl"
    monkeypatch.setenv(ALERT_LOG_ENV, str(path))
    monkeypatch.delenv(WEBHOOK_ENV, raising=False)
    return path


@pytest.fixture
def fast_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    """The production shell step (same retries and hooks) without the 10 s retry delay."""
    monkeypatch.setattr(fl, "shell_step", steps.shell_step.with_options(retry_delay_seconds=0))


def _count(path: Path) -> int:
    return int(path.read_text()) if path.exists() else 0


# (a) registration and plans ----------------------------------------------------------------


def test_six_flows_registered() -> None:
    assert list(fl.FLOWS) == NAMES
    deps = build_deployments()
    assert [d.name for d in deps] == NAMES
    rows = register_deployments()
    assert [r["deployment"] for r in rows] == [f"{n}/{n}" for n in NAMES]
    by_name = {r["deployment"].split("/")[0]: r for r in rows}
    assert by_name["drift_flow"]["schedule"] == "0 3 * * *"
    assert "gpu_queue" in by_name["train_rewriter_flow"]["tags"]
    assert by_name["data_flow"]["entrypoint"] == "terser.flows.flows.data_flow"


EXPECTED_PLANS: dict[str, list[tuple[str, str]]] = {
    "data_flow": [
        ("dvc:casualise", "llm"),
        ("dvc:dedupe", "shell"),
        ("dvc:split", "shell"),
        ("dvc:label_subset", "shell"),
        ("dvc:candidates", "llm"),
        ("dvc:token_counts", "shell"),
        ("respond:qwen2.5-7b", "llm"),
        ("respond:phi3.5-mini", "llm"),
        ("judge", "llm"),
    ],
    "train_classical_flow": [("pair_features", "shell"), ("fast_path_onnx", "shell")],
    "train_rewriter_flow": [
        ("cpu_smoke_test", "shell"),
        ("gpu_queue_submit", "gpu_queue"),
        ("gpu_worker", "handoff"),
    ],
    "evaluate_flow": [("gate_checkpoint", "shell"), ("rewriter_val_report", "shell")],
    "promote_flow": [("register", "shell"), ("promotion_guard", "python")],
    "drift_flow": [("tokenizer_drift", "python"), ("input_drift", "python")],
}


@pytest.mark.parametrize("name", NAMES)
def test_every_flow_plans_without_running(name: str, project_root: Path) -> None:
    queue_before = sorted(p.name for p in (project_root / "gpu_queue").iterdir())
    results = fl.plan(name)
    assert [(r.name, r.kind) for r in results] == EXPECTED_PLANS[name]
    # Only the read-only promotion guard evaluates in a dry run.
    assert [r.name for r in results if r.executed] == (
        ["promotion_guard"] if name == "promote_flow" else []
    )
    assert sorted(p.name for p in (project_root / "gpu_queue").iterdir()) == queue_before
    cmds = {r.name: r.cmd for r in results}
    if name == "data_flow":
        assert cmds["dvc:split"] == ("dvc", "repro", "--single-item", "split")
        assert "dvc:collect" not in cmds  # stale by design, never rebuilt implicitly
        assert cmds["judge"][:3] == ("python", "-m", "terser.phase2.label_stage")
    if name == "train_rewriter_flow":
        assert cmds["gpu_queue_submit"][:3] == ("python", "-m", "terser.gpu_queue")
    if name == "promote_flow":
        guard = results[1]
        assert not guard.ok and guard.output.startswith("refused")  # no rung passes C8.3
    if name == "drift_flow":
        assert cmds["tokenizer_drift"] == ("python", "-m", "terser.monitoring.tokenizer_drift")
        assert cmds["input_drift"][:4] == (
            "python",
            "-m",
            "terser.monitoring.input_drift",
            "--name",
        )


def test_cli_list_and_dry_run(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli(["list"]) == 0
    out = capsys.readouterr().out
    assert all(f"{n}  [terser.flows.flows:{n}]" in out for n in NAMES)
    assert cli(["run", "evaluate_flow", "--dry-run"]) == 0
    assert "python -m terser.phase3.gate_checkpoint  (planned)" in capsys.readouterr().out


def test_pipelines_alias() -> None:
    from terser.pipelines import flows as alias

    assert alias.FLOWS is fl.FLOWS


# (b) retries --------------------------------------------------------------------------------


def test_retry_policy() -> None:
    assert (steps.shell_step.retries, steps.shell_step.retry_delay_seconds) == (2, 10.0)
    assert steps.llm_step.retries == 3
    assert steps.submit_gpu_job.retries == 0 and steps.promotion_guard.retries == 0


@pytest.mark.usefixtures("fast_retries")
def test_injected_transient_failure_recovers(tmp_path: Path, alert_log: Path) -> None:
    counter = tmp_path / "flaky.count"
    state = fl.evaluate_flow(steps={"flaky": flaky_command(counter, 2)}, return_state=True)
    assert state.is_completed()
    assert _count(counter) == 3  # two injected failures, recovered on the second retry
    result: list[StepResult] = state.result()
    assert result[0].executed and result[0].output == "attempt 3: ok"
    history = task_states(state.state_details.flow_run_id)["flaky"]
    # With a 0 s delay Prefect goes straight to Retrying (AwaitingRetry needs a real delay).
    assert history.count("Retrying") == 2 and history[-1] == "Completed"
    assert read_alerts(alert_log) == []  # a recovered failure is not an alert


# (c) failure notifications ---------------------------------------------------------------------


@pytest.mark.usefixtures("fast_retries")
def test_failure_hook_fires_on_permanent_failure(
    tmp_path: Path, alert_log: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import httpx

    posts: list[Any] = []
    monkeypatch.setattr(httpx, "post", lambda *a, **k: posts.append((a, k)))
    counter = tmp_path / "broken.count"
    state = fl.evaluate_flow(steps={"broken": flaky_command(counter, 99)}, return_state=True)
    assert state.is_failed()
    assert _count(counter) == 3  # 1 attempt + 2 retries, then give up
    alerts = read_alerts(alert_log)
    assert [(a["kind"], a["name"]) for a in alerts] == [
        ("task", "shell"),
        ("flow", "evaluate_flow"),
    ]
    assert alerts[0]["run_name"] == "broken" and alerts[0]["attempts"] == 3
    assert "injected failure 3/99" in alerts[0]["message"]
    assert posts == []  # TERSER_ALERT_WEBHOOK unset: no network call


def test_drift_alert_fails_flow_and_posts_webhook(
    alert_log: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import httpx

    import terser.tokenizers

    class Resp:
        def raise_for_status(self) -> None:
            return None

    posts: list[tuple[str, dict[str, Any]]] = []

    def fake_post(url: str, json: dict[str, Any], timeout: float) -> Resp:
        posts.append((url, json))
        return Resp()

    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setenv(WEBHOOK_ENV, "http://alerts.example.invalid/hook")
    monkeypatch.setattr(terser.tokenizers, "count_tokens", lambda s, t: len(s) + 1000)
    state = fl.drift_flow(targets=["gpt-4o"], input_drift=False, return_state=True)
    assert state.is_failed() and "TOKENIZER DRIFT" in str(state.message)
    assert [p[1]["kind"] for p in posts] == ["flow"]
    assert posts[0][0] == "http://alerts.example.invalid/hook"
    assert read_alerts(alert_log)[0]["name"] == "drift_flow"


def _fake_input_drift(
    monkeypatch: pytest.MonkeyPatch, returncode: int, stdout: str
) -> list[list[str]]:
    """Replace the input-drift subprocess (bge-small embeddings) with a canned CLI result."""
    calls: list[list[str]] = []

    def fake(cmd: list[str], cwd: Path = steps.REPO_ROOT) -> subprocess.CompletedProcess[str]:
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, returncode, stdout, "")

    monkeypatch.setattr(steps, "_run_process", fake)
    monkeypatch.setattr(steps, "_counter", lambda t: len)  # tokenizer drift: no network
    import terser.monitoring.tokenizer_drift as td

    strings, snapshot = td.load_fixture()
    monkeypatch.setattr(
        td, "load_fixture", lambda: (strings, {"gpt-4o": [len(s) for s in strings]})
    )
    return calls


def test_drift_flow_runs_input_drift(alert_log: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ok = 'no input drift\n{"alert": false, "reasons": [], "summary_json": "r/x.json"}\n'
    calls = _fake_input_drift(monkeypatch, 0, ok)
    results = fl.drift_flow(current="logs/requests.jsonl", name="2026-10-17")
    assert calls == [
        [
            "python",
            "-m",
            "terser.monitoring.input_drift",
            "--name",
            "2026-10-17",
            "--current",
            "logs/requests.jsonl",
        ]
    ]
    assert [(r.name, r.ok) for r in results] == [("tokenizer_drift", True), ("input_drift", True)]
    assert "summary_json=r/x.json" in results[1].output
    assert read_alerts(alert_log) == []


def test_drift_flow_input_drift_alert_fails_run(
    alert_log: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verdict = {"alert": True, "reasons": ["PSI category = 1.234 > 0.2"], "summary_json": "x"}
    calls = _fake_input_drift(monkeypatch, 1, "INPUT DRIFT ...\n" + json.dumps(verdict) + "\n")
    state = fl.drift_flow(return_state=True)
    assert state.is_failed() and "INPUT DRIFT PSI category = 1.234 > 0.2" in str(state.message)
    assert len(calls) == 1  # an alert is a result, not a retried failure
    assert calls[0][4] == datetime.now(UTC).strftime("%Y-%m-%d")  # default name: today
    assert [(a["kind"], a["name"]) for a in read_alerts(alert_log)] == [("flow", "drift_flow")]


def test_input_drift_crash_is_a_step_failure(
    alert_log: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(steps.input_drift_check, "retry_delay_seconds", 0)
    calls = _fake_input_drift(monkeypatch, 2, "Traceback ...\n")
    state = fl.drift_flow(name="d", return_state=True)
    assert state.is_failed() and "without a verdict" in str(state.message)
    assert len(calls) == 2  # crashed once, retried once
    assert [a["kind"] for a in read_alerts(alert_log)] == ["task", "flow"]


# (d) cached LLM steps ---------------------------------------------------------------------------


def test_llm_step_cached_on_input_hash(tmp_path: Path) -> None:
    data = tmp_path / "pairs.jsonl"
    data.write_text('{"a": 1}\n')
    execs = tmp_path / "llm.count"
    cmd = counter_command(execs)
    assert llm_cache_demo(cmd, [str(data)]) == ["Completed", "Cached"]
    assert _count(execs) == 1
    assert llm_cache_demo(cmd, [str(data)]) == ["Cached", "Cached"]  # new flow run, same inputs
    assert _count(execs) == 1
    data.write_text('{"a": 2}\n')
    assert llm_cache_demo(cmd, [str(data)]) == ["Completed", "Cached"]
    assert _count(execs) == 2


def test_llm_cache_key(tmp_path: Path) -> None:
    f = tmp_path / "in.jsonl"
    f.write_text("x")
    params: dict[str, Any] = {"name": "judge", "cmd": ["python", "-m", "m"], "inputs": [str(f)]}
    real = llm_cache_key(None, {**params, "dry_run": False})
    assert real == llm_cache_key(None, {**params, "dry_run": False})
    assert real != llm_cache_key(None, {**params, "dry_run": True})
    f.write_text("y")
    assert real != llm_cache_key(None, {**params, "dry_run": False})


# flow behaviour (real mode, heavy steps faked) ---------------------------------------------------


def _fake_shell(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    calls: list[list[str]] = []

    def fake(name: str, cmd: list[str], dry_run: bool = False) -> StepResult:
        calls.append(list(cmd))
        return StepResult(name, "shell", tuple(cmd), executed=True, output="done")

    monkeypatch.setattr(fl, "shell_step", fake)
    return calls


def test_train_rewriter_flow_submits_valid_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, alert_log: Path
) -> None:
    calls = _fake_shell(monkeypatch)
    params = {"experiment": "sft_tiny", "job_id": "sft-test", "queue_dir": str(tmp_path)}
    results = fl.train_rewriter_flow(**params)
    assert calls == [["python", "-m", "pytest", "-q", "tests/rewriter/test_train_tiny.py"]]
    assert (tmp_path / "sft-test.yaml").exists() and results[1].executed
    assert "sft-test" in results[1].output  # queue order table
    assert fl.train_rewriter_flow(**params)[1].executed  # identical resubmission: no-op
    too_big = fl.train_rewriter_flow(
        "sft_tiny", "too-big", est_vram_gb=999, queue_dir=str(tmp_path), return_state=True
    )
    assert too_big.is_failed() and "exceeds max_vram_gb" in str(too_big.message)
    assert not (tmp_path / "too-big.yaml").exists()
    changed = fl.train_rewriter_flow(**params, est_hours=99, return_state=True)
    assert changed.is_failed() and "different spec" in str(changed.message)


def test_promote_flow_guards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, alert_log: Path
) -> None:
    import terser.tracking

    cats = {c: 0.95 for c in ("code", "fact")}
    rung = {"median_savings": 0.3, "preserved_by_category": cats}
    report = tmp_path / "val.json"
    report.write_text(
        json.dumps(
            {
                "good": {**rung, "preserved": 0.95, "c8_3": True},
                "bad": {**rung, "preserved": 0.80, "c8_3": False},
            }
        )
    )
    promoted: list[tuple[str, Any, str]] = []
    monkeypatch.setattr(terser.tracking, "promote", lambda n, v, s: promoted.append((n, v, s)))
    monkeypatch.setattr(steps, "_version_at", lambda name, alias: "7")
    kw: dict[str, Any] = {"report": str(report), "sync_registry": False}
    refused = fl.promote_flow(rung="bad", **kw)
    assert [r.name for r in refused] == ["promotion_guard"] and not refused[0].ok
    assert promoted == []
    allowed = fl.promote_flow(rung="good", champion_rung="bad", **kw)
    assert [r.name for r in allowed] == ["promotion_guard", "registry_promote"]
    assert promoted == [("terser-rewriter", "7", "Production")]
