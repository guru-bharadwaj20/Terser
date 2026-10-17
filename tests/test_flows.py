"""Prefect flows (row 11.6.1), their reliability (row 11.6.2) and continuous training (row 11.4.3).

Runs against an isolated temporary Prefect API (``prefect_test_harness``); no GPU, data or LLM
server. Covers: (a) every flow's dry-run command plan and the eight registered deployments (with
the CT drift-alert trigger and the nightly canary), (b) an injected transient failure recovering
through the retry policy, (c) the failure hook (alert log, optional webhook) on a permanent
failure, (d) the LLM-step cache, (e) ``ct_flow``: no trigger, the drift-alert and new-batch
trigger paths, and Staging registration only when the promotion guard allows it.
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
from terser.flows.registry import (  # noqa: E402
    build_deployments,
    register_deployments,
    registered_triggers,
)
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
    "ct_flow",
    "canary_flow",
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


def test_flows_registered() -> None:
    assert list(fl.FLOWS) == NAMES
    deps = build_deployments()
    assert [d.name for d in deps] == NAMES
    rows = register_deployments()
    assert [r["deployment"] for r in rows] == [f"{n}/{n}" for n in NAMES]
    by_name = {r["deployment"].split("/")[0]: r for r in rows}
    assert by_name["drift_flow"]["schedule"] == "0 3 * * *"
    assert "gpu_queue" in by_name["train_rewriter_flow"]["tags"]
    assert by_name["data_flow"]["entrypoint"] == "terser.flows.flows.data_flow"
    assert by_name["ct_flow"]["schedule"] == "0 4 * * *"
    assert by_name["ct_flow"]["entrypoint"] == "terser.flows.flows.ct_flow"
    # Row 11.5.4: the quality canary runs nightly.
    assert by_name["canary_flow"]["schedule"] == "0 2 * * *"
    assert by_name["canary_flow"]["entrypoint"] == "terser.flows.flows.canary_flow"
    assert "row-11.5.4" in by_name["canary_flow"]["tags"]
    # The drift alert (a failed drift_flow run) starts CT through a Prefect automation.
    assert register_deployments() == rows  # re-applying is idempotent
    triggers = registered_triggers()
    assert len(triggers) == 1
    t = triggers[0]
    assert (t["automation"], t["deployment"]) == ("ct-on-drift-alert", "ct_flow/ct_flow")
    assert t["expect"] == "prefect.flow-run.Failed"
    assert "prefect.resource.name=drift_flow" in t["match_related"]
    assert json.loads(t["parameters"]) == {"trigger": "drift"}


_CLASSICAL = [
    ("unprotect_outputs", "shell"),
    ("pair_features", "shell"),
    ("fast_path_onnx", "shell"),
]
_DATA = [
    ("dvc:casualise", "llm"),
    ("dvc:dedupe", "shell"),
    ("dvc:split", "shell"),
    ("dvc:label_subset", "shell"),
    ("dvc:candidates", "llm"),
    ("dvc:token_counts", "shell"),
    ("respond:qwen2.5-7b", "llm"),
    ("respond:phi3.5-mini", "llm"),
    ("unprotect:labels", "shell"),
    ("judge", "llm"),
]
_REWRITER = [
    ("cpu_smoke_test", "shell"),
    ("gpu_queue_submit", "gpu_queue"),
    ("gpu_worker", "handoff"),
]
_EVALUATE = [("gate_checkpoint", "shell"), ("rewriter_val_report", "shell")]
_PROMOTE = [("register", "shell"), ("promotion_guard", "python")]


def _sub(prefix: str, steps_: list[tuple[str, str]]) -> list[tuple[str, str]]:
    return [(f"{prefix}/{n}", k) for n, k in steps_]


EXPECTED_PLANS: dict[str, list[tuple[str, str]]] = {
    "data_flow": _DATA,
    "train_classical_flow": _CLASSICAL,
    "train_rewriter_flow": _REWRITER,
    "evaluate_flow": _EVALUATE,
    "promote_flow": _PROMOTE,
    "drift_flow": [("tokenizer_drift", "python"), ("input_drift", "python")],
    "ct_flow": [
        ("tokenizer_drift", "python"),
        ("input_drift", "python"),
        ("new_data", "trigger"),
        ("ct_trigger", "trigger"),
        *_sub("data_flow", _DATA),
        *_sub("train_classical_flow", _CLASSICAL),
        *_sub("train_rewriter_flow", _REWRITER),
        *_sub("evaluate_flow", _EVALUATE),
        *_sub("promote_flow", _PROMOTE),
        ("record_ct_cycle", "python"),
    ],
    "canary_flow": [
        ("canary:rewrite", "shell"),
        ("canary:respond:qwen2.5-7b", "shell"),
        ("canary:respond:phi3.5-mini", "shell"),
        ("canary:judge", "shell"),
        ("canary:finalize", "shell"),
    ],
}
_GUARD = {"promote_flow": "promotion_guard", "ct_flow": "promote_flow/promotion_guard"}


@pytest.mark.parametrize("name", NAMES)
def test_every_flow_plans_without_running(name: str, project_root: Path) -> None:
    queue_before = sorted(p.name for p in (project_root / "gpu_queue").iterdir())
    state_before = (project_root / fl.CT_STATE).exists()
    results = fl.plan(name)
    assert [(r.name, r.kind) for r in results] == EXPECTED_PLANS[name]
    # Only the read-only promotion guard evaluates in a dry run.
    assert [r.name for r in results if r.executed] == ([_GUARD[name]] if name in _GUARD else [])
    assert sorted(p.name for p in (project_root / "gpu_queue").iterdir()) == queue_before
    assert (project_root / fl.CT_STATE).exists() == state_before
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
    if name == "ct_flow":
        by = {r.name: r for r in results}
        assert cmds["new_data"][:3] == ("dvc", "status", "--json")
        assert by["ct_trigger"].output.startswith("dry run")
        # The rewriter retrain is validated against the queue but only queued (never run).
        submit = by["train_rewriter_flow/gpu_queue_submit"]
        assert not submit.executed and "ct-sft-qwen3b-lora-" in submit.output
        assert by["train_rewriter_flow/gpu_worker"].kind == "handoff"
        assert not by["promote_flow/promotion_guard"].ok  # Staging refused: no rung passes C8.3
    if name == "drift_flow":
        assert cmds["tokenizer_drift"] == ("python", "-m", "terser.monitoring.tokenizer_drift")
        assert cmds["input_drift"][:4] == (
            "python",
            "-m",
            "terser.monitoring.input_drift",
            "--name",
        )
    if name == "canary_flow":
        mod = ("python", "-m", "terser.monitoring.canary")
        assert all(c[:3] == mod for c in cmds.values())
        rewrite = cmds["canary:rewrite"]
        assert rewrite[3] == "rewrite" and "--force" in rewrite
        assert rewrite[rewrite.index("--mode") + 1] == "quality"
        assert rewrite[rewrite.index("--canary") + 1] == "data/canary/canary_v1.jsonl"
        run_dir = rewrite[rewrite.index("--run-dir") + 1]
        assert all(c[c.index("--run-dir") + 1] == run_dir for c in cmds.values())
        final = cmds["canary:finalize"]
        assert final[final.index("--gateway") + 1] == "localhost:9091"


def test_canary_flow_endpoints() -> None:
    results = fl.canary_flow(
        base_url="http://svc:8080",
        mode="fast",
        targets=["qwen2.5-7b"],
        llm_url="http://all:8000/v1",
        target_urls={"qwen2.5-7b": "http://qwen:8000/v1"},
        judge_url="http://judge:8000/v1",
        gateway=None,
        run_id="r1",
        dry_run=True,
    )
    cmds = {r.name: r.cmd for r in results}
    assert list(cmds) == [
        "canary:rewrite",
        "canary:respond:qwen2.5-7b",
        "canary:judge",
        "canary:finalize",
    ]
    assert ("--base-url", "http://svc:8080") == cmds["canary:rewrite"][8:10]
    assert cmds["canary:respond:qwen2.5-7b"][-2:] == ("--llm-url", "http://qwen:8000/v1")
    assert cmds["canary:judge"][-2:] == ("--judge-url", "http://judge:8000/v1")
    assert "--gateway" not in cmds["canary:finalize"]
    assert cmds["canary:finalize"][4:6] == ("--run-dir", "logs/canary/r1")


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
    monkeypatch.setattr(steps, "lineage_missing", lambda name, version: [])
    kw: dict[str, Any] = {"report": str(report), "sync_registry": False}
    refused = fl.promote_flow(rung="bad", **kw)
    assert [r.name for r in refused] == ["promotion_guard"] and not refused[0].ok
    assert promoted == []
    allowed = fl.promote_flow(rung="good", champion_rung="bad", **kw)
    assert [r.name for r in allowed] == ["promotion_guard", "registry_promote"]
    assert promoted == [("terser-rewriter", "7", "Production")]


def test_promote_flow_forces_register_on_empty_registry(
    monkeypatch: pytest.MonkeyPatch, alert_log: Path
) -> None:
    """Fresh clone: ``mlruns/`` is gitignored, so the up-to-date ``register`` stage is forced."""
    calls = _fake_shell(monkeypatch)
    for known, expected in ((False, ["--force"]), (True, [])):
        monkeypatch.setattr(fl, "registry_has", lambda name, known=known: known)
        calls.clear()
        fl.promote_flow(rung="sft3b")
        assert calls == [["dvc", "repro", "--single-item", *expected, "register"]]


# (e) continuous training (row 11.4.3) -----------------------------------------------------------


def _val_report(tmp_path: Path) -> Path:
    cats = {c: 0.95 for c in ("code", "fact")}
    rung = {"median_savings": 0.3, "preserved_by_category": cats}
    report = tmp_path / "val.json"
    good = {**rung, "preserved": 0.95, "c8_3": True}
    bad = {**rung, "preserved": 0.80, "c8_3": False}
    report.write_text(json.dumps({"good": good, "bad": bad}))
    return report


class _CT:
    """``ct_flow`` with every heavy step faked: shell, LLM, input drift, dvc status, registry."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import terser.monitoring.tokenizer_drift as td
        import terser.tracking

        self.shell = _fake_shell(monkeypatch)
        self.llm: list[str] = []
        self.procs: list[list[str]] = []
        self.promoted: list[tuple[str, Any, str]] = []
        self.aliases: list[str] = []
        self.drift_alert = False
        self.dvc_status: dict[str, Any] = {}
        self.queue = tmp_path / "queue"
        self.queue.mkdir()
        self.state = tmp_path / "ct_state.json"
        self.report = _val_report(tmp_path)

        def fake_llm(name: str, cmd: list[str], inputs: Any = (), dry_run: bool = False) -> Any:
            self.llm.append(name)
            return StepResult(name, "llm", tuple(cmd), executed=True, output="cached")

        def fake_proc(cmd: list[str], cwd: Path = steps.REPO_ROOT) -> Any:
            self.procs.append(list(cmd))
            if cmd[:2] == ["dvc", "status"]:
                return subprocess.CompletedProcess(cmd, 0, json.dumps(self.dvc_status), "")
            verdict = {"alert": self.drift_alert, "reasons": ["PSI category = 0.31 > 0.2"]}
            if not self.drift_alert:
                verdict["reasons"] = []
            return subprocess.CompletedProcess(cmd, int(self.drift_alert), json.dumps(verdict), "")

        def version_at(name: str, alias: str) -> str:
            self.aliases.append(alias)
            return "9"

        monkeypatch.setattr(fl, "llm_step", fake_llm)
        monkeypatch.setattr(steps, "_run_process", fake_proc)
        monkeypatch.setattr(steps, "_counter", lambda t: len)
        strings, _ = td.load_fixture()
        snapshot = {"gpt-4o": [len(s) for s in strings]}
        monkeypatch.setattr(td, "load_fixture", lambda: (strings, snapshot))
        monkeypatch.setattr(
            terser.tracking, "promote", lambda n, v, s: self.promoted.append((n, v, s))
        )
        monkeypatch.setattr(steps, "_version_at", version_at)
        monkeypatch.setattr(steps, "lineage_missing", lambda name, version: self.missing)
        monkeypatch.setattr(steps, "rung_at", lambda name, alias: None)
        self.missing: list[str] = []

    def run(self, **params: Any) -> Any:
        kw: dict[str, Any] = {
            "drift_targets": ["gpt-4o"],
            "drift_name": "2026-10-17",
            "queue_dir": str(self.queue),
            "state_path": str(self.state),
            "report": str(self.report),
            "rung": "good",
            "sync_registry": False,
            "job_id": "ct-test",
            **params,
        }
        return fl.ct_flow(**kw, return_state=True)

    def work(self) -> list[list[str]]:
        """Shell commands of the cycle, without the (environment-dependent) unprotect steps."""
        return [c for c in self.shell if c[:2] != ["dvc", "unprotect"]]


def test_ct_flow_no_trigger_does_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, alert_log: Path
) -> None:
    ct = _CT(tmp_path, monkeypatch)
    state = ct.run()
    assert state.is_completed()
    results: list[StepResult] = state.result()
    assert [(r.name, r.ok) for r in results] == [
        ("tokenizer_drift", True),
        ("input_drift", True),
        ("new_data", True),
        ("ct_trigger", True),
    ]
    assert "nothing to do" in results[-1].output
    assert ct.shell == [] and ct.llm == [] and ct.promoted == []
    assert list(ct.queue.iterdir()) == [] and not ct.state.exists()
    assert read_alerts(alert_log) == []


def test_ct_flow_drift_alert_runs_full_cycle_and_registers_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, alert_log: Path
) -> None:
    ct = _CT(tmp_path, monkeypatch)
    ct.drift_alert = True
    state = ct.run()
    assert state.is_completed()
    results: list[StepResult] = state.result()
    by = {r.name: r for r in results}
    assert not by["input_drift"].ok and "INPUT DRIFT" in by["input_drift"].output
    assert "INPUT DRIFT PSI category" in by["ct_trigger"].output
    # judge labels -> classical retrain (CPU) -> rewriter smoke test -> evaluate, in that order.
    assert ct.llm == [
        "dvc:casualise",
        "dvc:candidates",
        "respond:qwen2.5-7b",
        "respond:phi3.5-mini",
        "judge",
    ]
    work = ct.work()
    stages = [c[-1] for c in work if c[:2] == ["dvc", "repro"]]
    assert stages == ["dedupe", "split", "label_subset", "token_counts"]
    modules = [c[2] for c in work if c[0] == "python"]
    assert modules == [
        "terser.phase3.build_features",
        "terser.phase11.train_fast_path",
        "pytest",
        "terser.phase3.gate_checkpoint",
        "terser.models.rewriter.val_report",
    ]
    # The GPU retrain is queued, never run here.
    job = ct.queue / "ct-test.yaml"
    assert job.exists() and "experiment=sft_qwen3b_lora" in job.read_text()
    assert not by["train_rewriter_flow/gpu_worker"].executed
    # Guard passes on "good" -> the newest version is registered as Staging.
    assert by["promote_flow/promotion_guard"].ok
    assert ct.aliases == ["latest"]
    assert ct.promoted == [("terser-rewriter", "9", "Staging")]
    record = json.loads(ct.state.read_text())
    assert record["trigger"] == "check" and record["staging"] == "registered"
    assert record["rewriter_job"].endswith("ct-test.yaml")
    assert read_alerts(alert_log) == []  # an alert that starts CT is a result, not a failure


def test_ct_flow_guard_refuses_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, alert_log: Path
) -> None:
    """Started by the drift-alert trigger; a candidate failing C8.3 is not registered."""
    ct = _CT(tmp_path, monkeypatch)
    state = ct.run(trigger="drift", rung="bad", label=False)
    assert state.is_completed()  # a refusal is a decision, not a failed run
    results: list[StepResult] = state.result()
    names = [r.name for r in results]
    assert names[0] == "ct_trigger" and "drift alert" in results[0].output
    assert not any(n.startswith(("tokenizer_drift", "input_drift", "new_data")) for n in names)
    assert "promote_flow/registry_promote" not in names
    assert ct.promoted == [] and ct.aliases == []
    assert ct.llm == ["dvc:casualise", "dvc:candidates"]  # label=False: no respond / judge
    record = json.loads(ct.state.read_text())
    assert record["trigger"] == "drift" and record["staging"].startswith("refused")
    assert "checkpoints not passed: ['C8.3']" in record["staging"]


def test_ct_flow_new_batch_triggers_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, alert_log: Path
) -> None:
    ct = _CT(tmp_path, monkeypatch)
    batch = tmp_path / "batch_2026-10-17.jsonl"
    batch.write_text('{"id": "p1"}\n')
    kw: dict[str, Any] = {"check_drift": False, "batch": str(batch), "label": False}
    first: list[StepResult] = ct.run(**kw).result()
    assert first[0].name == "new_data" and not first[0].ok
    assert first[0].output.startswith(f"NEW DATA new batch {batch}")
    assert ct.promoted == [("terser-rewriter", "9", "Staging")]
    assert (
        json.loads(ct.state.read_text())["batches"][str(batch)]
        == steps.input_digest([str(batch)])[str(batch)]
    )
    again: list[StepResult] = ct.run(**kw).result()
    assert [r.name for r in again] == ["new_data", "ct_trigger"]  # same batch: nothing to do
    assert "nothing to do" in again[-1].output
    batch.write_text('{"id": "p1"}\n{"id": "p2"}\n')
    assert ct.run(**kw).result()[1].name == "ct_trigger"  # changed batch: a new cycle
    # A DVC-tracked data change (e.g. after `dvc pull` of a new raw batch) also triggers.
    ct.dvc_status = {"split": [{"changed deps": {"data/interim/casual_dedup.jsonl": "modified"}}]}
    ct.shell.clear()
    dvc: list[StepResult] = ct.run(check_drift=False, label=False).result()
    assert dvc[0].output == "NEW DATA DVC data stages changed: split"
    assert ["dvc", "status", "--json", *fl.DATA_STAGES] in ct.procs
    assert ["dvc", "repro", "--single-item", "split"] in ct.shell


def test_ct_flow_rejects_unknown_trigger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, alert_log: Path
) -> None:
    ct = _CT(tmp_path, monkeypatch)
    state = ct.run(trigger="sometimes")
    assert state.is_failed() and "trigger must be one of" in str(state.message)
    assert ct.shell == [] and [a["name"] for a in read_alerts(alert_log)] == ["ct_flow"]


def test_ct_flow_refuses_a_version_without_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, alert_log: Path
) -> None:
    """Rule 7 runs on the exact version that would move; a missing lineage tag refuses it."""
    ct = _CT(tmp_path, monkeypatch)
    ct.missing = ["git_sha"]
    results: list[StepResult] = ct.run(trigger="force", label=False).result()
    by = {r.name: r for r in results}
    assert not by["promote_flow/promotion_guard"].ok
    assert "lineage incomplete" in by["promote_flow/promotion_guard"].output
    assert ct.promoted == []


def test_ct_flow_promotes_the_version_of_the_guarded_rung(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, alert_log: Path
) -> None:
    ct = _CT(tmp_path, monkeypatch)
    ct.report.write_text(ct.report.read_text().replace('"good"', '"sft3b"'))
    ct.run(trigger="force", label=False, rung="sft3b").result()
    assert ct.aliases == ["experiment:sft_qwen3b_lora"]


def test_cli_prints_fired_trigger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    ct = _CT(tmp_path, monkeypatch)
    ct.dvc_status = {"dedupe": []}
    args = ["run", "ct_flow", "--param", "check_drift=false", "--param", "label=false"]
    args += ["--param", f"queue_dir={json.dumps(str(ct.queue))}"]
    args += ["--param", f"state_path={json.dumps(str(ct.state))}"]
    args += ["--param", f"report={json.dumps(str(ct.report))}", "--param", 'rung="good"']
    args += ["--param", "sync_registry=false", "--param", 'job_id="ct-cli"']
    assert cli(args) == 0
    out = capsys.readouterr().out
    assert "[trigger] new_data: dvc status --json" in out and "(FIRED)" in out
    assert "promote_flow/registry_promote" in out
