"""The small, named Prefect tasks the flows are built from (plan rows **11.6.1**, **11.6.2**).

Every task wraps an entry point that already exists and returns a :class:`StepResult`. With
``dry_run=True`` a task records the command it would run (``executed=False``) instead of running
it, so a whole flow can be exercised as a command plan without data, a GPU or an LLM server.

==========================  ================================  ===========================
task                        wraps                             reliability
==========================  ================================  ===========================
:func:`shell_step`          any CLI (``dvc repro``, modules)  2 retries, 10 s apart
:func:`llm_step`            CLIs that call an LLM             3 retries (10/30/90 s),
                                                              cached on an input hash
:func:`submit_gpu_job`      :mod:`terser.gpu_queue` API       no retry (deterministic)
:func:`drift_check`         tokenizer drift (row 11.5.3)      1 retry (tokenizer download)
:func:`input_drift_check`   input drift CLI (row 11.5.2)      1 retry; an alert is a result
:func:`promotion_guard`     :mod:`terser.pipelines.promotion` no retry (deterministic)
:func:`registry_promote`    :func:`terser.tracking.promote`   2 retries
:func:`new_data_check`      ``dvc status`` + batch digest     1 retry; new data is a result
:func:`record_ct_cycle`     CT state file (row 11.4.3)        no retry
==========================  ================================  ===========================

Every task's ``on_failure`` hook is :func:`terser.flows.reliability.notify_failure`.

LLM-step cache: the key (:func:`llm_cache_key`) is a SHA-256 of the step name, the command and
the content of every input file or directory, so a retried or re-run flow returns the stored
result of an unchanged LLM step without starting it (Prefect state ``Cached``), and any change to
an input re-runs it. Results persist in Prefect's local result storage
(``PREFECT_LOCAL_STORAGE_PATH``, default ``~/.prefect/storage``); set
``PREFECT_TASKS_REFRESH_CACHE=true`` to force a re-run. Inside a step the individual LLM requests
are additionally disk-cached by :class:`terser.llm.ResponseCache`, so even a cache miss never
pays twice for a finished generation.
"""

from __future__ import annotations

import hashlib
import json
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from prefect import task
from prefect.cache_policies import NO_CACHE
from prefect.context import TaskRunContext

from terser.config import project_root
from terser.flows.reliability import (
    LLM_RETRIES,
    LLM_RETRY_DELAYS,
    SHELL_RETRIES,
    SHELL_RETRY_DELAY,
    notify_failure,
)
from terser.pipelines.promotion import EvalSummary, GuardConfig, decide

__all__ = [
    "REPO_ROOT",
    "RUNG_EXPERIMENTS",
    "StepError",
    "StepResult",
    "drift_check",
    "input_digest",
    "input_drift_check",
    "lineage_missing",
    "llm_cache_key",
    "llm_step",
    "load_eval_summary",
    "new_data_check",
    "promotion_guard",
    "read_ct_state",
    "record_ct_cycle",
    "registry_has",
    "registry_promote",
    "run_command",
    "rung_at",
    "shell_step",
    "submit_gpu_job",
]

REPO_ROOT = project_root()
OUTPUT_TAIL = 2000
VAL_REPORT = "reports/p8_rewriter_val.json"


class StepError(RuntimeError):
    """A step failed (non-zero exit, invalid queue spec, drift alert)."""


@dataclass(frozen=True)
class StepResult:
    """What one step ran (or, in a dry run, would run) and how it ended."""

    name: str
    kind: str
    cmd: tuple[str, ...]
    executed: bool
    ok: bool = True
    output: str = ""

    @property
    def command(self) -> str:
        """The command as one shell-quoted string."""
        return shlex.join(self.cmd)


def _resolve(cmd: Sequence[str]) -> list[str]:
    """Run ``python`` / ``dvc`` from the current environment, so a flow never escapes the venv."""
    head, rest = cmd[0], list(cmd[1:])
    if head == "python":
        return [sys.executable, *rest]
    if head == "dvc":
        for name in ("dvc", "dvc.exe"):
            local = Path(sys.executable).with_name(name)
            if local.exists():
                return [str(local), *rest]
    return [head, *rest]


def _short_cmd(cmd: Sequence[str]) -> str:
    return shlex.join(a if len(a) <= 60 else a[:57] + "..." for a in cmd)


def _run_process(cmd: Sequence[str], cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(_resolve(cmd), cwd=cwd, capture_output=True, text=True, check=False)


def run_command(cmd: Sequence[str], cwd: Path = REPO_ROOT) -> str:
    """Run ``cmd`` from the repo root; return the stdout tail or raise :class:`StepError`."""
    proc = _run_process(cmd, cwd)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip()[-OUTPUT_TAIL:]
        raise StepError(f"{_short_cmd(cmd)} exited {proc.returncode}: {tail}")
    return proc.stdout.strip()[-OUTPUT_TAIL:]


def _plan(name: str, kind: str, cmd: Sequence[str]) -> StepResult:
    print(f"[plan] {name}: {shlex.join(cmd)}")
    return StepResult(name, kind, tuple(cmd), executed=False)


@task(
    name="shell",
    task_run_name="{name}",
    retries=SHELL_RETRIES,
    retry_delay_seconds=SHELL_RETRY_DELAY,
    cache_policy=NO_CACHE,
    on_failure=[notify_failure],
)
def shell_step(name: str, cmd: Sequence[str], dry_run: bool = False) -> StepResult:
    """Run one CLI step (``dvc repro``, ``python -m ...``) with retries."""
    if dry_run:
        return _plan(name, "shell", cmd)
    return StepResult(name, "shell", tuple(cmd), executed=True, output=run_command(cmd))


def _digest_path(path: Path) -> str:
    if path.is_file():
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    if path.is_dir():
        h = hashlib.sha256()
        for p in sorted(path.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                h.update(f"{p.relative_to(path).as_posix()}\0{_digest_path(p)}\n".encode())
        return h.hexdigest()
    return "missing"


def input_digest(paths: Sequence[str], root: Path = REPO_ROOT) -> dict[str, str]:
    """``{path: sha256 of its content}`` (directories hash every file; absent paths: missing)."""
    return {p: _digest_path(root / p) for p in sorted(paths)}


def llm_cache_key(context: TaskRunContext | None, parameters: Mapping[str, Any]) -> str:
    """Cache key of an :func:`llm_step`: step name, command, input hashes, absent outputs.

    A dry run keys on the paths only (no hashing of large inputs) and never shares a key with a
    real run. An ``outputs`` path that does not exist changes the key, so a deleted output is
    regenerated instead of the step reporting ``Cached``; once it is back the key is the same as
    before and the cache hits again.
    """
    del context
    dry_run = bool(parameters.get("dry_run", False))
    paths = [str(p) for p in parameters.get("inputs", ())]
    absent = sorted(str(p) for p in parameters.get("outputs", ()) if not (REPO_ROOT / p).exists())
    payload = {
        "name": parameters["name"],
        "cmd": [str(c) for c in parameters["cmd"]],
        "dry_run": dry_run,
        "inputs": sorted(paths) if dry_run else input_digest(paths),
    }
    if absent and not dry_run:
        payload["absent_outputs"] = absent
    blob = json.dumps(payload, sort_keys=True).encode()
    return "terser-llm-" + hashlib.sha256(blob).hexdigest()


@task(
    name="llm",
    task_run_name="{name}",
    retries=LLM_RETRIES,
    retry_delay_seconds=LLM_RETRY_DELAYS,
    cache_key_fn=llm_cache_key,
    persist_result=True,
    on_failure=[notify_failure],
)
def llm_step(
    name: str,
    cmd: Sequence[str],
    inputs: Sequence[str] = (),
    dry_run: bool = False,
    outputs: Sequence[str] = (),
) -> StepResult:
    """Run one LLM-calling CLI step.

    ``inputs`` and ``outputs`` are not read here: they only feed :func:`llm_cache_key`, which
    caches the step on the content of ``inputs`` and re-runs it when an output is missing.
    """
    if dry_run:
        return _plan(name, "llm", cmd)
    return StepResult(name, "llm", tuple(cmd), executed=True, output=run_command(cmd))


@task(name="gpu_queue_submit", cache_policy=NO_CACHE, on_failure=[notify_failure])
def submit_gpu_job(
    job: Mapping[str, Any], queue_dir: str = "gpu_queue", dry_run: bool = False
) -> StepResult:
    """Validate ``job`` against the whole queue and write ``<queue_dir>/<id>.yaml``.

    The spec is first validated in a scratch copy of the queue (dry run: nothing else happens).
    A spec already queued unchanged is a no-op; a different spec under the same id is an error.
    """
    from terser.gpu_queue import dry_run as queue_order
    from terser.gpu_queue import format_dry_run, validate

    qdir = REPO_ROOT / queue_dir
    spec = dict(job)
    target = qdir / f"{spec['id']}.yaml"
    cmd = ("python", "-m", "terser.gpu_queue", "validate", "--dir", queue_dir)
    if target.exists() and yaml.safe_load(target.read_text(encoding="utf-8")) != spec:
        raise StepError(f"{target.name} is already queued with a different spec; use a new id")
    text = yaml.safe_dump(spec, sort_keys=False)
    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp)
        for p in [*qdir.glob("*.yaml"), *qdir.glob("*.yml")]:
            shutil.copy2(p, staged / p.name)
        (staged / target.name).write_text(text, encoding="utf-8")
        errors = validate(staged, REPO_ROOT)
        if errors:
            raise StepError("gpu_queue validation failed: " + "; ".join(errors))
        table = format_dry_run(queue_order(staged, REPO_ROOT))
    if dry_run:
        print(f"[plan] gpu_queue_submit: would write {queue_dir}/{target.name}")
        return StepResult("gpu_queue_submit", "gpu_queue", cmd, executed=False, output=table)
    if not target.exists():
        target.write_text(text, encoding="utf-8")
        errors = validate(qdir, REPO_ROOT)
        if errors:
            target.unlink()
            raise StepError("gpu_queue validation failed: " + "; ".join(errors))
    return StepResult("gpu_queue_submit", "gpu_queue", cmd, executed=True, output=table)


def _counter(target: str) -> Callable[[str], int]:
    from terser.tokenizers import count_tokens

    return lambda s: count_tokens(s, target)


@task(
    name="tokenizer_drift",
    retries=1,
    retry_delay_seconds=SHELL_RETRY_DELAY,
    cache_policy=NO_CACHE,
    on_failure=[notify_failure],
)
def drift_check(targets: Sequence[str] | None = None, dry_run: bool = False) -> StepResult:
    """Re-count the fixture strings on every target (row 11.5.3); ``ok=False`` on drift."""
    cmd = ["python", "-m", "terser.monitoring.tokenizer_drift"]
    for t in targets or ():
        cmd += ["--target", t]
    if dry_run:
        return _plan("tokenizer_drift", "python", cmd)
    from terser.monitoring.tokenizer_drift import check_tokenizer_drift, load_fixture

    strings, snapshot = load_fixture()
    counters = {t: _counter(t) for t in (targets or sorted(snapshot))}
    report = check_tokenizer_drift(strings, snapshot, counters)
    summary = report.summary()
    return StepResult("tokenizer_drift", "python", tuple(cmd), True, not report.alert, summary)


@task(
    name="input_drift",
    retries=1,
    retry_delay_seconds=SHELL_RETRY_DELAY,
    cache_policy=NO_CACHE,
    on_failure=[notify_failure],
)
def input_drift_check(name: str, current: str | None = None, dry_run: bool = False) -> StepResult:
    """Input drift report (row 11.5.2) through its CLI; ``ok=False`` on a drift alert.

    ``python -m terser.monitoring.input_drift`` exits 1 on an alert and prints a JSON verdict
    (``alert``, ``reasons``, artifact paths) as its last stdout line. An alert is a result, not a
    step failure, so it is not retried; any other exit (or a missing verdict) is a failure.
    """
    cmd = ["python", "-m", "terser.monitoring.input_drift", "--name", name]
    if current:
        cmd += ["--current", current]
    if dry_run:
        return _plan("input_drift", "python", cmd)
    proc = _run_process(cmd)
    lines = proc.stdout.strip().splitlines()
    verdict: dict[str, Any] | None = None
    if proc.returncode in (0, 1) and lines:
        try:
            verdict = json.loads(lines[-1])
        except json.JSONDecodeError:
            verdict = None
    if not isinstance(verdict, dict) or "alert" not in verdict:
        tail = (proc.stderr or proc.stdout).strip()[-OUTPUT_TAIL:]
        raise StepError(f"{_short_cmd(cmd)} exited {proc.returncode} without a verdict: {tail}")
    alert = bool(verdict["alert"]) or proc.returncode == 1
    reasons = "; ".join(str(r) for r in verdict.get("reasons", []))
    artifacts = ", ".join(f"{k}={v}" for k, v in verdict.items() if k not in ("alert", "reasons"))
    out = ("INPUT DRIFT " + reasons) if alert else f"no input drift ({artifacts})"
    return StepResult("input_drift", "python", tuple(cmd), True, not alert, out)


def load_eval_summary(
    rung: str, report: str = VAL_REPORT, has_lineage: bool = False
) -> EvalSummary:
    """Validation summary of a rewriter rung from the row 8.3.5 val report json.

    The report says nothing about lineage, so the caller states it (``has_lineage``); the
    default is the safe one, a summary without lineage is refused by rule 7.
    """
    d = json.loads((REPO_ROOT / report).read_text(encoding="utf-8"))[rung]
    return EvalSummary(
        preservation=float(d["preserved"]),
        savings=float(d["median_savings"]),
        per_category_preservation={k: float(v) for k, v in d["preserved_by_category"].items()},
        checkpoints_passed=("C8.3",) if d.get("c8_3") else (),
        has_lineage=has_lineage,
    )


@task(name="promotion_guard", cache_policy=NO_CACHE, on_failure=[notify_failure])
def promotion_guard(
    rung: str,
    champion_rung: str | None = None,
    report: str = VAL_REPORT,
    required_checkpoints: Sequence[str] = ("C8.3",),
    has_lineage: bool = True,
) -> StepResult:
    """Row 11.4.4 guard on validation summaries; ``ok`` is the decision (pure, also in dry run).

    ``has_lineage`` is the candidate's registry lineage (rule 7). It defaults to True here because
    :func:`terser.flows.flows.promote_flow` checks the lineage of the exact version it is about
    to move (:func:`lineage_missing`) once the metric rules have passed.
    """
    candidate = load_eval_summary(rung, report, has_lineage=has_lineage)
    champion = load_eval_summary(champion_rung, report, has_lineage=True) if champion_rung else None
    decision = decide(candidate, champion, GuardConfig(required_checkpoints=required_checkpoints))
    out = "allowed" if decision.promote else "refused: " + "; ".join(decision.reasons)
    print(f"promotion guard ({rung}): {out}")
    cmd = ("terser.pipelines.promotion.decide", rung, champion_rung or "-")
    return StepResult("promotion_guard", "python", cmd, True, decision.promote, out)


RUNG_EXPERIMENTS: Mapping[str, str] = {
    "sft05": "sft_qwen05b_full",
    "sft15": "sft_qwen15b_lora",
    "sft3b": "sft_qwen3b_lora",
    "sft15v2": "sft_qwen15b_lora_v2",
    "sft3bv2": "sft_qwen3b_lora_v2",
    "sft7bv2": "sft_qwen7b_qlora_v2",
}
"""Val-report rung tag -> the ``experiment`` run param its registry version was logged with."""


def _version_at(model_name: str, alias: str) -> str:
    """Version holding ``alias``.

    ``latest`` is the newest registered version; ``experiment:<name>`` the newest version whose
    registering run has ``experiment=<name>`` (the version a val-report rung actually measures).
    """
    from mlflow import MlflowClient

    from terser.tracking import setup_tracking

    setup_tracking()
    client = MlflowClient()
    if alias == "latest" or alias.startswith("experiment:"):
        versions = list(client.search_model_versions(f"name = '{model_name}'"))
        if alias != "latest":
            want = alias.split(":", 1)[1]
            versions = [
                v
                for v in versions
                if v.run_id and client.get_run(v.run_id).data.params.get("experiment") == want
            ]
        if not versions:
            raise StepError(f"{model_name} has no registered version for {alias!r}")
        return str(max(int(v.version) for v in versions))
    return str(client.get_model_version_by_alias(model_name, alias).version)


def rung_at(model_name: str, alias: str) -> str | None:
    """Val-report rung of the version holding ``alias`` (None if no such version or rung)."""
    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException

    from terser.tracking import setup_tracking

    setup_tracking()
    client = MlflowClient()
    try:
        mv = client.get_model_version_by_alias(model_name, alias)
    except MlflowException:
        return None
    if not mv.run_id:
        return None
    experiment = client.get_run(mv.run_id).data.params.get("experiment")
    return next((r for r, e in RUNG_EXPERIMENTS.items() if e == experiment), None)


def lineage_missing(model_name: str, version: str) -> list[str]:
    """Lineage tags (row 11.2.3) that ``version`` lacks; empty when the lineage is complete."""
    from mlflow import MlflowClient

    from terser.tracking import setup_tracking
    from terser.tracking.lineage import LINEAGE_TAGS

    setup_tracking()
    tags = MlflowClient().get_model_version(model_name, version).tags
    return [t for t in LINEAGE_TAGS if t not in tags or (not tags[t] and t != "checkpoints_passed")]


def registry_has(model_name: str) -> bool:
    """True if the MLflow registry (repo-local by default) knows ``model_name``.

    ``mlruns/`` is gitignored, so on a fresh clone the registry is empty while ``dvc.lock`` says
    the ``register`` stage is up to date; :func:`terser.flows.flows.promote_flow` then forces it.
    """
    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException

    from terser.tracking import setup_tracking

    setup_tracking()
    try:
        MlflowClient().get_registered_model(model_name)
    except MlflowException:
        return False
    return True


@task(
    name="registry_promote",
    retries=SHELL_RETRIES,
    retry_delay_seconds=SHELL_RETRY_DELAY,
    cache_policy=NO_CACHE,
    on_failure=[notify_failure],
)
def registry_promote(
    model_name: str,
    stage: str,
    from_alias: str = "staging",
    dry_run: bool = False,
    version: str | None = None,
) -> StepResult:
    """Move ``version`` (default: the one holding ``from_alias``) to ``stage``."""
    cmd = ("terser.tracking.promote", model_name, f"@{from_alias}", stage)
    if dry_run:
        return _plan("registry_promote", "python", cmd)
    from terser.tracking import promote

    version = version or _version_at(model_name, from_alias)
    promote(model_name, version, stage)
    out = f"{model_name} v{version} -> {stage}"
    return StepResult("registry_promote", "python", cmd, True, True, out)


# Continuous training (row 11.4.3) -----------------------------------------------------------


def read_ct_state(state_path: str) -> dict[str, Any]:
    """The last completed CT cycle (``{}`` if none): time, trigger, batch digests."""
    path = REPO_ROOT / state_path
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _dvc_status(stages: Sequence[str]) -> dict[str, Any]:
    proc = _run_process(["dvc", "status", "--json", *stages])
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip()[-OUTPUT_TAIL:]
        raise StepError(f"dvc status exited {proc.returncode}: {tail}")
    lines = [x for x in proc.stdout.strip().splitlines() if x.strip()]
    status = json.loads(lines[-1]) if lines else {}
    return status if isinstance(status, dict) else {}


@task(
    name="new_data_check",
    retries=1,
    retry_delay_seconds=SHELL_RETRY_DELAY,
    cache_policy=NO_CACHE,
    on_failure=[notify_failure],
)
def new_data_check(
    batch: str | None,
    stages: Sequence[str],
    state_path: str,
    dry_run: bool = False,
) -> StepResult:
    """CT trigger "new data batch": ``ok=False`` when new data has arrived.

    Two signals: a DVC-tracked change (``dvc status --json <data stages>`` is not empty, e.g.
    after ``dvc pull`` or ``dvc add`` of new raw data), and, when ``batch`` is given, a batch path
    whose content digest differs from the one recorded by the last completed cycle. Like a drift
    alert, new data is a result, not a step failure.
    """
    cmd = ["dvc", "status", "--json", *stages]
    if dry_run:
        return _plan("new_data", "trigger", cmd)
    reasons = []
    changed = sorted(_dvc_status(stages)) if stages else []
    if changed:
        reasons.append("DVC data stages changed: " + ", ".join(changed))
    if batch:
        digest = input_digest([batch])[batch]
        if digest == "missing":
            raise StepError(f"batch {batch} does not exist")
        seen = read_ct_state(state_path).get("batches", {}).get(batch)
        if digest != seen:
            reasons.append(f"new batch {batch} (sha256 {digest[:12]})")
    out = ("NEW DATA " + "; ".join(reasons)) if reasons else "no new data"
    return StepResult("new_data", "trigger", tuple(cmd), True, not reasons, out)


@task(name="record_ct_cycle", cache_policy=NO_CACHE, on_failure=[notify_failure])
def record_ct_cycle(
    state_path: str,
    trigger: str,
    reasons: str,
    batch: str | None = None,
    summary: Mapping[str, Any] | None = None,
    dry_run: bool = False,
) -> StepResult:
    """Write the completed cycle to ``state_path`` so the same batch does not trigger again."""
    cmd = ("terser.flows.steps.record_ct_cycle", state_path)
    if dry_run:
        return _plan("record_ct_cycle", "python", cmd)
    from datetime import UTC, datetime

    state = read_ct_state(state_path)
    batches = dict(state.get("batches", {}))
    if batch:
        batches[batch] = input_digest([batch])[batch]
    record = {
        "time": datetime.now(UTC).isoformat(timespec="seconds"),
        "trigger": trigger,
        "reasons": reasons,
        "batches": batches,
        **dict(summary or {}),
    }
    path = REPO_ROOT / state_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return StepResult("record_ct_cycle", "python", cmd, True, True, f"wrote {state_path}")
