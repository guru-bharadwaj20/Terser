"""The six Prefect flows of plan row **11.6.1**, with the reliability of row **11.6.2**.

Each flow is a thin, ordered list of the tasks in :mod:`terser.flows.steps`, so any run can be
reproduced by hand from its command plan (``python -m terser.flows plan``):

==========================  ====================================================================
``data_flow``               ``dvc repro --single-item`` per stage (row 2.1.7 pipeline), then the
                            ``respond`` and ``judge`` LLM steps (rows 2.2.3, 2.3.3), then pandera
                            validation of a pair table when given (row 11.1.4)
``train_classical_flow``    CPU training: pair features, then the fast-path gate and tagger
                            exported to ONNX (row 11.3.2)
``train_rewriter_flow``     CPU smoke test, then a validated job spec submitted to ``gpu_queue/``;
                            the GPU worker (``python -m terser.gpu_queue run``) trains it, never
                            the orchestrator
``evaluate_flow``           validation-split reports: gate checkpoint C3.3, rewriter C8.3
``promote_flow``            registry sync (``dvc repro register`` = :mod:`terser.phase11.register`),
                            the row 11.4.4 promotion guard, then the alias move if allowed
``drift_flow``              daily: tokenizer drift (row 11.5.3) and input drift (row 11.5.2,
                            ``python -m terser.monitoring.input_drift``); an alert fails the run
==========================  ====================================================================

Every flow takes ``dry_run``: tasks then record their commands without running them, and the
flow returns the plan as :class:`terser.flows.steps.StepResult` records. ``casualise`` and
``candidates`` (vLLM stages) and ``respond`` / ``judge`` run as cached LLM steps; every other
stage is a retried shell step. A failed flow run calls
:func:`terser.flows.reliability.notify_flow_failure`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any

import yaml
from prefect import flow

from terser.flows.reliability import notify_flow_failure
from terser.flows.steps import (
    REPO_ROOT,
    VAL_REPORT,
    StepError,
    StepResult,
    drift_check,
    input_drift_check,
    llm_step,
    promotion_guard,
    registry_promote,
    shell_step,
    submit_gpu_job,
)
from terser.tracking.lineage import MODEL_NAMES

__all__ = [
    "DATA_STAGES",
    "EVALUATE_STEPS",
    "FLOWS",
    "LLM_STAGES",
    "PLAN_PARAMS",
    "SCHEDULES",
    "TRAIN_CLASSICAL_STEPS",
    "data_flow",
    "drift_flow",
    "evaluate_flow",
    "module_path",
    "plan",
    "promote_flow",
    "rewriter_job",
    "train_classical_flow",
    "train_rewriter_flow",
]

# ``collect`` is stale by design (reports/p2_dvc_repro.md, "Known stale stage"), so it is not
# rebuilt unless named explicitly; ``--single-item`` keeps DVC from pulling it in upstream.
DATA_STAGES: tuple[str, ...] = (
    "casualise",
    "dedupe",
    "split",
    "label_subset",
    "candidates",
    "token_counts",
)
LLM_STAGES = frozenset({"casualise", "candidates"})
TARGETS: Mapping[str, str] = {
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
    "phi3.5-mini": "microsoft/Phi-3.5-mini-instruct",
}
JUDGE_MODEL = "Qwen/Qwen2.5-32B-Instruct-AWQ"
PROMPTS = "data/interim/casual_dedup.jsonl"
CANDIDATES = "data/interim/candidates.jsonl"
RESPONSES_DIR = "data/interim/responses"
LABELS = "data/processed/labels.jsonl"

TRAIN_CLASSICAL_STEPS: Mapping[str, Sequence[str]] = {
    "pair_features": ["python", "-m", "terser.phase3.build_features"],
    "fast_path_onnx": ["python", "-m", "terser.phase11.train_fast_path"],
}
_REWRITES = "data/interim/rewrites"
EVALUATE_STEPS: Mapping[str, Sequence[str]] = {
    "gate_checkpoint": ["python", "-m", "terser.phase3.gate_checkpoint"],
    "rewriter_val_report": [
        "python",
        "-m",
        "terser.models.rewriter.val_report",
        "--rung",
        f"sft05={_REWRITES}/sft-qwen0.5b-full.val.jsonl",
        "--rung",
        f"sft15={_REWRITES}/sft-qwen1.5b-lora.val.jsonl",
        "--rung",
        f"sft3b={_REWRITES}/sft-qwen3b-lora.val.jsonl",
        "--labels",
        f"{_REWRITES}/val_labels_all.jsonl",
        "--out",
        "reports/p8_rewriter_val.md",
    ],
}
SCHEDULES: Mapping[str, str] = {"drift_flow": "0 3 * * *"}  # daily; the others run on demand
# Example parameters for flows with required ones (plans, tests, the demo report).
PLAN_PARAMS: Mapping[str, Mapping[str, Any]] = {
    "train_rewriter_flow": {
        "experiment": "sft_tiny",
        "job_id": "sft-tiny-flow-check",
        "est_vram_gb": 2,
        "est_hours": 0.1,
        "depends_on": [],
    },
}


def _stage_inputs(stage: str) -> list[str]:
    """A DVC stage's deps plus ``params.yaml``: the LLM-step cache key inputs."""
    stages = yaml.safe_load((REPO_ROOT / "dvc.yaml").read_text(encoding="utf-8"))["stages"]
    return [*stages[stage].get("deps", []), "params.yaml"]


def _run_steps(steps: Mapping[str, Sequence[str]], dry_run: bool) -> list[StepResult]:
    return [shell_step(name, list(cmd), dry_run) for name, cmd in steps.items()]


@flow(name="data_flow", log_prints=True, on_failure=[notify_flow_failure])
def data_flow(
    stages: Sequence[str] = DATA_STAGES,
    label: bool = True,
    targets: Mapping[str, str] | None = None,
    judge_model: str = JUDGE_MODEL,
    pairs: str | None = None,
    dry_run: bool = False,
) -> list[StepResult]:
    """Rebuild the dataset stage by stage, answer and judge it, validate the pair table."""
    out: list[StepResult] = []
    for stage in stages:
        cmd = ["dvc", "repro", "--single-item", stage]
        if stage in LLM_STAGES:
            out.append(llm_step(f"dvc:{stage}", cmd, _stage_inputs(stage), dry_run))
        else:
            out.append(shell_step(f"dvc:{stage}", cmd, dry_run))
    if label:
        responses = []
        for tag, model in (targets or TARGETS).items():
            path = f"{RESPONSES_DIR}/{tag}.jsonl"
            responses.append(path)
            cmd = ["python", "-m", "terser.phase2.respond", "--prompts", PROMPTS]
            cmd += ["--candidates", CANDIDATES, "--target", model, "--out", path]
            inputs = [PROMPTS, CANDIDATES, "src/terser/phase2/respond.py", "src/terser/llm"]
            out.append(llm_step(f"respond:{tag}", cmd, inputs, dry_run))
        cmd = ["python", "-m", "terser.phase2.label_stage", "--prompts", PROMPTS]
        cmd += ["--responses", *responses, "--judge-model", judge_model]
        cmd += ["--splits-dir", "data/processed/splits", "--out", LABELS]
        inputs = [PROMPTS, *responses, "src/terser/phase2/label_stage.py", "src/terser/judge"]
        out.append(llm_step("judge", cmd, inputs, dry_run))
    if pairs:
        cmd = ["python", "-m", "terser.data.validation", pairs]
        out.append(shell_step("validate_pairs", cmd, dry_run))
    return out


@flow(name="train_classical_flow", log_prints=True, on_failure=[notify_flow_failure])
def train_classical_flow(
    steps: Mapping[str, Sequence[str]] | None = None, dry_run: bool = False
) -> list[StepResult]:
    """CPU training steps in order (default: pair features, then the ONNX gate and tagger)."""
    return _run_steps(TRAIN_CLASSICAL_STEPS if steps is None else steps, dry_run)


def rewriter_job(
    experiment: str,
    job_id: str | None = None,
    est_vram_gb: float = 14,
    est_hours: float = 8,
    depends_on: Sequence[str] = ("C7.1", "C2.3"),
    priority: int = 2,
    hardware: str = "rtx4500ada",
) -> dict[str, Any]:
    """A ``gpu_queue`` job spec (see ``gpu_queue/README.md``) for one rewriter experiment."""
    jid = job_id or experiment.replace("_", "-")
    return {
        "id": jid,
        "phase": 8,
        "depends_on": list(depends_on),
        "command": f"python -m terser.models.rewriter.train experiment={experiment} "
        f"hardware={hardware}",
        "est_vram_gb": est_vram_gb,
        "est_hours": est_hours,
        "exclusive": True,
        "priority": priority,
        "cpu_smoke_test": "tests/rewriter/test_train_tiny.py",
        "outputs": [f"models/rewriter/{jid}/", "mlruns/"],
    }


@flow(name="train_rewriter_flow", log_prints=True, on_failure=[notify_flow_failure])
def train_rewriter_flow(
    experiment: str,
    job_id: str | None = None,
    est_vram_gb: float = 14,
    est_hours: float = 8,
    depends_on: Sequence[str] = ("C7.1", "C2.3"),
    priority: int = 2,
    queue_dir: str = "gpu_queue",
    dry_run: bool = False,
) -> list[StepResult]:
    """Smoke-test on CPU, then submit the rewriter job to ``gpu_queue/`` for the GPU worker."""
    job = rewriter_job(experiment, job_id, est_vram_gb, est_hours, depends_on, priority)
    smoke = ["python", "-m", "pytest", "-q", job["cpu_smoke_test"]]
    out = [shell_step("cpu_smoke_test", smoke, dry_run)]
    out.append(submit_gpu_job(job, queue_dir, dry_run))
    worker = ("python", "-m", "terser.gpu_queue", "run", "--dir", queue_dir)
    out.append(StepResult("gpu_worker", "handoff", worker, executed=False, output="GPU machine"))
    return out


@flow(name="evaluate_flow", log_prints=True, on_failure=[notify_flow_failure])
def evaluate_flow(
    steps: Mapping[str, Sequence[str]] | None = None, dry_run: bool = False
) -> list[StepResult]:
    """Validation-split evaluation reports (default: gate C3.3, rewriter C8.3)."""
    return _run_steps(EVALUATE_STEPS if steps is None else steps, dry_run)


@flow(name="promote_flow", log_prints=True, on_failure=[notify_flow_failure])
def promote_flow(
    rung: str = "sft3b",
    champion_rung: str | None = None,
    component: str = "rewriter",
    stage: str = "Production",
    from_alias: str = "staging",
    sync_registry: bool = True,
    report: str = VAL_REPORT,
    dry_run: bool = False,
) -> list[StepResult]:
    """Sync the registry, then promote ``component`` only if the promotion guard allows it."""
    out: list[StepResult] = []
    if sync_registry:
        out.append(shell_step("register", ["dvc", "repro", "--single-item", "register"], dry_run))
    guard = promotion_guard(rung, champion_rung, report)
    out.append(guard)
    if guard.ok:
        out.append(registry_promote(MODEL_NAMES[component], stage, from_alias, dry_run))
    return out


@flow(name="drift_flow", log_prints=True, on_failure=[notify_flow_failure])
def drift_flow(
    targets: Sequence[str] | None = None,
    input_drift: bool = True,
    current: str | None = None,
    name: str | None = None,
    dry_run: bool = False,
) -> list[StepResult]:
    """Tokenizer drift (row 11.5.3) and input drift (row 11.5.2); raises on any alert.

    ``current`` is the prompts JSONL window compared with the train split (default: the input
    drift module's default, the val split); ``name`` names the artifacts in
    ``reports/monitoring/`` (default: today's UTC date). Both checks always run, so one alert
    never hides the other.
    """
    out = [drift_check(targets, dry_run)]
    if input_drift:
        run_name = name or datetime.now(UTC).strftime("%Y-%m-%d")
        out.append(input_drift_check(run_name, current, dry_run))
    alerts = [r.output for r in out if not r.ok]
    if alerts:
        raise StepError(" | ".join(alerts))
    return out


FLOWS: dict[str, Any] = {
    "data_flow": data_flow,
    "train_classical_flow": train_classical_flow,
    "train_rewriter_flow": train_rewriter_flow,
    "evaluate_flow": evaluate_flow,
    "promote_flow": promote_flow,
    "drift_flow": drift_flow,
}


def module_path(name: str) -> str:
    """``module:function`` of a registered flow (its deployment entrypoint)."""
    return f"{__name__}:{FLOWS[name].fn.__name__}"


def plan(name: str, **params: Any) -> list[StepResult]:
    """Run flow ``name`` in dry-run mode (with :data:`PLAN_PARAMS`) and return its plan."""
    kwargs = {**PLAN_PARAMS.get(name, {}), **params, "dry_run": True}
    return list(FLOWS[name](**kwargs))
