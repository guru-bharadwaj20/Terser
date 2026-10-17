"""The Prefect flows of plan rows **11.6.1** and **11.4.3**, with the reliability of row **11.6.2**.

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
``ct_flow``                 continuous training (row 11.4.3): on a drift alert or a new data batch,
                            the flows above in order: judge labels (``data_flow``), retrain gate
                            and tagger (``train_classical_flow``), queue the rewriter retrain
                            (``train_rewriter_flow``), evaluate, then ``promote_flow`` to
                            ``Staging`` behind the promotion guard
``canary_flow``             nightly quality canary (row 11.5.4): the fixed 200-prompt canary set
                            through the deployed service, both targets and the judge
                            (``python -m terser.monitoring.canary``); the run's preservation goes
                            to the history, the Pushgateway and ``reports/p11_canary.md``
==========================  ====================================================================

Every flow takes ``dry_run``: tasks then record their commands without running them, and the
flow returns the plan as :class:`terser.flows.steps.StepResult` records. ``casualise`` and
``candidates`` (vLLM stages) and ``respond`` / ``judge`` run as cached LLM steps; every other
stage is a retried shell step. A failed flow run calls
:func:`terser.flows.reliability.notify_flow_failure`.

DVC-tracked outputs that a flow rewrites (the label table, pair / span features, the ONNX fast
path) are checked out as read-only hardlinks into the DVC cache (``cache.type: hardlink``), so
the flows ``dvc unprotect`` them first; without that a fresh clone fails with ``PermissionError``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

import yaml
from prefect import flow

from terser.flows import steps
from terser.flows.reliability import notify_flow_failure
from terser.flows.steps import (
    REPO_ROOT,
    VAL_REPORT,
    StepError,
    StepResult,
    drift_check,
    input_drift_check,
    llm_step,
    new_data_check,
    promotion_guard,
    record_ct_cycle,
    registry_has,
    registry_promote,
    shell_step,
    submit_gpu_job,
)
from terser.tracking.lineage import MODEL_NAMES

__all__ = [
    "CLASSICAL_OUTPUTS",
    "CT_STATE",
    "CT_TRIGGERS",
    "DATA_STAGES",
    "EVALUATE_STEPS",
    "FLOWS",
    "LLM_STAGES",
    "PLAN_PARAMS",
    "SCHEDULES",
    "TRAIN_CLASSICAL_STEPS",
    "CANARY",
    "CANARY_TARGETS",
    "canary_flow",
    "ct_flow",
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
# The labelled subset (dvc.yaml respond@* / label), not the whole deduplicated pool.
PROMPTS = "data/interim/label_prompts.jsonl"
CANDIDATES = "data/interim/candidates.jsonl"
RESPONSES_DIR = "data/interim/responses"
LABELS = "data/processed/labels.jsonl"

TRAIN_CLASSICAL_STEPS: Mapping[str, Sequence[str]] = {
    # features depend on data/processed/oracle.csv (dvc.yaml), which follows the labels.
    "oracle": ["dvc", "repro", "--single-item", "oracle"],
    "pair_features": ["python", "-m", "terser.phase3.build_features"],
    "fast_path_onnx": ["python", "-m", "terser.phase11.train_fast_path"],
}
# DVC-tracked files the default classical steps rewrite (build_features, train_fast_path).
CLASSICAL_OUTPUTS: tuple[str, ...] = (
    "data/processed/pair_features.parquet",
    "data/processed/span_features.parquet",
    "models/fast",
)
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
        "--rung",
        f"sft15v2={_REWRITES}/sft15v2.val.jsonl",
        "--rung",
        f"sft3bv2={_REWRITES}/sft3bv2.val.jsonl",
        "--rung",
        f"sft7bv2={_REWRITES}/sft7bv2.val.jsonl",
        # v1 and v2 labels; the v1-only file would drop the v2 rungs from the committed report.
        "--labels",
        f"{_REWRITES}/val_labels_all_v2.jsonl",
        "--out",
        "reports/p8_rewriter_val.md",
    ],
}
# drift_flow daily; ct_flow polls for a new data batch daily after it (drift alerts start it
# through an event trigger, see terser.flows.registry); the quality canary nightly before both
# (UTC); the others run on demand.
SCHEDULES: Mapping[str, str] = {
    "drift_flow": "0 3 * * *",
    "ct_flow": "0 4 * * *",
    "canary_flow": "0 2 * * *",
}
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

CT_STATE = "logs/ct_state.json"
CT_TRIGGERS: tuple[str, ...] = ("check", "drift", "batch", "force")
CT_EXPERIMENT = "sft_qwen3b_lora"  # the best rung so far (row 8.3.4)


def _dvc_stage(stage: str) -> dict[str, Any]:
    stages = yaml.safe_load((REPO_ROOT / "dvc.yaml").read_text(encoding="utf-8"))["stages"]
    return dict(stages[stage])


def _stage_inputs(stage: str) -> list[str]:
    """A DVC stage's deps plus ``params.yaml``: the LLM-step cache key inputs."""
    return [*_dvc_stage(stage).get("deps", []), "params.yaml"]


def _stage_outputs(stage: str) -> list[str]:
    """A DVC stage's outs (path strings; ``{path: options}`` entries reduced to the path)."""
    outs = _dvc_stage(stage).get("outs", [])
    return [o if isinstance(o, str) else next(iter(o)) for o in outs]


def _params(section: str) -> dict[str, Any]:
    """One section of ``params.yaml`` (the values the DVC ``respond`` / ``label`` stages use)."""
    params = yaml.safe_load((REPO_ROOT / "params.yaml").read_text(encoding="utf-8"))
    return dict(params.get(section) or {})


def _run_steps(steps: Mapping[str, Sequence[str]], dry_run: bool) -> list[StepResult]:
    return [shell_step(name, list(cmd), dry_run) for name, cmd in steps.items()]


def _unprotect(name: str, paths: Sequence[str], dry_run: bool) -> list[StepResult]:
    """``dvc unprotect`` the existing ``paths`` so a step can rewrite them (none: no step)."""
    present = list(paths) if dry_run else [p for p in paths if (REPO_ROOT / p).exists()]
    if not present:
        return []
    return [shell_step(name, ["dvc", "unprotect", *present], dry_run)]


@flow(name="data_flow", log_prints=True, on_failure=[notify_flow_failure])
def data_flow(
    stages: Sequence[str] = DATA_STAGES,
    label: bool = True,
    targets: Mapping[str, str] | None = None,
    judge_model: str | None = None,
    pairs: str | None = None,
    dry_run: bool = False,
) -> list[StepResult]:
    """Rebuild the dataset stage by stage, answer and judge it, validate the pair table.

    ``label=False`` skips the ``respond`` / ``judge`` LLM steps and keeps the DVC-tracked label
    table (the CPU-only path on a machine without the LLM response cache or a vLLM server).
    The respond / judge commands take every setting from ``params.yaml`` (the values the DVC
    ``respond@*`` and ``label`` stages use); ``targets`` and ``judge_model`` override them.
    """
    out: list[StepResult] = []
    for stage in stages:
        cmd = ["dvc", "repro", "--single-item", stage]
        if stage in LLM_STAGES:
            inputs, outputs = _stage_inputs(stage), _stage_outputs(stage)
            out.append(llm_step(f"dvc:{stage}", cmd, inputs, dry_run, outputs=outputs))
        else:
            out.append(shell_step(f"dvc:{stage}", cmd, dry_run))
    if label:
        rp, lp = _params("respond"), _params("label")
        tmap: Mapping[str, str] = targets or rp.get("targets") or TARGETS
        responses = [f"{RESPONSES_DIR}/{tag}.jsonl" for tag in tmap]
        # Both are DVC outs (hardlinks after `dvc checkout`): unprotect before rewriting them.
        out += _unprotect("unprotect:responses", responses, dry_run)
        for (tag, model), path in zip(tmap.items(), responses, strict=True):
            cmd = ["python", "-m", "terser.phase2.respond", "--prompts", PROMPTS]
            cmd += ["--candidates", CANDIDATES, "--target", model, "--out", path]
            for flag, key in (
                ("--max-tokens", "max_tokens"),
                ("--creative-samples", "creative_samples"),
                ("--creative-temperature", "creative_temperature"),
                ("--workers", "workers"),
            ):
                if key in rp:
                    cmd += [flag, str(rp[key])]
            inputs = [PROMPTS, CANDIDATES, "params.yaml", "src/terser/phase2/respond.py"]
            inputs.append("src/terser/llm")
            out.append(llm_step(f"respond:{tag}", cmd, inputs, dry_run, outputs=[path]))
        out += _unprotect("unprotect:labels", [LABELS], dry_run)
        model = judge_model or lp.get("judge_model") or JUDGE_MODEL
        cmd = ["python", "-m", "terser.phase2.label_stage", "--prompts", PROMPTS]
        cmd += ["--responses", *responses, "--judge-model", model]
        cmd += ["--splits-dir", "data/processed/splits", "--out", LABELS]
        if "workers" in lp:
            cmd += ["--workers", str(lp["workers"])]
        inputs = [PROMPTS, *responses, "params.yaml", "src/terser/phase2/label_stage.py"]
        inputs.append("src/terser/judge")
        out.append(llm_step("judge", cmd, inputs, dry_run, outputs=[LABELS]))
    if pairs:
        cmd = ["python", "-m", "terser.data.validation", pairs]
        out.append(shell_step("validate_pairs", cmd, dry_run))
    return out


@flow(name="train_classical_flow", log_prints=True, on_failure=[notify_flow_failure])
def train_classical_flow(
    steps: Mapping[str, Sequence[str]] | None = None, dry_run: bool = False
) -> list[StepResult]:
    """CPU training steps in order (default: pair features, then the ONNX gate and tagger).

    With the default steps their DVC-tracked outputs are unprotected first.
    """
    if steps is not None:
        return _run_steps(steps, dry_run)
    out = _unprotect("unprotect_outputs", CLASSICAL_OUTPUTS, dry_run)
    return out + _run_steps(TRAIN_CLASSICAL_STEPS, dry_run)


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
    """Sync the registry, then promote ``component`` only if the promotion guard allows it.

    ``mlruns/`` is gitignored, so on a fresh clone ``dvc.lock`` calls the ``register`` stage up
    to date while the registry is empty; the sync then runs it with ``--force``.
    ``from_alias="latest"`` promotes the newest registered version, ``experiment:<name>`` the
    newest version of one experiment. The version is resolved once, its lineage checked
    (rule 7), and that same version is promoted.
    """
    out: list[StepResult] = []
    if sync_registry:
        cmd = ["dvc", "repro", "--single-item", "register"]
        if not dry_run and not registry_has(MODEL_NAMES[component]):
            cmd.insert(3, "--force")
        out.append(shell_step("register", cmd, dry_run))
    guard = promotion_guard(rung, champion_rung, report)
    name = MODEL_NAMES[component]
    version: str | None = None
    if guard.ok and not dry_run:
        # Rule 7 on the exact version that would move, not on the report row.
        version = steps._version_at(name, from_alias)
        missing = steps.lineage_missing(name, version)
        if missing:
            msg = f"refused: {name} v{version} lineage incomplete (missing {missing})"
            print(f"promotion guard ({rung}): {msg}")
            guard = replace(guard, ok=False, output=msg)
    out.append(guard)
    if guard.ok:
        out.append(registry_promote(name, stage, from_alias, dry_run, version=version))
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


def _prefixed(prefix: str, results: Sequence[StepResult]) -> list[StepResult]:
    return [replace(r, name=f"{prefix}/{r.name}") for r in results]


@flow(name="ct_flow", log_prints=True, on_failure=[notify_flow_failure])
def ct_flow(
    trigger: str = "check",
    check_drift: bool = True,
    batch: str | None = None,
    drift_targets: Sequence[str] | None = None,
    drift_current: str | None = None,
    drift_name: str | None = None,
    data_stages: Sequence[str] = DATA_STAGES,
    label: bool = True,
    pairs: str | None = None,
    experiment: str = CT_EXPERIMENT,
    job_id: str | None = None,
    est_vram_gb: float = 20,
    est_hours: float = 3,
    depends_on: Sequence[str] = ("C7.1", "C2.3"),
    queue_dir: str = "gpu_queue",
    rung: str = "sft3b",
    champion_rung: str | None = None,
    candidate_alias: str | None = None,
    sync_registry: bool = True,
    report: str = VAL_REPORT,
    state_path: str = CT_STATE,
    dry_run: bool = False,
) -> list[StepResult]:
    """Continuous training (row 11.4.3): one cycle on a drift alert or a new data batch.

    ``trigger``:

    * ``check`` (default, the daily schedule): run the detectors (tokenizer and input drift when
      ``check_drift``; :func:`terser.flows.steps.new_data_check` on the DVC data stages and on
      ``batch``) and run the cycle only if one fires;
    * ``drift``: started by the drift alert (the deployment's event trigger on a failed
      ``drift_flow`` run), no re-check;
    * ``batch``: a new batch declared by the caller (``batch`` path, or DVC data already updated);
    * ``force``: run the cycle unconditionally.

    The cycle: ``data_flow`` (stages, then the judge labels unless ``label=False``),
    ``train_classical_flow`` (gate and tagger, CPU), ``train_rewriter_flow`` (CPU smoke test, then
    the job is *queued* in ``queue_dir`` for the GPU worker; nothing trains here),
    ``evaluate_flow``, then ``promote_flow(stage="Staging")``: the registry version that the
    ``rung`` row of the validation report measures (``candidate_alias``, default
    ``experiment:<rung's experiment>``) moves to Staging only if the row 11.4.4 guard passes
    (C8.3 required; rules 3-5 against ``champion_rung``, default the rung now holding Staging;
    rule 7 on that version's lineage). The rewriter queued in this cycle is evaluated by the next
    cycle, once the GPU worker has trained it. A completed cycle is recorded in ``state_path`` so
    an unchanged ``batch`` does not trigger again. In a dry run the detectors only plan and the
    cycle is planned as if triggered.
    """
    if trigger not in CT_TRIGGERS:
        raise ValueError(f"trigger must be one of {CT_TRIGGERS}, got {trigger!r}")
    out: list[StepResult] = []
    reasons: list[str] = []
    if trigger == "check":
        if check_drift:
            out.append(drift_check(drift_targets, dry_run))
            run_name = drift_name or datetime.now(UTC).strftime("%Y-%m-%d")
            out.append(input_drift_check(run_name, drift_current, dry_run))
        out.append(new_data_check(batch, list(data_stages), state_path, dry_run))
        reasons = [r.output for r in out if r.executed and not r.ok]
        if not reasons and not dry_run:
            msg = "no drift alert and no new data: nothing to do"
            print(f"ct_flow: {msg}")
            return [
                *out,
                StepResult("ct_trigger", "trigger", ("ct_flow", trigger), True, True, msg),
            ]
    elif trigger == "drift":
        reasons = ["drift alert (drift_flow failed)"]
    elif trigger == "batch":
        reasons = [f"new data batch {batch}" if batch else "new data batch (DVC)"]
    else:
        reasons = ["forced"]
    why = "; ".join(reasons) or "dry run: cycle planned as if triggered"
    print(f"ct_flow: cycle started ({why})")
    out.append(StepResult("ct_trigger", "trigger", ("ct_flow", trigger), not dry_run, True, why))

    out += _prefixed(
        "data_flow", data_flow(stages=data_stages, label=label, pairs=pairs, dry_run=dry_run)
    )
    out += _prefixed("train_classical_flow", train_classical_flow(dry_run=dry_run))
    jid = job_id or f"ct-{experiment.replace('_', '-')}-{datetime.now(UTC):%Y%m%d}"
    queued = train_rewriter_flow(
        experiment,
        jid,
        est_vram_gb,
        est_hours,
        depends_on,
        queue_dir=queue_dir,
        dry_run=dry_run,
    )
    out += _prefixed("train_rewriter_flow", queued)
    out += _prefixed("evaluate_flow", evaluate_flow(dry_run=dry_run))
    # The guard reads ``rung``'s row of the val report, so promote the registry version that
    # row measures (its experiment), not whatever was registered last.
    if candidate_alias is None:
        exp = steps.RUNG_EXPERIMENTS.get(rung)
        candidate_alias = f"experiment:{exp}" if exp else "latest"
    # Rules 3-5 compare with the current champion: the rung holding Staging, when known.
    if champion_rung is None and not dry_run:
        held = steps.rung_at(MODEL_NAMES["rewriter"], "staging")
        champion_rung = held if held != rung else None
    staged = promote_flow(
        rung=rung,
        champion_rung=champion_rung,
        component="rewriter",
        stage="Staging",
        from_alias=candidate_alias,
        sync_registry=sync_registry,
        report=report,
        dry_run=dry_run,
    )
    out += _prefixed("promote_flow", staged)
    guard = next(r for r in staged if r.name == "promotion_guard")
    summary = {
        "rewriter_job": f"{queue_dir}/{jid}.yaml",
        "staging": "registered" if guard.ok else guard.output,
    }
    out.append(record_ct_cycle(state_path, trigger, why, batch, summary, dry_run))
    return out


CANARY = "data/canary/canary_v1.jsonl"
CANARY_TARGETS: tuple[str, ...] = ("qwen2.5-7b", "phi3.5-mini")  # terser.monitoring.canary


@flow(name="canary_flow", log_prints=True, on_failure=[notify_flow_failure])
def canary_flow(
    base_url: str = "http://localhost:8080",
    mode: str = "quality",
    targets: Sequence[str] = CANARY_TARGETS,
    llm_url: str | None = None,
    target_urls: Mapping[str, str] | None = None,
    judge_url: str | None = None,
    judge_model: str = JUDGE_MODEL,
    gateway: str | None = "localhost:9091",
    canary: str = CANARY,
    run_id: str | None = None,
    dry_run: bool = False,
) -> list[StepResult]:
    """Nightly quality canary (row 11.5.4): service -> targets -> judge -> push.

    Runs :mod:`terser.monitoring.canary` stage by stage in ``logs/canary/<run_id>`` (default
    ``run_id``: today's UTC date): ``rewrite`` (every canary prompt through ``POST /rewrite`` of
    the service at ``base_url`` in ``mode``, once per target), ``respond`` per target (endpoint
    ``target_urls[tag]``, else ``llm_url``, else ``TERSER_LOCAL_LLM_URL``), ``judge``
    (``judge_url``), ``finalize`` (history ``logs/canary_history.jsonl``, push to ``gateway``,
    ``reports/p11_canary.md``). The endpoints must be up for the whole run; on the single-GPU
    workstation ``/home/temp/canary_chain.sh`` runs the same stages, serving one model at a time.
    The rewrite step always re-queries the service (``--force``, which clears the run's
    downstream files): its output is what is measured. Every step is a retried shell step, not
    an input-hash-cached LLM step, so a rerun never skips writing a stage's outputs; the target
    and judge requests themselves are disk-cached by :class:`terser.llm.LLMClient`, so a retry
    does not pay twice. A low rolling value is not a flow failure; the Grafana alert (row 11.5.5)
    handles it.
    """
    rid = run_id or datetime.now(UTC).strftime("%Y-%m-%d")
    run_dir = f"logs/canary/{rid}"
    mod = ["python", "-m", "terser.monitoring.canary"]
    rewrite = [*mod, "rewrite", "--run-dir", run_dir, "--canary", canary, "--base-url", base_url]
    rewrite += ["--mode", mode, "--targets", *targets, "--judge-model", judge_model, "--force"]
    out = [shell_step("canary:rewrite", rewrite, dry_run)]
    for tag in targets:
        cmd = [*mod, "respond", "--run-dir", run_dir, "--target", tag]
        url = (target_urls or {}).get(tag, llm_url)
        if url:
            cmd += ["--llm-url", url]
        out.append(shell_step(f"canary:respond:{tag}", cmd, dry_run))
    cmd = [*mod, "judge", "--run-dir", run_dir, "--judge-model", judge_model]
    if judge_url:
        cmd += ["--judge-url", judge_url]
    out.append(shell_step("canary:judge", cmd, dry_run))
    cmd = [*mod, "finalize", "--run-dir", run_dir]
    if gateway:
        cmd += ["--gateway", gateway]
    out.append(shell_step("canary:finalize", cmd, dry_run))
    return out


FLOWS: dict[str, Any] = {
    "data_flow": data_flow,
    "train_classical_flow": train_classical_flow,
    "train_rewriter_flow": train_rewriter_flow,
    "evaluate_flow": evaluate_flow,
    "promote_flow": promote_flow,
    "drift_flow": drift_flow,
    "ct_flow": ct_flow,
    "canary_flow": canary_flow,
}


def module_path(name: str) -> str:
    """``module:function`` of a registered flow (its deployment entrypoint)."""
    return f"{__name__}:{FLOWS[name].fn.__name__}"


def plan(name: str, **params: Any) -> list[StepResult]:
    """Run flow ``name`` in dry-run mode (with :data:`PLAN_PARAMS`) and return its plan."""
    kwargs = {**PLAN_PARAMS.get(name, {}), **params, "dry_run": True}
    return list(FLOWS[name](**kwargs))
