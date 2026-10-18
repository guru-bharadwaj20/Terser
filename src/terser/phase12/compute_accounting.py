"""Row 12.10: compute accounting, generated tokens, GPU-hours and energy per phase.

Regenerates ``reports/compute_accounting.md`` and ``reports/compute_accounting.json`` from the
evidence that the GPU work left on disk. Every number in the report names its source file.

Sources (all read-only):

* **LLM ledger** ``.cache/llm_ledger.json`` (:mod:`terser.llm.budget`): one entry per uncached
  LLM call with a UTC timestamp, model, input and output tokens. Cache hits are never ledgered,
  so ledger output tokens are the tokens the GPU actually generated. A stage's tokens are the
  ledger output tokens of its model inside its time window.
* **Stage summaries**: the JSON object each stage prints at the end of its log in ``logs/``
  (``wall_seconds``, ``generated_tokens``, ``records``), cross-checked against the ledger.
* **vLLM server logs** ``logs/vllm_*.log`` (and ``judge32.log``, ``judge_phi.log``): first to
  last timestamped line = the time the card was reserved by that server (startup, idle time and
  shutdown included). vLLM prints ``MM-DD hh:mm:ss`` without a year; the year is taken from the
  ledger (2026). The workstation clock is UTC: ledger (UTC) and vLLM windows agree to the second.
* **Training logs** ``logs/sft-*.log``: first to last Python-logging timestamp.
* **MLflow** ``mlruns/mlflow.db`` (SQLite, opened read-only): Phase 1 pilot, judge-bias study and
  Phase 7 benchmark runs, which predate or bypass the stage summaries.
* **Rewriter validation files** ``data/interim/rewrites/*.val.jsonl``: per-prompt
  ``latency_s`` (generation seconds) and ``raw_rewrite`` (tokenised with the Qwen2.5 tokenizer
  for generated tokens).
* **RTX 4090 smoke** ``reports/p2_gpu_smoke.md`` (worker-1; its raw logs are not on this machine).
* **Estimates**: ``configs/experiment/p2_compute_estimates.yaml`` (row 2.4.3 plan; flagged
  ``placeholder: true``), ``reports/p2_full_scale_plan.md`` (smoke-based projection) and the
  ``est_hours`` of the ``gpu_queue/`` job specs.
* **Optuna** ``experiments/p10_rewriter_hpo/optuna.db``: Phase 10 rewriter-HPO trial durations.

Measures:

* ``active_s``: GPU-seconds the stage itself measured (summary ``wall_seconds``, MLflow
  ``wall_seconds``, or the training/generation span). One card, so GPU-seconds = wall seconds.
* ``reserved_s``: server or training-process lifetime (>= active). For stages with no surviving
  server log, the phase total uses ``active_s`` as a lower bound and says so.
* ``ledger_busy_s``: cross-check, the summed span of ledger call bursts (gaps > 300 s split).
* Energy: **estimated**, not measured. No power was logged during the runs, so energy =
  GPU-hours x 210 W, the RTX 4500 Ada board power limit (``nvidia-smi power.limit``). That is an
  upper bound on board energy; a 60 s ``nvidia-smi`` sample taken during an (excluded) QLoRA run
  is kept in ``reports/compute_power_sample.csv`` and reported as a sensitivity.

Coverage: Phases 1, 2, 7, 8 (rungs 1-3, v2 rungs incl. 7B QLoRA, q7 quality chain), 10 (rewriter
HPO), 11 (quality-path benchmarks, canary) and 12 (test evaluation). Ledger calls are attributed
by model and time window, never by ledger bucket (Phase 12 respond and the first test judge were
charged to ``phase2``). Stages with ``pending_since`` (the row 12.4 seed run) are counted only
once their logs are complete; until then they are listed as pending and later ledger calls are
held back.

CLI::

    python -m terser.phase12.compute_accounting            # writes reports/compute_accounting.*
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from terser.config import project_root

__all__ = [
    "BOARD_POWER_W",
    "LEDGER_CUTOFF",
    "SESSIONS",
    "STAGES",
    "Session",
    "Stage",
    "build",
    "busy_seconds",
    "json_blobs",
    "log_span",
    "md_table",
    "render_markdown",
    "vllm_span",
]

ROOT = project_root()
# Board power limits used for the energy ESTIMATE. RTX 4500 Ada: 210 W (CONTRIBUTING hardware
# table; nvidia-smi power.limit in reports/compute_power_sample.csv). RTX 4090: 450 W (NVIDIA
# reference board power; the worker's limit was not recorded).
BOARD_POWER_W: dict[str, float] = {"rtx4500ada": 210.0, "rtx4090": 450.0}
# Start of the row 12.4 seed run (logs/p12_seeds.status "start 07:07" IST = 01:37 UTC). Ledger
# calls from the earliest ``pending_since`` of a still-pending stage on are held back; once
# nothing is pending the whole ledger is counted.
LEDGER_CUTOFF = "2026-10-20T01:37:00"
BUSY_GAP_S = 300.0
LEDGER_GRACE_S = 30.0  # ledger entries are written just after the server logs the response
LOG_YEAR = 2026

M32 = "Qwen/Qwen2.5-32B-Instruct-AWQ"
M14 = "Qwen/Qwen2.5-14B-Instruct-AWQ"
M7 = "Qwen/Qwen2.5-7B-Instruct"
PHI = "microsoft/Phi-3.5-mini-instruct"


@dataclass(frozen=True)
class Session:
    """A GPU reservation: one vLLM server lifetime or one training process."""

    id: str
    phase: int
    log: str
    kind: str  # "vllm" | "pylog"
    note: str = ""


@dataclass(frozen=True)
class Stage:
    """One unit of GPU work and where each of its numbers comes from."""

    id: str
    phase: int
    row: str
    what: str
    hardware: str = "rtx4500ada"
    model: str | None = None
    window: tuple[str, str] | str | None = None  # UTC ISO pair, or "@<session id>"
    active: dict[str, Any] = field(default_factory=dict)
    reported_tokens: dict[str, Any] | None = None
    tokens: str = "ledger"  # "ledger" | "reported" | "tokenize"
    labels: dict[str, Any] | None = None
    sessions: tuple[str, ...] = ()
    group: str = ""  # labelling round this stage belongs to (GPU-seconds per labelled pair)
    attempt: str = "final"  # "final" | "retry" (work redone or interrupted)
    note: str = ""
    # In-progress work: UTC start of the run. The stage is counted only once ``done`` (a
    # (path, regex) that must match) holds, or, without ``done``, once ``active`` resolves;
    # until then it is listed as pending and ledger calls from ``pending_since`` on are held back.
    pending_since: str | None = None
    done: tuple[str, str] | None = None


def _s(path: str, key: str = "wall_seconds") -> dict[str, Any]:
    return {"kind": "summary", "path": path, "key": key}


def _mlflow(runs: Sequence[str], metric: str) -> dict[str, Any]:
    return {"kind": "mlflow", "runs": list(runs), "metric": metric}


SESSIONS: tuple[Session, ...] = (
    Session("vllm_14b", 2, "logs/vllm_14b.log", "vllm"),
    Session("vllm_32b", 2, "logs/vllm_32b.log", "vllm", "casualise + candidates, one server"),
    Session("vllm_7b", 2, "logs/vllm_7b.log", "vllm", "left up after the stage finished"),
    Session("vllm_phi35", 2, "logs/vllm_phi35.log", "vllm", "stage paused and resumed"),
    Session("vllm_32b_judge", 2, "logs/vllm_32b_judge.log", "vllm", "first full-judge attempt"),
    Session("judge32", 2, "logs/judge32.log", "vllm"),
    Session("judge_phi", 2, "logs/judge_phi.log", "vllm"),
    Session("sft05", 8, "logs/sft-qwen0.5b-full.log", "pylog"),
    Session("sft15", 8, "logs/sft-qwen1.5b-lora.log", "pylog"),
    Session("sft3b", 8, "logs/sft-qwen3b.log", "pylog"),
    Session("vllm_eval_qwen1", 8, "logs/vllm_eval_qwen1.log", "vllm", "failed startup"),
    Session("vllm_eval_qwen2", 8, "logs/vllm_eval_qwen2.log", "vllm", "failed startup"),
    Session("vllm_eval_qwen", 8, "logs/vllm_eval_qwen.log", "vllm"),
    Session("vllm_eval_phi", 8, "logs/vllm_eval_phi.log", "vllm"),
    Session("vllm_eval_judge", 8, "logs/vllm_eval_judge.log", "vllm"),
    Session("vllm_r3_qwen", 8, "logs/vllm_r3_qwen.log", "vllm"),
    Session("vllm_r3_phi", 8, "logs/vllm_r3_phi.log", "vllm"),
    Session("vllm_r3_judge", 8, "logs/vllm_r3_judge.log", "vllm", "restart; first log lost"),
    Session("vllm_aug_gen", 8, "logs/vllm_aug_gen.log", "vllm"),
    Session("vllm_aug_qwen", 8, "logs/vllm_aug_qwen.log", "vllm"),
    Session("vllm_aug_phi", 8, "logs/vllm_aug_phi.log", "vllm"),
    Session("vllm_aug_judge", 8, "logs/vllm_aug_judge.log", "vllm"),
    # Phase 8 v2 rewriter rungs (v2_chain, then the 7B QLoRA + q7 quality chain)
    Session("sft15v2", 8, "logs/sft_qwen15b_lora_v2.log", "pylog"),
    Session("sft3bv2", 8, "logs/sft_qwen3b_lora_v2.log", "pylog"),
    Session("sft7bv2", 8, "logs/sft_qwen7b_qlora_v2.log", "pylog"),
    Session("vllm_v2_qwen", 8, "logs/vllm_v2_qwen.log", "vllm"),
    Session("vllm_v2_phi", 8, "logs/vllm_v2_phi.log", "vllm"),
    Session("vllm_v2_judge", 8, "logs/vllm_v2_judge.log", "vllm"),
    Session("vllm_q7_qwen", 8, "logs/vllm_q7_qwen2.5-7b.log", "vllm", "7b + hmm + rl sets"),
    Session("vllm_q7_phi", 8, "logs/vllm_q7_phi3.5.log", "vllm", "7b + hmm + rl sets"),
    Session("vllm_q7_judge", 8, "logs/vllm_q7_judge.log", "vllm", "7b + hmm + rl sets"),
    # Phase 10: rewriter HPO (Optuna trials train LoRA adapters on the GPU)
    Session(
        "hpo_trials",
        10,
        "experiments/p10_rewriter_hpo/optuna.db",
        "optuna",
        "first trial start to last trial end",
    ),
    Session("vllm_hpo_qwen", 10, "logs/vllm_hpo_qwen2.5-7b.log", "vllm"),
    Session("vllm_hpo_phi", 10, "logs/vllm_hpo_phi3.5.log", "vllm"),
    Session("vllm_hpo_judge", 10, "logs/vllm_hpo_judge.log", "vllm"),
    # Phase 11: quality-path load benchmarks and the canary
    Session("vllm_qb_3b", 11, "logs/vllm_qb_3b.log", "vllm"),
    Session("vllm_qb_15v2", 11, "logs/vllm_qb_15v2.log", "vllm"),
    Session("vllm_qraw_3b", 11, "logs/vllm_qraw_3b.log", "vllm"),
    Session("vllm_qraw_15v2", 11, "logs/vllm_qraw_15v2.log", "vllm"),
    Session("vllm_canary_rw", 11, "logs/vllm_canary_rewriter.log", "vllm"),
    Session("vllm_canary_qwen", 11, "logs/vllm_canary_qwen2.5-7b.log", "vllm"),
    Session("vllm_canary_phi", 11, "logs/vllm_canary_phi3.5-mini.log", "vllm"),
    Session("vllm_canary_judge", 11, "logs/vllm_canary_judge.log", "vllm"),
    # Phase 12: test-split evaluation
    Session("vllm_p12_zs", 12, "logs/vllm_p12_zs.log", "vllm"),
    Session("vllm_p12_qwen", 12, "logs/vllm_p12_qwen2.5-7b.log", "vllm"),
    Session("vllm_p12_phi", 12, "logs/vllm_p12_phi3.5.log", "vllm"),
    Session("vllm_p12_judge", 12, "logs/vllm_p12_judge.log", "vllm", "stopped by the token cap"),
    Session("vllm_p12_judge2", 12, "logs/vllm_p12_judge2.log", "vllm", "resumed judge"),
    # Phase 12 row 12.4 seed run (logs/p12_seeds.status); pending until its logs are complete
    Session("p12s_train_s1", 12, "logs/p12_train_sft3bv2_s1.log", "pylog"),
    Session("p12s_train_s2", 12, "logs/p12_train_sft3bv2_s2.log", "pylog"),
    Session("vllm_p12s_qwen", 12, "logs/vllm_p12s_qwen2.5-7b.log", "vllm"),
    Session("vllm_p12s_phi", 12, "logs/vllm_p12s_phi3.5.log", "vllm"),
    Session("vllm_p12s_judge", 12, "logs/vllm_p12s_judge.log", "vllm"),
)

RW = "data/interim/rewrites"
# fmt: off
STAGES: tuple[Stage, ...] = (
    # ---- Phase 1: pilot labels (row 1.3.6), MLflow runs e71803a6 / 6b1831d5 / 508e0d17
    Stage(
        "p1-pilot-respond", 1, "1.3.6", "pilot target outputs", model=M7,
        window=("2026-10-01T06:40:00", "2026-10-01T06:46:00"),
        active=_mlflow(["e71803a6"], "wall_seconds"),
        reported_tokens=_mlflow(["e71803a6"], "generated_tokens"),
        labels={"kind": "const", "value": 836, "source": "MLflow e71803a6 n_items"},
        group="p1", note="no server log kept",
    ),
    Stage(
        "p1-pilot-judge", 1, "1.3.6", "pilot judge (32B)", model=M32,
        window=("2026-10-01T06:46:00", "2026-10-01T06:56:00"),
        active=_mlflow(["6b1831d5", "508e0d17"], "wall_seconds"),
        reported_tokens=_mlflow(["6b1831d5", "508e0d17"], "generated_tokens"),
        group="p1", note="no server log kept",
    ),
    # ---- Phase 2
    Stage(
        "p2-judgebias-14b", 2, "2.3.3", "judge-bias study, 14B pairwise + pointwise", model=M14,
        window="@vllm_14b", active=_mlflow(["d8b9796c", "c96954c4"], "wall_seconds"),
        reported_tokens=_mlflow(["d8b9796c", "c96954c4"], "generated_tokens"),
        sessions=("vllm_14b",),
    ),
    Stage(
        "p2-judgebias-32b", 2, "2.3.3", "judge-bias study, 32B pairwise + pointwise", model=M32,
        window=("2026-10-01T07:00:00", "2026-10-01T07:19:00"),
        active=_mlflow(["09d66708", "3494f098"], "wall_seconds"),
        reported_tokens=_mlflow(["09d66708", "3494f098"], "generated_tokens"),
        note="server log overwritten by the next 32B start",
    ),
    Stage(
        "p2-casualise", 2, "2.1.4", "casualise 1,680 clean prompts x 5 styles (32B)", model=M32,
        window=("2026-10-01T07:24:41", "2026-10-01T08:30:00"),
        active=_s("logs/dvc_casualise.log"),
        reported_tokens=_s("logs/dvc_casualise.log", "generated_tokens"),
        sessions=("vllm_32b",), group="p2",
    ),
    Stage(
        "p2-candidates", 2, "2.2.1", "LLM rewrite candidates low/medium/high (32B)", model=M32,
        window=("2026-10-01T15:50:00", "2026-10-01T17:25:00"),
        active={"kind": "log_span",
                "paths": ["logs/p2_candidates.log", "logs/p2_candidates_2.log"]},
        sessions=("vllm_32b",), group="p2",
        note="two runs (low; medium + high); spans include CPU operator/LLMLingua steps",
    ),
    Stage(
        "p2-respond-qwen", 2, "2.2.3", "target outputs, Qwen2.5-7B", model=M7, window="@vllm_7b",
        active=_s("logs/p2_respond_qwen7b.log"),
        reported_tokens=_s("logs/p2_respond_qwen7b.log", "generated_tokens"),
        sessions=("vllm_7b",), group="p2",
    ),
    Stage(
        "p2-respond-phi", 2, "2.2.3", "target outputs, Phi-3.5-mini", model=PHI,
        window="@vllm_phi35", active=_s("logs/p2_respond_phi35.log"),
        reported_tokens=_s("logs/p2_respond_phi35.log", "generated_tokens"),
        sessions=("vllm_phi35",), group="p2",
    ),
    Stage(
        "p2-judge-smoke", 2, "2.3.3", "label-stage smoke on the full data (32B)", model=M32,
        window=("2026-10-02T15:00:00", "2026-10-02T16:10:00"),
        active={"kind": "shell_real", "path": "logs/p2_label_smoke.log"},
        group="p2", attempt="retry", note="server log not kept",
    ),
    Stage(
        "p2-judge-attempt1", 2, "2.3.3", "first full-judge attempt (32B), stopped", model=M32,
        window="@vllm_32b_judge", active={"kind": "ledger_busy"},
        sessions=("vllm_32b_judge",), group="p2", attempt="retry",
        note="no summary (logs/p2_label.log empty); its calls became cache hits of the final run",
    ),
    Stage(
        "p2-judge", 2, "2.3.3", "position-swapped pairwise judge, final run (32B)", model=M32,
        window="@judge32", active=_s("logs/judge_full.log"),
        reported_tokens=_s("logs/judge_full.log", "generated_tokens"),
        labels={"kind": "summary", "path": "logs/judge_full.log", "key": "records"},
        sessions=("judge32",), group="p2",
    ),
    Stage(
        "p2-selfpref", 2, "2.2.5", "second judge Phi-3.5 on 1,000 pairs", model=PHI,
        window="@judge_phi", active=_s("logs/selfpref.log"),
        reported_tokens=_s("logs/selfpref.log", "generated_tokens"),
        labels={"kind": "summary", "path": "logs/selfpref.log", "key": "records"},
        sessions=("judge_phi",),
    ),
    Stage(
        "p2-smoke-rtx4090", 2, "2.2.x", "20-prompt hardware smoke on worker-1", hardware="rtx4090",
        active={"kind": "md_sum", "path": "reports/p2_gpu_smoke.md", "header": "Stage",
                "column": "Stage seconds", "exclude": ["Token counts (CPU)"]},
        reported_tokens={"kind": "md_sum", "path": "reports/p2_gpu_smoke.md", "header": "Stage",
                         "column": "Generated tokens", "exclude": ["Token counts (CPU)"]},
        tokens="reported",
        note="reserved = 'Server-reserved GPU-hours' column of the same report",
    ),
    # ---- Phase 7: C7 benchmarks (rows 7.5-7.7); MLflow records tokens, not GPU time
    Stage(
        "p7-benchmarks", 7, "7.5-7.7", "vLLM smoke, serving and training benchmarks",
        active={"kind": "none"},
        reported_tokens={"kind": "mlflow_phase", "phase": "7", "metric": "generated_tokens"},
        tokens="reported", note="GPU time not recorded (runs logged after the fact)",
    ),
    # ---- Phase 8: rewriter rungs, validation, target augmentation
    Stage("p8-sft-0.5b", 8, "8.3.3", "SFT Qwen2.5-0.5B full FT, 3,000 steps",
          active={"kind": "log_span", "paths": ["logs/sft-qwen0.5b-full.log"]}, tokens="none",
          sessions=("sft05",)),
    Stage("p8-sft-1.5b", 8, "8.3.4", "SFT Qwen2.5-1.5B LoRA, 3,000 steps",
          active={"kind": "log_span", "paths": ["logs/sft-qwen1.5b-lora.log"]}, tokens="none",
          sessions=("sft15",)),
    Stage("p8-sft-3b", 8, "8.3.4", "SFT Qwen2.5-3B LoRA, 1,000 steps",
          active={"kind": "log_span", "paths": ["logs/sft-qwen3b.log"]}, tokens="none",
          sessions=("sft3b",)),
    Stage("p8-gen-0.5b", 8, "8.3.3", "val rewrites, 285 prompts (0.5B)",
          active={"kind": "latency_sum", "path": f"{RW}/sft-qwen0.5b-full.val.jsonl"},
          tokens="tokenize", reported_tokens={"kind": "tokenize",
                                              "path": f"{RW}/sft-qwen0.5b-full.val.jsonl"}),
    Stage("p8-gen-1.5b", 8, "8.3.4", "val rewrites, 285 prompts (1.5B)",
          active={"kind": "latency_sum", "path": f"{RW}/sft-qwen1.5b-lora.val.jsonl"},
          tokens="tokenize", reported_tokens={"kind": "tokenize",
                                              "path": f"{RW}/sft-qwen1.5b-lora.val.jsonl"}),
    Stage("p8-gen-3b", 8, "8.3.4", "val rewrites, 285 prompts (3B)",
          active={"kind": "latency_sum", "path": f"{RW}/sft-qwen3b-lora.val.jsonl"},
          tokens="tokenize", reported_tokens={"kind": "tokenize",
                                              "path": f"{RW}/sft-qwen3b-lora.val.jsonl"}),
    Stage("p8-eval-qwen", 8, "8.3.3-8.3.4", "val target outputs, rungs 1-2 (Qwen2.5-7B)",
          model=M7, window="@vllm_eval_qwen", active=_s("logs/eval_resp_qwen.log"),
          reported_tokens=_s("logs/eval_resp_qwen.log", "generated_tokens"),
          sessions=("vllm_eval_qwen", "vllm_eval_qwen1", "vllm_eval_qwen2"), group="p8-eval"),
    Stage("p8-eval-phi", 8, "8.3.3-8.3.4", "val target outputs, rungs 1-2 (Phi-3.5)",
          model=PHI, window="@vllm_eval_phi", active=_s("logs/eval_resp_phi.log"),
          reported_tokens=_s("logs/eval_resp_phi.log", "generated_tokens"),
          sessions=("vllm_eval_phi",), group="p8-eval"),
    Stage("p8-eval-judge", 8, "8.3.3-8.3.4", "val judge, rungs 1-2 (32B)", model=M32,
          window="@vllm_eval_judge", active=_s("logs/eval_judge.log"),
          reported_tokens=_s("logs/eval_judge.log", "generated_tokens"),
          labels={"kind": "summary", "path": "logs/eval_judge.log", "key": "records"},
          sessions=("vllm_eval_judge",), group="p8-eval"),
    Stage("p8-r3-qwen", 8, "8.3.4", "val target outputs, rung 3 (Qwen2.5-7B)", model=M7,
          window="@vllm_r3_qwen", active=_s("logs/r3_resp_qwen.log"),
          reported_tokens=_s("logs/r3_resp_qwen.log", "generated_tokens"),
          sessions=("vllm_r3_qwen",), group="p8-r3"),
    Stage("p8-r3-phi", 8, "8.3.4", "val target outputs, rung 3 (Phi-3.5)", model=PHI,
          window="@vllm_r3_phi", active=_s("logs/r3_resp_phi.log"),
          reported_tokens=_s("logs/r3_resp_phi.log", "generated_tokens"),
          sessions=("vllm_r3_phi",), group="p8-r3"),
    Stage("p8-r3-judge", 8, "8.3.4", "val judge, rung 3 (32B), incl. interrupted first try",
          model=M32, window=("2026-10-17T02:49:00", "2026-10-17T02:59:00"),
          active={"kind": "ledger_busy"},
          reported_tokens=_s("logs/r3_judge.log", "generated_tokens"),
          labels={"kind": "summary", "path": "logs/r3_judge.log", "key": "records"},
          sessions=("vllm_r3_judge",), group="p8-r3",
          note="summary wall_seconds (16 s) covers only the resumed run"),
    Stage("p8-aug-candidates", 8, "8.3.1", "llm_max / llm_min target augmentation (32B)",
          model=M32, window="@vllm_aug_gen", active=_s("logs/aug_cand.log"),
          sessions=("vllm_aug_gen",), group="p8-aug"),
    Stage("p8-aug-qwen", 8, "8.3.1", "augmentation target outputs (Qwen2.5-7B)", model=M7,
          window="@vllm_aug_qwen", active=_s("logs/aug_resp_qwen.log"),
          reported_tokens=_s("logs/aug_resp_qwen.log", "generated_tokens"),
          sessions=("vllm_aug_qwen",), group="p8-aug"),
    Stage("p8-aug-phi", 8, "8.3.1", "augmentation target outputs (Phi-3.5)", model=PHI,
          window="@vllm_aug_phi", active=_s("logs/aug_resp_phi.log"),
          reported_tokens=_s("logs/aug_resp_phi.log", "generated_tokens"),
          sessions=("vllm_aug_phi",), group="p8-aug"),
    Stage("p8-aug-judge", 8, "8.3.1", "augmentation judge (32B)", model=M32,
          window="@vllm_aug_judge", active=_s("logs/aug_judge.log"),
          reported_tokens=_s("logs/aug_judge.log", "generated_tokens"),
          labels={"kind": "summary", "path": "logs/aug_judge.log", "key": "records"},
          sessions=("vllm_aug_judge",), group="p8-aug",
          note="restarted at 05:37; wall_seconds and summary tokens cover the restarted run"),
    # ---- Phase 8 v2: retrained rungs on sft_v2 data (logs/v2_chain.status, q7_chain.status)
    Stage("p8-sft-1.5b-v2", 8, "8.3.4", "SFT Qwen2.5-1.5B LoRA v2, 3,000 steps",
          active={"kind": "log_span", "paths": ["logs/sft_qwen15b_lora_v2.log"]}, tokens="none",
          sessions=("sft15v2",)),
    Stage("p8-sft-3b-v2", 8, "8.3.4", "SFT Qwen2.5-3B LoRA v2, 1,000 steps",
          active={"kind": "log_span", "paths": ["logs/sft_qwen3b_lora_v2.log"]}, tokens="none",
          sessions=("sft3bv2",)),
    Stage("p8-sft-7b-v2", 8, "8.3.5", "SFT Qwen2.5-7B QLoRA v2, 1,000 steps",
          active={"kind": "log_span", "paths": ["logs/sft_qwen7b_qlora_v2.log"]}, tokens="none",
          sessions=("sft7bv2",)),
    Stage("p8-gen-1.5b-v2", 8, "8.3.4", "val rewrites, 285 prompts (1.5B v2)",
          active={"kind": "latency_sum", "path": f"{RW}/sft15v2.val.jsonl"}, tokens="tokenize",
          reported_tokens={"kind": "tokenize", "path": f"{RW}/sft15v2.val.jsonl"}),
    Stage("p8-gen-3b-v2", 8, "8.3.4", "val rewrites, 285 prompts (3B v2)",
          active={"kind": "latency_sum", "path": f"{RW}/sft3bv2.val.jsonl"}, tokens="tokenize",
          reported_tokens={"kind": "tokenize", "path": f"{RW}/sft3bv2.val.jsonl"}),
    Stage("p8-gen-7b-v2", 8, "8.3.5", "val rewrites, 285 prompts (7B QLoRA v2)",
          active={"kind": "latency_sum", "path": f"{RW}/sft7bv2.val.jsonl"}, tokens="tokenize",
          reported_tokens={"kind": "tokenize", "path": f"{RW}/sft7bv2.val.jsonl"}),
    Stage("p8-v2-qwen", 8, "8.3.4", "val target outputs, v2 rungs (Qwen2.5-7B)", model=M7,
          window="@vllm_v2_qwen", active=_s("logs/v2_resp_qwen.log"),
          reported_tokens=_s("logs/v2_resp_qwen.log", "generated_tokens"),
          sessions=("vllm_v2_qwen",), group="p8-v2"),
    Stage("p8-v2-phi", 8, "8.3.4", "val target outputs, v2 rungs (Phi-3.5)", model=PHI,
          window="@vllm_v2_phi", active=_s("logs/v2_resp_phi.log"),
          reported_tokens=_s("logs/v2_resp_phi.log", "generated_tokens"),
          sessions=("vllm_v2_phi",), group="p8-v2"),
    Stage("p8-v2-judge", 8, "8.3.4", "val judge, v2 rungs (32B)", model=M32,
          window="@vllm_v2_judge", active=_s("logs/v2_judge.log"),
          reported_tokens=_s("logs/v2_judge.log", "generated_tokens"),
          labels={"kind": "summary", "path": "logs/v2_judge.log", "key": "records"},
          sessions=("vllm_v2_judge",), group="p8-v2"),
    *(
        Stage(f"p8-q7-{s}-{t}", 8, row, f"val target outputs, {what} ({tn})", model=m,
              window=f"@vllm_q7_{t}" if s == "7b" else None,
              active=_s(f"logs/q7_resp_{s}_{lg}.log"), tokens="reported",
              reported_tokens=_s(f"logs/q7_resp_{s}_{lg}.log", "generated_tokens"),
              sessions=(f"vllm_q7_{t}",), group="p8-q7",
              note="one server per target ran the 7b, hmm and rl sets; its ledger window "
              "(all three sets) is on the 7b stage" if s == "7b" else "shares the 7b server")
        for s, row, what in (("7b", "8.3.5", "7B v2 rung"), ("hmm", "C5.6", "HMM rewriter set"),
                             ("rl", "C9.2", "RL (Q-learning) operator set"))
        for t, m, tn, lg in (("qwen", M7, "Qwen2.5-7B", "qwen2.5-7b"),
                             ("phi", PHI, "Phi-3.5", "phi3.5"))
    ),
    *(
        Stage(f"p8-q7-judge-{s}", 8, row, f"val judge, {what} (32B)", model=M32,
              window=win, active=_s(f"logs/q7_judge_{s}.log"),
              reported_tokens=_s(f"logs/q7_judge_{s}.log", "generated_tokens"),
              labels={"kind": "summary", "path": f"logs/q7_judge_{s}.log", "key": "records"},
              sessions=("vllm_q7_judge",), group="p8-q7")
        for s, row, what, win in (
            ("7b", "8.3.5", "7B v2 rung", ("2026-10-17T18:16:00", "2026-10-17T18:24:30")),
            ("hmm", "C5.6", "HMM rewriter set", ("2026-10-17T18:24:30", "2026-10-17T18:48:30")),
            ("rl", "C9.2", "RL operator set", ("2026-10-17T18:48:30", "2026-10-17T18:58:20")),
        )
    ),
    # ---- Phase 10: rewriter HPO (logs/hpo_chain.status; Optuna study p10_rewriter_hpo)
    Stage("p10-hpo-trials", 10, "10.x", "Optuna LoRA HPO, 12 trials (5 complete, 7 pruned)",
          active={"kind": "optuna_sum", "path": "experiments/p10_rewriter_hpo/optuna.db"},
          tokens="none", sessions=("hpo_trials",),
          note="val rewrites of hpo00/02/08 generated inside the trials"),
    Stage("p10-hpo-qwen", 10, "10.x", "val target outputs, HPO candidates (Qwen2.5-7B)",
          model=M7, window="@vllm_hpo_qwen", active=_s("logs/hpo_resp_qwen2.5-7b.log"),
          reported_tokens=_s("logs/hpo_resp_qwen2.5-7b.log", "generated_tokens"),
          sessions=("vllm_hpo_qwen",), group="p10-hpo"),
    Stage("p10-hpo-phi", 10, "10.x", "val target outputs, HPO candidates (Phi-3.5)", model=PHI,
          window="@vllm_hpo_phi", active=_s("logs/hpo_resp_phi3.5.log"),
          reported_tokens=_s("logs/hpo_resp_phi3.5.log", "generated_tokens"),
          sessions=("vllm_hpo_phi",), group="p10-hpo"),
    Stage("p10-hpo-judge", 10, "10.x", "val judge, HPO candidates (32B)", model=M32,
          window="@vllm_hpo_judge", active=_s("logs/hpo_judge.log"),
          reported_tokens=_s("logs/hpo_judge.log", "generated_tokens"),
          labels={"kind": "summary", "path": "logs/hpo_judge.log", "key": "records"},
          sessions=("vllm_hpo_judge",), group="p10-hpo"),
    # ---- Phase 11: quality path (vLLM-served rewriter) benchmarks and the canary
    *(
        Stage(f"p11-{k}-{r}", 11, row, f"{what}, rewriter {r}", tokens="none",
              active={"kind": "session_span", "session": f"vllm_{k}_{r}"},
              sessions=(f"vllm_{k}_{r}",),
              note="no stage timer; active = server span (rewriter calls not ledgered)")
        for k, row, what in (("qb", "11.x", "quality-path load benchmark (C11.3/C11.4)"),
                             ("qraw", "11.x", "raw quality-path benchmark"))
        for r in ("3b", "15v2")
    ),
    Stage("p11-canary-rewrite", 11, "11.x", "canary, 400 prompts via the quality path",
          active=_s("logs/canary_rewrite.log", "rewrite.wall_seconds"), tokens="none",
          sessions=("vllm_canary_rw",), group="p11-canary"),
    Stage("p11-canary-qwen", 11, "11.x", "canary target outputs (Qwen2.5-7B)", model=M7,
          active=_s("logs/canary_respond_qwen2.5-7b.log"), tokens="reported",
          reported_tokens=_s("logs/canary_respond_qwen2.5-7b.log", "generated_tokens"),
          sessions=("vllm_canary_qwen",), group="p11-canary", note="not ledgered"),
    Stage("p11-canary-phi", 11, "11.x", "canary target outputs (Phi-3.5)", model=PHI,
          active=_s("logs/canary_respond_phi3.5-mini.log"), tokens="reported",
          reported_tokens=_s("logs/canary_respond_phi3.5-mini.log", "generated_tokens"),
          sessions=("vllm_canary_phi",), group="p11-canary", note="not ledgered"),
    Stage("p11-canary-judge", 11, "11.x", "canary judge (32B)", model=M32,
          active=_s("logs/canary_judge.log"), tokens="reported",
          reported_tokens=_s("logs/canary_judge.log", "generated_tokens"),
          labels={"kind": "summary", "path": "logs/canary_judge.log", "key": "records"},
          sessions=("vllm_canary_judge",), group="p11-canary", note="not ledgered"),
    # ---- Phase 12: test split (logs/p12_chain.status). The ledger charged respond and the first
    # judge run to bucket "phase2" and the resumed judge to "phase12"; attribution is by model and
    # time window, so all of it lands here.
    *(
        Stage(f"p12-gen-{tag}", 12, "12.x", f"test rewrites, 1,992 prompts ({tag})",
              active={"kind": "latency_sum", "path": f"data/interim/final/test_{tag}.jsonl"},
              tokens="tokenize",
              reported_tokens={"kind": "tokenize", "path": f"data/interim/final/test_{tag}.jsonl"})
        for tag in ("sft3bv2", "sft15v2", "sfthpo")
    ),
    Stage("p12-zeroshot-32b", 12, "12.x", "zero-shot 32B candidates low/medium/high, 1,992 prompts",
          model=M32, window="@vllm_p12_zs", active=_s("logs/p12_llm.log"),
          sessions=("vllm_p12_zs",)),
    Stage("p12-resp-qwen", 12, "12.x", "test target outputs (Qwen2.5-7B)", model=M7,
          window="@vllm_p12_qwen", active=_s("logs/p12_resp_qwen2.5-7b.log"),
          reported_tokens=_s("logs/p12_resp_qwen2.5-7b.log", "generated_tokens"),
          sessions=("vllm_p12_qwen",), group="p12", note="ledger bucket phase2"),
    Stage("p12-resp-phi", 12, "12.x", "test target outputs (Phi-3.5)", model=PHI,
          window="@vllm_p12_phi", active=_s("logs/p12_resp_phi3.5.log"),
          reported_tokens=_s("logs/p12_resp_phi3.5.log", "generated_tokens"),
          sessions=("vllm_p12_phi",), group="p12", note="ledger bucket phase2"),
    Stage("p12-judge-run1", 12, "12.x", "test judge (32B), stopped by the phase2 token cap",
          model=M32, window="@vllm_p12_judge", active={"kind": "ledger_busy"},
          sessions=("vllm_p12_judge",), group="p12", attempt="retry",
          note="BudgetExceededError (logs/p12_judge.log); ledger bucket phase2; its calls are "
          "cache hits of the resumed run"),
    Stage("p12-judge", 12, "12.x", "test judge (32B), resumed run", model=M32,
          window="@vllm_p12_judge2", active=_s("logs/p12_judge2.log"),
          reported_tokens=_s("logs/p12_judge2.log", "generated_tokens"),
          labels={"kind": "summary", "path": "logs/p12_judge2.log", "key": "records"},
          sessions=("vllm_p12_judge2",), group="p12", note="ledger bucket phase12"),
    # ---- Phase 12 row 12.4 seeds (/home/temp/p12_seeds.sh, logs/p12_seeds.status)
    *(
        st
        for k in (1, 2)
        for st in (
            Stage(f"p12-seed-train-s{k}", 12, "12.4", f"SFT Qwen2.5-3B LoRA v2, seed {k}",
                  active={"kind": "log_span", "paths": [f"logs/p12_train_sft3bv2_s{k}.log"]},
                  tokens="none", sessions=(f"p12s_train_s{k}",), pending_since=LEDGER_CUTOFF,
                  done=(f"logs/p12_train_sft3bv2_s{k}.log", r"\]\[__main__\]\[INFO\] - trained ")),
            Stage(f"p12-seed-gen-s{k}", 12, "12.4", f"test rewrites, 1,992 prompts (seed {k})",
                  active={"kind": "latency_sum",
                          "path": f"data/interim/final/test_sft3bv2_s{k}.jsonl"},
                  tokens="tokenize",
                  reported_tokens={"kind": "tokenize",
                                   "path": f"data/interim/final/test_sft3bv2_s{k}.jsonl"},
                  pending_since=LEDGER_CUTOFF,
                  done=("logs/p12_seeds.status", rf"^generate sft3bv2_s{k} rc=0")),
        )
    ),
    Stage("p12-seed-resp-qwen", 12, "12.4", "seed test target outputs (Qwen2.5-7B)", model=M7,
          window="@vllm_p12s_qwen", active=_s("logs/p12s_resp_qwen2.5-7b.log"),
          reported_tokens=_s("logs/p12s_resp_qwen2.5-7b.log", "generated_tokens"),
          sessions=("vllm_p12s_qwen",), pending_since=LEDGER_CUTOFF),
    Stage("p12-seed-resp-phi", 12, "12.4", "seed test target outputs (Phi-3.5)", model=PHI,
          window="@vllm_p12s_phi", active=_s("logs/p12s_resp_phi3.5.log"),
          reported_tokens=_s("logs/p12s_resp_phi3.5.log", "generated_tokens"),
          sessions=("vllm_p12s_phi",), pending_since=LEDGER_CUTOFF),
    Stage("p12-seed-judge", 12, "12.4", "seed test judge (32B)", model=M32,
          window="@vllm_p12s_judge", active=_s("logs/p12s_judge.log"),
          reported_tokens=_s("logs/p12s_judge.log", "generated_tokens"),
          sessions=("vllm_p12s_judge",), pending_since=LEDGER_CUTOFF,
          note="summary records = whole test_labels.jsonl, so no label count"),
)
# fmt: on

CPU_PHASES: dict[int, str] = {
    0: "scaffolding, CI",
    3: "classical models (Compute Allocation: CPU)",
    4: "perceptron / MLP studies (CPU)",
    5: "ensembles, NB, GMM, HMM (CPU; its HMM rewriter C5.6 val labels ran on the q7 chain, "
    "counted in Phase 8)",
    6: "clustering (CPU)",
    9: "tabular Q-learning (CPU; its C9.2 val labels ran on the q7 chain, counted in Phase 8)",
}
# GPU phases that also had CPU-only work (no GPU time to count).
CPU_PARTS: dict[int, str] = {
    10: "GBM / MLP HPO studies (MLflow hardware tag cpu)",
    11: "fast path, ONNX serving (MLflow hardware tag cpu)",
    12: "CPU candidates (logs/p12_cpu_candidates*.log) and results",
}

# ---------------------------------------------------------------- parsing helpers

_PY_TS = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+\]")
_PY_TS2 = re.compile(r"^(\d{4}/\d\d/\d\d \d\d:\d\d:\d\d) ")
_VLLM_TS = re.compile(r"(?:INFO|WARNING|ERROR|DEBUG) (\d\d-\d\d \d\d:\d\d:\d\d)")
_REAL = re.compile(r"^real\s+(?:(\d+)m)?([\d.]+)s", re.M)


def _utc(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=UTC)


def json_blobs(text: str) -> list[dict[str, Any]]:
    """Top-level JSON objects printed one key per line (``{`` and ``}`` at column 0)."""
    blobs: list[dict[str, Any]] = []
    buf: list[str] | None = None
    for line in text.splitlines():
        if buf is None and line.startswith("{") and line.rstrip().endswith("}") and line != "{":
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                blobs.append(obj)
        elif line == "{":
            buf = [line]
        elif buf is not None:
            buf.append(line)
            if line == "}":
                try:
                    blobs.append(json.loads("\n".join(buf)))
                except json.JSONDecodeError:
                    pass
                buf = None
    return blobs


def log_span(lines: Iterable[str]) -> tuple[datetime, datetime] | None:
    """First and last Python-logging / MLflow timestamp of a log."""
    first = last = None
    for line in lines:
        m = _PY_TS.match(line)
        t = None
        if m:
            t = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
        else:
            m2 = _PY_TS2.match(line)
            if m2:
                t = datetime.strptime(m2.group(1), "%Y/%m/%d %H:%M:%S")
        if t is not None:
            t = t.replace(tzinfo=UTC)
            first = first or t
            last = t
    return (first, last) if first and last else None


def vllm_span(lines: Iterable[str], year: int = LOG_YEAR) -> tuple[datetime, datetime] | None:
    """First and last timestamped line of a vLLM server log (``MM-DD hh:mm:ss``)."""
    first = last = None
    for line in lines:
        m = _VLLM_TS.search(line)
        if m:
            t = datetime.strptime(f"{year}-{m.group(1)}", "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC)
            first = first or t
            last = t
    return (first, last) if first and last else None


def busy_seconds(times: Sequence[datetime], gap_s: float = BUSY_GAP_S) -> float:
    """Summed span of bursts of call timestamps; a gap longer than ``gap_s`` splits bursts."""
    if not times:
        return 0.0
    ts = sorted(times)
    total, start, prev = 0.0, ts[0], ts[0]
    for t in ts[1:]:
        if (t - prev).total_seconds() > gap_s:
            total += (prev - start).total_seconds()
            start = t
        prev = t
    return total + (prev - start).total_seconds()


def md_table(text: str, header: str) -> list[dict[str, str]]:
    """Rows of the first Markdown table whose first header cell is ``header``."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if line.startswith("|") and cells and cells[0] == header:
            rows = []
            for row in lines[i + 2 :]:
                if not row.startswith("|"):
                    break
                vals = [c.strip() for c in row.strip().strip("|").split("|")]
                rows.append(dict(zip(cells, vals, strict=False)))
            return rows
    return []


def _optuna_trials(p: Path) -> list[tuple[datetime, datetime]]:
    """(start, complete) of every finished trial in an Optuna SQLite study (naive UTC times)."""
    if not p.exists():
        return []
    db = sqlite3.connect(f"file:{p.as_posix()}?mode=ro", uri=True)
    try:
        rows = db.execute(
            "select datetime_start, datetime_complete from trials "
            "where datetime_start is not null and datetime_complete is not null"
        ).fetchall()
    finally:
        db.close()
    return [
        (
            datetime.fromisoformat(a).replace(tzinfo=UTC),
            datetime.fromisoformat(b).replace(tzinfo=UTC),
        )
        for a, b in rows
    ]


def _num(s: str) -> float:
    return float(s.replace(",", "").replace("`", ""))


# ---------------------------------------------------------------- evidence store


class Evidence:
    """Lazy, cached access to the source files under ``root``."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._ledger: list[tuple[datetime, str, int, int]] | None = None
        self._ledger_meta: dict[str, Any] = {}
        self._db: sqlite3.Connection | None = None
        self._spans: dict[str, tuple[datetime, datetime] | None] = {}

    def path(self, rel: str) -> Path:
        return self.root / rel

    def text(self, rel: str) -> str | None:
        p = self.path(rel)
        return p.read_text(encoding="utf-8", errors="replace") if p.exists() else None

    @property
    def ledger(self) -> list[tuple[datetime, str, int, int]]:
        if self._ledger is None:
            p = self.path(".cache/llm_ledger.json")
            rows: list[tuple[datetime, str, int, int]] = []
            if p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
                for name, ph in data.get("phases", {}).items():
                    self._ledger_meta[name] = {"calls": len(ph["calls"]), "tokens": ph["tokens"]}
                    for e in ph["calls"]:
                        rows.append(
                            (
                                datetime.fromisoformat(e["timestamp"]),
                                e["model"],
                                int(e.get("input_tokens") or 0),
                                int(e.get("output_tokens") or 0),
                            )
                        )
            self._ledger = sorted(rows)
        return self._ledger

    def ledger_meta(self) -> dict[str, Any]:
        _ = self.ledger
        return self._ledger_meta

    def db(self) -> sqlite3.Connection | None:
        if self._db is None:
            p = self.path("mlruns/mlflow.db")
            if not p.exists():
                return None
            self._db = sqlite3.connect(f"file:{p.as_posix()}?mode=ro", uri=True)
        return self._db

    def mlflow_metric(self, prefixes: Sequence[str], metric: str) -> float | None:
        db = self.db()
        if db is None:
            return None
        total, found = 0.0, False
        for pre in prefixes:
            row = db.execute(
                "select value from latest_metrics where key=? and run_uuid like ?",
                (metric, f"{pre}%"),
            ).fetchone()
            if row is not None:
                total += float(row[0])
                found = True
        return total if found else None

    def mlflow_phase_metric(self, phase: str, metric: str) -> tuple[float | None, int]:
        db = self.db()
        if db is None:
            return None, 0
        rows = db.execute(
            "select m.value from latest_metrics m join tags t on t.run_uuid=m.run_uuid "
            "where m.key=? and t.key='phase' and t.value=?",
            (metric, phase),
        ).fetchall()
        return (sum(float(r[0]) for r in rows) if rows else None), len(rows)

    def span(self, session: Session) -> tuple[datetime, datetime] | None:
        if session.id not in self._spans and session.kind == "optuna":
            rows = _optuna_trials(self.path(session.log))
            self._spans[session.id] = (
                (min(a for a, _ in rows), max(b for _, b in rows)) if rows else None
            )
        if session.id not in self._spans:
            txt = self.text(session.log)
            if txt is None:
                self._spans[session.id] = None
            else:
                lines = txt.splitlines()
                fn = vllm_span if session.kind == "vllm" else log_span
                self._spans[session.id] = fn(lines)
        return self._spans[session.id]


# ---------------------------------------------------------------- per-stage resolution

_SESSIONS = {s.id: s for s in SESSIONS}


def _window(stage: Stage, ev: Evidence) -> tuple[datetime, datetime, str] | None:
    w = stage.window
    if w is None:
        return None
    if isinstance(w, str):
        sess = _SESSIONS[w.lstrip("@")]
        sp = ev.span(sess)
        if sp is None:
            return None
        return sp[0], sp[1] + timedelta(seconds=LEDGER_GRACE_S), f"{sess.log} span + 30 s"
    return _utc(w[0]), _utc(w[1]), "fixed window (see STAGES)"


def _value(src: dict[str, Any] | None, ev: Evidence, stage: Stage) -> tuple[float | None, str]:
    """Resolve one source spec to (value, human-readable source)."""
    if not src:
        return None, ""
    kind = src["kind"]
    if kind == "summary":
        txt = ev.text(src["path"])
        if txt is None:
            return None, f"{src['path']} (missing)"
        for blob in reversed(json_blobs(txt)):
            val: Any = blob
            for part in src["key"].split("."):
                val = val.get(part) if isinstance(val, dict) else None
            if isinstance(val, int | float):
                return float(val), f"{src['path']}: {src['key']}"
        return None, f"{src['path']}: no {src['key']}"
    if kind == "optuna_sum":
        trials = _optuna_trials(ev.path(src["path"]))
        if not trials:
            return None, f"{src['path']} (missing)"
        total = sum((b - a).total_seconds() for a, b in trials)
        return total, f"{src['path']}: sum of trial datetime_complete - datetime_start"
    if kind == "session_span":
        sess = _SESSIONS[src["session"]]
        sp = ev.span(sess)
        if sp is None:
            return None, f"{sess.log} (missing)"
        return (sp[1] - sp[0]).total_seconds(), f"{sess.log}: server span"
    if kind == "mlflow":
        v = ev.mlflow_metric(src["runs"], src["metric"])
        return v, f"mlruns/mlflow.db runs {', '.join(src['runs'])}: {src['metric']}"
    if kind == "mlflow_phase":
        v, n = ev.mlflow_phase_metric(src["phase"], src["metric"])
        return v, f"mlruns/mlflow.db phase={src['phase']} ({n} runs): {src['metric']}"
    if kind == "log_span":
        total, parts = 0.0, []
        for rel in src["paths"]:
            txt = ev.text(rel)
            sp = log_span(txt.splitlines()) if txt else None
            if sp is None:
                return None, f"{rel} (missing or no timestamps)"
            total += (sp[1] - sp[0]).total_seconds()
            parts.append(rel)
        return total, "first-last timestamp of " + " + ".join(parts)
    if kind == "shell_real":
        txt = ev.text(src["path"])
        m = _REAL.search(txt or "")
        if not m:
            return None, f"{src['path']}: no `real` line"
        return float(m.group(1) or 0) * 60 + float(m.group(2)), f"{src['path']}: shell `real`"
    if kind == "latency_sum":
        p = ev.path(src["path"])
        if not p.exists():
            return None, f"{src['path']} (missing)"
        rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
        return sum(float(r["latency_s"]) for r in rows), f"{src['path']}: sum latency_s"
    if kind == "tokenize":
        p = ev.path(src["path"])
        if not p.exists():
            return None, f"{src['path']} (missing)"
        from terser.tokenizers.count import count_tokens

        rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
        n = sum(count_tokens(r.get("raw_rewrite") or "", "qwen2.5") for r in rows)
        return float(n), f"{src['path']}: Qwen2.5 tokens of raw_rewrite"
    if kind == "md_sum":
        txt = ev.text(src["path"])
        rows = md_table(txt or "", src["header"])
        vals = [_num(r[src["column"]]) for r in rows if r[src["header"]] not in src["exclude"]]
        if not vals:
            return None, f"{src['path']} (table missing)"
        return sum(vals), f"{src['path']}: sum of '{src['column']}'"
    if kind == "const":
        return float(src["value"]), str(src["source"])
    if kind == "ledger_busy":
        win = _window(stage, ev)
        if win is None or stage.model is None:
            return None, "ledger (no window)"
        ts = [c[0] for c in ev.ledger if win[0] <= c[0] <= win[1] and c[1] == stage.model]
        return busy_seconds(ts), f".cache/llm_ledger.json busy span ({win[2]})"
    if kind == "none":
        return None, "not recorded"
    raise ValueError(f"unknown source kind {kind}")


def _ledger_window(stage: Stage, ev: Evidence) -> dict[str, Any]:
    win = _window(stage, ev)
    if win is None or stage.model is None:
        return {}
    sel = [c for c in ev.ledger if win[0] <= c[0] <= win[1] and c[1] == stage.model]
    return {
        "calls": len(sel),
        "input_tokens": sum(c[2] for c in sel),
        "output_tokens": sum(c[3] for c in sel),
        "busy_s": busy_seconds([c[0] for c in sel]),
        "window": [win[0].isoformat(), win[1].isoformat()],
        "window_source": win[2],
        "first_call": sel[0][0].isoformat() if sel else None,
        "last_call": sel[-1][0].isoformat() if sel else None,
    }


def _stage_record(stage: Stage, ev: Evidence) -> dict[str, Any]:
    active, active_src = _value(stage.active, ev, stage)
    reported, reported_src = _value(stage.reported_tokens, ev, stage)
    labels, labels_src = _value(stage.labels, ev, stage)
    led = _ledger_window(stage, ev)
    if stage.tokens == "ledger":
        tokens, tokens_src = led.get("output_tokens"), ".cache/llm_ledger.json output_tokens"
    elif stage.tokens in ("reported", "tokenize"):
        tokens, tokens_src = reported, reported_src
    else:
        tokens, tokens_src = None, "n/a (training)"
    return {
        "id": stage.id,
        "phase": stage.phase,
        "row": stage.row,
        "what": stage.what,
        "hardware": stage.hardware,
        "model": stage.model,
        "attempt": stage.attempt,
        "group": stage.group,
        "active_s": active,
        "active_source": active_src,
        "generated_tokens": int(tokens) if tokens is not None else None,
        "tokens_source": tokens_src,
        "reported_tokens": int(reported) if reported is not None else None,
        "reported_tokens_source": reported_src,
        "labels": int(labels) if labels is not None else None,
        "labels_source": labels_src,
        "ledger": led,
        "sessions": list(stage.sessions),
        "note": stage.note,
    }


# ---------------------------------------------------------------- aggregation


def _h(s: float | None) -> float | None:
    return None if s is None else s / 3600.0


def _kwh(s: float | None, watts: float) -> float | None:
    return None if s is None else s / 3600.0 * watts / 1000.0


def _power_sample(ev: Evidence) -> dict[str, Any] | None:
    txt = ev.text("reports/compute_power_sample.csv")
    if not txt:
        return None
    rows = [r.split(", ") for r in txt.splitlines() if r.count(",") >= 5]
    watts = [float(r[2].split()[0]) for r in rows]
    util = [float(r[4].split()[0]) for r in rows]
    limit = float(rows[0][3].split()[0])
    return {
        "source": "reports/compute_power_sample.csv (nvidia-smi -l 1)",
        "workload": "sft_qwen7b_qlora_v2 training (stage p8-sft-7b-v2), plus other users' CPU jobs",
        "n": len(watts),
        "first": rows[0][0],
        "last": rows[-1][0],
        "mean_w": statistics.mean(watts),
        "median_w": statistics.median(watts),
        "min_w": min(watts),
        "max_w": max(watts),
        "mean_util_pct": statistics.mean(util),
        "power_limit_w": limit,
    }


def _estimates(ev: Evidence) -> dict[str, Any]:
    out: dict[str, Any] = {}
    p = ev.path("configs/experiment/p2_compute_estimates.yaml")
    if p.exists():
        cfg = yaml.safe_load(p.read_text(encoding="utf-8"))["params"]
        stages = cfg["stages"]
        gpu_s = sum(float(v["gpu_seconds"]) for v in stages.values())
        out["p2_plan"] = {
            "source": "configs/experiment/p2_compute_estimates.yaml",
            "placeholder": bool(cfg.get("placeholder")),
            "labelled_pairs": int(cfg["labelled_pairs"]),
            "stages": stages,
            "gpu_seconds": gpu_s,
            "gpu_s_per_pair": gpu_s / float(cfg["labelled_pairs"]),
            "tolerance": float(cfg.get("tolerance", 0.2)),
        }
    txt = ev.text("reports/p2_full_scale_plan.md")
    if txt:
        m = re.search(r"Projected aggregate stage GPU allocation is ([\d.]+) hours", txt)
        m2 = re.search(r"Candidate labels total at most ([\d,]+) records", txt)
        if m and m2:
            hours, labels = float(m.group(1)), int(m2.group(1).replace(",", ""))
            out["p2_projection"] = {
                "source": "reports/p2_full_scale_plan.md (RTX 4090 smoke x 1,800/20)",
                "gpu_hours": hours,
                "labels": labels,
                "gpu_s_per_pair": hours * 3600 / labels,
                "rows": md_table(txt, "Work"),
            }
    queue = {}
    for q in sorted(ev.path("gpu_queue").glob("*.yaml")):
        spec = yaml.safe_load(q.read_text(encoding="utf-8"))
        if isinstance(spec, dict) and "est_hours" in spec:
            queue[spec["id"]] = float(spec["est_hours"])
    out["gpu_queue_est_hours"] = {"source": "gpu_queue/*.yaml est_hours", "jobs": queue}
    return out


def _sum(xs: Iterable[float | None]) -> float | None:
    vals = [x for x in xs if x is not None]
    return sum(vals) if vals else None


def build(root: Path = ROOT) -> dict[str, Any]:
    """Collect every number of the report into one JSON-serialisable dict."""
    ev = Evidence(root)
    counted: list[Stage] = []
    pending: list[Stage] = []
    for stg in STAGES:
        (pending if _is_pending(stg, ev) else counted).append(stg)
    stages = [_stage_record(stg, ev) for stg in counted]
    pending_sessions = {x for stg in pending for x in stg.sessions} - {
        x for stg in counted for x in stg.sessions
    }
    excluded = [
        {
            "id": stg.id,
            "evidence": (stg.done[0] if stg.done else _src_path(stg.active)) or "n/a",
            "since": stg.pending_since or "",
        }
        for stg in pending
    ]
    sessions: list[dict[str, Any]] = []
    for s in SESSIONS:
        if s.id in pending_sessions:
            continue
        sp = ev.span(s)
        sessions.append(
            {
                "id": s.id,
                "phase": s.phase,
                "log": s.log,
                "start": sp[0].isoformat() if sp else None,
                "end": sp[1].isoformat() if sp else None,
                "reserved_s": (sp[1] - sp[0]).total_seconds() if sp else None,
                "note": s.note,
            }
        )
    sess_by_id: dict[str, dict[str, Any]] = {str(s["id"]): s for s in sessions}

    smoke_reserved_h, _ = _value(
        {"kind": "md_sum", "path": "reports/p2_gpu_smoke.md", "header": "Server / workload",
         "column": "Server-reserved GPU-hours", "exclude": []},
        ev, STAGES[0],
    )  # fmt: skip
    for st in stages:
        if st["id"] == "p2-smoke-rtx4090" and smoke_reserved_h is not None:
            st["reserved_override_s"] = smoke_reserved_h * 3600
            st["reserved_source"] = "reports/p2_gpu_smoke.md: sum of 'Server-reserved GPU-hours'"

    phases: dict[int, dict[str, Any]] = {}
    for st in stages:
        ph = phases.setdefault(st["phase"], {"phase": st["phase"], "stage_ids": []})
        ph["stage_ids"].append(st["id"])
    for ph in phases.values():
        sts = [_stage(stages, i) for i in ph["stage_ids"]]
        ph.update(_aggregate(sts, sess_by_id))
        ph["hardware"] = sorted({s["hardware"] for s in sts})

    groups: dict[str, dict[str, Any]] = {}
    for st in stages:
        if st["group"]:
            groups.setdefault(st["group"], {"stage_ids": []})["stage_ids"].append(st["id"])
    for g in groups.values():
        sts = [_stage(stages, i) for i in g["stage_ids"]]
        g.update(_aggregate(sts, sess_by_id))
        g["final_active_s"] = sum(s["active_s"] or 0.0 for s in sts if s["attempt"] == "final")
        g["labels"] = sum(s["labels"] or 0 for s in sts)
        lab = g["labels"] or None
        g["gpu_s_per_label"] = g["active_s"] / lab if lab else None
        g["final_gpu_s_per_label"] = g["final_active_s"] / lab if lab else None
        g["reserved_gpu_s_per_label"] = g["reserved_s"] / lab if lab else None

    since = [s.pending_since for s in pending if s.pending_since]
    cutoff_s = min(since) if since else None
    if cutoff_s:  # completed stages of the same run keep their calls: cut after their windows
        ends = [st["ledger"]["window"][1][:19] for st in stages if st["ledger"]]
        cutoff_s = max([cutoff_s, *ends])
    cutoff = _utc(cutoff_s) if cutoff_s else datetime.max.replace(tzinfo=UTC)
    kept = [c for c in ev.ledger if c[0] < cutoff]
    late = [c for c in ev.ledger if c[0] >= cutoff]
    attributed = sum(st["ledger"].get("output_tokens", 0) for st in stages)
    covered = [
        (_utc(st["ledger"]["window"][0][:19]), _utc(st["ledger"]["window"][1][:19]), st["model"])
        for st in stages
        if st["ledger"]
    ]
    unattributed: dict[str, int] = {}
    for c in kept:
        if not any(a <= c[0] <= b and m == c[1] for a, b, m in covered):
            unattributed[c[1]] = unattributed.get(c[1], 0) + c[3]
    budget = {}
    bp = ev.path("configs/budget.yaml")
    if bp.exists():
        caps = yaml.safe_load(bp.read_text(encoding="utf-8"))["phases"]
        for name, meta in ev.ledger_meta().items():
            budget[name] = {**meta, "max_tokens": caps.get(name, {}).get("max_tokens")}

    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "board_power_w": dict(BOARD_POWER_W),
        "stages": stages,
        "sessions": sessions,
        "smoke_rtx4090_reserved_h": smoke_reserved_h,
        "phases": [phases[k] for k in sorted(phases)],
        "cpu_only_phases": CPU_PHASES,
        "cpu_parts": CPU_PARTS,
        "groups": groups,
        "estimates": _estimates(ev),
        "ledger": {
            "source": ".cache/llm_ledger.json",
            "cutoff": cutoff_s,
            "calls": len(kept),
            "output_tokens": sum(c[3] for c in kept),
            "attributed_output_tokens": attributed,
            "unattributed_output_tokens": sum(unattributed.values()),
            "unattributed_by_model": unattributed,
            "after_cutoff_calls": len(late),
            "after_cutoff_output_tokens": sum(c[3] for c in late),
            "budget": budget,
        },
        "power_sample": _power_sample(ev),
        "excluded": excluded,
    }


def _src_path(src: dict[str, Any]) -> str | None:
    if "path" in src:
        return str(src["path"])
    if src.get("paths"):
        return str(src["paths"][0])
    return None


def _is_pending(stage: Stage, ev: Evidence) -> bool:
    """An in-progress stage stays pending until its completion evidence exists."""
    if stage.pending_since is None:
        return False
    if stage.done is not None:
        txt = ev.text(stage.done[0])
        return txt is None or re.search(stage.done[1], txt, re.M) is None
    return _value(stage.active, ev, stage)[0] is None


def _aggregate(sts: list[dict[str, Any]], sess_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Active and reserved GPU-seconds and estimated energy of a set of stages.

    Reserved: each server/training session counts once, at least as long as the active time of
    the stages that ran on it; a stage with no surviving session log counts its active time
    (a lower bound), or its own reported reservation (the RTX 4090 smoke)."""
    active = sum(s["active_s"] or 0.0 for s in sts)
    e_active = sum(_kwh(s["active_s"] or 0.0, _watts(s["hardware"])) or 0.0 for s in sts)
    reserved, e_reserved, lower_bound = 0.0, 0.0, []
    on_session: dict[str, float] = {}
    for s in sts:
        if s["sessions"]:
            on_session[s["sessions"][0]] = on_session.get(s["sessions"][0], 0.0) + (
                s["active_s"] or 0.0
            )
            for extra in s["sessions"][1:]:
                on_session.setdefault(extra, 0.0)
        else:
            r = s.get("reserved_override_s")
            if r is None:
                r = s["active_s"] or 0.0
                if s["active_s"]:
                    lower_bound.append(s["id"])
            reserved += r
            e_reserved += _kwh(r, _watts(s["hardware"])) or 0.0
    for sid, act in on_session.items():
        r = max(sess_by_id[sid]["reserved_s"] or 0.0, act)
        if act > (sess_by_id[sid]["reserved_s"] or 0.0):
            lower_bound.append(sid)
        reserved += r
        e_reserved += _kwh(r, BOARD_POWER_W["rtx4500ada"]) or 0.0
    return {
        "active_s": active,
        "gpu_time_recorded": any(s["active_s"] is not None for s in sts),
        "reserved_s": reserved,
        "active_gpu_h": active / 3600,
        "reserved_gpu_h": reserved / 3600,
        "energy_active_kwh": e_active,
        "energy_reserved_kwh": e_reserved,
        "tokens": sum(s["generated_tokens"] or 0 for s in sts),
        "sessions": sorted(on_session),
        "reserved_lower_bound_from": sorted(lower_bound),
    }


def _watts(hardware: str) -> float:
    return BOARD_POWER_W.get(hardware, BOARD_POWER_W["rtx4500ada"])


def _stage(stages: list[dict[str, Any]], sid: str) -> dict[str, Any]:
    return next(s for s in stages if s["id"] == sid)


# ---------------------------------------------------------------- rendering


def _f(x: float | None, nd: int = 2) -> str:
    return "n/a" if x is None else f"{x:,.{nd}f}"


def _i(x: int | float | None) -> str:
    return "n/a" if x is None else f"{int(round(x)):,}"


def render_markdown(r: dict[str, Any]) -> str:
    """The human-readable report."""
    w = r["board_power_w"]
    out: list[str] = []
    a = out.append
    a("# Compute accounting (row 12.10)")
    a("")
    a(
        "Generated by `python -m terser.phase12.compute_accounting` from the logs, LLM ledger, "
        "MLflow database and data files on the RTX 4500 Ada workstation "
        f"({r['generated_at']}). Every number below names its source in "
        "[compute_accounting.json](compute_accounting.json) (`*_source` fields)."
    )
    a("")
    a(
        "**Method.** *Active GPU-hours* = the wall time each stage measured itself (one card, so "
        "GPU-seconds = wall seconds). *Reserved GPU-hours* = the lifetime of the vLLM server or "
        "training process from the first to the last timestamp of its log (startup, idle and "
        "shutdown included); where no server log survives, the stage's active time is used as a "
        "lower bound. *Generated tokens* = output tokens in the LLM ledger for the stage's model "
        "inside its time window (cache hits are not ledgered, so this is work the GPU did); stage "
        "summaries are shown alongside as a cross-check. **Energy is an estimate**: no power was "
        f"logged during the runs, so kWh = GPU-hours x board power limit ({w['rtx4500ada']:.0f} W "
        f"for the RTX 4500 Ada, {w['rtx4090']:.0f} W for the RTX 4090 smoke), an upper bound. "
        "Ledger calls are attributed by model and time window, not by the ledger's budget bucket: "
        "the Phase 12 test respond runs and the first test judge run were charged to bucket "
        "`phase2` (which their spend pushed to its cap) and the resumed judge to `phase12`; all "
        "of it is counted under Phase 12. All times are UTC (the `*.status` files print IST, "
        "UTC+5:30)."
        + (
            f" Ledger calls from {r['ledger']['cutoff']} UTC on belong to in-progress work and "
            "are held back."
            if r["ledger"]["cutoff"]
            else ""
        )
    )
    a("")
    a("## Per phase")
    a("")
    a(
        "| Phase | Hardware | Stages | Generated tokens | Active GPU-h | Reserved GPU-h | "
        "Energy, active (kWh, est.) | Energy, reserved (kWh, est.) |"
    )
    a("| :---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |")
    tot = {"tok": 0, "act": 0.0, "res": 0.0, "ea": 0.0, "er": 0.0}
    for ph in r["phases"]:
        known = ph.get("gpu_time_recorded", True)
        na = "not recorded"
        a(
            f"| {ph['phase']} | {', '.join(ph['hardware'])} | {len(ph['stage_ids'])} | "
            f"{_i(ph['tokens'])} | {_f(ph['active_gpu_h']) if known else na} | "
            f"{_f(ph['reserved_gpu_h']) if known else na} | "
            f"{_f(ph['energy_active_kwh']) if known else na} | "
            f"{_f(ph['energy_reserved_kwh']) if known else na} |"
        )
        tot["tok"] += ph["tokens"]
        tot["act"] += ph["active_s"]
        tot["res"] += ph["reserved_s"]
        tot["ea"] += ph["energy_active_kwh"]
        tot["er"] += ph["energy_reserved_kwh"]
    for p in sorted(r["cpu_only_phases"]):
        a(f"| {p} | cpu | - | 0 | 0 | 0 | 0 | 0 |")
    a(
        f"| **all** | | | **{_i(tot['tok'])}** | **{_f(tot['act'] / 3600)}** | "
        f"**{_f(tot['res'] / 3600)}** | **{_f(tot['ea'])}** | **{_f(tot['er'])}** |"
    )
    a("")
    cpu = "; ".join(f"{p}: {w_}" for p, w_ in sorted(r["cpu_only_phases"].items()))
    parts = "; ".join(f"{p}: {w_}" for p, w_ in sorted(r.get("cpu_parts", {}).items()))
    a(
        f"CPU-only phases (no GPU work): {cpu}. CPU work inside GPU phases, not counted: {parts}. "
        "Phase 11 benchmark stages have no stage timer, so their active time is the server span. "
        "Phase 7 benchmark runs logged tokens but not "
        "their GPU time. Phase 2 reserved hours include the RTX 4090 smoke's server-reserved "
        f"{_f(r['smoke_rtx4090_reserved_h'], 3)} h from `reports/p2_gpu_smoke.md`. Reserved "
        "hours are a lower bound where a server log was lost: "
        + ", ".join(f"`{x}`" for ph in r["phases"] for x in ph["reserved_lower_bound_from"])
        + "."
    )
    a("")
    ps = r.get("power_sample")
    if ps:
        a(
            f"Power sensitivity: a {ps['n']}-sample `nvidia-smi` reading "
            f"({ps['first']} to {ps['last']} UTC, {ps['workload']}) averaged "
            f"{ps['mean_w']:.1f} W (median {ps['median_w']:.1f}, range {ps['min_w']:.0f}-"
            f"{ps['max_w']:.0f} W, {ps['mean_util_pct']:.0f}% mean utilisation) against a "
            f"{ps['power_limit_w']:.0f} W limit. At that draw the active-hours energy would be "
            f"{_f(_kwh(tot['act'], ps['mean_w']))} kWh instead of {_f(tot['ea'])} kWh. "
            "One sample of a different workload: a sensitivity, not a measurement of these runs."
        )
        a("")
    a("## Per stage")
    a("")
    a(
        "| Stage | Row | Hardware | Model | Generated tokens | Stage-reported tokens | "
        "Active s | Ledger busy s | Labels | Note |"
    )
    a("| :--- | :---: | :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- |")
    for st in r["stages"]:
        model = (st["model"] or "").split("/")[-1]
        tag = " (retry)" if st["attempt"] == "retry" else ""
        a(
            f"| `{st['id']}`{tag} | {st['row']} | {st['hardware']} | {model} | "
            f"{_i(st['generated_tokens'])} | {_i(st['reported_tokens'])} | {_f(st['active_s'], 0)} "
            f"| {_f(st['ledger'].get('busy_s'), 0)} | {_i(st['labels'])} | {st['note']} |"
        )
    a("")
    led = r["ledger"]
    a(
        f"Ledger check: {_i(led['calls'])} calls, {_i(led['output_tokens'])} output tokens, of "
        f"which {_i(led['attributed_output_tokens'])} fall in a stage window "
        f"({_i(led['unattributed_output_tokens'])} unattributed"
        + "".join(f", {m.split('/')[-1]} {_i(v)}" for m, v in led["unattributed_by_model"].items())
        + f"); {_i(led['after_cutoff_calls'])} later calls of in-progress work with "
        f"{_i(led['after_cutoff_output_tokens'])} output tokens are held back. Where the "
        "stage-reported count differs from the ledger, the ledger also holds calls of an "
        "interrupted or repeated run in the same window (casualise, rung-3 and augmentation "
        "judges), or the stage's own count includes work the ledger never saw (canary stages "
        "are not ledgered, so they use the stage-reported count)."
    )
    a("")
    a("## GPU reservations")
    a("")
    a("| Session | Phase | Log | Start (UTC) | End (UTC) | Reserved h | Note |")
    a("| :--- | :---: | :--- | :--- | :--- | ---: | :--- |")
    for s in r["sessions"]:
        a(
            f"| {s['id']} | {s['phase']} | `{s['log']}` | {(s['start'] or 'n/a')[:19]} | "
            f"{(s['end'] or 'n/a')[:19]} | {_f(_h(s['reserved_s']))} | {s['note']} |"
        )
    a("")
    a("## GPU-seconds per labelled pair: estimate vs actual")
    a("")
    est = r["estimates"]
    a("| Basis | GPU-seconds | Labelled pairs | GPU-s per pair | Source |")
    a("| :--- | ---: | ---: | ---: | :--- |")
    plan = est.get("p2_plan")
    if plan:
        flag = " (placeholder)" if plan["placeholder"] else ""
        a(
            f"| Estimate: row 2.4.3 plan{flag} | {_i(plan['gpu_seconds'])} | "
            f"{_i(plan['labelled_pairs'])} | {_f(plan['gpu_s_per_pair'], 3)} | `{plan['source']}` |"
        )
    proj = est.get("p2_projection")
    if proj:
        a(
            f"| Estimate: RTX 4090 smoke projection | {_i(proj['gpu_hours'] * 3600)} | "
            f"{_i(proj['labels'])} | {_f(proj['gpu_s_per_pair'], 3)} | "
            "`reports/p2_full_scale_plan.md` |"
        )
    names = {
        "p1": "Actual: Phase 1 pilot (respond + judge)",
        "p2": "Actual: Phase 2 labels (casualise -> candidates -> respond -> judge)",
        "p8-aug": "Actual: Phase 8 augmentation round (candidates -> respond -> judge)",
        "p8-eval": "Actual: Phase 8 val labels, rungs 1-2 (respond -> judge)",
        "p8-r3": "Actual: Phase 8 val labels, rung 3 (respond -> judge)",
        "p8-v2": "Actual: Phase 8 val labels, v2 rungs (respond -> judge)",
        "p8-q7": "Actual: Phase 8 val labels, 7B v2 + HMM + RL sets (respond -> judge)",
        "p10-hpo": "Actual: Phase 10 HPO val labels (respond -> judge)",
        "p11-canary": "Actual: Phase 11 canary (rewrite -> respond -> judge)",
        "p12": "Actual: Phase 12 test labels (respond -> judge)",
    }
    for gid, g in r["groups"].items():
        a(
            f"| {names.get(gid, gid)}, all attempts | {_i(g['active_s'])} | {_i(g['labels'])} | "
            f"{_f(g['gpu_s_per_label'], 3)} | stages {', '.join(g['stage_ids'])} |"
        )
        if g["final_active_s"] != g["active_s"]:
            a(
                f"| {names.get(gid, gid)}, final runs only | {_i(g['final_active_s'])} | "
                f"{_i(g['labels'])} | {_f(g['final_gpu_s_per_label'], 3)} | "
                "retries excluded; understates cost, the final judge reused their cache |"
            )
        if g["sessions"]:
            a(
                f"| {names.get(gid, gid)}, reserved | {_i(g['reserved_s'])} | {_i(g['labels'])} | "
                f"{_f(g['reserved_gpu_s_per_label'], 3)} | sessions {', '.join(g['sessions'])} |"
            )
    a("")
    p2 = r["groups"].get("p2")
    if plan and p2 and p2["gpu_s_per_label"]:
        ratio = p2["gpu_s_per_label"] / plan["gpu_s_per_pair"]
        a(
            f"Phase 2 actual / plan estimate = **{ratio:.2f}x** per labelled pair "
            f"(risk R09 trigger: > 1.2x). The plan assumed {_i(plan['labelled_pairs'])} labelled "
            f"pairs; the label subset produced {_i(p2['labels'])} records."
        )
        if proj:
            out[-1] += (
                f" Against the smoke projection the ratio is "
                f"**{p2['gpu_s_per_label'] / proj['gpu_s_per_pair']:.2f}x**."
            )
        a("")
        a(
            "| Phase 2 stage | Est. tokens | Actual tokens (ledger) | Est. GPU-s | "
            "Actual active GPU-s |"
        )
        a("| :--- | ---: | ---: | ---: | ---: |")
        mapping = {
            "casualise": ["p2-casualise"],
            "candidates": ["p2-candidates"],
            "respond": ["p2-respond-qwen", "p2-respond-phi"],
            "judge": ["p2-judge-smoke", "p2-judge-attempt1", "p2-judge"],
        }
        for name, ids in mapping.items():
            e = plan["stages"].get(name, {})
            sts = [_stage(r["stages"], i) for i in ids]
            a(
                f"| {name} | {_i(e.get('generated_tokens'))} | "
                f"{_i(_sum(s['generated_tokens'] for s in sts))} | {_i(e.get('gpu_seconds'))} | "
                f"{_i(_sum(s['active_s'] for s in sts))} |"
            )
        a("")
    q = est.get("gpu_queue_est_hours", {}).get("jobs", {})
    if q:
        a("## GPU queue estimates vs actual")
        a("")
        a("| Job | est_hours (`gpu_queue/`) | Actual active h | Stage |")
        a("| :--- | ---: | ---: | :--- |")
        jm = {
            "p2-candidates-qwen32b": "p2-candidates",
            "sft-qwen0.5b-full": "p8-sft-0.5b",
            "sft-qwen1.5b-lora": "p8-sft-1.5b",
            "sft-qwen3b-lora": "p8-sft-3b",
        }
        for job, hours in q.items():
            sid = jm.get(job)
            act = _stage(r["stages"], sid)["active_s"] if sid else None
            a(f"| {job} | {hours:g} | {_f(_h(act))} | `{sid}` |")
        a("")
    if led["budget"]:
        a("## Token budget (configs/budget.yaml)")
        a("")
        a(
            "| Ledger phase key | Calls (whole ledger) | Tokens, input + output (whole ledger) | "
            "max_tokens cap | Used |"
        )
        a("| :--- | ---: | ---: | ---: | ---: |")
        for name, b in led["budget"].items():
            cap = b["max_tokens"]
            used = f"{b['tokens'] / cap:.1%}" if cap else "n/a"
            a(f"| {name} | {_i(b['calls'])} | {_i(b['tokens'])} | {_i(cap)} | {used} |")
        a("")
        a(
            "LLM calls after Phase 1 were ledgered under the `phase2` key up to the Phase 12 test "
            "judge, which hit that key's cap and resumed under `phase12`; Phase 8, 10 and 12 work "
            "is therefore inside the `phase2` totals. This table reads the ledger's own running "
            "totals, so it also includes calls of in-progress work."
        )
        a("")
    a("## Pending")
    a("")
    if r["excluded"]:
        a(
            "In-progress stages are not counted yet; they are added automatically when their "
            "logs are complete and the report is regenerated:"
        )
        a("")
        for e in r["excluded"]:
            a(f"- `{e['id']}` (waiting for `{e['evidence']}`)")
    else:
        a("None: every GPU stage on record is counted.")
    a("")
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Row 12.10 compute accounting report")
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--out-md", type=Path, default=Path("reports/compute_accounting.md"))
    ap.add_argument("--out-json", type=Path, default=Path("reports/compute_accounting.json"))
    args = ap.parse_args(argv)
    report = build(args.root)
    md_path = args.out_md if args.out_md.is_absolute() else args.root / args.out_md
    js_path = args.out_json if args.out_json.is_absolute() else args.root / args.out_json
    js_path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    tot = sum(p["active_s"] for p in report["phases"]) / 3600
    print(f"wrote {md_path} and {js_path}: {tot:.2f} active GPU-hours")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
