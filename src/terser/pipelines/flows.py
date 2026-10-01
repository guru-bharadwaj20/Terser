"""Prefect orchestration flows (plan rows **11.6.1** and **11.6.2**).

Six flows, each a thin layer over code that already exists, so a flow can always be reproduced by
hand:

==========================  ====================================================================
``data_flow``               ``dvc repro <stage>`` (row 11.1.2 pipeline), then pandera validation
                            of the pair table when it exists (row 11.1.4)
``train_classical_flow``    runs the configured CPU training commands (gate, tagger)
``train_rewriter_flow``     writes a ``gpu_queue/`` job spec and validates the queue; the GPU
                            worker drains it (the flow never trains on the orchestrator)
``evaluate_flow``           runs the configured validation commands
``promote_flow``            :func:`terser.pipelines.promotion.decide` then registry promotion
``drift_flow``              tokenizer drift check (row 11.5.3); raises on an alert
==========================  ====================================================================

Reliability (row 11.6.2): shell steps are Prefect tasks with ``retries`` and ``retry_delay``;
LLM calls inside those steps are disk-cached by :mod:`terser.llm` so a retry never re-pays for a
completed generation; a failed flow run triggers Prefect's notification automations.

Prefect is optional (``pip install -e ".[mlops]"``). Without it the decorators below are
no-ops, so every flow is still a plain callable (``flow.fn`` works in both cases). To register
the flows with a Prefect server as deployments::

    python -m terser.pipelines.flows serve
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar

import yaml

from terser.pipelines.promotion import Decision, EvalSummary, GuardConfig, decide

__all__ = [
    "FLOWS",
    "StepError",
    "data_flow",
    "drift_flow",
    "evaluate_flow",
    "promote_flow",
    "train_classical_flow",
    "train_rewriter_flow",
]

F = TypeVar("F", bound=Callable[..., Any])
REPO_ROOT = Path(__file__).resolve().parents[3]

try:  # pragma: no cover - exercised only where the mlops extra is installed
    from prefect import flow as _pflow
    from prefect import task as _ptask

    HAVE_PREFECT = True
except ImportError:  # pragma: no cover
    HAVE_PREFECT = False


def _flow(name: str) -> Callable[[F], F]:
    def deco(fn: F) -> F:
        if HAVE_PREFECT:
            return _pflow(name=name, log_prints=True)(fn)  # type: ignore[return-value]
        fn.fn = fn  # type: ignore[attr-defined]
        return fn

    return deco


def _task(retries: int = 0, retry_delay_seconds: float = 0) -> Callable[[F], F]:
    def deco(fn: F) -> F:
        if HAVE_PREFECT:
            return _ptask(  # type: ignore[return-value]
                retries=retries, retry_delay_seconds=retry_delay_seconds
            )(fn)
        fn.fn = fn  # type: ignore[attr-defined]
        return fn

    return deco


class StepError(RuntimeError):
    """A pipeline step exited non-zero."""


def run_step(cmd: Sequence[str], cwd: Path = REPO_ROOT, attempts: int = 1) -> str:
    """Run ``cmd``; retry up to ``attempts`` times; return stdout or raise :class:`StepError`."""
    last = ""
    for _ in range(max(1, attempts)):
        proc = subprocess.run(list(cmd), cwd=cwd, capture_output=True, text=True)
        if proc.returncode == 0:
            return proc.stdout
        last = proc.stderr.strip()[-2000:]
    raise StepError(f"{' '.join(cmd)} failed: {last}")


@_task(retries=2, retry_delay_seconds=30)
def shell_task(cmd: Sequence[str]) -> str:
    """A retried shell step."""
    return run_step(cmd)


def _py(*args: str) -> list[str]:
    return [sys.executable, "-m", *args]


@_flow("data_flow")
def data_flow(stage: str = "split", pairs: str | None = None) -> str:
    """Rebuild data with DVC up to ``stage``; validate the pair table if given."""
    out = str(shell_task(["dvc", "repro", stage]))
    if pairs:
        shell_task(_py("terser.data.validation", pairs))
    return out


@_flow("train_classical_flow")
def train_classical_flow(commands: Sequence[Sequence[str]]) -> list[str]:
    """Run CPU training commands (e.g. the gate and tagger trainers) in order."""
    return [str(shell_task(list(c))) for c in commands]


@_flow("train_rewriter_flow")
def train_rewriter_flow(
    job: Mapping[str, Any], queue_dir: str | Path = REPO_ROOT / "gpu_queue"
) -> Path:
    """Submit a rewriter training job to ``gpu_queue/`` and validate the queue."""
    from terser.gpu_queue.queue import validate

    queue_dir = Path(queue_dir)
    path = queue_dir / f"{job['id']}.yaml"
    path.write_text(yaml.safe_dump(dict(job), sort_keys=False), encoding="utf-8")
    errors = validate(queue_dir, REPO_ROOT)
    if errors:
        path.unlink()
        raise StepError("gpu_queue validation failed: " + "; ".join(errors))
    return path


@_flow("evaluate_flow")
def evaluate_flow(commands: Sequence[Sequence[str]]) -> list[str]:
    """Run validation-split evaluation commands."""
    return [str(shell_task(list(c))) for c in commands]


@_flow("promote_flow")
def promote_flow(
    candidate: EvalSummary,
    champion: EvalSummary | None,
    model_name: str,
    version: int | str,
    stage: str = "Staging",
    cfg: GuardConfig | None = None,
) -> Decision:
    """Promote ``model_name`` v``version`` to ``stage`` only if the guard allows it."""
    decision = decide(candidate, champion, cfg)
    if decision.promote:
        from terser.tracking import promote

        promote(model_name, version, stage)
    else:
        print("promotion refused: " + "; ".join(decision.reasons))
    return decision


@_flow("drift_flow")
def drift_flow(targets: Sequence[str] = ("gpt-4o", "qwen2.5", "llama-3.1")) -> str:
    """Tokenizer drift check; raises :class:`StepError` on an alert so the run fails loudly."""
    from terser.monitoring.tokenizer_drift import check_tokenizer_drift, load_fixture
    from terser.tokenizers import count_tokens

    strings, snapshot = load_fixture()
    counters = {t: (lambda s, t=t: count_tokens(s, t)) for t in targets}
    report = check_tokenizer_drift(strings, snapshot, counters)
    if report.alert:
        raise StepError(report.summary())
    return report.summary()


FLOWS: dict[str, Any] = {
    "data_flow": data_flow,
    "train_classical_flow": train_classical_flow,
    "train_rewriter_flow": train_rewriter_flow,
    "evaluate_flow": evaluate_flow,
    "promote_flow": promote_flow,
    "drift_flow": drift_flow,
}


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - needs a Prefect server
    """``serve``: register every flow as a deployment (drift daily, data on demand)."""
    args = list(sys.argv[1:] if argv is None else argv)
    if args != ["serve"]:
        print("usage: python -m terser.pipelines.flows serve", file=sys.stderr)
        return 2
    if not HAVE_PREFECT:
        print('Prefect is not installed: pip install -e ".[mlops]"', file=sys.stderr)
        return 1
    from prefect import serve

    deployments = [
        f.to_deployment(name=name, cron="0 3 * * *" if name == "drift_flow" else None)
        for name, f in FLOWS.items()
    ]
    serve(*deployments)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
