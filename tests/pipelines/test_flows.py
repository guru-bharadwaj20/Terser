"""Orchestration flows (11.6.1) as plain callables; retry recovery of an injected failure."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from terser.pipelines import flows
from terser.pipelines.promotion import EvalSummary

CATS = {"code": 0.96, "fact": 0.97}
GOOD = EvalSummary(0.96, 0.31, CATS)


def test_all_six_flows_defined() -> None:
    assert set(flows.FLOWS) == {
        "data_flow",
        "train_classical_flow",
        "train_rewriter_flow",
        "evaluate_flow",
        "promote_flow",
        "drift_flow",
    }
    assert all(callable(getattr(f, "fn", None)) for f in flows.FLOWS.values())


def test_run_step_recovers_from_injected_failure(tmp_path: Path) -> None:
    """The first attempt fails (marker absent), the retry succeeds."""
    marker = tmp_path / "attempted"
    code = (
        "import pathlib,sys; p=pathlib.Path(sys.argv[1]);"
        "ok=p.exists(); p.touch(); print('ok' if ok else 'fail'); sys.exit(0 if ok else 3)"
    )
    cmd = [sys.executable, "-c", code, str(marker)]
    with pytest.raises(flows.StepError):
        flows.run_step(cmd, attempts=1)
    marker.unlink()
    assert flows.run_step(cmd, attempts=2).strip() == "ok"


def _fake_shell(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    calls: list[list[str]] = []

    def fake(cmd: list[str]) -> str:
        calls.append(list(cmd))
        return "done"

    monkeypatch.setattr(flows, "shell_task", fake)
    return calls


def test_data_train_evaluate_flows(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_shell(monkeypatch)
    flows.data_flow.fn("split", pairs="pairs.jsonl")
    assert calls[0] == ["dvc", "repro", "split"] and calls[1][-2:] == [
        "terser.data.validation",
        "pairs.jsonl",
    ]
    assert flows.train_classical_flow.fn([["a"], ["b"]]) == ["done", "done"]
    assert flows.evaluate_flow.fn([["c"]]) == ["done"]


def test_train_rewriter_flow_submits_valid_job(tmp_path: Path) -> None:
    job: dict[str, Any] = {
        "id": "sft-test",
        "phase": 8,
        "depends_on": [],
        "command": "python -m terser.models.rewriter.train experiment=tiny",
        "est_vram_gb": 12,
        "est_hours": 1,
        "exclusive": True,
        "priority": 2,
        "cpu_smoke_test": "tests/pipelines/test_flows.py",
        "outputs": ["models/rewriter/sft-test/"],
    }
    path = flows.train_rewriter_flow.fn(job, tmp_path)
    assert path.exists()
    bad = {**job, "id": "too-big", "est_vram_gb": 999}
    with pytest.raises(flows.StepError):
        flows.train_rewriter_flow.fn(bad, tmp_path)
    assert not (tmp_path / "too-big.yaml").exists()


def test_promote_flow_guards(monkeypatch: pytest.MonkeyPatch) -> None:
    promoted: list[tuple[str, Any, str]] = []
    import terser.tracking

    monkeypatch.setattr(terser.tracking, "promote", lambda n, v, s: promoted.append((n, v, s)))
    bad = EvalSummary(0.80, 0.31, CATS)
    assert not flows.promote_flow.fn(bad, GOOD, "terser-gate", 2).promote
    assert promoted == []
    assert flows.promote_flow.fn(GOOD, GOOD, "terser-gate", 2).promote
    assert promoted == [("terser-gate", 2, "Staging")]


def test_drift_flow_raises_on_alert(monkeypatch: pytest.MonkeyPatch) -> None:
    import terser.tokenizers

    monkeypatch.setattr(terser.tokenizers, "count_tokens", lambda s, t: len(s) + 1000)
    with pytest.raises(flows.StepError, match="TOKENIZER DRIFT"):
        flows.drift_flow.fn(("gpt-4o",))
