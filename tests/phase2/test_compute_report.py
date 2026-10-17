"""Tests for row 2.4.3: the compute report (synthetic ledgers and runs only)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from terser.llm.budget import Ledger, Usage
from terser.phase2.compute_report import (
    DEFAULT_ESTIMATES,
    STAGES,
    UNATTRIBUTED,
    ComputeEstimates,
    build_report,
    check_c29,
    count_labelled_pairs,
    load_estimates,
    load_ledger_calls,
    load_mlflow_runs,
    main,
    normalise_runs,
    write_report,
)
from terser.phase2.oracle import Phase2InputError

REPO = Path(__file__).resolve().parents[2]


def _estimates(**kw: Any) -> ComputeEstimates:
    base: dict[str, Any] = {
        "stages": {
            "casualise": {"generated_tokens": 1000.0, "gpu_seconds": 100.0},
            "candidates": {"generated_tokens": 2000.0, "gpu_seconds": 200.0},
            "respond": {"generated_tokens": 4000.0, "gpu_seconds": 300.0},
            "judge": {"generated_tokens": 3000.0, "gpu_seconds": 400.0},
        },
        "labelled_pairs": 100.0,
        "tolerance": 0.20,
        "placeholder": False,
        "models": {"Qwen/Qwen2.5-7B-Instruct": "respond"},
        "ledger_phases": {"judge_phase": "judge"},
        "stand_in_models": ("qwen2.5:1.5b",),
    }
    base.update(kw)
    return ComputeEstimates(**base)


def _write_ledger(path: Path, calls: list[tuple[str, str, int, int, bool]]) -> Path:
    """calls: (phase, model, input_tokens, output_tokens, usage_reported), via the real Ledger."""
    ledger = Ledger(path)
    for i, (phase, model, tin, tout, reported) in enumerate(calls):
        ledger.add(phase, Usage("local", model, tin, tout, f"key{i:03d}", reported), 0.0)
    return path


def _runs(rows: list[tuple[str, str, str, float | None]], **cols: list[Any]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["run_id", "task", "hardware", "duration_s"])
    for name, values in cols.items():
        df[name] = values
    return df


GPU = "NVIDIA RTX 4500 Ada Generation"


@pytest.fixture()
def scenario(tmp_path: Path) -> dict[str, Any]:
    est = _estimates()
    cas = _write_ledger(
        tmp_path / "cas.json",
        [("phase2", "Qwen/Qwen2.5-32B-Instruct-AWQ", 50, 500, True)] * 2,
    )
    shared = _write_ledger(
        tmp_path / "shared.json",
        [
            ("phase2", "Qwen/Qwen2.5-7B-Instruct", 100, 4000, True),
            ("judge_phase", "Qwen/Qwen2.5-32B-Instruct-AWQ", 10, 3000, False),
            ("phase2", "qwen2.5:1.5b", 10, 999, True),  # stand-in, excluded
            ("phase2", "mystery-model", 1, 7, True),  # unattributed
        ],
    )
    cand = _write_ledger(tmp_path / "cand.json", [("phase2", "any", 20, 2000, False)])
    calls = pd.concat(
        [
            load_ledger_calls(cas, est, stage="casualise"),
            load_ledger_calls(shared, est),
            load_ledger_calls(cand, est, stage="candidates"),
        ],
        ignore_index=True,
    )
    runs = normalise_runs(
        _runs(
            [
                ("r1", "casualise", GPU, 100.0),
                ("r2", "candidates", GPU, 100.0),
                ("r3", "respond", GPU, 150.0),
                ("r4", "judge", GPU, 400.0),
                ("r5", "judge", "cpu", 50.0),
                ("r6", "judge", "auto", 20.0),
                ("r7", "respond", GPU, 999.0),  # stand-in
                ("r8", "respond", GPU, None),  # unfinished
                ("r9", "judgevalidation", "cpu", 5.0),  # not a stage
            ],
            stand_in=[False] * 6 + ["true", False, False],
            n_gpus=[1, 2, 2, 1, 1, 1, 1, 1, 1],
        ),
        est,
    )
    return {"est": est, "calls": calls, "runs": runs}


# --- loaders ---------------------------------------------------------------------------------


def test_ledger_attribution_and_flags(scenario: dict[str, Any]) -> None:
    calls = scenario["calls"]
    by_model = calls.set_index("model")
    assert by_model.loc["Qwen/Qwen2.5-7B-Instruct", "stage"] == "respond"  # models map
    assert set(calls.loc[calls["phase"] == "judge_phase", "stage"]) == {"judge"}  # phase map
    assert calls["stage"].value_counts()["casualise"] == 2  # --ledger STAGE=PATH
    assert by_model.loc["mystery-model", "stage"] == UNATTRIBUTED
    assert bool(by_model.loc["qwen2.5:1.5b", "stand_in"])
    assert int((~calls["usage_reported"]).sum()) == 2


def test_ledger_entry_stage_key_and_old_entries(tmp_path: Path) -> None:
    path = tmp_path / "l.json"
    path.write_text(
        json.dumps(
            {
                "phases": {
                    "phase2": {
                        "calls": [
                            {"model": "m", "input_tokens": 1, "output_tokens": 2, "stage": "judge"},
                            {"model": "m", "input_tokens": 1, "output_tokens": 3},
                        ]
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    calls = load_ledger_calls(path, _estimates())
    assert calls["stage"].tolist() == ["judge", UNATTRIBUTED]
    assert calls["usage_reported"].all()  # missing key = reported
    with pytest.raises(Phase2InputError):
        load_ledger_calls(path, _estimates(), stage="bogus")
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(Phase2InputError, match="phases"):
        load_ledger_calls(path, _estimates())


def test_normalise_runs_classifies_hardware() -> None:
    runs = normalise_runs(
        _runs([("b", "respond", GPU, 1.0), ("a", "judge", "cpu", 2.0), ("c", "x", None, 3.0)]),
        _estimates(),
    )
    assert runs["run_id"].tolist() == ["a", "b", "c"]
    assert runs["hw_class"].tolist() == ["cpu", "gpu", "unknown"]
    assert runs["stage"].tolist() == ["judge", "respond", ""]
    assert runs["n_gpus"].tolist() == [1, 1, 1]
    with pytest.raises(Phase2InputError, match="duplicate"):
        normalise_runs(_runs([("a", "judge", "cpu", 1.0)] * 2), _estimates())
    with pytest.raises(Phase2InputError, match="missing"):
        normalise_runs(pd.DataFrame({"run_id": ["a"]}), _estimates())
    with pytest.raises(Phase2InputError, match="negative"):
        normalise_runs(_runs([("a", "judge", "cpu", -1.0)]), _estimates())


def test_load_estimates_repo_file_is_placeholder() -> None:
    est = load_estimates(REPO / DEFAULT_ESTIMATES)
    assert est.placeholder is True
    assert set(est.stages) == set(STAGES)
    assert est.tolerance == pytest.approx(0.20)
    assert est.total_gpu_seconds > 0 and est.labelled_pairs


def test_load_estimates_validation(tmp_path: Path) -> None:
    p = tmp_path / "e.yaml"
    p.write_text("params:\n  stages:\n    train: {gpu_seconds: 1}\n", encoding="utf-8")
    with pytest.raises(Phase2InputError, match="unknown stage"):
        load_estimates(p)
    p.write_text("stages:\n  judge: {gpu_seconds: -1}\n", encoding="utf-8")
    with pytest.raises(Phase2InputError, match="negative"):
        load_estimates(p)
    p.write_text("attribution:\n  models: {m: nowhere}\n", encoding="utf-8")
    with pytest.raises(Phase2InputError, match="unknown stage"):
        load_estimates(p)


def test_count_labelled_pairs() -> None:
    labels = pd.DataFrame(
        {
            "prompt_id": ["p1", "p1", "p2", "p3"],
            "candidate_id": ["c1", "c2", "c1", "c1"],
            "target": ["q"] * 4,
            "label": ["preserved", "degraded", None, "preserved"],
            "stand_in": [False, False, False, True],
        }
    )
    assert count_labelled_pairs(labels) == (2, 1)


# --- report ----------------------------------------------------------------------------------


def test_report_aggregates_per_stage(scenario: dict[str, Any]) -> None:
    rep = build_report(scenario["calls"], scenario["runs"], scenario["est"], labelled_pairs=100)
    s = rep.stages.set_index("stage")
    assert s.loc["casualise", "generated_tokens"] == 1000
    assert s.loc["candidates", "generated_tokens"] == 2000
    assert s.loc["respond", "generated_tokens"] == 4000
    assert s.loc["judge", "generated_tokens"] == 3000
    assert s.loc[UNATTRIBUTED, "generated_tokens"] == 7
    # GPU-seconds = duration x n_gpus; cpu / auto / stand-in / unfinished runs not counted.
    assert s.loc["casualise", "gpu_seconds"] == pytest.approx(100.0)
    assert s.loc["candidates", "gpu_seconds"] == pytest.approx(200.0)
    assert s.loc["respond", "gpu_seconds"] == pytest.approx(300.0)
    assert s.loc["judge", "gpu_seconds"] == pytest.approx(400.0)
    assert s.loc["judge", "cpu_seconds"] == pytest.approx(50.0)
    assert s.loc["judge", "unknown_hw_seconds"] == pytest.approx(20.0)
    assert s.loc["total", "gpu_seconds"] == pytest.approx(1000.0)
    assert s.loc["total", "gpu_hours"] == pytest.approx(1000 / 3600)
    assert s.loc["total", "gpu_seconds_per_pair"] == pytest.approx(10.0)
    assert s.loc["judge", "gpu_seconds_per_pair"] == pytest.approx(4.0)
    assert s.loc["total", "unreported_calls"] == 2
    # Exact estimates -> PASS.
    assert rep.c29.passed
    assert "C2.9 Cost within estimate: PASS" in rep.markdown
    assert "PLACEHOLDER" not in rep.markdown
    md = rep.markdown
    assert "usage_reported: false" in md and "**2 entries**" in md
    assert "could not be attributed" in md
    assert "hardware `auto`/missing" in md
    for reason in ("stand-in", "unfinished", "not a stage"):
        assert reason in md
    unrep = rep.table[rep.table["section"] == "usage_unreported"]
    assert len(unrep) == 2


def test_c29_tolerance_and_placeholder() -> None:
    assert check_c29(120.0, 100.0, 0.2, placeholder=False).passed
    assert check_c29(80.0, 100.0, 0.2, placeholder=False).passed
    assert not check_c29(121.0, 100.0, 0.2, placeholder=False).passed
    assert not check_c29(79.0, 100.0, 0.2, placeholder=False).passed
    assert not check_c29(0.0, 0.0, 0.2, placeholder=False).passed
    res = check_c29(100.0, 100.0, 0.2, placeholder=True)
    assert "not valid" in res.detail and "dvc repro" in res.detail


def test_report_over_budget_and_placeholder(scenario: dict[str, Any]) -> None:
    est = _estimates(
        placeholder=True,
        stages={s: {"generated_tokens": 1.0, "gpu_seconds": 100.0} for s in STAGES},
    )
    rep = build_report(scenario["calls"], scenario["runs"], est, labelled_pairs=None)
    assert not rep.c29.passed  # 1000 actual vs 400 estimate
    assert "PLACEHOLDER ESTIMATES" in rep.markdown
    ev = rep.table[rep.table["section"] == "estimate_vs_actual"]
    total_gpu = ev[(ev["stage"] == "total") & (ev["metric"] == "gpu_seconds")].iloc[0]
    assert total_gpu["actual_over_estimate"] == pytest.approx(2.5)
    assert not bool(total_gpu["within_tolerance"])
    assert rep.stages["gpu_seconds_per_pair"].isna().all()


def test_empty_inputs_do_not_crash() -> None:
    est = _estimates()
    calls = pd.DataFrame(
        columns=[
            "stage",
            "ledger",
            "phase",
            "timestamp",
            "provider",
            "model",
            "input_tokens",
            "output_tokens",
            "cache_key",
            "usage_reported",
            "stand_in",
        ]
    )
    runs = normalise_runs(pd.DataFrame(columns=["run_id", "task", "hardware", "duration_s"]), est)
    rep = build_report(calls, runs, est, labelled_pairs=0)
    assert not rep.c29.passed
    assert "None: every ledger entry" in rep.markdown


def test_write_report_deterministic_lf(scenario: dict[str, Any], tmp_path: Path) -> None:
    outs = []
    for run in ("a", "b"):
        rep = build_report(scenario["calls"], scenario["runs"], scenario["est"], labelled_pairs=100)
        outs.append(write_report(rep, tmp_path / run / "c.md", tmp_path / run / "c.csv"))
    for i in range(2):
        data = outs[0][i].read_bytes()
        assert data == outs[1][i].read_bytes()
        assert b"\r\n" not in data


@pytest.fixture
def mlflow_uri(tmp_path: Path) -> Iterator[str]:
    """A per-test SQLite tracking URI; the process-global URI is reset afterwards."""
    import mlflow

    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(uri)
    yield uri
    mlflow.set_tracking_uri(None)


def test_mlflow_loader(tmp_path: Path, mlflow_uri: str) -> None:
    import mlflow

    uri = mlflow_uri
    exp = mlflow.create_experiment("p2test", artifact_location=(tmp_path / "art").as_uri())
    with mlflow.start_run(experiment_id=exp, run_name="2-judge-q-full-0") as run:
        mlflow.set_tags({"task": "judge", "hardware": GPU, "stand_in": "false", "n_gpus": "2"})
    raw = load_mlflow_runs("p2test", uri)
    assert load_mlflow_runs("missing-experiment", uri).empty
    runs = normalise_runs(raw, _estimates())
    assert runs["run_id"].tolist() == [run.info.run_id]
    row = runs.iloc[0]
    assert row["stage"] == "judge" and row["hw_class"] == "gpu" and row["n_gpus"] == 2
    assert row["duration_s"] >= 0


def test_cli(scenario: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    est_path = tmp_path / "est.yaml"
    est_path.write_text(
        "params:\n"
        "  placeholder: false\n"
        "  labelled_pairs: 100\n"
        "  stages:\n"
        "    casualise: {generated_tokens: 1000, gpu_seconds: 100}\n"
        "    candidates: {generated_tokens: 2000, gpu_seconds: 200}\n"
        "    respond: {generated_tokens: 4000, gpu_seconds: 300}\n"
        "    judge: {generated_tokens: 3000, gpu_seconds: 400}\n",
        encoding="utf-8",
    )
    ledger = _write_ledger(tmp_path / "judge.json", [("phase2", "m", 1, 3000, False)])
    runs = _runs([("r1", "judge", GPU, 1000.0)])
    runs_path = tmp_path / "runs.csv"
    runs.to_csv(runs_path, index=False)
    labels = pd.DataFrame(
        {
            "prompt_id": [f"p{i}" for i in range(100)],
            "candidate_id": ["c1"] * 100,
            "target": ["q"] * 100,
            "label": ["preserved"] * 100,
        }
    )
    labels_path = tmp_path / "labels.csv"
    labels.to_csv(labels_path, index=False)
    md, csv = tmp_path / "out" / "cost.md", tmp_path / "out" / "cost.csv"
    args = [
        "--estimates",
        str(est_path),
        "--ledger",
        f"judge={ledger}",
        "--runs",
        str(runs_path),
        "--labels",
        str(labels_path),
        "--out-md",
        str(md),
        "--out-csv",
        str(csv),
    ]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "C2.9 Cost within estimate: PASS" in out  # 1000 GPU-s total vs 1000 estimate
    assert "1 ledger entry usage_reported=false" in out
    text = md.read_text(encoding="utf-8")
    assert "GPU-seconds per labelled pair: **10.000**" in text
    first = md.read_bytes()
    assert main(args) == 0
    assert md.read_bytes() == first

    assert (
        main(
            [
                *args[:4],
                "--no-mlflow",
                "--labelled-pairs",
                "5",
                "--out-md",
                str(md),
                "--out-csv",
                str(csv),
            ]
        )
        == 0
    )
    assert "C2.9 Cost within estimate: FAIL" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["--estimates", str(est_path), "--ledger", f"train={ledger}", "--no-mlflow"])
