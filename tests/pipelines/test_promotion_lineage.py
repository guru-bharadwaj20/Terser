"""Promotion guard (11.4.4) and registry lineage (11.2.2 / 11.2.3) tests."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import mlflow
import numpy as np
import pytest

from terser.pipelines.promotion import EvalSummary, GuardConfig, decide, guarded_promote
from terser.tracking import promote, start_run
from terser.tracking.lineage import LineageError, register_with_lineage, trace

CATS = {"code": 0.96, "fact": 0.97, "reason": 0.95, "explain": 0.97, "creative": 0.93}
CHAMP = EvalSummary(
    0.955, 0.31, CATS, gate_auc=0.86, checkpoints_passed=("C8.3",), has_lineage=True
)


# ---- promotion guard ---------------------------------------------------------------------------
def test_equal_or_better_candidate_is_promoted() -> None:
    better = replace(CHAMP, savings=0.33, gate_auc=0.87)
    calls: list[int] = []
    d = guarded_promote(
        better, CHAMP, lambda: calls.append(1), GuardConfig(required_checkpoints=("C8.3",))
    )
    assert d.promote and calls == [1]
    assert decide(CHAMP, None).promote  # first model: floors only


@pytest.mark.parametrize(
    ("regression", "fragment"),
    [
        ({"preservation": 0.90}, "floor"),
        ({"preservation": 0.94}, "preservation regressed"),
        ({"per_category_preservation": {**CATS, "code": 0.89}}, "code preservation"),
        ({"per_category_preservation": {**CATS, "reason": 0.92}}, "reason preservation regressed"),
        ({"per_category_preservation": {k: v for k, v in CATS.items() if k != "fact"}}, "missing"),
        ({"savings": 0.28}, "savings regressed"),
        ({"gate_auc": 0.84}, "AUC regressed"),
        ({"checkpoints_passed": ()}, "checkpoints not passed"),
        ({"has_lineage": False}, "lineage"),
    ],
)
def test_synthetic_regression_is_refused(regression: dict[str, object], fragment: str) -> None:
    """C11.7: every injected regression is refused and the registry is never touched."""
    cand = replace(CHAMP, **regression)  # type: ignore[arg-type]
    calls: list[int] = []
    d = guarded_promote(
        cand, CHAMP, lambda: calls.append(1), GuardConfig(required_checkpoints=("C8.3",))
    )
    assert not d.promote and calls == []
    assert any(fragment in r for r in d.reasons), d.reasons


def test_all_failures_reported() -> None:
    cand = replace(CHAMP, preservation=0.5, savings=0.0, gate_auc=0.5)
    assert len(decide(cand, CHAMP).reasons) >= 4


# ---- registry and lineage ----------------------------------------------------------------------
@pytest.fixture()
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.chdir(tmp_path)
    yield uri
    mlflow.set_tracking_uri(None)


@pytest.mark.slow  # logs, registers and promotes real MLflow models
def test_register_promote_trace(store: str) -> None:
    from sklearn.dummy import DummyClassifier

    cfg = {"model": "lgbm", "params": {"num_leaves": 31}, "seed": 0}
    with start_run(phase=11, row="11.2.3", task="lineage", run_name="11-lineage-gate-dummy-0") as r:
        clf = DummyClassifier().fit(np.zeros((2, 1)), np.array([0, 1]))
        mlflow.sklearn.log_model(clf, name="model", pip_requirements=["scikit-learn"])
        run_tags = mlflow.get_run(r.info.run_id).data.tags
    mv = register_with_lineage(r.info.run_id, "model", "gate", cfg, ["C5.2", "C10.2"])
    assert mv.name == "terser-gate"
    with pytest.raises(LineageError):
        trace("gate")  # nothing in production yet
    promote("terser-gate", mv.version, "Staging")
    promote("terser-gate", mv.version, "Production")
    lin = trace("gate")
    assert lin["version"] == int(mv.version) and lin["config"] == cfg
    assert lin["git_sha"] == run_tags["git_sha"] and lin["data_hash"] == run_tags["data_hash"]
    assert lin["checkpoints_passed"] == ["C5.2", "C10.2"]
    assert lin["source_run_id"] == r.info.run_id
    with pytest.raises(LineageError):  # promotion moved the version off staging
        trace("terser-gate", "staging")


@pytest.mark.slow
def test_untraceable_version_detected(store: str) -> None:
    from mlflow import MlflowClient
    from sklearn.dummy import DummyClassifier

    with start_run(phase=11, row="11.2.3", task="lineage", run_name="11-lineage-tag-dummy-0") as r:
        clf = DummyClassifier().fit(np.zeros((2, 1)), np.array([0, 1]))
        mlflow.sklearn.log_model(clf, name="model", pip_requirements=["scikit-learn"])
    mv = register_with_lineage(r.info.run_id, "model", "tagger", {"k": 1}, [])
    client = MlflowClient()
    client.set_model_version_tag("terser-tagger", mv.version, "config_json", '{"k": 2}')
    promote("terser-tagger", mv.version, "Production")
    with pytest.raises(LineageError, match="config hash"):
        trace("tagger")
    with pytest.raises(KeyError):
        register_with_lineage(r.info.run_id, "model", "router", {}, [])
