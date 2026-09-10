"""Tests for terser.tracking against a temporary SQLite MLflow store."""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pytest
from mlflow import MlflowClient
from sklearn.dummy import DummyClassifier

from terser.tracking import runs
from terser.tracking.runs import REQUIRED_TAGS, promote, register_model, start_run


@pytest.fixture(scope="module")
def store_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One SQLite store per module: schema migrations take several seconds per new database."""
    return tmp_path_factory.mktemp("mlflow")


@pytest.fixture()
def store(store_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """Point MLflow at the throwaway SQLite database (never the repo's mlruns/)."""
    uri = f"sqlite:///{(store_dir / 'mlflow.db').as_posix()}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.chdir(store_dir)
    yield uri
    mlflow.set_tracking_uri(None)


def test_default_uri_is_repo_local_sqlite() -> None:
    uri = runs.default_tracking_uri()
    assert uri.startswith("sqlite:///") and uri.endswith("mlruns/mlflow.db")
    assert "\\" not in uri


@pytest.mark.slow
def test_run_has_tags_params_metrics_artifact(store: str, store_dir: Path) -> None:
    with start_run(
        phase=0, row="0.2.4", task="tracking", run_name="0-tracking-dummy-base-0", params={"a": 1}
    ) as run:
        mlflow.log_metric("loss", 0.5, step=1)
        mlflow.log_text("hi", "notes/x.txt")
    client = MlflowClient(store)
    got = client.get_run(run.info.run_id)
    assert set(REQUIRED_TAGS) <= set(got.data.tags)
    assert got.data.tags["phase"] == "0" and got.data.tags["hardware"] == "cpu"
    assert got.data.params == {"a": "1"}
    assert got.data.metrics["loss"] == 0.5
    assert [a.path for a in client.list_artifacts(run.info.run_id, "notes")] == ["notes/x.txt"]
    assert (store_dir / "artifacts").is_dir()


def test_bad_run_name_warns(store: str) -> None:
    with pytest.warns(UserWarning, match="does not follow"):
        with start_run(phase=0, row="0.2.4", task="tracking", run_name="oops"):
            pass


def test_missing_git_is_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a: object, **_k: object) -> None:
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", boom)
    assert runs.get_git_sha() == "unknown"


def test_data_hash(tmp_path: Path) -> None:
    assert runs.data_hash(tmp_path) == "none"
    (tmp_path / "dvc.lock").write_text("schema: '2.0'\n")
    assert len(runs.data_hash(tmp_path)) == 16


@pytest.mark.slow
def test_register_and_promote(store: str) -> None:
    with start_run(phase=0, row="0.2.4", task="tracking", run_name="0-tracking-dummy-reg-0") as run:
        clf = DummyClassifier().fit(np.zeros((2, 1)), np.array([0, 1]))
        mlflow.sklearn.log_model(clf, name="model", pip_requirements=["scikit-learn"])
    v1 = register_model(run.info.run_id, "model", "m")
    v2 = register_model(run.info.run_id, "model", "m")
    client = MlflowClient(store)
    promote("m", v1.version, "Production")
    assert client.get_model_version_by_alias("m", "production").version == v1.version
    promote("m", v2.version, "Production")
    promote("m", v1.version, "Archived")
    assert client.get_model_version_by_alias("m", "production").version == v2.version
    mv1 = client.get_model_version("m", v1.version)
    assert mv1.aliases == ["archived"] and mv1.tags["stage"] == "Archived"
    promote("m", v1.version, "None")
    assert client.get_model_version("m", v1.version).aliases == []
    promote("m", v2.version, "champion")
    assert client.get_model_version_by_alias("m", "champion").version == v2.version
