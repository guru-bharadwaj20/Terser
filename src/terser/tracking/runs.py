"""MLflow run and Model Registry helpers.

Backend: MLflow 3.x deprecates the plain file store and the Model Registry requires a database
backend, so the default tracking URI is SQLite at ``<repo>/mlruns/mlflow.db`` with artifacts under
``<repo>/mlruns/artifacts``. Set ``MLFLOW_TRACKING_URI`` to override.

Registry: stage transitions (``transition_model_version_stage``) are deprecated since MLflow 2.9.
The plan's stages ``None -> Staging -> Production -> Archived`` are therefore mapped to registered
model *aliases* (``staging``, ``production``, ``archived``). An alias points at exactly one version,
so promoting a version moves the alias. ``None`` removes every stage alias from that version.
Each version also gets a ``stage`` tag mirroring its latest stage for easy filtering.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow
from mlflow import MlflowClient
from mlflow.entities import Run
from mlflow.entities.model_registry import ModelVersion
from mlflow.exceptions import MlflowException

from terser.config import project_root

REPO_ROOT: Path = project_root()
REQUIRED_TAGS: tuple[str, ...] = ("phase", "row", "task", "hardware", "data_hash", "git_sha")
STAGES: tuple[str, ...] = ("None", "Staging", "Production", "Archived")
RUN_NAME_RE = re.compile(r"^\d+-[A-Za-z0-9_.]+-[A-Za-z0-9_.]+-[A-Za-z0-9_.]+-\d+$")


def default_tracking_uri(root: Path = REPO_ROOT) -> str:
    """Return the SQLite tracking URI under ``<root>/mlruns`` (forward slashes, Windows-safe)."""
    return f"sqlite:///{(root / 'mlruns' / 'mlflow.db').resolve().as_posix()}"


def setup_tracking() -> str:
    """Point MLflow at ``MLFLOW_TRACKING_URI`` or the repo-local SQLite store; return the URI."""
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        (REPO_ROOT / "mlruns").mkdir(parents=True, exist_ok=True)
        uri = default_tracking_uri()
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    return uri


def get_git_sha(cwd: Path = REPO_ROOT) -> str:
    """Return ``git rev-parse HEAD`` (read-only) or ``unknown`` if git/the repo is unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    sha = out.stdout.strip()
    return sha if out.returncode == 0 and sha else "unknown"


def data_hash(root: Path = REPO_ROOT) -> str:
    """Return the sha256 (first 16 hex chars) of ``<root>/dvc.lock``, or ``none`` if absent."""
    lock = root / "dvc.lock"
    if not lock.is_file():
        return "none"
    return hashlib.sha256(lock.read_bytes()).hexdigest()[:16]


def validate_run_name(run_name: str) -> bool:
    """Check ``<phase>-<task>-<model>-<variant>-<seed>``; warn (never raise) if it doesn't match."""
    ok = bool(RUN_NAME_RE.match(run_name))
    if not ok:
        warnings.warn(
            f"Run name {run_name!r} does not follow <phase>-<task>-<model>-<variant>-<seed>",
            UserWarning,
            stacklevel=3,
        )
    return ok


def _ensure_experiment(name: str) -> str:
    """Get or create an experiment.

    For SQLite stores, artifacts go to ``<db dir>/artifacts/<name>`` (i.e. ``mlruns/artifacts`` by
    default) instead of MLflow's CWD-relative ``./mlruns``.
    """
    exp = mlflow.get_experiment_by_name(name)
    if exp is not None:
        return str(exp.experiment_id)
    location = None
    uri = mlflow.get_tracking_uri()
    if uri.startswith("sqlite:///"):
        db = Path(uri.removeprefix("sqlite:///"))
        location = f"{(db.parent / 'artifacts').resolve().as_uri()}/{name}"
    return str(mlflow.create_experiment(name, artifact_location=location))


@contextmanager
def start_run(
    *,
    phase: int,
    row: str,
    task: str,
    run_name: str,
    hardware: str = "cpu",
    params: dict[str, Any] | None = None,
    experiment: str | None = None,
) -> Iterator[Run]:
    """Start an MLflow run with the six required tags and ``params`` logged; yield the run."""
    setup_tracking()
    validate_run_name(run_name)
    exp_id = _ensure_experiment(experiment or f"phase{phase}")
    tags = {
        "phase": str(phase),
        "row": row,
        "task": task,
        "hardware": hardware,
        "data_hash": data_hash(),
        "git_sha": get_git_sha(),
    }
    with mlflow.start_run(experiment_id=exp_id, run_name=run_name, tags=tags) as run:
        if params:
            mlflow.log_params(params)
        yield run


def register_model(run_id: str, artifact_path: str, name: str) -> ModelVersion:
    """Register ``runs:/<run_id>/<artifact_path>`` as a new version of model ``name``.

    In MLflow 3, ``log_model(name=...)`` stores models as logged models (``models:/m-...``) rather
    than run artifacts; MLflow resolves the ``runs:/`` URI to that logged model (with a warning).
    """
    setup_tracking()
    return mlflow.register_model(f"runs:/{run_id}/{artifact_path}", name)


def promote(name: str, version: int | str, stage_or_alias: str) -> None:
    """Move ``name`` version ``version`` to a stage (mapped to an alias) or an arbitrary alias.

    ``Staging``/``Production``/``Archived`` become lowercase aliases and drop the version's other
    stage aliases; ``None`` only drops them. Any other value is set verbatim as an alias.
    """
    setup_tracking()
    client = MlflowClient()
    version = str(version)
    stage = next((s for s in STAGES if s.lower() == stage_or_alias.lower()), None)
    if stage is None:
        client.set_registered_model_alias(name, stage_or_alias, version)
        return
    current = client.get_model_version(name, version).aliases
    for s in STAGES[1:]:
        if s.lower() in current and s != stage:
            client.delete_registered_model_alias(name, s.lower())
    if stage != "None":
        # The alias moves here: its previous holder no longer has this stage.
        try:
            previous = client.get_model_version_by_alias(name, stage.lower()).version
        except MlflowException:
            previous = None
        if previous is not None and str(previous) != version:
            client.set_model_version_tag(name, str(previous), "stage", "None")
        client.set_registered_model_alias(name, stage.lower(), version)
    client.set_model_version_tag(name, version, "stage", stage)
