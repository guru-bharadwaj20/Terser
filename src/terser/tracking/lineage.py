"""Model registry entries with lineage (plan rows **11.2.2** and **11.2.3**).

Registered models (row 11.2.2): :data:`MODEL_NAMES` maps the three served components to their
registry names. Stages ``None -> Staging -> Production -> Archived`` are the aliases managed by
:func:`terser.tracking.runs.promote`.

Lineage (row 11.2.3): :func:`register_with_lineage` registers a run's model and writes, as
model-version tags,

* ``git_sha``: code that trained it (copied from the source run's tag, else the current HEAD);
* ``data_hash``: the DVC data hash (sha256 prefix of ``dvc.lock``) from the source run;
* ``config_json`` and ``config_sha256``: the resolved training config;
* ``checkpoints_passed``: comma-separated checkpoint ids the model passed (e.g. ``C5.2,C8.3``);
* ``source_run_id``: the MLflow run holding metrics and artifacts.

:func:`trace` resolves a stage alias (default ``production``) to its version and returns that
lineage, raising :class:`LineageError` if any field is missing. A model without full lineage
cannot be traced and so must not be promoted (see :mod:`terser.pipelines.promotion`).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from terser.tracking.runs import data_hash, get_git_sha, register_model, setup_tracking

__all__ = [
    "LINEAGE_TAGS",
    "MODEL_NAMES",
    "LineageError",
    "register_with_lineage",
    "trace",
]

MODEL_NAMES: Mapping[str, str] = {
    "gate": "terser-gate",
    "tagger": "terser-tagger",
    "rewriter": "terser-rewriter",
}
LINEAGE_TAGS: tuple[str, ...] = (
    "git_sha",
    "data_hash",
    "config_json",
    "config_sha256",
    "checkpoints_passed",
    "source_run_id",
)


class LineageError(LookupError):
    """A registered model version is missing lineage."""


def _config_json(config: Mapping[str, Any]) -> str:
    return json.dumps(config, sort_keys=True, default=str)


def register_with_lineage(
    run_id: str,
    artifact_path: str,
    component: str,
    config: Mapping[str, Any],
    checkpoints_passed: Sequence[str],
) -> Any:
    """Register ``runs:/run_id/artifact_path`` under ``MODEL_NAMES[component]`` with lineage."""
    from mlflow import MlflowClient

    if component not in MODEL_NAMES:
        raise KeyError(f"unknown component {component!r}; choose from {sorted(MODEL_NAMES)}")
    setup_tracking()
    client = MlflowClient()
    run_tags = client.get_run(run_id).data.tags
    name = MODEL_NAMES[component]
    mv = register_model(run_id, artifact_path, name)
    cfg = _config_json(config)
    tags = {
        "git_sha": run_tags.get("git_sha") or get_git_sha(),
        "data_hash": run_tags.get("data_hash") or data_hash(),
        "config_json": cfg,
        "config_sha256": hashlib.sha256(cfg.encode()).hexdigest(),
        "checkpoints_passed": ",".join(checkpoints_passed),
        "source_run_id": run_id,
        "component": component,
    }
    for k, v in tags.items():
        client.set_model_version_tag(name, mv.version, k, v)
    client.set_model_version_tag(name, mv.version, "stage", "None")
    return mv


def trace(component_or_name: str, alias: str = "production") -> dict[str, Any]:
    """Lineage of the version holding ``alias``; :class:`LineageError` if incomplete."""
    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException

    setup_tracking()
    name = MODEL_NAMES.get(component_or_name, component_or_name)
    client = MlflowClient()
    try:
        mv = client.get_model_version_by_alias(name, alias)
    except MlflowException as exc:
        raise LineageError(f"{name} has no version with alias {alias!r}") from exc
    missing = [t for t in LINEAGE_TAGS if not mv.tags.get(t) and t != "checkpoints_passed"]
    if "checkpoints_passed" not in mv.tags:
        missing.append("checkpoints_passed")
    if missing:
        raise LineageError(f"{name} v{mv.version} lacks lineage tags {missing}")
    if mv.tags["config_sha256"] != hashlib.sha256(mv.tags["config_json"].encode()).hexdigest():
        raise LineageError(f"{name} v{mv.version}: config hash does not match config_json")
    return {
        "name": name,
        "version": int(mv.version),
        "alias": alias,
        "git_sha": mv.tags["git_sha"],
        "data_hash": mv.tags["data_hash"],
        "config": json.loads(mv.tags["config_json"]),
        "config_sha256": mv.tags["config_sha256"],
        "checkpoints_passed": [c for c in mv.tags["checkpoints_passed"].split(",") if c],
        "source_run_id": mv.tags["source_run_id"],
    }
