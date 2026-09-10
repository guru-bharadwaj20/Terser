"""GPU job spec schema: parsing and per-file field validation (no cross-job checks)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

CHECKPOINT_RE = re.compile(r"^C\d+(\.\d+)?$")

REQUIRED_FIELDS: tuple[str, ...] = (
    "id",
    "phase",
    "depends_on",
    "command",
    "est_vram_gb",
    "est_hours",
    "exclusive",
    "priority",
    "cpu_smoke_test",
    "outputs",
)


@dataclass(frozen=True)
class JobSpec:
    """One queued GPU job, as declared in ``gpu_queue/<id>.yaml``."""

    id: str
    phase: int
    depends_on: list[str]
    command: str
    est_vram_gb: float
    est_hours: float
    exclusive: bool
    priority: int
    cpu_smoke_test: str
    outputs: list[str]
    path: Path = field(default=Path(), compare=False)

    @property
    def gates(self) -> list[str]:
        """External checkpoint gates (e.g. ``C7.1``) this job requires."""
        return [d for d in self.depends_on if is_checkpoint(d)]

    @property
    def job_deps(self) -> list[str]:
        """Dependencies on other queued jobs."""
        return [d for d in self.depends_on if not is_checkpoint(d)]


def is_checkpoint(dep: str) -> bool:
    """True if ``dep`` names an external checkpoint gate rather than a job id."""
    return bool(CHECKPOINT_RE.match(dep))


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _is_str_list(v: Any) -> bool:
    return isinstance(v, list) and all(isinstance(x, str) and x for x in v)


def parse_spec(path: Path) -> tuple[JobSpec | None, list[str]]:
    """Parse and validate one spec file. Returns ``(spec, errors)``; spec is None on any error."""
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        return None, [f"{path.name}: cannot parse YAML: {exc}"]
    if not isinstance(data, dict):
        return None, [f"{path.name}: top level must be a mapping"]

    errors: list[str] = []

    def err(fld: str, msg: str) -> None:
        errors.append(f"{path.name}: field '{fld}': {msg}")

    for key in data:
        if key not in REQUIRED_FIELDS:
            err(str(key), "unknown field")
    for key in REQUIRED_FIELDS:
        if key not in data:
            err(key, "missing required field")
    if errors:
        return None, errors

    if not isinstance(data["id"], str) or not data["id"]:
        err("id", "must be a non-empty string")
    elif data["id"] != path.stem:
        err("id", f"'{data['id']}' must equal filename stem '{path.stem}'")
    if not _is_int(data["phase"]) or data["phase"] < 0:
        err("phase", "must be a non-negative integer")
    if not _is_str_list(data["depends_on"]):
        err("depends_on", "must be a list of non-empty strings")
    if not isinstance(data["command"], str) or not data["command"].strip():
        err("command", "must be a non-empty string")
    for fld in ("est_vram_gb", "est_hours"):
        if not _is_number(data[fld]) or data[fld] <= 0:
            err(fld, "must be a number > 0")
    if not isinstance(data["exclusive"], bool):
        err("exclusive", "must be a boolean")
    if not _is_int(data["priority"]) or data["priority"] < 1:
        err("priority", "must be an integer >= 1 (1 = highest)")
    if not isinstance(data["cpu_smoke_test"], str) or not data["cpu_smoke_test"]:
        err("cpu_smoke_test", "must be a non-empty path string")
    if not _is_str_list(data["outputs"]):
        err("outputs", "must be a list of non-empty strings")
    if errors:
        return None, errors

    spec = JobSpec(
        id=data["id"],
        phase=data["phase"],
        depends_on=list(data["depends_on"]),
        command=data["command"],
        est_vram_gb=float(data["est_vram_gb"]),
        est_hours=float(data["est_hours"]),
        exclusive=data["exclusive"],
        priority=data["priority"],
        cpu_smoke_test=data["cpu_smoke_test"],
        outputs=list(data["outputs"]),
        path=path,
    )
    return spec, []
