"""GPU job queue: cross-job validation, priority-aware topological ordering, and the runner.

Jobs execute strictly sequentially, so two ``exclusive`` jobs (and, by extension, vLLM serving and
training) never share the card.
"""

from __future__ import annotations

import heapq
import json
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import yaml

from terser.gpu_queue.schema import JobSpec, parse_spec

DEFAULT_MAX_VRAM_GB = 22.0
HARDWARE_YAML = Path("configs/hardware/rtx4500ada.yaml")
GATES_FILE = "gates.yaml"
STATE_FILE = Path(".cache/gpu_queue_state.json")
NO_GPU_MSG = "GPU not connected — jobs remain queued"


class QueueError(RuntimeError):
    """Raised when the queue is invalid or cannot run."""


def spec_files(queue_dir: Path) -> list[Path]:
    """Job spec files in ``queue_dir`` (``gates.yaml`` excluded), sorted by name."""
    if not queue_dir.is_dir():
        return []
    files = list(queue_dir.glob("*.yaml")) + list(queue_dir.glob("*.yml"))
    return sorted(p for p in files if p.name != GATES_FILE)


def max_vram_gb(root: Path) -> float:
    """``max_vram_gb`` from the RTX 4500 Ada hardware config if present, else 22."""
    path = root / HARDWARE_YAML
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        value = data.get("max_vram_gb", DEFAULT_MAX_VRAM_GB)
        return float(value)
    except (OSError, yaml.YAMLError, AttributeError, TypeError, ValueError):
        return DEFAULT_MAX_VRAM_GB


def load_jobs(queue_dir: Path) -> tuple[list[JobSpec], list[str]]:
    """Parse all spec files; returns successfully parsed jobs and per-file errors."""
    jobs: list[JobSpec] = []
    errors: list[str] = []
    for path in spec_files(queue_dir):
        spec, errs = parse_spec(path)
        errors.extend(errs)
        if spec is not None:
            jobs.append(spec)
    return jobs, errors


def _find_cycles(jobs: dict[str, JobSpec]) -> list[list[str]]:
    color: dict[str, int] = {j: 0 for j in jobs}  # 0 unvisited, 1 on stack, 2 done
    stack: list[str] = []
    cycles: list[list[str]] = []

    def visit(node: str) -> None:
        color[node] = 1
        stack.append(node)
        for dep in jobs[node].job_deps:
            if dep not in jobs:
                continue
            if color[dep] == 1:
                cycles.append(stack[stack.index(dep) :] + [dep])
            elif color[dep] == 0:
                visit(dep)
        stack.pop()
        color[node] = 2

    for node in sorted(jobs):
        if color[node] == 0:
            visit(node)
    return cycles


def validate(queue_dir: Path | str, root: Path | str | None = None) -> list[str]:
    """Validate every spec in ``queue_dir``. Returns a list of error messages (empty = valid).

    ``root`` is the repo root used to resolve ``cpu_smoke_test`` and the hardware config
    (default: current working directory).
    """
    queue_dir, root = Path(queue_dir), Path(root) if root else Path.cwd()
    jobs, errors = load_jobs(queue_dir)
    limit = max_vram_gb(root)

    by_id: dict[str, JobSpec] = {}
    for job in jobs:
        if job.id in by_id:
            errors.append(f"{job.path.name}: field 'id': duplicate id '{job.id}'")
        else:
            by_id[job.id] = job
    for job in jobs:
        name = job.path.name
        for dep in job.job_deps:
            if dep not in by_id:
                errors.append(f"{name}: field 'depends_on': unknown job dependency '{dep}'")
        if job.est_vram_gb > limit:
            errors.append(
                f"{name}: field 'est_vram_gb': {job.est_vram_gb:g} GB exceeds "
                f"max_vram_gb {limit:g} GB"
            )
        if not (root / job.cpu_smoke_test).exists():
            errors.append(
                f"{name}: field 'cpu_smoke_test': path '{job.cpu_smoke_test}' does not exist"
            )
    for cycle in _find_cycles(by_id):
        errors.append(f"dependency cycle: {' -> '.join(cycle)}")
    return errors


def order_jobs(jobs: list[JobSpec]) -> list[JobSpec]:
    """Topological order; among ready jobs the lowest priority number goes first (tie -> id).

    Raises ``QueueError`` on cycles. Unknown job dependencies are ignored here (``validate``
    reports them).
    """
    by_id = {j.id: j for j in jobs}
    indeg = {j.id: sum(1 for d in j.job_deps if d in by_id) for j in jobs}
    children: dict[str, list[str]] = {j.id: [] for j in jobs}
    for j in jobs:
        for d in j.job_deps:
            if d in by_id:
                children[d].append(j.id)
    ready = [(j.priority, j.id) for j in jobs if indeg[j.id] == 0]
    heapq.heapify(ready)
    out: list[JobSpec] = []
    while ready:
        _, jid = heapq.heappop(ready)
        out.append(by_id[jid])
        for child in children[jid]:
            indeg[child] -= 1
            if indeg[child] == 0:
                heapq.heappush(ready, (by_id[child].priority, child))
    if len(out) != len(jobs):
        raise QueueError("dependency cycle detected; run `validate` for details")
    return out


def _load_valid(queue_dir: Path, root: Path) -> list[JobSpec]:
    errors = validate(queue_dir, root)
    if errors:
        raise QueueError("queue is invalid:\n  " + "\n  ".join(errors))
    return order_jobs(load_jobs(queue_dir)[0])


def format_dry_run(ordered: list[JobSpec]) -> str:
    """Render the dry-run table plus total estimated GPU-hours."""
    if not ordered:
        return "queue empty"
    header = ("#", "id", "phase", "prio", "VRAM GB", "hours", "exclusive", "gates")
    rows = [
        (
            str(i),
            j.id,
            str(j.phase),
            str(j.priority),
            f"{j.est_vram_gb:g}",
            f"{j.est_hours:g}",
            "yes" if j.exclusive else "no",
            ", ".join(j.gates) or "-",
        )
        for i, j in enumerate(ordered, 1)
    ]
    widths = [max(len(r[c]) for r in [header, *rows]) for c in range(len(header))]
    lines = ["  ".join(cell.ljust(w) for cell, w in zip(r, widths)).rstrip() for r in rows]
    head = "  ".join(h.ljust(w) for h, w in zip(header, widths)).rstrip()
    total = sum(j.est_hours for j in ordered)
    return "\n".join([head, "-" * len(head), *lines, "", f"Total estimated GPU-hours: {total:g}"])


def dry_run(queue_dir: Path | str, root: Path | str | None = None) -> list[JobSpec]:
    """Validate and return jobs in execution order (raises ``QueueError`` if invalid)."""
    return _load_valid(Path(queue_dir), Path(root) if root else Path.cwd())


def cuda_available() -> bool:
    """True if torch is importable and sees a CUDA device (lazy, optional import)."""
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 - broken driver installs count as "no GPU"
        return False


def passed_gates(queue_dir: Path) -> set[str]:
    """Checkpoint ids listed in ``<queue_dir>/gates.yaml`` (missing file = none passed)."""
    path = queue_dir / GATES_FILE
    if not path.exists():
        return set()
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return set()
    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        raise QueueError(f"{path}: must be a YAML list of passed checkpoint ids")
    return set(data)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def run(
    queue_dir: Path | str,
    root: Path | str | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, dict[str, object]]:
    """Run queued jobs sequentially in dry-run order and return the final state.

    Raises ``QueueError`` if no CUDA device is visible or the queue is invalid. Jobs whose gates
    are not passed, or whose job dependencies did not finish, are marked ``blocked``. Jobs already
    ``done`` in the state file are skipped.
    """
    queue_dir, root = Path(queue_dir), Path(root) if root else Path.cwd()
    if not cuda_available():
        raise QueueError(NO_GPU_MSG)
    ordered = _load_valid(queue_dir, root)
    gates = passed_gates(queue_dir)
    state_path = root / STATE_FILE
    state_path.parent.mkdir(parents=True, exist_ok=True)
    previous: dict[str, dict[str, object]] = {}
    if state_path.exists():
        previous = json.loads(state_path.read_text(encoding="utf-8")).get("jobs", {})
    state: dict[str, dict[str, object]] = {
        j.id: (
            previous[j.id]
            if previous.get(j.id, {}).get("status") == "done"
            else {"status": "pending"}
        )
        for j in ordered
    }

    def save() -> None:
        state_path.write_text(json.dumps({"jobs": state}, indent=2), encoding="utf-8")

    save()
    for job in ordered:
        if state[job.id]["status"] == "done":
            log(f"[skip] {job.id}: already done")
            continue
        missing_gates = [g for g in job.gates if g not in gates]
        bad_deps = [d for d in job.job_deps if state[d]["status"] != "done"]
        if missing_gates or bad_deps:
            reason = []
            if missing_gates:
                reason.append(f"gates not passed: {', '.join(missing_gates)}")
            if bad_deps:
                reason.append(f"dependencies not done: {', '.join(bad_deps)}")
            state[job.id] = {"status": "blocked", "reason": "; ".join(reason)}
            log(f"[blocked] {job.id}: {state[job.id]['reason']}")
            save()
            continue
        state[job.id] = {"status": "running", "started": _now()}
        save()
        log(f"[run] {job.id}: {job.command}")
        proc = subprocess.run(job.command, shell=True, cwd=root, check=False)
        state[job.id].update(
            status="done" if proc.returncode == 0 else "failed",
            returncode=proc.returncode,
            finished=_now(),
        )
        log(f"[{state[job.id]['status']}] {job.id} (exit {proc.returncode})")
        save()
    return state
