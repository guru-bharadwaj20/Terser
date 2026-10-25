"""Tests for the GPU job queue: schema, validation, ordering, CLI, and the runner."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import yaml

from terser.gpu_queue import (
    NO_GPU_MSG,
    QueueError,
    dry_run,
    format_dry_run,
    run,
    validate,
)
from terser.gpu_queue.__main__ import main
from terser.gpu_queue.queue import STATE_FILE, cuda_available

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "gpu_queue"
PY = f'"{sys.executable}"'


def spec(job_id: str, **over: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": job_id,
        "phase": 8,
        "depends_on": [],
        "command": f'{PY} -c "print(1)"',
        "est_vram_gb": 4,
        "est_hours": 1,
        "exclusive": True,
        "priority": 1,
        "cpu_smoke_test": "smoke.py",
        "outputs": [],
    }
    base.update(over)
    return base


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / "smoke.py").write_text("", encoding="utf-8")
    (tmp_path / "gpu_queue").mkdir()
    return tmp_path


def write(root: Path, data: dict[str, Any], name: str | None = None) -> None:
    path = root / "gpu_queue" / f"{name or data['id']}.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")


def ids(root: Path) -> list[str]:
    return [j.id for j in dry_run(root / "gpu_queue", root)]


# --- fixtures / DoD -------------------------------------------------------------------------


def test_valid_fixture_orders_sample_jobs() -> None:
    assert validate(FIXTURES / "valid", ROOT) == []
    ordered = dry_run(FIXTURES / "valid", ROOT)
    assert [j.id for j in ordered] == ["sft-tiny-a", "sft-tiny-b"]
    assert ordered[0].gates == ["C7.1"] and ordered[1].gates == []
    table = format_dry_run(ordered)
    assert "sft-tiny-a" in table and "C7.1" in table
    assert "Total estimated GPU-hours: 2" in table


def test_cycle_fixture_detected() -> None:
    errors = validate(FIXTURES / "invalid" / "cycle", ROOT)
    assert any("dependency cycle" in e for e in errors)
    with pytest.raises(QueueError):
        dry_run(FIXTURES / "invalid" / "cycle", ROOT)


# --- ordering -------------------------------------------------------------------------------


def test_priority_among_ready_jobs(root: Path) -> None:
    write(root, spec("b", priority=2))
    write(root, spec("a", priority=2))
    write(root, spec("z", priority=1))
    assert ids(root) == ["z", "a", "b"]


def test_dependency_overrides_priority(root: Path) -> None:
    write(root, spec("low", priority=5))
    write(root, spec("urgent", priority=1, depends_on=["low"]))
    write(root, spec("mid", priority=2))
    # 'urgent' cannot start before 'low'; 'mid' is ready and beats 'low'.
    assert ids(root) == ["mid", "low", "urgent"]


def test_gates_do_not_affect_order(root: Path) -> None:
    write(root, spec("gated", priority=1, depends_on=["C7.1", "C2"]))
    write(root, spec("free", priority=2))
    assert ids(root) == ["gated", "free"]


# --- validation errors ----------------------------------------------------------------------


def test_empty_queue(root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert validate(root / "gpu_queue", root) == []
    assert main(["dry-run", "--dir", str(root / "gpu_queue"), "--root", str(root)]) == 0
    assert "queue empty" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("override", "field"),
    [
        ({"phase": "eight"}, "phase"),
        ({"depends_on": "C7.1"}, "depends_on"),
        ({"command": ""}, "command"),
        ({"est_vram_gb": 0}, "est_vram_gb"),
        ({"est_hours": -1}, "est_hours"),
        ({"exclusive": "yes"}, "exclusive"),
        ({"priority": 0}, "priority"),
        ({"priority": True}, "priority"),
        ({"outputs": "models/"}, "outputs"),
        ({"cpu_smoke_test": 3}, "cpu_smoke_test"),
    ],
)
def test_bad_field_types(root: Path, override: dict[str, Any], field: str) -> None:
    write(root, spec("job", **override))
    errors = validate(root / "gpu_queue", root)
    assert any("job.yaml" in e and f"'{field}'" in e for e in errors), errors


def test_unknown_and_missing_fields(root: Path) -> None:
    data = spec("job", gpu="a100")
    del data["priority"]
    write(root, data)
    errors = validate(root / "gpu_queue", root)
    assert any("'gpu'" in e and "unknown field" in e for e in errors)
    assert any("'priority'" in e and "missing" in e for e in errors)


def test_id_must_match_filename(root: Path) -> None:
    write(root, spec("job"), name="other")
    assert any("other.yaml" in e and "'id'" in e for e in validate(root / "gpu_queue", root))


def test_not_a_mapping(root: Path) -> None:
    (root / "gpu_queue" / "bad.yaml").write_text("- 1\n- 2\n", encoding="utf-8")
    assert any("bad.yaml" in e for e in validate(root / "gpu_queue", root))


def test_duplicate_ids(root: Path) -> None:
    write(root, spec("job"), name="job")
    (root / "gpu_queue" / "job.yml").write_text(yaml.safe_dump(spec("job")), encoding="utf-8")
    assert any("duplicate id 'job'" in e for e in validate(root / "gpu_queue", root))


def test_unknown_job_dependency(root: Path) -> None:
    write(root, spec("job", depends_on=["C7.1", "ghost"]))
    errors = validate(root / "gpu_queue", root)
    assert len(errors) == 1 and "unknown job dependency 'ghost'" in errors[0]


def test_cycle_detection(root: Path) -> None:
    write(root, spec("a", depends_on=["c"]))
    write(root, spec("b", depends_on=["a"]))
    write(root, spec("c", depends_on=["b"]))
    write(root, spec("self", depends_on=["self"]))
    errors = [e for e in validate(root / "gpu_queue", root) if "cycle" in e]
    assert len(errors) == 2
    assert any("self -> self" in e for e in errors)


def test_vram_over_default_budget(root: Path) -> None:
    write(root, spec("big", est_vram_gb=23.5))
    write(root, spec("ok", est_vram_gb=23))
    errors = validate(root / "gpu_queue", root)
    assert len(errors) == 1 and "big.yaml" in errors[0] and "'est_vram_gb'" in errors[0]


def test_vram_budget_read_from_hardware_yaml(root: Path) -> None:
    hw = root / "configs" / "hardware"
    hw.mkdir(parents=True)
    (hw / "rtx4500ada.yaml").write_text("max_vram_gb: 10\n", encoding="utf-8")
    write(root, spec("mid", est_vram_gb=12))
    assert any("exceeds max_vram_gb 10" in e for e in validate(root / "gpu_queue", root))


def test_missing_smoke_test(root: Path) -> None:
    write(root, spec("job", cpu_smoke_test="tests/nope.py"))
    errors = validate(root / "gpu_queue", root)
    assert len(errors) == 1 and "'cpu_smoke_test'" in errors[0]


def test_cli_validate_exit_codes(root: Path) -> None:
    args = ["validate", "--dir", str(root / "gpu_queue"), "--root", str(root)]
    write(root, spec("job"))
    assert main(args) == 0
    write(root, spec("job2", priority=0))
    assert main(args) == 1


# --- run ------------------------------------------------------------------------------------


@pytest.fixture
def no_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)


def fake_torch(monkeypatch: pytest.MonkeyPatch, available: bool) -> None:
    mod = types.ModuleType("torch")
    mod.cuda = types.SimpleNamespace(is_available=lambda: available)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", mod)


def test_run_refuses_without_torch(
    root: Path, no_torch: None, capsys: pytest.CaptureFixture[str]
) -> None:
    write(root, spec("job"))
    assert not cuda_available()
    with pytest.raises(QueueError, match="GPU not connected"):
        run(root / "gpu_queue", root)
    assert main(["run", "--dir", str(root / "gpu_queue"), "--root", str(root)]) != 0
    assert NO_GPU_MSG in capsys.readouterr().err
    assert not (root / STATE_FILE).exists()


def test_run_refuses_when_cuda_unavailable(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch(monkeypatch, available=False)
    with pytest.raises(QueueError, match="GPU not connected"):
        run(root / "gpu_queue", root)


def test_run_with_fake_cuda(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake_torch(monkeypatch, available=True)
    write(root, spec("first", priority=1, command=f"{PY} -c \"open('first.txt', 'w')\""))
    write(root, spec("gated", priority=2, depends_on=["C7.1"]))
    write(root, spec("after-gated", priority=1, depends_on=["gated"]))
    write(root, spec("boom", priority=3, command=f'{PY} -c "import sys; sys.exit(3)"'))
    write(root, spec("after-boom", priority=1, depends_on=["boom", "first"]))
    write(root, spec("unrelated", priority=9))

    state = run(root / "gpu_queue", root, log=lambda _: None)
    status = {k: v["status"] for k, v in state.items()}
    assert status == {
        "first": "done",
        "gated": "blocked",
        "after-gated": "blocked",
        "boom": "failed",
        "after-boom": "blocked",
        "unrelated": "done",
    }
    assert (root / "first.txt").exists()  # commands run with cwd = root
    saved = json.loads((root / STATE_FILE).read_text(encoding="utf-8"))["jobs"]
    assert {k: v["status"] for k, v in saved.items()} == status
    assert saved["boom"]["returncode"] == 3
    assert "C7.1" in saved["gated"]["reason"]


def test_run_gates_file_unblocks_and_skips_done(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_torch(monkeypatch, available=True)
    write(root, spec("gated", depends_on=["C7.1"]))
    assert run(root / "gpu_queue", root, log=lambda _: None)["gated"]["status"] == "blocked"
    (root / "gpu_queue" / "gates.yaml").write_text("- C7.1\n", encoding="utf-8")
    assert validate(root / "gpu_queue", root) == []  # gates.yaml is not a job spec
    assert run(root / "gpu_queue", root, log=lambda _: None)["gated"]["status"] == "done"
    logs: list[str] = []
    run(root / "gpu_queue", root, log=logs.append)
    assert any("already done" in line for line in logs)
