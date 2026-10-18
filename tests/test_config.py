"""Tests for the Hydra config system (terser.config, configs/, terser.hello)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from terser.config import (
    RESOLVER_NAME,
    configs_dir,
    load_config,
    project_root,
    register_resolvers,
)


def test_compose_defaults() -> None:
    cfg = load_config()
    assert cfg.seed == 0
    assert cfg.experiment.name == "hello"
    assert cfg.experiment.phase == 0
    assert cfg.hardware.device == "cpu"


def test_overrides() -> None:
    cfg = load_config(["seed=5", "experiment.params.repeat=3"])
    assert cfg.seed == 5
    assert cfg.experiment.params.repeat == 3


def test_paths_absolute_and_inside_repo() -> None:
    root = project_root()
    assert (configs_dir() / "config.yaml").is_file()
    cfg = load_config()
    for key, value in cfg.paths.items():
        path = Path(value)
        assert path.is_absolute(), key
        assert path.resolve().is_relative_to(root), key


def test_resolver_idempotent() -> None:
    register_resolvers()
    register_resolvers()
    assert OmegaConf.has_resolver(RESOLVER_NAME)
    assert load_config().paths.root == project_root().as_posix()


def test_root_env_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TERSER_ROOT", str(tmp_path))
    assert project_root() == tmp_path.resolve()


def test_hardware_switch_rtx4500ada() -> None:
    cfg = load_config(["hardware=rtx4500ada"])
    assert cfg.hardware.device == "cuda:0"
    assert cfg.hardware.precision == "bf16"
    assert cfg.hardware.total_vram_gb == 24


def test_hello_subprocess(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "terser.hello",
            "experiment=hello",
            "seed=7",
            f"hydra.run.dir='{run_dir.as_posix()}'",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "seed: 7" in result.stdout
    assert "name: hello" in result.stdout
