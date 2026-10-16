"""Serving config model paths resolve against the repo root, whatever the cwd."""

from __future__ import annotations

from pathlib import Path

import pytest

from terser.serving.api import REPO_ROOT, resolve_path


def test_relative_paths_resolve_to_repo_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    assert resolve_path("models/fast/gate.onnx") == REPO_ROOT / "models/fast/gate.onnx"
    assert (REPO_ROOT / "configs" / "serving" / "default.yaml").is_file()


def test_absolute_paths_are_kept(tmp_path: Path) -> None:
    p = tmp_path / "gate.onnx"
    assert resolve_path(p) == p
