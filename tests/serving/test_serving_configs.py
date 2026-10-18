"""The committed serving configs: priced targets exist, the ONNX gate threshold is traceable."""

from __future__ import annotations

import json
from typing import Any

import pytest

from terser.serving.api import REPO_ROOT, build_service, load_config
from terser.tokenizers.count import TARGETS

CONFIGS = [
    REPO_ROOT / "configs" / "serving" / "default.yaml",
    REPO_ROOT / "configs" / "serving" / "fast_onnx.yaml",
    REPO_ROOT / "docker" / "serving.compose.yaml",
]


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_priced_targets_are_countable(path: object) -> None:
    prices = load_config(str(path))["pricing"]["target_input_per_mtok"]
    assert set(prices) <= set(TARGETS)
    assert "phi3.5" in prices and "llama-3.1" not in prices  # ADR-0011


def test_fast_onnx_threshold_is_the_reported_operating_point() -> None:
    cfg = load_config(str(CONFIGS[1]))
    report = json.loads((REPO_ROOT / "reports" / "p11_fast_path.json").read_text("utf-8"))
    chosen = report["operating_point"]["chosen"]
    assert cfg["gate_threshold"] == chosen["threshold"]
    assert chosen["sensitivity"] >= 0.95


def test_onnx_gate_requires_explicit_threshold(require_dvc: Any) -> None:
    cfg = load_config(str(CONFIGS[1]))
    require_dvc(REPO_ROOT / cfg["gate_onnx"], "dvc pull models/fast")
    cfg.pop("gate_threshold")
    cfg.pop("tagger_onnx")
    with pytest.raises(KeyError, match="gate_threshold"):
        build_service(cfg)
