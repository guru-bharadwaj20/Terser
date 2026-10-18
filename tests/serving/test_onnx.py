"""ONNX fast-path export (row 11.3.2): a sklearn gate runs under ONNX Runtime on CPU."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("onnxruntime")
pytest.importorskip("skl2onnx")

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from terser.features.shared import GATE_FEATURE_NAMES, extract_features
from terser.serving.onnx_export import check_parity, export_sklearn
from terser.serving.paths import OnnxGate


def test_export_and_serve_gate(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = len(GATE_FEATURE_NAMES)
    x = rng.normal(size=(200, n)).astype(np.float32)
    y = (x[:, -1] + 0.3 * rng.normal(size=200) < 0).astype(int)
    model = make_pipeline(StandardScaler(), LogisticRegression()).fit(x, y)
    path = export_sklearn(model, n, tmp_path / "gate.onnx")
    assert check_parity(model, path, x) <= 1e-4
    gate = OnnxGate(path)
    assert gate.session.get_providers() == ["CPUExecutionProvider"]

    def words(s: str) -> int:
        return len(s.split())

    orig = extract_features("please only use Python 3 and never numpy", counter=words)
    rew = extract_features("use Python 3", counter=words)
    p = gate.score(orig, rew)
    assert 0.0 <= p <= 1.0
