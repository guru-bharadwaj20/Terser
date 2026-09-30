"""The worked example in docs/derivations/backprop.md section 4 (row 4.1.7)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from terser.models.neural.numpy_mlp import NumpyMLP

DOC = Path(__file__).resolve().parents[1] / "docs" / "derivations" / "backprop.md"


def _example() -> tuple[NumpyMLP, np.ndarray, np.ndarray]:
    m = NumpyMLP((2, 2, 1), seed=0)
    m.weights = [np.array([[0.5, -0.3], [0.2, 0.8]]), np.array([[1.0], [-1.5]])]
    m.biases = [np.array([0.1, -0.1]), np.array([0.2])]
    return m, np.array([[1.0, 2.0]]), np.array([1.0])


def test_worked_example_numbers() -> None:
    m, X, y = _example()
    zs, acts = m._forward(X)
    np.testing.assert_allclose(zs[0], [[1.0, 1.2]])
    assert zs[1][0, 0] == pytest.approx(-0.6)
    assert acts[-1][0, 0] == pytest.approx(0.354344, abs=5e-7)
    assert m.loss(X, y) == pytest.approx(1.037488, abs=5e-7)
    gws, gbs = m.backward(X, y)
    np.testing.assert_allclose(gws[1][:, 0], [-0.645656, -0.774788], atol=5e-7)
    np.testing.assert_allclose(gbs[1], [-0.645656], atol=5e-7)
    np.testing.assert_allclose(gbs[0], [-0.645656, 0.968484], atol=5e-7)
    np.testing.assert_allclose(gws[0], [[-0.645656, 0.968484], [-1.291313, 1.936969]], atol=5e-7)


def test_doc_quotes_the_same_numbers() -> None:
    text = DOC.read_text(encoding="utf-8")
    for number in ("0.354344", "1.037488", "-0.774788", "0.968484", "-1.291313", "1.936969"):
        assert number in text
