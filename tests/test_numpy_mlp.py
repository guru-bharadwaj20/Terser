"""Tests for terser.models.neural.numpy_mlp (row 4.1.3)."""

from __future__ import annotations

import numpy as np
import pytest

from terser.models.neural.numpy_mlp import NumpyMLP, gradient_check


def _xor(n: int = 200, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (n, 2))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(float)
    return X, y


@pytest.mark.parametrize("activation", ["relu", "tanh", "sigmoid"])
def test_gradient_check(activation: str) -> None:
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (16, 6))
    y = rng.integers(0, 2, 16)
    m = NumpyMLP((6, 5, 4, 1), activation=activation, seed=2)  # type: ignore[arg-type]
    for b in m.biases:
        b += 0.1  # keep ReLU pre-activations away from the kink
    assert gradient_check(m, X, y) < 1e-6


def test_matches_torch_autograd() -> None:
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(3)
    X = rng.normal(0, 1, (32, 8))
    y = rng.integers(0, 2, 32).astype(float)
    m = NumpyMLP((8, 16, 4, 1), seed=4)
    ws = [torch.tensor(w, requires_grad=True) for w in m.weights]
    bs = [torch.tensor(b, requires_grad=True) for b in m.biases]
    a = torch.tensor(X)
    for i, (w, b) in enumerate(zip(ws, bs, strict=True)):
        a = a @ w + b
        if i < len(ws) - 1:
            a = torch.relu(a)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(a[:, 0], torch.tensor(y))
    loss.backward()
    assert m.loss(X, y) == pytest.approx(loss.item(), rel=1e-12)
    gws, gbs = m.backward(X, y)
    for ours, t in zip(gws + gbs, ws + bs, strict=True):
        np.testing.assert_allclose(ours, t.grad.numpy(), rtol=1e-10, atol=1e-14)


def test_learns_xor() -> None:
    X, y = _xor()
    m = NumpyMLP((2, 16, 1), activation="tanh", seed=0)
    history = m.fit(X, y, lr=0.5, epochs=2000)
    assert history[-1] < history[0] / 3
    assert np.mean((m.predict_proba(X) >= 0.5) == y) > 0.9


def test_param_count_and_validation() -> None:
    assert NumpyMLP((1536, 256, 64, 1)).n_params == 409_985
    with pytest.raises(ValueError):
        NumpyMLP((4, 2))
    with pytest.raises(ValueError):
        NumpyMLP((4,))
