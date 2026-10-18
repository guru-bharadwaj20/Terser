"""Multi-layer perceptron with hand-written backpropagation, NumPy only (plan row **4.1.3**).

Hidden layers use ReLU (or tanh/sigmoid), the output is one sigmoid unit, and the loss is mean
binary cross-entropy. There is no autograd: :meth:`NumpyMLP.backward` applies the chain rule
derived in ``docs/derivations/backprop.md`` (row 4.1.7). With a sigmoid output and BCE the output
error simplifies to ``delta_L = (p - y) / n``; each hidden layer then receives
``delta_l = (delta_{l+1} W_{l+1}^T) * f'(z_l)``, and ``dW_l = a_{l-1}^T delta_l``,
``db_l = sum(delta_l)``.

Correctness is tested two ways: central finite differences (relative error < 1e-6 in float64) and
``torch.autograd`` on the same weights.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import numpy.typing as npt

__all__ = ["NumpyMLP", "gradient_check"]

FloatArray = npt.NDArray[np.float64]
Activation = Literal["relu", "tanh", "sigmoid", "step"]


def _sigmoid(z: FloatArray) -> FloatArray:
    return np.asarray(0.5 * (1.0 + np.tanh(0.5 * z)))


def _act(name: Activation, z: FloatArray) -> FloatArray:
    if name == "relu":
        return np.maximum(z, 0.0)
    if name == "tanh":
        return np.tanh(z)
    if name == "step":
        return (z > 0).astype(np.float64)
    return _sigmoid(z)


def _act_grad(name: Activation, z: FloatArray) -> FloatArray:
    """Local derivative ``f'(z)``.

    ``step`` has zero derivative almost everywhere, so it uses the clipped straight-through
    estimator ``1[|z| <= 1]`` (row 4.2.1); its gradients are therefore *not* the true gradients
    and :func:`gradient_check` does not apply to it.
    """
    if name == "relu":
        return (z > 0).astype(np.float64)
    if name == "tanh":
        return 1.0 - np.tanh(z) ** 2
    if name == "step":
        return (np.abs(z) <= 1.0).astype(np.float64)
    s = _sigmoid(z)
    return s * (1.0 - s)


@dataclass
class NumpyMLP:
    """Fully connected binary classifier.

    Args:
        sizes: Layer widths including input and output, e.g. ``(1536, 256, 64, 1)``.
        activation: Hidden-layer activation.
        seed: Seed for He (ReLU, N(0, 2/fan_in)) / LeCun (tanh, sigmoid, N(0, 1/fan_in))
            initialisation.
    """

    sizes: Sequence[int]
    activation: Activation = "relu"
    seed: int = 0
    weights: list[FloatArray] = field(init=False)
    biases: list[FloatArray] = field(init=False)

    def __post_init__(self) -> None:
        if len(self.sizes) < 2 or self.sizes[-1] != 1:
            raise ValueError(f"sizes must have >= 2 layers ending in 1, got {tuple(self.sizes)}")
        rng = np.random.default_rng(self.seed)
        self.weights, self.biases = [], []
        for fan_in, fan_out in zip(self.sizes[:-1], self.sizes[1:], strict=True):
            scale = np.sqrt((2.0 if self.activation == "relu" else 1.0) / fan_in)
            self.weights.append(rng.normal(0.0, scale, (fan_in, fan_out)))
            self.biases.append(np.zeros(fan_out))

    @property
    def n_params(self) -> int:
        """Total trainable parameters (see row 4.1.5)."""
        return sum(w.size + b.size for w, b in zip(self.weights, self.biases, strict=True))

    def _forward(self, X: FloatArray) -> tuple[list[FloatArray], list[FloatArray]]:
        """Pre-activations ``zs`` and activations ``acts`` (``acts[0] = X``)."""
        acts, zs = [X], []
        last = len(self.weights) - 1
        for i, (w, b) in enumerate(zip(self.weights, self.biases, strict=True)):
            z = acts[-1] @ w + b
            zs.append(z)
            acts.append(_sigmoid(z) if i == last else _act(self.activation, z))
        return zs, acts

    def hidden_unit_stats(self, X: npt.ArrayLike) -> dict[str, float]:
        """Fraction of saturated sigmoid/tanh units and dead ReLU units over ``X`` (row 4.2.1).

        A unit is *saturated* on an input when ``f'(z) < 0.01 * max f'`` (``|z| > ~2.99`` for
        tanh, ``~5.99`` for sigmoid) and *dead* when it outputs 0 on **every** input in ``X``.
        """
        zs, _ = self._forward(np.asarray(X, dtype=np.float64))
        hidden = np.concatenate(zs[:-1], axis=1) if len(zs) > 1 else np.empty((0, 0))
        if hidden.size == 0:
            return {"saturated_fraction": 0.0, "dead_fraction": 0.0}
        peak = {"sigmoid": 0.25, "tanh": 1.0}.get(self.activation)
        sat = (
            float(np.mean(_act_grad(self.activation, hidden) < 0.01 * peak))
            if peak is not None
            else 0.0
        )
        dead = float(np.mean((hidden <= 0).all(axis=0))) if self.activation == "relu" else 0.0
        return {"saturated_fraction": sat, "dead_fraction": dead}

    def predict_proba(self, X: npt.ArrayLike) -> FloatArray:
        """P(degraded) per row, shape (n,)."""
        return self._forward(np.asarray(X, dtype=np.float64))[1][-1][:, 0]

    def loss(self, X: npt.ArrayLike, y: npt.ArrayLike) -> float:
        """Mean binary cross-entropy, computed from logits for stability."""
        zs, _ = self._forward(np.asarray(X, dtype=np.float64))
        z = zs[-1][:, 0]
        ya = np.asarray(y, dtype=np.float64)
        return float(np.mean(np.logaddexp(0.0, z) - ya * z))

    def backward(
        self, X: npt.ArrayLike, y: npt.ArrayLike
    ) -> tuple[list[FloatArray], list[FloatArray]]:
        """Gradients of :meth:`loss` w.r.t. every weight matrix and bias vector."""
        _, gws, gbs = self._loss_and_grads(X, y)
        return gws, gbs

    def _loss_and_grads(
        self, X: npt.ArrayLike, y: npt.ArrayLike
    ) -> tuple[float, list[FloatArray], list[FloatArray]]:
        """:meth:`loss` and its gradients from a single forward pass."""
        Xa = np.asarray(X, dtype=np.float64)
        ya = np.asarray(y, dtype=np.float64).reshape(-1, 1)
        zs, acts = self._forward(Xa)
        z = zs[-1][:, 0]
        loss = float(np.mean(np.logaddexp(0.0, z) - ya[:, 0] * z))
        delta = (acts[-1] - ya) / len(ya)
        gws: list[FloatArray] = [np.empty(0)] * len(self.weights)
        gbs: list[FloatArray] = [np.empty(0)] * len(self.biases)
        for layer in range(len(self.weights) - 1, -1, -1):
            gws[layer] = acts[layer].T @ delta
            gbs[layer] = delta.sum(axis=0)
            if layer:
                delta = (delta @ self.weights[layer].T) * _act_grad(self.activation, zs[layer - 1])
        return loss, gws, gbs

    def step(self, X: npt.ArrayLike, y: npt.ArrayLike, lr: float) -> float:
        """One gradient-descent update; returns the loss before the update."""
        loss, gws, gbs = self._loss_and_grads(X, y)
        for w, b, gw, gb in zip(self.weights, self.biases, gws, gbs, strict=True):
            np.subtract(w, lr * gw, out=w)  # in place: w is the model's own array
            np.subtract(b, lr * gb, out=b)
        return loss

    def fit(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        lr: float = 0.1,
        epochs: int = 100,
        batch_size: int | None = None,
    ) -> list[float]:
        """Mini-batch (or full-batch when ``batch_size`` is None) gradient descent.

        Returns the per-epoch mean training loss.
        """
        Xa = np.asarray(X, dtype=np.float64)
        ya = np.asarray(y, dtype=np.float64)
        rng = np.random.default_rng(self.seed)
        bs = batch_size or len(ya)
        history = []
        for _ in range(epochs):
            order = rng.permutation(len(ya))
            losses = [
                self.step(Xa[order[i : i + bs]], ya[order[i : i + bs]], lr)
                for i in range(0, len(ya), bs)
            ]
            history.append(float(np.mean(losses)))
        return history


def gradient_check(model: NumpyMLP, X: npt.ArrayLike, y: npt.ArrayLike, h: float = 1e-5) -> float:
    """Max relative error between :meth:`NumpyMLP.backward` and central finite differences.

    Relative error per parameter is ``|g_a - g_n| / max(|g_a| + |g_n|, 1e-12)``; parameters where
    both gradients are below ``1e-10`` (e.g. dead ReLU units) are skipped.
    """
    gws, gbs = model.backward(X, y)
    worst = 0.0
    for params, grads in ((model.weights, gws), (model.biases, gbs)):
        for p, g in zip(params, grads, strict=True):
            it = np.nditer(p, flags=["multi_index"])
            for _ in it:
                idx = it.multi_index
                old = p[idx]
                p[idx] = old + h
                up = model.loss(X, y)
                p[idx] = old - h
                down = model.loss(X, y)
                p[idx] = old
                num = (up - down) / (2 * h)
                ana = g[idx]
                if abs(num) < 1e-10 and abs(ana) < 1e-10:
                    continue
                worst = max(worst, abs(ana - num) / max(abs(ana) + abs(num), 1e-12))
    return worst
