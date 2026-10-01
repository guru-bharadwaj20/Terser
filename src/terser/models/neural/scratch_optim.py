"""First-order optimizers from scratch for the NumPy MLP (plan row **4.3.1**).

Each optimizer updates a list of parameter arrays in place from their gradients ``g``:

* **GD / SGD**: ``theta <- theta - lr g`` (full batch vs mini-batch is the caller's choice).
* **Momentum**: ``v <- beta v + g``; ``theta <- theta - lr v``. Velocity accumulates a running sum
  of past gradients, damping oscillation across steep directions and accelerating along shallow
  consistent ones (effective step ``lr / (1 - beta)`` in a steady direction).
* **Adagrad**: ``G <- G + g^2``; ``theta <- theta - lr g / (sqrt(G) + eps)``. Per-coordinate step
  sizes shrink with accumulated squared gradient, so rare features keep larger steps but the
  global step decays monotonically.
* **Adam**: ``m <- b1 m + (1-b1) g``; ``v <- b2 v + (1-b2) g^2``; bias-corrected
  ``m_hat = m / (1 - b1^t)``, ``v_hat = v / (1 - b2^t)``;
  ``theta <- theta - lr m_hat / (sqrt(v_hat) + eps)``.

Each matches the corresponding ``torch.optim`` update (no weight decay) step for step.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

__all__ = ["SGD", "Adagrad", "Adam", "Momentum", "Optimizer", "make_optimizer"]

FloatArray = npt.NDArray[np.float64]


@dataclass
class Optimizer:
    """Plain gradient descent; subclasses override :meth:`_direction`."""

    lr: float
    t: int = field(init=False, default=0)
    _state: list[dict[str, FloatArray]] = field(init=False, default_factory=list, repr=False)

    def step(self, params: Sequence[FloatArray], grads: Sequence[FloatArray]) -> None:
        if not self._state:
            self._state = [{} for _ in params]
        self.t += 1
        for p, g, s in zip(params, grads, self._state, strict=True):
            p -= self.lr * self._direction(g, s)

    def _direction(self, g: FloatArray, s: dict[str, FloatArray]) -> FloatArray:
        return g


class SGD(Optimizer):
    """``theta <- theta - lr g``; full-batch GD is SGD with the whole set as the batch."""


@dataclass
class Momentum(Optimizer):
    beta: float = 0.9

    def _direction(self, g: FloatArray, s: dict[str, FloatArray]) -> FloatArray:
        s["v"] = self.beta * s.get("v", np.zeros_like(g)) + g
        return s["v"]


@dataclass
class Adagrad(Optimizer):
    eps: float = 1e-10

    def _direction(self, g: FloatArray, s: dict[str, FloatArray]) -> FloatArray:
        s["G"] = s.get("G", np.zeros_like(g)) + g * g
        return np.asarray(g / (np.sqrt(s["G"]) + self.eps))


@dataclass
class Adam(Optimizer):
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    def _direction(self, g: FloatArray, s: dict[str, FloatArray]) -> FloatArray:
        s["m"] = self.beta1 * s.get("m", np.zeros_like(g)) + (1 - self.beta1) * g
        s["v"] = self.beta2 * s.get("v", np.zeros_like(g)) + (1 - self.beta2) * g * g
        m_hat = s["m"] / (1 - self.beta1**self.t)
        v_hat = s["v"] / (1 - self.beta2**self.t)
        return np.asarray(m_hat / (np.sqrt(v_hat) + self.eps))


def make_optimizer(name: str, lr: float, **kwargs: float) -> Optimizer:
    """``name`` in ``{"gd", "sgd", "momentum", "adagrad", "adam"}``."""
    table: dict[str, type[Optimizer]] = {
        "gd": SGD,
        "sgd": SGD,
        "momentum": Momentum,
        "adagrad": Adagrad,
        "adam": Adam,
    }
    if name not in table:
        raise ValueError(f"unknown optimizer {name!r}; choose from {sorted(table)}")
    return table[name](lr=lr, **kwargs)
