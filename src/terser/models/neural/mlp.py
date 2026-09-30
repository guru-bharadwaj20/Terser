"""PyTorch MLP quality gate and its parameter count (plan rows **4.1.4**, **4.1.5**).

Architecture ``1536 -> 256 -> 64 -> 1`` on the pair embedding of row 4.1.2
(``[e_o, e_r, |e_o - e_r|, e_o * e_r]`` from 384-d bge-small). A dense layer with ``m`` inputs and
``k`` outputs has ``m*k`` weights and ``k`` biases:

====================  ==========================  =========
Layer                 Weights + biases            Params
====================  ==========================  =========
1536 -> 256           1536 x 256 + 256            393,472
256 -> 64             256 x 64 + 64               16,448
64 -> 1               64 x 1 + 1                  65
**Total**                                         **409,985**
====================  ==========================  =========

:func:`dense_param_count` computes this closed form; a test checks it equals
``sum(p.numel() for p in QualityGateMLP().parameters())``. ``torch`` is imported lazily so the
rest of the package works without it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from torch import nn

__all__ = ["GATE_SIZES", "build_quality_gate_mlp", "dense_param_count"]

GATE_SIZES: Final[tuple[int, ...]] = (1536, 256, 64, 1)


def dense_param_count(sizes: Sequence[int]) -> int:
    """Parameters of a fully connected net with layer widths ``sizes``: ``sum(m*k + k)``."""
    if len(sizes) < 2 or any(s <= 0 for s in sizes):
        raise ValueError(f"need >= 2 positive layer widths, got {tuple(sizes)}")
    return sum(m * k + k for m, k in zip(sizes[:-1], sizes[1:], strict=True))


def build_quality_gate_mlp(sizes: Sequence[int] = GATE_SIZES, dropout: float = 0.0) -> nn.Module:
    """``nn.Sequential`` of Linear-ReLU(-Dropout) blocks ending in one logit (no sigmoid).

    Train with ``BCEWithLogitsLoss``; dropout adds no parameters.
    """
    from torch import nn

    dense_param_count(sizes)  # validates sizes
    layers: list[nn.Module] = []
    for i, (m, k) in enumerate(zip(sizes[:-1], sizes[1:], strict=True)):
        layers.append(nn.Linear(m, k))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
            if dropout:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)
