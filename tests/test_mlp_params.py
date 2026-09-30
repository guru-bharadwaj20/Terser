"""Row 4.1.5: the hand-computed parameter count equals the PyTorch module's."""

from __future__ import annotations

import pytest

from terser.models.neural.mlp import GATE_SIZES, build_quality_gate_mlp, dense_param_count


def test_closed_form() -> None:
    assert dense_param_count(GATE_SIZES) == 1536 * 256 + 256 + 256 * 64 + 64 + 64 * 1 + 1
    assert dense_param_count(GATE_SIZES) == 409_985
    with pytest.raises(ValueError):
        dense_param_count((10,))
    with pytest.raises(ValueError):
        dense_param_count((10, 0))


@pytest.mark.parametrize("dropout", [0.0, 0.2])
def test_equals_torch_numel(dropout: float) -> None:
    pytest.importorskip("torch")
    model = build_quality_gate_mlp(dropout=dropout)
    assert sum(p.numel() for p in model.parameters()) == 409_985
    per_layer = [p.numel() for p in model.parameters()]
    assert per_layer == [1536 * 256, 256, 256 * 64, 64, 64, 1]
