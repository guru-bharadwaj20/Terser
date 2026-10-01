"""Rows 8.1.2 (manual conv1d), 8.2.1-8.2.3 (TextCNN architecture, parameters, pooling)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from terser.models.neural.conv import conv1d, conv1d_output_length, same_padding
from terser.models.neural.textcnn import TextCNNConfig, build_textcnn, textcnn_param_count

DOC = Path(__file__).resolve().parents[1] / "docs" / "derivations" / "deep_learning_intro.md"


def _fixture(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(2, 4, 11)), rng.normal(size=(3, 4, 3)), rng.normal(size=3)


def test_hand_computed_example() -> None:
    x = np.array([[1.0, 2.0, 3.0, 4.0]])
    w = np.array([[[1.0, 0.0, -1.0]]])
    np.testing.assert_allclose(conv1d(x, w), [[-2.0, -2.0]])
    np.testing.assert_allclose(conv1d(x, w, padding=1), [[-2.0, -2.0, -2.0, 3.0]])
    np.testing.assert_allclose(conv1d(x, w, bias=[0.5], padding=1, stride=2), [[-1.5, -1.5]])


def test_output_length_formula() -> None:
    assert conv1d_output_length(11, 3) == 9
    assert conv1d_output_length(11, 3, padding=1) == 11
    assert conv1d_output_length(11, 5, stride=2, padding=2) == 6
    assert conv1d_output_length(11, 3, dilation=2) == 7
    assert same_padding(5) == 2
    with pytest.raises(ValueError):
        same_padding(4)
    with pytest.raises(ValueError):
        conv1d_output_length(2, 5)
    with pytest.raises(ValueError):
        conv1d(np.zeros((3, 5)), np.zeros((1, 2, 3)))


@pytest.mark.parametrize(
    ("stride", "padding", "dilation"), [(1, 0, 1), (1, 1, 1), (2, 1, 1), (3, 2, 1), (1, 2, 2)]
)
def test_equals_torch_conv1d(stride: int, padding: int, dilation: int) -> None:
    torch = pytest.importorskip("torch")
    x, w, b = _fixture()
    conv = torch.nn.Conv1d(4, 3, 3, stride=stride, padding=padding, dilation=dilation).double()
    with torch.no_grad():
        conv.weight.copy_(torch.from_numpy(w))
        conv.bias.copy_(torch.from_numpy(b))
        expected = conv(torch.from_numpy(x)).numpy()
    ours = conv1d(x, w, b, stride=stride, padding=padding, dilation=dilation)
    assert ours.shape == expected.shape
    np.testing.assert_allclose(ours, expected, atol=1e-12)
    np.testing.assert_allclose(conv1d(x[0], w, b, stride, padding, dilation), expected[0])


def test_closed_form_parameter_count() -> None:
    assert 384 * 3 * 128 + 128 == 147_584
    assert 384 * 5 * 128 + 128 == 245_888
    assert textcnn_param_count() == 147_584 + 245_888 + 257 + 257 == 393_986
    assert textcnn_param_count(TextCNNConfig(pooling="concat")) == 393_986 - 257 + 513
    with pytest.raises(ValueError):
        TextCNNConfig(pooling="min")
    with pytest.raises(ValueError):
        TextCNNConfig(kernels=(2,))


@pytest.mark.parametrize("pooling", ["max", "avg", "concat"])
def test_param_count_equals_numel(pooling: str) -> None:
    pytest.importorskip("torch")
    cfg = TextCNNConfig(pooling=pooling)
    model = build_textcnn(cfg)
    assert sum(p.numel() for p in model.parameters()) == textcnn_param_count(cfg)
    if pooling != "concat":
        assert sum(p.numel() for p in model.parameters()) == 393_986
    else:
        assert model.prompt_head.weight.numel() + model.prompt_head.bias.numel() == 513


def test_forward_shapes_and_masked_pooling() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    for pooling in ("max", "avg", "concat"):
        model = build_textcnn(TextCNNConfig(pooling=pooling)).eval()
        x = torch.randn(2, 7, 384)
        mask = torch.ones(2, 7, dtype=torch.bool)
        mask[1, 4:] = False
        tok, prompt = model(x, mask)
        assert tok.shape == (2, 7) and prompt.shape == (2,)
        # Changing padded positions must not change the pooled prompt logit.
        x2 = x.clone()
        x2[1, 6:] = 100.0  # beyond the conv receptive field of real tokens (k=5 -> 2 each side)
        _, prompt2 = model(x2, mask)
        torch.testing.assert_close(prompt[1], prompt2[1])
        _, p_nomask = model(x)
        assert p_nomask.shape == (2,)


def test_doc_quotes_the_derivation() -> None:
    text = DOC.read_text(encoding="utf-8")
    for number in ("147,584", "245,888", "393,986", "513"):
        assert number in text
