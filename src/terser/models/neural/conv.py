"""Manual 1-D convolution (plan row **8.1.2**).

For input ``x`` of shape ``(C_in, L)``, weights ``w`` of shape ``(C_out, C_in, k)`` and bias
``b`` of shape ``(C_out,)``, with zero padding ``p`` on both sides, stride ``s`` and dilation
``d``:

    L_out = floor((L + 2p - d(k - 1) - 1) / s) + 1
    y[o, t] = b[o] + sum_{c, j} w[o, c, j] * x_pad[c, t*s + j*d]

This is cross-correlation (the kernel is not flipped), which is what deep-learning libraries
call convolution. A test checks it equals ``torch.nn.Conv1d`` on a fixture input. "Same"
padding for odd ``k``, stride 1, dilation 1 is ``p = (k - 1) / 2``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = ["conv1d", "conv1d_output_length", "same_padding"]

FloatArray = npt.NDArray[np.float64]


def conv1d_output_length(
    length: int, kernel: int, stride: int = 1, padding: int = 0, dilation: int = 1
) -> int:
    """``floor((L + 2p - d(k-1) - 1) / s) + 1``; raises if the result is < 1."""
    if kernel < 1 or stride < 1 or dilation < 1 or padding < 0:
        raise ValueError("kernel, stride, dilation must be >= 1 and padding >= 0")
    out = (length + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1
    if out < 1:
        raise ValueError(f"input length {length} too short for kernel {kernel}")
    return out


def same_padding(kernel: int) -> int:
    """Padding that keeps the length unchanged at stride 1 (odd kernels only)."""
    if kernel % 2 == 0:
        raise ValueError("'same' padding needs an odd kernel")
    return (kernel - 1) // 2


def conv1d(
    x: npt.ArrayLike,
    weight: npt.ArrayLike,
    bias: npt.ArrayLike | None = None,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> FloatArray:
    """Cross-correlate ``x`` (C_in, L) or (N, C_in, L) with ``weight`` (C_out, C_in, k)."""
    xa = np.asarray(x, dtype=np.float64)
    w = np.asarray(weight, dtype=np.float64)
    batched = xa.ndim == 3
    if not batched:
        xa = xa[None]
    n, c_in, length = xa.shape
    c_out, w_in, k = w.shape
    if w_in != c_in:
        raise ValueError(f"weight expects {w_in} input channels, got {c_in}")
    l_out = conv1d_output_length(length, k, stride, padding, dilation)
    xp = np.pad(xa, ((0, 0), (0, 0), (padding, padding)))
    y = np.zeros((n, c_out, l_out))
    for t in range(l_out):
        start = t * stride
        window = xp[:, :, start : start + dilation * (k - 1) + 1 : dilation]  # (n, c_in, k)
        y[:, :, t] = np.einsum("nck,ock->no", window, w)
    if bias is not None:
        y += np.asarray(bias, dtype=np.float64)[None, :, None]
    return y if batched else y[0]
