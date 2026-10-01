"""GPU environment import test (row 7.3)."""

from __future__ import annotations

import importlib

import pytest

GPU_PACKAGES = ("torch", "vllm", "peft", "trl", "bitsandbytes")


@pytest.mark.gpu
@pytest.mark.parametrize("name", GPU_PACKAGES)
def test_gpu_package_imports(name: str) -> None:
    module = importlib.import_module(name)
    assert getattr(module, "__version__", None)


@pytest.mark.gpu
def test_torch_sees_cuda_bf16() -> None:
    import torch

    assert torch.cuda.is_available()
    assert torch.cuda.is_bf16_supported()
