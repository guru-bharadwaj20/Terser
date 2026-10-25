"""Tests for terser.hardware.get_device with torch absent, faked, and (optionally) real."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest
from omegaconf import OmegaConf

from terser.hardware import DeviceInfo, HardwareUnavailableError, get_device

HW_DIR = Path(__file__).resolve().parents[1] / "configs" / "hardware"


def load(name: str) -> Any:
    return OmegaConf.load(HW_DIR / f"{name}.yaml")


def assert_hint(exc: BaseException) -> None:
    msg = str(exc)
    assert "hardware=cpu" in msg and "hardware=auto" in msg
    assert "RTX 4500 Ada" in msg


@pytest.fixture
def no_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)  # makes `import torch` raise ImportError


def make_fake_torch(available: bool = True, count: int = 1, bf16: bool = True) -> Any:
    cuda = types.SimpleNamespace(
        is_available=lambda: available,
        device_count=lambda: count,
        is_bf16_supported=lambda: bf16,
        get_device_name=lambda i=0: "NVIDIA RTX 4500 Ada Generation",
    )
    return types.SimpleNamespace(cuda=cuda, device=lambda s: f"torch.device({s})")


@pytest.fixture
def fake_torch(monkeypatch: pytest.MonkeyPatch) -> Any:
    def install(**kw: Any) -> Any:
        mod = make_fake_torch(**kw)
        monkeypatch.setitem(sys.modules, "torch", mod)
        return mod

    return install


def test_yaml_files_load() -> None:
    assert load("cpu").precision == "fp32" and load("cpu").max_vram_gb == 0
    gpu = load("rtx4500ada")
    assert gpu.device == "cuda:0" and gpu.precision == "bf16" and gpu.max_vram_gb == 23
    assert gpu.total_vram_gb == 24 and gpu.compute_capability == "8.9"
    assert load("auto").device == "auto" and load("auto").precision == "auto"


def test_cpu_profile(no_torch: None) -> None:
    info = get_device(load("cpu"))
    assert info == DeviceInfo(type="cpu", index=None, precision="fp32", name="cpu")


def test_root_config_and_plain_mapping(no_torch: None) -> None:
    root = OmegaConf.create({"hardware": load("cpu"), "seed": 0})
    assert get_device(root).type == "cpu"
    assert get_device({"hardware": {"device": "cpu"}}).precision == "fp32"


def test_auto_falls_back_to_cpu_without_torch(no_torch: None) -> None:
    info = get_device(load("auto"))
    assert (info.type, info.precision) == ("cpu", "fp32")


def test_auto_falls_back_to_cpu_without_cuda(fake_torch: Any) -> None:
    fake_torch(available=False, count=0)
    assert get_device(load("auto")).type == "cpu"


@pytest.mark.parametrize("bf16,expected", [(True, "bf16"), (False, "fp16")])
def test_auto_uses_cuda_when_available(fake_torch: Any, bf16: bool, expected: str) -> None:
    fake_torch(bf16=bf16)
    info = get_device(load("auto"))
    assert (info.type, info.index, info.precision) == ("cuda", 0, expected)
    assert str(info) == "cuda:0"
    assert info.torch_device() == "torch.device(cuda:0)"


def test_rtx_profile_on_cpu_without_torch(no_torch: None) -> None:
    with pytest.raises(HardwareUnavailableError, match="no CUDA device is visible") as exc:
        get_device(load("rtx4500ada"))
    assert_hint(exc.value)


def test_rtx_profile_without_cuda(fake_torch: Any) -> None:
    fake_torch(available=False, count=0)
    with pytest.raises(HardwareUnavailableError, match="no CUDA device is visible") as exc:
        get_device(load("rtx4500ada"))
    assert_hint(exc.value)


def test_rtx_profile_with_gpu(fake_torch: Any) -> None:
    fake_torch()
    info = get_device(load("rtx4500ada"))
    assert info == DeviceInfo("cuda", 0, "bf16", "NVIDIA RTX 4500 Ada Generation")


def test_cuda_index_out_of_range(fake_torch: Any) -> None:
    fake_torch(count=1)
    with pytest.raises(HardwareUnavailableError, match="only 1 CUDA device") as exc:
        get_device({"device": "cuda:1"})
    assert_hint(exc.value)


def test_bare_cuda_means_index_zero(fake_torch: Any) -> None:
    fake_torch(count=2)
    assert get_device({"device": "cuda", "precision": "fp16"}).index == 0


def test_bf16_unsupported(fake_torch: Any) -> None:
    fake_torch(bf16=False)
    with pytest.raises(HardwareUnavailableError, match="bfloat16") as exc:
        get_device(load("rtx4500ada"))
    assert_hint(exc.value)


@pytest.mark.parametrize("precision", ["fp16", "bf16"])
def test_cpu_rejects_half_precision(no_torch: None, precision: str) -> None:
    with pytest.raises(HardwareUnavailableError, match="not supported on CPU") as exc:
        get_device({"device": "cpu", "precision": precision})
    assert_hint(exc.value)


@pytest.mark.parametrize("device", ["mps", "cuda:x", "GPU0", ""])
def test_unknown_device(no_torch: None, device: str) -> None:
    with pytest.raises(ValueError, match="Unknown device"):
        get_device({"device": device})


def test_unknown_precision(no_torch: None) -> None:
    with pytest.raises(ValueError, match="Unknown precision"):
        get_device({"device": "cpu", "precision": "int8"})


def test_torch_device_without_torch(no_torch: None) -> None:
    with pytest.raises(ImportError, match="torch is not installed"):
        get_device(load("cpu")).torch_device()


@pytest.mark.gpu
def test_real_gpu() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device visible")
    info = get_device(load("auto"))
    assert info.type == "cuda"
    assert info.torch_device().type == "cuda"
    assert torch.zeros(1, device=info.torch_device()).device.type == "cuda"
