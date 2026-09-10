"""Resolve a ``hardware`` config node into a :class:`DeviceInfo`.

This is the only place in Terser that decides between CPU and CUDA. Other modules call
:func:`get_device` instead of hard-coding device strings. ``torch`` is optional and imported lazily.
"""

from __future__ import annotations

import importlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import ModuleType
from typing import Any

__all__ = ["DeviceInfo", "HardwareUnavailableError", "get_device"]

PRECISIONS = ("fp32", "fp16", "bf16")
_CUDA_RE = re.compile(r"^cuda(?::(\d+))?$")
_HINT = (
    "The NVIDIA RTX 4500 Ada may not be connected yet. "
    "Run with `hardware=cpu` or `hardware=auto` (falls back to CPU), "
    "or queue the job in gpu_queue/."
)


class HardwareUnavailableError(RuntimeError):
    """Raised when the requested hardware (device or precision) is not usable here."""


@dataclass(frozen=True)
class DeviceInfo:
    """A resolved device: ``type`` is ``"cpu"`` or ``"cuda"``; ``index`` is None on CPU."""

    type: str
    index: int | None
    precision: str
    name: str

    def __str__(self) -> str:
        return self.type if self.index is None else f"{self.type}:{self.index}"

    def torch_device(self) -> Any:
        """Return the matching ``torch.device``; raises ImportError if torch is not installed."""
        torch = _import_torch()
        if torch is None:
            raise ImportError(
                "torch is not installed; install the GPU extras (`pip install -e .[gpu]`) "
                "to obtain a torch.device."
            )
        return torch.device(str(self))


def _import_torch() -> ModuleType | None:
    try:
        return importlib.import_module("torch")
    except ImportError:
        return None


def _hardware_node(cfg: Any) -> Mapping[str, Any]:
    """Accept a hardware node or a root config with a ``hardware`` key (Mapping or DictConfig)."""
    try:
        from omegaconf import DictConfig, OmegaConf

        if isinstance(cfg, DictConfig):
            cfg = OmegaConf.to_container(cfg, resolve=True)
    except ImportError:  # pragma: no cover - omegaconf is a core dependency
        pass
    if not isinstance(cfg, Mapping):
        raise TypeError(f"hardware config must be a Mapping or DictConfig, got {type(cfg)!r}")
    if "hardware" in cfg and isinstance(cfg["hardware"], Mapping):
        cfg = cfg["hardware"]
    return cfg


def _cuda_count(torch: ModuleType | None) -> int:
    if torch is None or not torch.cuda.is_available():
        return 0
    return int(torch.cuda.device_count())


def get_device(cfg: Any) -> DeviceInfo:
    """Resolve ``cfg`` (``device``: auto|cpu|cuda|cuda:N, ``precision``: auto|fp32|fp16|bf16)."""
    node = _hardware_node(cfg)
    device = str(node.get("device", "auto")).strip().lower()
    precision = str(node.get("precision", "auto")).strip().lower()
    if precision != "auto" and precision not in PRECISIONS:
        raise ValueError(f"Unknown precision {precision!r}; expected auto or one of {PRECISIONS}.")

    if device == "cpu":
        return _cpu(precision)

    torch = _import_torch()
    if device == "auto":
        if torch is None or _cuda_count(torch) == 0:
            return _cpu(precision)
        return _cuda(torch, 0, precision)

    match = _CUDA_RE.match(device)
    if match is None:
        raise ValueError(f"Unknown device {device!r}; expected 'auto', 'cpu', 'cuda' or 'cuda:N'.")
    index = int(match.group(1) or 0)
    if torch is None:
        raise HardwareUnavailableError(
            f"GPU config requested device {device!r} but torch is not installed, so no CUDA "
            f"device is visible. {_HINT}"
        )
    count = _cuda_count(torch)
    if count == 0:
        raise HardwareUnavailableError(
            f"GPU config requested device {device!r} but no CUDA device is visible. {_HINT}"
        )
    if index >= count:
        raise HardwareUnavailableError(
            f"GPU config requested device {device!r} but only {count} CUDA device(s) are "
            f"visible (valid indices 0..{count - 1}). {_HINT}"
        )
    return _cuda(torch, index, precision)


def _cpu(precision: str) -> DeviceInfo:
    if precision not in ("auto", "fp32"):
        raise HardwareUnavailableError(
            f"Precision {precision!r} is not supported on CPU; use precision fp32 or auto "
            f"(e.g. `hardware=cpu`). {_HINT}"
        )
    return DeviceInfo(type="cpu", index=None, precision="fp32", name="cpu")


def _cuda(torch: ModuleType, index: int, precision: str) -> DeviceInfo:
    bf16_ok = bool(torch.cuda.is_bf16_supported())  # checks the current device
    name = str(torch.cuda.get_device_name(index))
    if precision == "auto":
        precision = "bf16" if bf16_ok else "fp16"
    elif precision == "bf16" and not bf16_ok:
        raise HardwareUnavailableError(
            f"Precision bf16 was requested but CUDA device {index} ({name}) does not support "
            f"bfloat16; set precision fp16 or auto. {_HINT}"
        )
    return DeviceInfo(type="cuda", index=index, precision=precision, name=name)
