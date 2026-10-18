"""Hardware tag for MLflow runs (the ``hardware`` tag of :mod:`terser.tracking`)."""

from __future__ import annotations

import os

__all__ = ["hardware_tag"]

_KNOWN = {
    "rtx 4500 ada": "rtx4500ada",
    "rtx 6000 ada": "rtx6000ada",
    "a100": "a100",
    "h100": "h100",
}


def hardware_tag(index: int = 0) -> str:
    """``$TERSER_HARDWARE`` if set, else a short name for CUDA device ``index``, else ``cpu``."""
    env = os.environ.get("TERSER_HARDWARE")
    if env:
        return env
    try:
        import torch

        if not torch.cuda.is_available():
            return "cpu"
        name = torch.cuda.get_device_name(index).lower()
    except Exception:  # no torch / broken driver: not a GPU host
        return "cpu"
    for key, tag in _KNOWN.items():
        if key in name:
            return tag
    return name.replace("nvidia", "").strip().replace(" ", "")
