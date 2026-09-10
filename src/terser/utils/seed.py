"""Global seeding and explicit RNG helpers for reproducible experiments.

Notes on ``PYTHONHASHSEED``: string hashing is randomized once at interpreter start-up, so
setting ``os.environ["PYTHONHASHSEED"]`` inside a running process does *not* change ``hash()``
for that process; it only affects child processes spawned afterwards (subprocess, multiprocessing
with ``spawn``). Code that must be deterministic in-process should therefore never depend on
set/dict-of-str iteration order derived from hashes: sort before iterating, or use a stable hash
(e.g. ``hashlib``) instead of ``hash()``.

Torch is optional: it is imported lazily and skipped when absent.
"""

from __future__ import annotations

import importlib
import os
import random
import sys
from types import ModuleType

import numpy as np

__all__ = ["rng", "set_seed"]


def _import_torch() -> ModuleType | None:
    """Return the torch module if importable, else None (never raises)."""
    if "torch" in sys.modules:
        return sys.modules["torch"]
    try:
        return importlib.import_module("torch")
    except ImportError:
        return None


def set_seed(seed: int, deterministic: bool = True) -> int:
    """Seed Python ``random``, NumPy's global RNG and (if installed) torch.

    Also sets ``PYTHONHASHSEED`` (affects child processes only, see module docstring). With
    ``deterministic=True`` and torch present, enables cuDNN determinism, deterministic algorithms
    (``warn_only=True``) and ``CUBLAS_WORKSPACE_CONFIG=:4096:8``.

    Returns the seed, for logging convenience.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch = _import_torch()
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)
    return seed


def rng(seed: int | None) -> np.random.Generator:
    """Return an independent ``numpy.random.Generator``; prefer passing this explicitly."""
    return np.random.default_rng(seed)
