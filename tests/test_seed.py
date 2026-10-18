"""Tests for the determinism harness (terser.utils.seed and dummy_experiment)."""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from terser.utils import rng, set_seed


def _run_experiment(seed: int, hashseed: str) -> str:
    env = {**os.environ, "PYTHONHASHSEED": hashseed}
    out = subprocess.run(
        [sys.executable, "-m", "terser.utils.dummy_experiment", "--seed", str(seed)],
        capture_output=True,
        text=True,
        env=env,
        check=True,
        timeout=600,
    )
    lines = out.stdout.strip().splitlines()
    assert len(lines) == 1
    json.loads(lines[0])
    return lines[0]


def test_experiment_identical_across_hash_seeds() -> None:
    assert _run_experiment(0, "1") == _run_experiment(0, "12345")


def test_experiment_differs_across_seeds() -> None:
    assert _run_experiment(0, "1") != _run_experiment(1, "1")


def test_set_seed_repeats_random_and_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    # setenv first so monkeypatch records the original state; delenv of an absent variable
    # records nothing, and the value set_seed writes would leak into later tests.
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.delenv("PYTHONHASHSEED")
    assert set_seed(7) == 7
    a = ([random.random() for _ in range(5)], np.random.rand(5))
    set_seed(7)
    b = ([random.random() for _ in range(5)], np.random.rand(5))
    assert a[0] == b[0]
    np.testing.assert_array_equal(a[1], b[1])
    assert os.environ["PYTHONHASHSEED"] == "7"


def test_rng_helper() -> None:
    np.testing.assert_array_equal(rng(3).random(4), rng(3).random(4))
    assert not np.array_equal(rng(3).random(4), rng(4).random(4))


def test_torch_branch_with_fake_module(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, tuple]] = []

    def rec(name: str):
        return lambda *args, **kwargs: calls.append((name, args, kwargs))

    fake = SimpleNamespace(
        manual_seed=rec("manual_seed"),
        cuda=SimpleNamespace(manual_seed_all=rec("manual_seed_all")),
        backends=SimpleNamespace(cudnn=SimpleNamespace(deterministic=False, benchmark=True)),
        use_deterministic_algorithms=rec("use_deterministic_algorithms"),
    )
    monkeypatch.setitem(sys.modules, "torch", fake)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)

    set_seed(11)

    assert ("manual_seed", (11,), {}) in calls
    assert ("manual_seed_all", (11,), {}) in calls
    assert ("use_deterministic_algorithms", (True,), {"warn_only": True}) in calls
    assert fake.backends.cudnn.deterministic is True
    assert fake.backends.cudnn.benchmark is False
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
