"""Shared pytest configuration for Terser.

Registers the project's markers, adds opt-in command-line switches for expensive test classes,
and provides common fixtures. Safety default: unit tests never see real API keys unless the run
was started with ``--run-api``, so a plain ``pytest`` invocation can never spend money.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

API_KEY_ENV_VARS: tuple[str, ...] = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")
"""Environment variables holding paid LLM API credentials."""

MARKERS: dict[str, str] = {
    "gpu": "needs the RTX 4500 Ada / CUDA (skipped unless --run-gpu and CUDA is available)",
    "api": "calls a paid LLM API (skipped unless --run-api and an API key is set)",
    "network": "downloads files, e.g. HF tokenizers (skipped with --no-network)",
    "slow": "slow test (deselect with -m 'not slow')",
    "dvc": "needs DVC-tracked artifacts (skipped if not pulled; fails with --require-dvc)",
    "llmlingua": "loads the ~700 MB LLMLingua-2 model on CPU (skipped unless --run-llmlingua)",
}

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CASUAL_PROMPTS_PATH = Path(__file__).resolve().parent / "fixtures" / "casual_prompts.jsonl"


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add opt-in switches for GPU/API tests and an opt-out for network tests."""
    group = parser.getgroup("terser")
    group.addoption("--run-gpu", action="store_true", default=False, help="run tests marked gpu")
    group.addoption(
        "--run-api", action="store_true", default=False, help="run tests marked api (costs money)"
    )
    group.addoption(
        "--no-network", action="store_true", default=False, help="skip tests marked network"
    )
    group.addoption(
        "--require-dvc",
        action="store_true",
        default=False,
        help="fail (instead of skip) tests whose DVC-tracked inputs are missing",
    )
    group.addoption(
        "--run-llmlingua",
        action="store_true",
        default=False,
        help="run tests marked llmlingua (downloads and loads the LLMLingua-2 model)",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register markers (so no pyproject config is required) and enable ``pytester``."""
    for name, description in MARKERS.items():
        config.addinivalue_line("markers", f"{name}: {description}")
    if not config.pluginmanager.has_plugin("pytester"):
        config.pluginmanager.import_plugin("pytester")


def _cuda_available() -> bool:
    """Return True if torch is importable and reports a CUDA device. Never hard-requires torch."""
    try:
        import torch  # lazy import keeps torch optional

        return bool(torch.cuda.is_available())
    except Exception:  # any import/driver failure means "no CUDA"
        return False


def _api_key_present() -> bool:
    """Return True if at least one paid-API key is set in the environment."""
    return any(os.environ.get(var) for var in API_KEY_ENV_VARS)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Attach skip markers to gpu/api/network tests according to options and environment."""
    run_gpu = bool(config.getoption("--run-gpu"))
    run_api = bool(config.getoption("--run-api"))
    no_network = bool(config.getoption("--no-network"))

    gpu_skip: pytest.MarkDecorator | None = None
    if any(item.get_closest_marker("gpu") for item in items):
        if not run_gpu:
            gpu_skip = pytest.mark.skip(reason="gpu test: pass --run-gpu to run")
        elif not _cuda_available():
            gpu_skip = pytest.mark.skip(reason="gpu test: CUDA not available")

    api_skip: pytest.MarkDecorator | None = None
    if not run_api:
        api_skip = pytest.mark.skip(reason="api test: pass --run-api to run (costs money)")
    elif not _api_key_present():
        api_skip = pytest.mark.skip(
            reason=f"api test: none of {', '.join(API_KEY_ENV_VARS)} is set"
        )

    network_skip = pytest.mark.skip(reason="network test: --no-network given")
    run_llmlingua = bool(config.getoption("--run-llmlingua")) or (
        os.environ.get("TERSER_RUN_LLMLINGUA") == "1"
    )
    llmlingua_skip = pytest.mark.skip(reason="llmlingua test: pass --run-llmlingua to run")

    for item in items:
        if gpu_skip is not None and item.get_closest_marker("gpu"):
            item.add_marker(gpu_skip)
        if api_skip is not None and item.get_closest_marker("api"):
            item.add_marker(api_skip)
        if no_network and item.get_closest_marker("network"):
            item.add_marker(network_skip)
        if not run_llmlingua and item.get_closest_marker("llmlingua"):
            item.add_marker(llmlingua_skip)


@pytest.fixture
def require_dvc(request: pytest.FixtureRequest) -> Any:
    """``require_dvc(path, hint)``: skip if a DVC-tracked ``path`` is missing, or fail under
    ``--require-dvc`` so a pulled checkout (or a CI job that pulls) cannot skip silently."""
    strict = bool(request.config.getoption("--require-dvc"))

    def check(path: Path, hint: str) -> None:
        if path.exists():
            return
        if strict:
            pytest.fail(f"{path} missing under --require-dvc ({hint})")
        pytest.skip(f"{path} not pulled ({hint})")

    return check


@pytest.fixture(autouse=True)
def _safe_environment(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Silence HF symlink warnings and scrub API keys unless the run opted in with --run-api."""
    monkeypatch.setenv("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    if not request.config.getoption("--run-api"):
        for var in API_KEY_ENV_VARS:
            monkeypatch.delenv(var, raising=False)


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Absolute path of the repository root."""
    return _PROJECT_ROOT


@pytest.fixture(scope="session")
def casual_prompts() -> list[dict[str, Any]]:
    """The 20 hand-written casual prompts from ``tests/fixtures/casual_prompts.jsonl``."""
    with _CASUAL_PROMPTS_PATH.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


@pytest.fixture
def tmp_cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A fresh cache directory exported as ``TERSER_CACHE`` for the duration of a test."""
    cache = tmp_path / "terser_cache"
    cache.mkdir()
    monkeypatch.setenv("TERSER_CACHE", str(cache))
    yield cache
