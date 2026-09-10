"""Tests for the test scaffold itself: fixture schema, markers, skip logic and key scrubbing."""

from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from conftest import API_KEY_ENV_VARS

CATEGORIES: frozenset[str] = frozenset(
    {"code", "fact", "reason", "explain", "creative", "transform"}
)
REQUIRED_FIELDS: dict[str, type] = {
    "id": str,
    "category": str,
    "prompt": str,
    "efficient": str,
    "constraints": list,
}
CONFTEST_SOURCE = Path(__file__).with_name("conftest.py")


# --- fixture data -------------------------------------------------------------------------------


def test_casual_prompts_count(casual_prompts: list[dict[str, Any]]) -> None:
    """There are exactly 20 fixture prompts."""
    assert len(casual_prompts) == 20


def test_casual_prompts_schema(casual_prompts: list[dict[str, Any]]) -> None:
    """Every record has exactly the required fields with the right types and valid values."""
    for rec in casual_prompts:
        assert set(rec) == set(REQUIRED_FIELDS), rec.get("id")
        for field, typ in REQUIRED_FIELDS.items():
            assert isinstance(rec[field], typ), (rec["id"], field)
        assert rec["category"] in CATEGORIES, rec["id"]
        assert rec["id"].startswith(f"{rec['category']}-"), rec["id"]
        assert rec["prompt"].strip() and rec["efficient"].strip(), rec["id"]
        assert rec["constraints"], rec["id"]
        assert all(isinstance(c, str) and c.strip() for c in rec["constraints"]), rec["id"]


def test_casual_prompts_unique_ids(casual_prompts: list[dict[str, Any]]) -> None:
    """Ids are unique."""
    ids = [rec["id"] for rec in casual_prompts]
    assert len(ids) == len(set(ids))


def test_casual_prompts_category_balance(casual_prompts: list[dict[str, Any]]) -> None:
    """All six categories are present with at least 3 prompts each."""
    counts = Counter(rec["category"] for rec in casual_prompts)
    assert set(counts) == CATEGORIES
    assert all(n >= 3 for n in counts.values()), counts


def test_efficient_is_not_longer(casual_prompts: list[dict[str, Any]]) -> None:
    """The hand-written efficient version is never longer than the casual prompt."""
    for rec in casual_prompts:
        assert len(rec["efficient"]) <= len(rec["prompt"]), rec["id"]


def test_constraints_survive_rewrite(casual_prompts: list[dict[str, Any]]) -> None:
    """Every constraint phrase appears (case-insensitively) in both prompt and efficient."""
    for rec in casual_prompts:
        prompt, efficient = rec["prompt"].lower(), rec["efficient"].lower()
        for constraint in rec["constraints"]:
            key = constraint.lower()
            assert key in prompt, (rec["id"], "prompt", constraint)
            assert key in efficient, (rec["id"], "efficient", constraint)


# --- fixtures and environment -------------------------------------------------------------------


def test_project_root(project_root: Path) -> None:
    """``project_root`` points at the repo root."""
    assert (project_root / "pyproject.toml").is_file()
    assert (project_root / "tests" / "conftest.py").is_file()


def test_tmp_cache_dir(tmp_cache_dir: Path) -> None:
    """``tmp_cache_dir`` exists and is exported as TERSER_CACHE."""
    assert tmp_cache_dir.is_dir()
    assert os.environ["TERSER_CACHE"] == str(tmp_cache_dir)


def test_hf_symlink_warning_disabled() -> None:
    """The autouse fixture silences the HF Hub symlink warning."""
    assert os.environ.get("HF_HUB_DISABLE_SYMLINKS_WARNING") == "1"


def test_api_keys_scrubbed(request: pytest.FixtureRequest) -> None:
    """Without --run-api, no paid-API key is visible to tests."""
    if request.config.getoption("--run-api"):
        pytest.skip("keys are intentionally kept with --run-api")
    for var in API_KEY_ENV_VARS:
        assert var not in os.environ, var


# --- markers and skip logic (in an isolated inner pytest run) ----------------------------------

_INNER_TESTS = """
import os
import pytest

@pytest.mark.gpu
def test_gpu():
    pass

@pytest.mark.api
def test_api():
    pass

@pytest.mark.network
def test_network():
    pass

@pytest.mark.slow
def test_slow():
    pass

def test_sees_no_keys():
    assert "ANTHROPIC_API_KEY" not in os.environ
    assert "OPENAI_API_KEY" not in os.environ
"""


@pytest.fixture
def inner(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> pytest.Pytester:
    """A pytester sandbox using this project's conftest, with fake API keys in the environment."""
    pytester.makeconftest(CONFTEST_SOURCE.read_text(encoding="utf-8"))
    pytester.makepyfile(test_inner=_INNER_TESTS)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake-not-a-real-key")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-not-a-real-key")
    return pytester


def test_markers_registered(inner: pytest.Pytester) -> None:
    """All four markers are registered, so ``--strict-markers`` passes."""
    result = inner.runpytest("--markers")
    for name in ("gpu", "api", "network", "slow"):
        result.stdout.fnmatch_lines([f"@pytest.mark.{name}:*"])
    inner.runpytest("--strict-markers", "--no-network").assert_outcomes(passed=2, skipped=3)


def test_default_run_skips_gpu_and_api(inner: pytest.Pytester) -> None:
    """By default gpu and api tests are skipped (even with keys set) and keys are scrubbed."""
    result = inner.runpytest("-rs")
    result.assert_outcomes(passed=3, skipped=2)
    result.stdout.fnmatch_lines(["*--run-gpu*", "*--run-api*"])


def test_run_gpu_without_cuda_still_skips(
    inner: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--run-gpu`` on a machine without CUDA still skips gpu tests."""
    monkeypatch.setitem(sys.modules, "torch", None)  # `import torch` now raises ImportError
    result = inner.runpytest("-rs", "--run-gpu", "-k", "gpu")
    result.assert_outcomes(skipped=1)
    result.stdout.fnmatch_lines(["*CUDA not available*"])


def test_run_api_requires_key(inner: pytest.Pytester, monkeypatch: pytest.MonkeyPatch) -> None:
    """``--run-api`` runs api tests only when a key is present."""
    inner.runpytest("--run-api", "-k", "test_api").assert_outcomes(passed=1)
    for var in API_KEY_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    inner.runpytest("--run-api", "-k", "test_api").assert_outcomes(skipped=1)


def test_no_network_skips_network(inner: pytest.Pytester) -> None:
    """``--no-network`` skips network tests; they run otherwise."""
    inner.runpytest("-k", "network").assert_outcomes(passed=1)
    inner.runpytest("--no-network", "-k", "network").assert_outcomes(skipped=1)
