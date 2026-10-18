"""Tests for terser.judge.code_runner (plan row 2.3.1): extraction and the sandboxed runner.

Each sandbox run starts a fresh interpreter (about a second on Windows), so the run tests are kept
few and small.
"""

from __future__ import annotations

import os
import sys

import pytest

from terser.judge.code_runner import (
    CodeRunResult,
    CodeTests,
    SandboxLimits,
    extract_code,
    extract_code_blocks,
    pass_rate,
    run_code_tests,
    run_output_tests,
)

HUMANEVAL_TEST = """
METADATA = {}


def check(candidate):
    assert candidate([1, 2, 3]) == 6
    assert candidate([]) == 0
    assert candidate([-1, 1]) == 0
"""
HUMANEVAL = CodeTests.humaneval(HUMANEVAL_TEST, "sum_list")
MBPP = CodeTests.mbpp(
    [
        "assert is_palindrome('A man a plan a canal Panama')",
        "assert not is_palindrome('hello')",
        "assert is_palindrome('')",
    ]
)
FAST = SandboxLimits(timeout_s=20.0)


# ----------------------------------------------------------------------------------------------
# extraction
# ----------------------------------------------------------------------------------------------


def test_extract_code_blocks_tags_and_unclosed_fence() -> None:
    text = "Intro\n```py\nx = 1\n```\ntext\n~~~js\nlet y = 2;\n~~~\n```python\ndef f():\n    pass"
    blocks = extract_code_blocks(text)
    assert [(b.language, b.closed) for b in blocks] == [
        ("python", True),
        ("javascript", True),
        ("python", False),
    ]
    assert blocks[0].code == "x = 1\n"


def test_extract_code_prefers_entry_point_and_adds_helpers() -> None:
    text = (
        "Helper:\n```python\nimport math\n\ndef _sq(x):\n    return x * x\n```\n"
        "Solution:\n```python\ndef hyp(a, b):\n    return math.sqrt(_sq(a) + _sq(b))\n```\n"
        "Usage:\n```python\nprint(hyp(3, 4))\n```"
    )
    code = extract_code(text, "python", entry_point="hyp")
    assert code is not None
    assert "def hyp" in code and "def _sq" in code and "import math" in code
    assert "print(hyp" not in code


def test_extract_code_bare_and_other_languages() -> None:
    bare = "Sure! Here it is:\n\ndef double(x):\n    return 2 * x\n\nThis doubles x."
    assert extract_code(bare) == "def double(x):\n    return 2 * x\n"
    sql = "```sql\nSELECT 1;\n```"
    assert extract_code(sql, "python") is None
    assert extract_code(sql, "sql") == "SELECT 1;\n"
    assert extract_code("no code at all") is None


def test_code_tests_validation_and_records() -> None:
    with pytest.raises(ValueError):
        CodeTests("humaneval", ("def check(c): pass",))
    with pytest.raises(ValueError):
        CodeTests.mbpp([])
    rec = CodeTests.from_record(
        {"test_list": ["assert f() == 1"], "test_setup_code": "import re", "test_imports": []}
    )
    assert rec.style == "mbpp" and rec.setup == "import re"
    he = CodeTests.from_record({"test": HUMANEVAL_TEST, "entry_point": "sum_list"})
    assert he.style == "humaneval" and he.n_tests == 1
    with pytest.raises(ValueError):
        CodeTests.from_record({"prompt": "x"})
    with pytest.raises(ValueError):
        SandboxLimits(timeout_s=0)


def test_pass_rate_over_runs() -> None:
    ok = CodeRunResult("passed", 2, 2)
    bad = CodeRunResult("failed", 2, 1)
    assert ok.pass_rate == 1.0 and bad.pass_rate == 0.5
    assert pass_rate([ok, bad, ok, bad]) == 0.5
    assert pass_rate([]) == 0.0


# ----------------------------------------------------------------------------------------------
# sandboxed runs
# ----------------------------------------------------------------------------------------------


def test_humaneval_passing_and_failing() -> None:
    good = run_code_tests("def sum_list(xs):\n    return sum(xs)\n", HUMANEVAL, FAST)
    assert good.status == "passed" and good.passed and good.pass_rate == 1.0
    assert "reliability_guard" in good.limits_applied
    bad = run_code_tests("def sum_list(xs):\n    return len(xs)\n", HUMANEVAL, FAST)
    assert bad.status == "failed" and bad.pass_rate == 0.0
    assert "AssertionError" in bad.stderr_tail


def test_mbpp_per_test_results_from_model_output() -> None:
    output = (
        "Here's a palindrome checker:\n\n```python\n"
        "def is_palindrome(s):\n"
        "    t = s.replace(' ', '').lower()\n"
        "    return t == t[::-1] and len(t) > 0\n"
        "```\nIt ignores spaces and case."
    )
    result = run_output_tests(output, MBPP, limits=FAST)
    assert result is not None
    assert result.per_test == (True, True, False)
    assert result.status == "failed" and result.pass_rate == pytest.approx(2 / 3)
    fixed = output.replace(" and len(t) > 0", "")
    fixed_result = run_output_tests(fixed, MBPP, limits=FAST)
    assert fixed_result is not None and fixed_result.passed
    assert run_output_tests(output, MBPP, language="java") is None


def test_exception_is_captured_in_stderr_tail() -> None:
    code = "def sum_list(xs):\n    raise RuntimeError('boom from candidate')\n"
    result = run_code_tests(code, HUMANEVAL, FAST)
    assert result.status == "failed"
    assert "RuntimeError" in result.stderr_tail and "boom from candidate" in result.stderr_tail


def test_load_error_and_no_code() -> None:
    result = run_code_tests("def broken(:\n", MBPP, FAST)
    assert result.status == "error" and "SyntaxError" in result.stderr_tail
    assert run_code_tests(None, MBPP).status == "no_code"
    assert run_code_tests("   ", MBPP).pass_rate == 0.0


def test_infinite_loop_is_killed_by_timeout() -> None:
    tests = CodeTests.mbpp(["assert quick() == 1", "assert spin() == 1"])
    code = "def quick():\n    return 1\n\ndef spin():\n    while True:\n        pass\n"
    result = run_code_tests(code, tests, SandboxLimits(timeout_s=4.0))
    assert result.status == "timeout"
    assert result.duration_s < 15
    assert result.per_test == (True, False)  # tests finished before the kill still count
    assert "killed after 4s" in result.stderr_tail


def test_environment_is_scrubbed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TERSER_TEST_SECRET_TOKEN", "do-not-leak")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-do-not-leak")
    monkeypatch.setenv("PYTHONPATH", os.getcwd())
    code = (
        "import os, sys\n"
        "def env():\n"
        "    return dict(os.environ)\n"
        "def cwd():\n"
        "    return os.getcwd()\n"
    )
    tests = CodeTests.mbpp(
        [
            "assert 'TERSER_TEST_SECRET_TOKEN' not in env(), sorted(env())",
            "assert 'ANTHROPIC_API_KEY' not in env()",
            "assert not any('do-not-leak' in v for v in env().values())",
            "assert 'terser-sandbox-' in cwd(), cwd()",
        ]
        # Without a virtualenv (as in CI) the base interpreter has the project installed.
        + (
            ["import importlib.util\nassert importlib.util.find_spec('terser') is None"]
            if sys.prefix != sys.base_prefix
            else []
        )
    )
    result = run_code_tests(code, tests, FAST)
    assert result.passed, result.stderr_tail


def test_network_and_subprocess_are_blocked() -> None:
    code = (
        "import socket, subprocess, sys\n"
        "def net():\n"
        "    try:\n"
        "        socket.create_connection(('127.0.0.1', 9), timeout=1)\n"
        "    except OSError as exc:\n"
        "        return 'disabled' in str(exc)\n"
        "    return False\n"
        "def spawn():\n"
        "    try:\n"
        "        subprocess.run([sys.executable, '-c', 'pass'])\n"
        "    except PermissionError:\n"
        "        return True\n"
        "    return False\n"
    )
    result = run_code_tests(code, CodeTests.mbpp(["assert net()", "assert spawn()"]), FAST)
    assert result.passed, result.stderr_tail
    assert "network:socket_guard" in result.limits_applied
