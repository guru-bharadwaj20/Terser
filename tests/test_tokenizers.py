"""Tests for ground-truth token counting (checkpoint C0.2)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from terser.tokenizers import available_targets, count_tokens
from terser.tokenizers import count as tc

FIXTURES = Path(__file__).parent / "fixtures"
MONITORING = Path(__file__).parent.parent / "data" / "monitoring"
STRINGS = [
    json.loads(line)["text"]
    # split on "\n" only: str.splitlines() would also break on U+2028 etc. inside strings
    for line in (MONITORING / "tokenizer_strings.jsonl").read_text(encoding="utf-8").split("\n")
    if line
]
SNAPSHOT = json.loads((MONITORING / "tokenizer_counts.json").read_text(encoding="utf-8"))
LOCAL_TARGETS = ["gpt-4o", "qwen2.5", "llama-3.1", "phi3.5"]
HF_REPOS = {
    "qwen2.5": "Qwen/Qwen2.5-7B-Instruct",
    "llama-3.1": tc.LLAMA_MIRROR,
    "phi3.5": "microsoft/Phi-3.5-mini-instruct",
}


def test_fixture_size() -> None:
    assert len(STRINGS) == 200 and "" in STRINGS
    assert all(len(SNAPSHOT[t]) == 200 for t in LOCAL_TARGETS)


def _reference(target: str) -> Any:
    if target == "gpt-4o":
        import tiktoken

        enc = tiktoken.get_encoding("o200k_base")
        return lambda s: len(enc.encode(s, disallowed_special=()))
    from tokenizers import Tokenizer

    tok = Tokenizer.from_pretrained(HF_REPOS[target])
    return lambda s: len(tok.encode(s, add_special_tokens=False).ids)


@pytest.mark.network
@pytest.mark.parametrize("target", LOCAL_TARGETS)
def test_matches_official_tokenizer(target: str) -> None:
    ref = _reference(target)
    mismatches = [i for i, s in enumerate(STRINGS) if count_tokens(s, target) != ref(s)]
    assert mismatches == []


@pytest.mark.network
@pytest.mark.parametrize("target", LOCAL_TARGETS)
def test_snapshot_no_drift(target: str) -> None:
    assert [count_tokens(s, target) for s in STRINGS] == SNAPSHOT[target]


@pytest.mark.network
@pytest.mark.parametrize("target", LOCAL_TARGETS)
def test_empty_is_zero_and_no_bos_added(target: str) -> None:
    assert count_tokens("", target) == 0
    assert count_tokens("hello", target) == 1


# --- Claude path (mocked; never calls the API) ---------------------------------------------

OVERHEAD = 7


class FakeClient:
    """Counts one token per character plus fixed framing, and records calls."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.messages = SimpleNamespace(count_tokens=self._count)

    def _count(self, model: str, messages: list[dict[str, str]]) -> SimpleNamespace:
        text = messages[0]["content"]
        assert text, "API rejects empty content"
        self.calls.append(text)
        return SimpleNamespace(input_tokens=OVERHEAD + len(text))


@pytest.fixture
def fake_claude(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[FakeClient]:
    import anthropic

    client = FakeClient()
    monkeypatch.setattr(anthropic, "Anthropic", lambda *a, **k: client)
    monkeypatch.setenv("TERSER_CACHE", str(tmp_path))
    monkeypatch.setenv("TERSER_CLAUDE_MODEL", "claude-test")
    tc._encoder.cache_clear()
    yield client
    tc._encoder.cache_clear()


def test_claude_overhead_subtracted_and_cached(fake_claude: FakeClient) -> None:
    assert count_tokens("hey so can you like explain recursion", "claude") == 37
    n_calls = len(fake_claude.calls)
    assert count_tokens("hey so can you like explain recursion", "claude") == 37
    assert len(fake_claude.calls) == n_calls


def test_claude_disk_cache_survives_new_process(fake_claude: FakeClient) -> None:
    count_tokens("persist me", "claude")
    tc._encoder.cache_clear()  # simulate a fresh process: calibration + text from disk
    fake_claude.calls.clear()
    assert count_tokens("persist me", "claude") == len("persist me")
    assert fake_claude.calls == []


def test_claude_empty_string_is_zero(fake_claude: FakeClient) -> None:
    assert count_tokens("", "claude") == 0
    assert "" not in fake_claude.calls


def test_calibration_rejects_nonlinear_counts() -> None:
    with pytest.raises(RuntimeError):
        tc.calibrate_overhead(lambda s: 5 + len(s) ** 2)


def test_calibration_recovers_overhead_without_unit_assumption() -> None:
    # Probe unit costs 3 tokens here; a "1 char = 1 token" assumption would be wrong.
    assert tc.calibrate_overhead(lambda s: 11 + 3 * s.count("\n")) == 11


# --- misc ------------------------------------------------------------------------------------


def test_unknown_target_raises() -> None:
    with pytest.raises(KeyError):
        count_tokens("hi", "gpt-2")


def test_available_targets_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert available_targets() == LOCAL_TARGETS
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert available_targets() == [*LOCAL_TARGETS, "claude"]


def test_llama_prefers_official_repo_with_token(monkeypatch: pytest.MonkeyPatch) -> None:
    import tokenizers

    class FakeTokenizer:
        @staticmethod
        def from_pretrained(repo: str, token: str | None = None) -> str:
            if repo == tc.LLAMA_OFFICIAL and token != "good":
                raise OSError("gated")
            return repo

    monkeypatch.setattr(tokenizers, "Tokenizer", FakeTokenizer)
    monkeypatch.setenv("HF_TOKEN", "good")
    assert tc.load_hf_tokenizer(tc.LLAMA_MIRROR) == tc.LLAMA_OFFICIAL
    monkeypatch.setenv("HF_TOKEN", "bad")
    assert tc.load_hf_tokenizer(tc.LLAMA_MIRROR) == tc.LLAMA_MIRROR
    monkeypatch.delenv("HF_TOKEN")
    assert tc.load_hf_tokenizer(tc.LLAMA_MIRROR) == tc.LLAMA_MIRROR
