"""Tests for row 2.2.1 candidate generation.

Unit tests are fast and offline: the LLM transport, operators, tokenizer and LLMLingua-2
compressor are faked. The one real LLMLingua-2 test downloads/loads a ~700 MB model, so besides
the ``network``/``slow`` markers it is opt-in via ``TERSER_RUN_LLMLINGUA=1``.
"""

from __future__ import annotations

import faulthandler
import importlib.util
import json
import os
import sys
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

from terser.data.candidates import (
    CANDIDATE_FIELDS,
    DEFAULT_METHODS,
    CandidateServerUnavailableError,
    CandidateSettings,
    LLMLingua2Compressor,
    ResumeMismatchError,
    candidate_id,
    clean_rewrite,
    method_kind,
    output_path_for,
    payload_altered,
    read_candidates,
    render_rewrite_prompt,
    run_candidates,
    settings_from_config,
    validate_rewrite,
)
from terser.llm import LLMClient, TransportResult

MODEL = "fake-rewriter"
UNIT_TIMEOUT_S = 60
SLOW_TIMEOUT_S = 1800


@pytest.fixture(autouse=True)
def _timeout_guard(request: pytest.FixtureRequest) -> Iterator[None]:
    """Per-test watchdog (no pytest-timeout dependency): dump stacks and abort on a hang."""
    seconds = SLOW_TIMEOUT_S if request.node.get_closest_marker("slow") else UNIT_TIMEOUT_S
    faulthandler.dump_traceback_later(seconds, exit=True, file=sys.stderr)
    try:
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()


PROMPTS = [
    {"id": "code-01", "category": "code", "prompt": 'um please fix "a + c" for me thanks'},
    {"id": "fact-01", "category": "fact", "prompt": "what is the capital of France"},
]


def word_count(text: str, target: str) -> int:
    """Stand-in tokenizer: one token per whitespace-separated word."""
    return len(text.split())


def fake_apply_ops(text: str, codes: str, target: str | None) -> str:
    return text.replace("um ", "") if "F" in codes else text


def fake_compressor(text: str) -> str:
    return " ".join(text.split()[::2])


class FakeRewriter:
    """Returns a scripted reply per level (found in the instruction), wrapped like a chat model."""

    def __init__(self, replies: dict[str, str] | None = None) -> None:
        self.replies = replies or {}
        self.calls: list[str] = []

    def __call__(
        self,
        provider: str,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        system: str | None,
        params: dict[str, Any],
    ) -> TransportResult:
        assert params.get("temperature") == 0.0
        content = messages[0]["content"]
        self.calls.append(content)
        level = content.split("Aggressiveness: ", 1)[1].split("\n", 1)[0]
        prompt = content.split("[Prompt]\n", 1)[1].rsplit("\n[End of prompt]", 1)[0]
        reply = self.replies.get(level, "Here is the rewritten prompt:\n" + prompt.split(" ", 1)[1])
        return TransportResult(reply, 40, 8)


def write_prompts(tmp_path: Path, rows: list[dict[str, Any]] = PROMPTS) -> Path:
    path = tmp_path / "prompts.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def make_client(tmp_path: Path, transport: Any = None, **kwargs: Any) -> LLMClient:
    budget = tmp_path / "budget.yaml"
    budget.write_text(
        yaml.safe_dump(
            {
                "phases": {"phase2": {"max_usd": 0, "max_tokens": 10_000_000}},
                "providers": {"local": {"base_url": "http://127.0.0.1:9/v1", "priced": False}},
            }
        ),
        encoding="utf-8",
    )
    return LLMClient(
        "phase2",
        budget_path=budget,
        transport=transport,
        ledger_path=tmp_path / "ledger.json",
        cache_dir=tmp_path / "cache",
        **kwargs,
    )


def make_settings(tmp_path: Path, **kwargs: Any) -> CandidateSettings:
    base = CandidateSettings(
        prompts_path=write_prompts(tmp_path),
        output_path=tmp_path / "out" / "candidates.jsonl",
        smoke_output_path=tmp_path / "out" / "smoke" / "candidates.jsonl",
        rewrite_model=MODEL,
        length_targets=("words",),
    )
    return replace(base, **kwargs)


def run(settings: CandidateSettings, client: LLMClient | None = None) -> Any:
    return run_candidates(
        settings,
        client=client,
        apply_ops=fake_apply_ops,
        count=word_count,
        compressor=fake_compressor,
    )


# ----------------------------------------------------------------------------- unit helpers


def test_method_kind_and_ids() -> None:
    assert [method_kind(m) for m in DEFAULT_METHODS] == [
        "ops",
        "ops",
        "llmlingua2",
        "llm",
        "llm",
        "llm",
    ]
    assert candidate_id("code-001", "llm_low") == "code-001:llm_low"
    for bad in ("llm_extreme", "ops_", "gzip"):
        with pytest.raises(ValueError):
            method_kind(bad)


@pytest.mark.parametrize("level", ["low", "medium", "high"])
def test_render_rewrite_prompt(level: str) -> None:
    prompt = 'fix {this} "code" please'
    text = render_rewrite_prompt(prompt, level)
    assert f"Aggressiveness: {level}" in text
    assert text.count(prompt) == 1
    assert "[Prompt]\n" + prompt + "\n[End of prompt]" in text
    assert "{level" not in text and "{prompt}" not in text
    assert "verbatim" in text and "Output only the rewritten prompt" in text
    with pytest.raises(ValueError):
        render_rewrite_prompt(prompt, "extreme")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Write a sort function.", "Write a sort function."),
        ("Here is the rewritten prompt:\nWrite a sort function.", "Write a sort function."),
        ("Sure! Here's the concise version:\n\nWrite X.", "Write X."),
        ("Rewritten prompt: Write X.", "Write X."),
        ('"Write a sort function."', "Write a sort function."),
        ("“Write X.”", "Write X."),
        ("```\nWrite X.\n```", "Write X."),
        ("[Prompt]\nWrite X.\n[End of prompt]", "Write X."),
        ('Here is the rewritten prompt:\n"Write X."', "Write X."),
        ('"Fix" this and "that"', '"Fix" this and "that"'),  # inner quotes: not a wrapper
    ],
)
def test_clean_rewrite(raw: str, expected: str) -> None:
    assert clean_rewrite(raw, "um can you write a sort function") == expected


def test_clean_rewrite_keeps_wrappers_the_original_had() -> None:
    fenced = "```\nprint(1)\n```"
    assert clean_rewrite(fenced, fenced) == fenced
    quoted = '"translate this"'
    assert clean_rewrite(quoted, quoted) == quoted


def test_payload_altered() -> None:
    original = 'please fix `x + 1` and translate "Bonjour tout le monde"\n```py\na = 1\n```'
    kept = 'Translate "Bonjour tout le monde"; fix `x + 1`.\n```py\na = 1\n```'
    assert not payload_altered(original, kept)
    assert payload_altered(original, kept.replace("tout le monde", "le monde"))
    assert payload_altered(original, kept.replace("a = 1", "a=1"))
    assert payload_altered(
        original, kept.replace('"Bonjour tout le monde"', "Bonjour tout le monde")
    )
    assert not payload_altered("no payload here", "none")


def test_validate_rewrite_reasons() -> None:
    original = 'um please translate "hola amigo" to English'
    assert validate_rewrite(
        'Translate "hola amigo" to English', original, targets=["w"], count=word_count
    ) == (
        'Translate "hola amigo" to English',
        None,
    )
    assert validate_rewrite("  ", original, targets=["w"], count=word_count)[1] == "empty"
    assert (
        validate_rewrite(
            "Translate hola amigo to English", original, targets=["w"], count=word_count
        )[1]
        == "payload_altered"
    )
    longer = original + " right now and quickly please thanks"
    assert validate_rewrite(longer, original, targets=["a", "b"], count=word_count)[1] == (
        "longer_than_original:a"
    )


def test_settings_guards(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Ollama"):
        make_settings(tmp_path, stand_in=False, rewrite_model="qwen2.5:1.5b")
    with pytest.raises(ValueError, match="duplicate"):
        make_settings(tmp_path, methods=("ops_F", "ops_F"))
    with pytest.raises(ValueError):
        make_settings(tmp_path, methods=("llm_huge",))
    s = make_settings(tmp_path, smoke_output_path=tmp_path / "out" / "candidates.jsonl")
    with pytest.raises(ValueError, match="differ"):
        output_path_for(s, "ops_F")


# ----------------------------------------------------------------------------- pipeline


def test_run_writes_six_candidates_with_schema(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, stand_in=False)
    fake = FakeRewriter({"high": "Capital of France?"})
    summary = run(settings, make_client(tmp_path, fake))

    rows = read_candidates(settings.output_path)
    assert not settings.smoke_output_path.exists()
    assert len(rows) == 2 * 6
    assert {tuple(r) for r in rows} == {CANDIDATE_FIELDS}
    ids = [r["candidate_id"] for r in rows]
    assert len(set(ids)) == len(ids)
    assert {r["method"] for r in rows} == set(DEFAULT_METHODS)
    by_id = {r["candidate_id"]: r for r in rows}

    ops = by_id["code-01:ops_FHC"]
    assert ops["text"] == 'please fix "a + c" for me thanks'
    assert ops["generator"] == "operators:FHC" and ops["stand_in"] is False
    assert by_id["fact-01:ops_CHFIDSL"]["text"] == PROMPTS[1]["prompt"]
    assert summary.identical_to_original["ops_FHC"] == 1

    assert by_id["fact-01:llmlingua2"]["text"] == "what the of"
    low = by_id["fact-01:llm_low"]
    assert low["text"] == "is the capital of France"  # preamble stripped
    assert low["generator"] == f"{MODEL}|rewrite_v1"
    assert low["rejected"] is False and low["rejection_reason"] is None
    # code-01 high reply drops the quoted payload -> rejected but recorded
    bad = by_id["code-01:llm_high"]
    assert bad["rejected"] is True and bad["rejection_reason"] == "payload_altered"
    assert summary.rejected["llm_high"] == 1
    assert summary.written["llm_low"] == 2 and len(fake.calls) == 6


def test_run_is_resumable_and_cached(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, stand_in=False)
    fake = FakeRewriter()
    run(settings, make_client(tmp_path, fake))
    first = settings.output_path.read_text(encoding="utf-8")

    summary = run(settings, make_client(tmp_path, fake))
    assert settings.output_path.read_text(encoding="utf-8") == first
    assert sum(summary.written.values()) == 0
    assert sum(summary.skipped_existing.values()) == 12
    assert len(fake.calls) == 6

    # Drop two lines: only those are regenerated; the LLM one comes from the response cache.
    lines = first.splitlines(keepends=True)
    kept = [ln for ln in lines if '"code-01:llm_low"' not in ln and '"fact-01:ops_FHC"' not in ln]
    settings.output_path.write_text("".join(kept), encoding="utf-8")
    summary = run(settings, make_client(tmp_path, fake))
    assert summary.written == {"llm_low": 1, "ops_FHC": 1}
    assert len(fake.calls) == 6
    assert sorted(settings.output_path.read_text(encoding="utf-8").splitlines()) == sorted(
        first.splitlines()
    )


def test_resume_mismatch_raises(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, stand_in=False, methods=("llm_low",))
    run(settings, make_client(tmp_path, FakeRewriter()))
    other = replace(settings, rewrite_model="other-model")
    with pytest.raises(ResumeMismatchError):
        run(other, make_client(tmp_path, FakeRewriter()))


def test_stand_in_llm_rewrites_go_to_smoke_file_only(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)  # stand_in defaults to True
    run(settings, make_client(tmp_path, FakeRewriter()))
    real = read_candidates(settings.output_path)
    smoke = read_candidates(settings.smoke_output_path)
    assert {r["method"] for r in real} == {"ops_FHC", "ops_CHFIDSL", "llmlingua2"}
    assert all(r["stand_in"] is False for r in real)
    assert {r["method"] for r in smoke} == {"llm_low", "llm_medium", "llm_high"}
    assert all(r["stand_in"] is True for r in smoke)


def test_non_llm_methods_need_no_client(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, methods=("ops_FHC", "llmlingua2"), limit=1)
    summary = run(settings, client=None)
    assert summary.n_prompts == 1
    assert [r["candidate_id"] for r in read_candidates(settings.output_path)] == [
        "code-01:ops_FHC",
        "code-01:llmlingua2",
    ]


def test_server_unavailable_keeps_earlier_candidates(tmp_path: Path) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = make_client(tmp_path, http_client=httpx.Client(transport=httpx.MockTransport(refuse)))
    settings = make_settings(tmp_path, stand_in=False)
    with pytest.raises(CandidateServerUnavailableError, match="server"):
        run(settings, client)
    methods = {r["method"] for r in read_candidates(settings.output_path)}
    assert methods == {"ops_FHC", "ops_CHFIDSL", "llmlingua2"}


def test_settings_from_config() -> None:
    from terser.config import load_config

    cfg = load_config(["experiment=p2_candidates"])
    s = settings_from_config(cfg)
    assert s.methods == DEFAULT_METHODS
    assert s.stand_in is True and s.rewrite_model == "qwen2.5:1.5b"
    assert s.output_path.as_posix().endswith("data/interim/pilot_candidates.jsonl")
    assert "smoke" in s.smoke_output_path.as_posix()
    assert s.length_targets == ("gpt-4o", "qwen2.5", "llama-3.1")

    cfg = load_config(
        ["experiment=p2_candidates", "experiment.params.methods=[ops_FHC,llmlingua2]"]
    )
    assert settings_from_config(cfg).methods == ("ops_FHC", "llmlingua2")


def test_llmlingua_adapter_uses_cache_without_package(tmp_path: Path) -> None:
    comp = LLMLingua2Compressor("some/model", rate=0.5, cache_dir=tmp_path / "ll")
    assert comp("   ") == "   "
    comp.cache.put(comp._key("hello there"), {"text": "hello", "revision": "abc"})  # type: ignore[union-attr]
    assert comp("hello there") == "hello"
    assert comp.generator == "some/model@abc?rate=0.5"


def test_llmlingua2_resume_checks_model_and_rate(tmp_path: Path) -> None:
    settings = make_settings(tmp_path, methods=("llmlingua2",), limit=1)
    settings.output_path.parent.mkdir(parents=True)
    rec = {
        "prompt_id": "code-01",
        "candidate_id": "code-01:llmlingua2",
        "method": "llmlingua2",
        "text": "fix",
        "generator": f"{settings.llmlingua_model}@abc123?rate=0.5",
        "stand_in": False,
        "rejected": False,
        "rejection_reason": None,
    }
    settings.output_path.write_text(json.dumps(rec) + "\n", encoding="utf-8")
    # same model and rate (revision unknown before loading): skipped, the model is never loaded
    summary = run_candidates(settings, compressor=LLMLingua2Compressor(settings.llmlingua_model))
    assert summary.skipped_existing["llmlingua2"] == 1
    with pytest.raises(ResumeMismatchError):
        run(replace(settings, llmlingua_rate=0.33))
    with pytest.raises(ResumeMismatchError):
        run(replace(settings, llmlingua_model="microsoft/other"))


def test_real_operator_stacks_keep_payload(tmp_path: Path) -> None:
    rows = [
        {"id": "p1", "prompt": 'hey so um could you please translate "Hola, ¿qué tal?" thanks!'},
        {"id": "p2", "prompt": "I was wondering if you could fix `x = 1 +` for me please"},
    ]
    settings = make_settings(tmp_path, methods=("ops_FHC", "ops_CHFIDSL"))
    settings = replace(settings, prompts_path=write_prompts(tmp_path, rows))
    run_candidates(settings, count=word_count)
    out = read_candidates(settings.output_path)
    assert len(out) == 4
    by_pid = {r["prompt_id"]: r["text"] for r in out if r["method"] == "ops_CHFIDSL"}
    assert '"Hola, ¿qué tal?"' in by_pid["p1"] and "`x = 1 +`" in by_pid["p2"]
    assert all(
        not payload_altered(rows[0 if r["prompt_id"] == "p1" else 1]["prompt"], r["text"])
        for r in out
    )


@pytest.mark.network
@pytest.mark.slow
@pytest.mark.skipif(importlib.util.find_spec("llmlingua") is None, reason="llmlingua not installed")
@pytest.mark.skipif(
    os.environ.get("TERSER_RUN_LLMLINGUA") != "1",
    reason="loads a ~700 MB model on CPU: set TERSER_RUN_LLMLINGUA=1 to run",
)
def test_llmlingua2_real_model_compresses(tmp_path: Path) -> None:
    comp = LLMLingua2Compressor(cache_dir=tmp_path / "ll")
    text = PROMPTS[0]["prompt"] + " I really need this to work by tomorrow morning okay"
    out = comp(text)
    assert out and len(out.split()) < len(text.split())
    assert "@" in comp.generator


def test_concurrent_workers_match_sequential_output(tmp_path: Path) -> None:
    seq_dir, par_dir = tmp_path / "seq", tmp_path / "par"
    seq_dir.mkdir()
    par_dir.mkdir()
    seq = make_settings(seq_dir)
    par = make_settings(par_dir, workers=4)
    run(seq, make_client(seq_dir, FakeRewriter()))
    fake = FakeRewriter()
    run(par, make_client(par_dir, fake))
    assert seq.output_path.read_text() == par.output_path.read_text()
    with pytest.raises(ValueError):
        make_settings(tmp_path, workers=0)
