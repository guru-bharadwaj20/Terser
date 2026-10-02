"""Row 2.2.3: the respond stage (offline; the LLM transport is faked)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

from terser.data.respond import (
    RespondServerUnavailableError,
    RespondSettings,
    iter_texts,
    response_id,
    run_respond,
)
from terser.llm import LLMClient
from terser.llm.client import TransportResult

PROMPTS = [
    {"id": "code-001", "category": "code", "prompt": "hey can u write a palindrome fn"},
    {"id": "creative-001", "category": "creative", "prompt": "um write a haiku about rain"},
]
CANDIDATES = [
    {"prompt_id": "code-001", "candidate_id": "code-001:ops_FHC", "method": "ops_FHC",
     "text": "Write a palindrome function.", "rejected": False, "stand_in": False},
    {"prompt_id": "code-001", "candidate_id": "code-001:llm_high", "method": "llm_high",
     "text": "", "rejected": True, "stand_in": False},
    {"prompt_id": "creative-001", "candidate_id": "creative-001:llm_low", "method": "llm_low",
     "text": "Haiku about rain.", "rejected": False, "stand_in": True},
]  # fmt: skip


class FakeTransport:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def __call__(
        self,
        provider: str,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        system: str | None,
        params: dict[str, Any],
    ) -> TransportResult:
        self.calls.append((model, messages[0]["content"], params))
        return TransportResult(f"{model} says: {messages[0]['content']}", 10, 5)


def make_client(tmp_path: Path, transport: Any = None, **kw: Any) -> LLMClient:
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
        **kw,
    )


def settings(tmp_path: Path, **kw: Any) -> RespondSettings:
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text("".join(json.dumps(r) + "\n" for r in PROMPTS), encoding="utf-8")
    cands = tmp_path / "cands.jsonl"
    cands.write_text("".join(json.dumps(r) + "\n" for r in CANDIDATES), encoding="utf-8")
    return RespondSettings(
        prompts_path=prompts,
        candidates_path=cands,
        output_path=tmp_path / "out" / "responses.jsonl",
        smoke_output_path=tmp_path / "smoke" / "responses.jsonl",
        targets=("org/Target-A", "org/Target-B"),
        **kw,
    )


def read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_iter_texts_skips_rejected_and_stand_in_candidates() -> None:
    ids = [t["text_id"] for t in iter_texts(PROMPTS, CANDIDATES)]
    assert ids == ["code-001:original", "code-001:ops_FHC", "creative-001:original"]
    no_orig = [t["text_id"] for t in iter_texts(PROMPTS, CANDIDATES, include_original=False)]
    assert no_orig == ["code-001:ops_FHC"]


def test_greedy_once_and_creative_sampled_three_times(tmp_path: Path) -> None:
    fake = FakeTransport()
    s = settings(tmp_path)
    summary = run_respond(s, client=make_client(tmp_path, fake))
    rows = read(s.output_path)
    # per target: 2 code texts x 1 + 1 creative text x 3 = 5
    assert len(rows) == 10 and summary.written["org/Target-A"] == 5
    creative = [r for r in rows if r["category"] == "creative" and r["target"] == "org/Target-A"]
    assert [r["sample"] for r in creative] == [0, 1, 2]
    assert [r["seed"] for r in creative] == [0, 1, 2]
    assert all(r["temperature"] == 0.7 for r in creative)
    code = [r for r in rows if r["category"] == "code"]
    assert all(r["temperature"] == 0.0 and r["seed"] is None for r in code)
    assert all(r["stand_in"] is False for r in rows)
    assert rows[0]["response_id"] == response_id("code-001:original", "org/Target-A", 0)
    assert rows[0]["output"].startswith("org/Target-A says:")
    assert not s.smoke_output_path.exists()


def test_resume_skips_existing_responses(tmp_path: Path) -> None:
    fake = FakeTransport()
    s = settings(tmp_path)
    run_respond(s, client=make_client(tmp_path, fake))
    n_calls = len(fake.calls)
    again = run_respond(s, client=make_client(tmp_path, fake))
    assert len(fake.calls) == n_calls and again.skipped_existing == 10
    assert len(read(s.output_path)) == 10


def test_stand_in_goes_to_smoke_file_only(tmp_path: Path) -> None:
    s = settings(tmp_path, stand_in=True)
    run_respond(s, client=make_client(tmp_path, FakeTransport()))
    assert not s.output_path.exists()
    assert all(r["stand_in"] for r in read(s.smoke_output_path))


def test_settings_validation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Ollama"):
        settings(tmp_path).__class__(
            prompts_path=tmp_path / "p",
            candidates_path=tmp_path / "c",
            output_path=tmp_path / "o",
            smoke_output_path=tmp_path / "s",
            targets=("qwen2.5:0.5b",),
        )
    with pytest.raises(ValueError, match="differ"):
        RespondSettings(
            prompts_path=tmp_path / "p",
            candidates_path=tmp_path / "c",
            output_path=tmp_path / "o",
            smoke_output_path=tmp_path / "o",
        )


def test_unreachable_server_is_a_clear_error(tmp_path: Path) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client = make_client(tmp_path, http_client=httpx.Client(transport=httpx.MockTransport(refuse)))
    with pytest.raises(RespondServerUnavailableError):
        run_respond(settings(tmp_path), client=client)


def test_config_composes() -> None:
    from terser.config import load_config
    from terser.data.respond import settings_from_config

    cfg = load_config(["experiment=p2_respond"])
    s = settings_from_config(cfg)
    assert s.targets == ("meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen2.5-7B-Instruct")
    assert s.creative_samples == 3 and not s.stand_in
