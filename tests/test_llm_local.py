"""Tests for local / OpenAI-compatible providers in terser.llm. httpx.MockTransport only."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

from terser.llm import (
    Budget,
    BudgetExceededError,
    LLMClient,
    LocalLLMError,
    LocalLLMUnavailableError,
    parse_providers,
)

MSGS = [{"role": "user", "content": "make this prompt shorter please"}]
REPLY = {
    "choices": [{"message": {"role": "assistant", "content": "shorter"}}],
    "usage": {"prompt_tokens": 12, "completion_tokens": 3},
}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("TERSER_LOCAL_LLM_URL", "TERSER_LOCAL_LLM_KEY", "HOSTED_KEY"):
        monkeypatch.delenv(var, raising=False)


class Server:
    """Records requests and answers with a fixed JSON body."""

    def __init__(self, body: dict[str, Any] | None = None, status: int = 200) -> None:
        self.requests: list[httpx.Request] = []
        self.body = REPLY if body is None else body
        self.status = status

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return httpx.Response(self.status, json=self.body)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handler))

    def payload(self, i: int = -1) -> dict[str, Any]:
        return json.loads(self.requests[i].content)


def make_client(
    tmp_path: Path,
    http: httpx.Client,
    providers: dict[str, Any] | None = None,
    **phase: Any,
) -> LLMClient:
    spec = {"max_usd": 0, "max_tokens": 10_000, "allow_unpriced": False, **phase}
    cfg: dict[str, Any] = {
        "ledger_path": str(tmp_path / "ledger.json"),
        "cache_dir": str(tmp_path / "responses"),
        "phases": {"p": spec},
        "prices": {"paid-model": {"input_per_mtok": 1.0, "output_per_mtok": 2.0}},
    }
    if providers is not None:
        cfg["providers"] = providers
    path = tmp_path / "budget.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return LLMClient("p", budget_path=path, http_client=http)


def ledger(tmp_path: Path) -> dict[str, Any]:
    return json.loads((tmp_path / "ledger.json").read_text(encoding="utf-8"))


def test_request_shape_and_parsing(tmp_path: Path) -> None:
    srv = Server()
    c = make_client(tmp_path, srv.client())
    resp = c.complete(
        "local", "qwen2.5:0.5b", MSGS, max_tokens=64, system="be terse", temperature=0, seed=1
    )
    req = srv.requests[0]
    assert req.method == "POST"
    assert str(req.url) == "http://localhost:11434/v1/chat/completions"
    assert "authorization" not in req.headers
    body = srv.payload()
    assert body["model"] == "qwen2.5:0.5b" and body["max_tokens"] == 64
    assert body["messages"] == [{"role": "system", "content": "be terse"}, *MSGS]
    assert body["temperature"] == 0 and body["seed"] == 1
    assert resp.text == "shorter" and (resp.input_tokens, resp.output_tokens) == (12, 3)
    assert resp.usd == 0.0 and resp.usage_reported and not resp.cached
    rec = ledger(tmp_path)["phases"]["p"]
    assert rec["tokens"] == 15 and rec["usd"] == 0


def test_no_system_message_when_system_is_none(tmp_path: Path) -> None:
    srv = Server()
    make_client(tmp_path, srv.client()).complete("local", "m", MSGS, max_tokens=8, stop=["\n"])
    body = srv.payload()
    assert body["messages"] == MSGS and body["stop"] == ["\n"]


def test_missing_usage_marks_response(tmp_path: Path) -> None:
    srv = Server({"choices": [{"message": {"content": "abcdefgh"}}]})
    c = make_client(tmp_path, srv.client())
    resp = c.complete("local", "m", MSGS, max_tokens=8, input_tokens_estimate=40)
    assert (resp.input_tokens, resp.output_tokens) == (0, 0)
    assert resp.usage_reported is False
    call = ledger(tmp_path)["phases"]["p"]["calls"][0]
    # ledger charges approximate counts so token caps still bound compute
    assert call["usage_reported"] is False and call["input_tokens"] == 40
    assert call["output_tokens"] == 2
    again = c.complete("local", "m", MSGS, max_tokens=8, input_tokens_estimate=40)
    assert again.cached and again.usage_reported is False and len(srv.requests) == 1


def test_local_ignores_usd_and_unpriced_rules(tmp_path: Path) -> None:
    srv = Server()
    # max_usd 0 and allow_unpriced false would block a priced provider
    c = make_client(tmp_path, srv.client(), max_usd=0, allow_unpriced=False)
    resp = c.complete("local", "never-priced", MSGS, max_tokens=16)
    assert resp.usd == 0.0 and len(srv.requests) == 1


def test_local_blocked_by_token_cap(tmp_path: Path) -> None:
    srv = Server()
    c = make_client(tmp_path, srv.client(), max_tokens=50)
    with pytest.raises(BudgetExceededError, match="token cap"):
        c.complete("local", "m", MSGS, max_tokens=100)
    assert srv.requests == []


def test_local_blocked_in_zero_cap_phase(tmp_path: Path) -> None:
    srv = Server()
    c = make_client(tmp_path, srv.client(), max_usd=100, max_tokens=0, allow_unpriced=True)
    with pytest.raises(BudgetExceededError, match="not allowed"):
        c.complete("local", "m", MSGS, max_tokens=10)
    assert srv.requests == []


def test_cache_hit_costs_zero_and_skips_server(tmp_path: Path) -> None:
    srv = Server()
    c = make_client(tmp_path, srv.client())
    first = c.complete("local", "m", MSGS, max_tokens=16, temperature=0)
    before = ledger(tmp_path)
    second = c.complete("local", "m", MSGS, max_tokens=16, temperature=0)
    assert len(srv.requests) == 1
    assert second.cached and second.usd == 0 and second.text == first.text
    assert ledger(tmp_path) == before


def test_env_overrides_url_and_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TERSER_LOCAL_LLM_URL", "http://gpu-box:8000/v1/")
    monkeypatch.setenv("TERSER_LOCAL_LLM_KEY", "sekrit")
    srv = Server()
    make_client(tmp_path, srv.client()).complete("local", "m", MSGS, max_tokens=8)
    req = srv.requests[0]
    assert str(req.url) == "http://gpu-box:8000/v1/chat/completions"
    assert req.headers["authorization"] == "Bearer sekrit"


def test_yaml_base_url_used(tmp_path: Path) -> None:
    srv = Server()
    providers = {"local": {"kind": "openai_compatible", "base_url": "http://127.0.0.1:8080/v1"}}
    make_client(tmp_path, srv.client(), providers).complete("local", "m", MSGS, max_tokens=8)
    assert str(srv.requests[0].url) == "http://127.0.0.1:8080/v1/chat/completions"


def test_priced_hosted_provider_follows_usd_rules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOSTED_KEY", "k1")
    srv = Server()
    providers = {
        "hosted": {
            "kind": "openai_compatible",
            "base_url": "https://api.example.com/v1",
            "priced": True,
            "api_key_env": "HOSTED_KEY",
        }
    }
    c = make_client(tmp_path, srv.client(), providers, max_usd=1.0)
    with pytest.raises(BudgetExceededError, match="no price"):
        c.complete("hosted", "unlisted", MSGS, max_tokens=8)
    assert srv.requests == []
    resp = c.complete("hosted", "paid-model", MSGS, max_tokens=8)
    assert resp.usd == pytest.approx((12 * 1 + 3 * 2) / 1e6)
    assert srv.requests[0].headers["authorization"] == "Bearer k1"


def failing(exc: Callable[[httpx.Request], Exception]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        raise exc(request)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_unreachable_server_clear_error(tmp_path: Path) -> None:
    http = failing(lambda r: httpx.ConnectError("connection refused", request=r))
    c = make_client(tmp_path, http)
    with pytest.raises(LocalLLMUnavailableError, match="Ollama") as info:
        c.complete("local", "m", MSGS, max_tokens=8)
    assert "llama-server" in str(info.value) and "vLLM" in str(info.value)
    assert "localhost:11434" in str(info.value)
    # the reservation is released: nothing spent, nothing left pending
    data = json.loads((tmp_path / "ledger.json").read_text(encoding="utf-8"))
    assert data["phases"] == {} and not any(data["reserved"].values())


def test_timeout_is_unavailable_error(tmp_path: Path) -> None:
    http = failing(lambda r: httpx.ReadTimeout("slow", request=r))
    with pytest.raises(LocalLLMUnavailableError, match="timed out"):
        make_client(tmp_path, http).complete("local", "m", MSGS, max_tokens=8)


def test_http_error_and_bad_body(tmp_path: Path) -> None:
    c = make_client(tmp_path, Server({"error": "model not found"}, status=404).client())
    with pytest.raises(LocalLLMError, match="HTTP 404"):
        c.complete("local", "m", MSGS, max_tokens=8)
    c = make_client(tmp_path, Server({"nope": 1}).client())
    with pytest.raises(LocalLLMError, match="unexpected response"):
        c.complete("local", "m", MSGS, max_tokens=8)


def test_repo_budget_yaml_providers(tmp_path: Path) -> None:
    path = Path(__file__).resolve().parents[1] / "configs" / "budget.yaml"
    local_phases = ["phase1", "phase2", "phase5", "phase8", "phase9", "phase11", "phase12"]
    for phase in local_phases:
        b = Budget.load(path, phase, ledger_path=tmp_path / "l.json")
        assert b.max_tokens > 0 and b.is_free("local")
    b = Budget.load(path, "phase3", ledger_path=tmp_path / "l.json")
    assert b.providers["local"].base_url == "http://localhost:11434/v1"
    assert b.providers["local"].timeout_s == 300 and not b.is_free("anthropic")


def test_parse_providers_validation() -> None:
    assert "local" in parse_providers(None)
    with pytest.raises(ValueError, match="kind"):
        parse_providers({"x": {"kind": "grpc", "base_url": "http://h"}})
    with pytest.raises(ValueError, match="base_url"):
        parse_providers({"x": {"kind": "openai_compatible"}})


def test_parse_logprobs_reads_top_logprobs() -> None:
    from terser.llm.providers import parse_logprobs

    choice = {
        "logprobs": {
            "content": [
                {
                    "token": "A",
                    "logprob": -0.1,
                    "top_logprobs": [
                        {"token": "A", "logprob": -0.1},
                        {"token": "B", "logprob": -2.5},
                    ],
                },
                {"token": "}", "logprob": 0.0},
            ]
        }
    }
    assert parse_logprobs(choice) == (("A", (("A", -0.1), ("B", -2.5))), ("}", ()))
    assert parse_logprobs({"message": {"content": "x"}}) is None
    assert parse_logprobs({"logprobs": None}) is None
