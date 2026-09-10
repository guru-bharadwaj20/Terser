"""Budget-guarded, disk-cached LLM client: cache lookup -> budget check -> call -> record."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from terser.llm.budget import Budget, Usage, approx_tokens
from terser.llm.cache import ResponseCache, cache_key

DEFAULT_BUDGET_PATH = Path(__file__).resolve().parents[3] / "configs" / "budget.yaml"


@dataclass(frozen=True)
class TransportResult:
    text: str
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    usd: float | None
    cached: bool
    cache_key: str


class Transport(Protocol):
    """Performs the actual provider call. Injected in tests to avoid real API traffic."""

    def __call__(
        self,
        provider: str,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        system: str | None,
        params: dict[str, Any],
    ) -> TransportResult: ...


def anthropic_transport(
    provider: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    system: str | None,
    params: dict[str, Any],
) -> TransportResult:
    import anthropic

    kwargs: dict[str, Any] = dict(params)
    if system is not None:
        kwargs["system"] = system
    resp = anthropic.Anthropic().messages.create(
        model=model, messages=messages, max_tokens=max_tokens, **kwargs
    )
    text = "".join(getattr(b, "text", "") for b in resp.content if b.type == "text")
    return TransportResult(text, resp.usage.input_tokens, resp.usage.output_tokens)


def openai_transport(
    provider: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    system: str | None,
    params: dict[str, Any],
) -> TransportResult:
    try:
        import openai
    except ImportError as exc:  # optional dependency
        raise RuntimeError("OpenAI provider requires the 'openai' package") from exc

    msgs = ([{"role": "system", "content": system}] if system is not None else []) + messages
    resp = openai.OpenAI().chat.completions.create(
        model=model, messages=msgs, max_tokens=max_tokens, **params
    )
    text = resp.choices[0].message.content or ""
    return TransportResult(text, resp.usage.prompt_tokens, resp.usage.completion_tokens)


def default_transport(
    provider: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    system: str | None,
    params: dict[str, Any],
) -> TransportResult:
    transports = {"anthropic": anthropic_transport, "openai": openai_transport}
    if provider not in transports:
        raise ValueError(f"unknown provider {provider!r}; expected one of {sorted(transports)}")
    return transports[provider](provider, model, messages, max_tokens, system, params)


class LLMClient:
    """Every hosted-LLM call in Terser goes through here."""

    def __init__(
        self,
        phase: str,
        *,
        budget_path: str | Path = DEFAULT_BUDGET_PATH,
        transport: Transport | None = None,
        ledger_path: str | Path | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.budget = Budget.load(budget_path, phase, ledger_path=ledger_path, cache_dir=cache_dir)
        self.cache = ResponseCache(self.budget.cache_dir)
        self.transport: Transport = transport or default_transport

    def complete(
        self,
        provider: str,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        system: str | None = None,
        *,
        input_tokens_estimate: int | None = None,
        **params: Any,
    ) -> LLMResponse:
        key = cache_key(provider, model, messages, {**params, "max_tokens": max_tokens}, system)
        hit = self.cache.get(key)
        if hit is not None:
            tokens_in, tokens_out = hit["input_tokens"], hit["output_tokens"]
            return LLMResponse(hit["text"], tokens_in, tokens_out, 0.0, True, key)

        if input_tokens_estimate is None:
            prompt = json.dumps(messages, ensure_ascii=False) + (system or "")
            input_tokens_estimate = approx_tokens(prompt)
        self.budget.check(Usage(provider, model, input_tokens_estimate, max_tokens, key))

        result = self.transport(provider, model, messages, max_tokens, system, dict(params))
        usd = self.budget.record(
            Usage(provider, model, result.input_tokens, result.output_tokens, key)
        )
        self.cache.put(
            key,
            {
                "provider": provider,
                "model": model,
                "text": result.text,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "usd": usd,
            },
        )
        return LLMResponse(result.text, result.input_tokens, result.output_tokens, usd, False, key)
