"""Budget-guarded, disk-cached LLM client: cache lookup -> budget check -> call -> record."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

from terser.llm.budget import Budget, Usage, approx_tokens
from terser.llm.cache import ResponseCache, cache_key
from terser.llm.providers import (
    ProviderConfig,
    TokenLogprobs,
    default_providers,
    openai_compatible_request,
)

if TYPE_CHECKING:
    import httpx

DEFAULT_BUDGET_PATH = Path(__file__).resolve().parents[3] / "configs" / "budget.yaml"


@dataclass(frozen=True)
class TransportResult:
    text: str
    input_tokens: int
    output_tokens: int
    usage_reported: bool = True  # False when the server omitted usage (counts are then 0)
    logprobs: TokenLogprobs | None = None  # only when requested and returned by the server


@dataclass(frozen=True)
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    usd: float | None
    cached: bool
    cache_key: str
    usage_reported: bool = True
    logprobs: TokenLogprobs | None = None


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
        model=model, messages=cast(Any, messages), max_tokens=max_tokens, **kwargs
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


class ProviderTransport:
    """Dispatches to Anthropic, OpenAI or a configured OpenAI-compatible provider."""

    def __init__(
        self,
        providers: dict[str, ProviderConfig] | None = None,
        *,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.providers = providers if providers is not None else default_providers()
        self.http_client = http_client

    def __call__(
        self,
        provider: str,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        system: str | None,
        params: dict[str, Any],
    ) -> TransportResult:
        if provider in self.providers:
            text, tokens_in, tokens_out, reported, logprobs = openai_compatible_request(
                self.providers[provider],
                model,
                messages,
                max_tokens,
                system,
                params,
                http_client=self.http_client,
            )
            return TransportResult(text, tokens_in, tokens_out, reported, logprobs)
        hosted = {"anthropic": anthropic_transport, "openai": openai_transport}
        if provider not in hosted:
            known = sorted({*hosted, *self.providers})
            raise ValueError(f"unknown provider {provider!r}; expected one of {known}")
        return hosted[provider](provider, model, messages, max_tokens, system, params)


def default_transport(
    provider: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    system: str | None,
    params: dict[str, Any],
) -> TransportResult:
    """Transport with built-in providers only (``anthropic``, ``openai``, ``local``)."""
    return ProviderTransport()(provider, model, messages, max_tokens, system, params)


class LLMClient:
    """Every LLM call in Terser (hosted or local) goes through here."""

    def __init__(
        self,
        phase: str,
        *,
        budget_path: str | Path = DEFAULT_BUDGET_PATH,
        transport: Transport | None = None,
        ledger_path: str | Path | None = None,
        cache_dir: str | Path | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.budget = Budget.load(budget_path, phase, ledger_path=ledger_path, cache_dir=cache_dir)
        self.cache = ResponseCache(self.budget.cache_dir)
        self.transport: Transport = transport or ProviderTransport(
            self.budget.providers, http_client=http_client
        )

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
            reported = bool(hit.get("usage_reported", True))
            return LLMResponse(
                hit["text"],
                tokens_in,
                tokens_out,
                0.0,
                True,
                key,
                reported,
                _logprobs_from_json(hit.get("logprobs")),
            )

        if input_tokens_estimate is None:
            prompt = json.dumps(messages, ensure_ascii=False) + (system or "")
            input_tokens_estimate = approx_tokens(prompt)
        self.budget.check(Usage(provider, model, input_tokens_estimate, max_tokens, key))

        result = self.transport(provider, model, messages, max_tokens, system, dict(params))
        if result.usage_reported:
            actual = Usage(provider, model, result.input_tokens, result.output_tokens, key)
        else:
            # Server omitted usage: charge approximate counts so token caps still bound compute.
            out_estimate = approx_tokens(result.text) if result.text else 0
            actual = Usage(provider, model, input_tokens_estimate, out_estimate, key, False)
        usd = self.budget.record(actual)
        payload: dict[str, Any] = {
            "provider": provider,
            "model": model,
            "text": result.text,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "usd": usd,
            "usage_reported": result.usage_reported,
        }
        if result.logprobs is not None:
            payload["logprobs"] = result.logprobs
        self.cache.put(key, payload)
        return LLMResponse(
            result.text,
            result.input_tokens,
            result.output_tokens,
            usd,
            False,
            key,
            result.usage_reported,
            result.logprobs,
        )


def _logprobs_from_json(raw: Any) -> TokenLogprobs | None:
    """Rebuild :data:`TokenLogprobs` from its cached JSON form (lists instead of tuples)."""
    if raw is None:
        return None
    return tuple((str(token), tuple((str(c), float(lp)) for c, lp in top)) for token, top in raw)
