"""OpenAI-compatible providers (Ollama, llama.cpp ``llama-server``, vLLM, hosted endpoints).

``budget.yaml`` maps a provider name to ``{kind: openai_compatible, base_url, priced, ...}``.
Unpriced (self-hosted) providers cost $0 by definition; only token caps apply to them.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import httpx

LOCAL_PROVIDER = "local"
OPENAI_COMPATIBLE = "openai_compatible"
DEFAULT_LOCAL_URL = "http://localhost:11434/v1"
DEFAULT_TIMEOUT_S = 300.0
LOCAL_URL_ENV = "TERSER_LOCAL_LLM_URL"
LOCAL_KEY_ENV = "TERSER_LOCAL_LLM_KEY"


class LocalLLMError(RuntimeError):
    """An OpenAI-compatible server returned an error or an unparseable response."""


class LocalLLMUnavailableError(LocalLLMError):
    """The OpenAI-compatible server could not be reached (not started, wrong URL, timeout)."""


@dataclass(frozen=True)
class ProviderConfig:
    """One named OpenAI-compatible endpoint."""

    name: str
    base_url: str
    kind: str = OPENAI_COMPATIBLE
    priced: bool = False
    timeout_s: float = DEFAULT_TIMEOUT_S
    api_key_env: str | None = None

    def resolved_base_url(self) -> str:
        """``base_url``, overridden by ``TERSER_LOCAL_LLM_URL`` for the ``local`` provider."""
        if self.name == LOCAL_PROVIDER:
            env = os.environ.get(LOCAL_URL_ENV)
            if env:
                return env.rstrip("/")
        return self.base_url.rstrip("/")

    def api_key(self) -> str | None:
        """Bearer key from ``api_key_env`` (``TERSER_LOCAL_LLM_KEY`` for ``local``), if set."""
        env_name = self.api_key_env or (LOCAL_KEY_ENV if self.name == LOCAL_PROVIDER else None)
        return (os.environ.get(env_name) or None) if env_name else None


def default_providers() -> dict[str, ProviderConfig]:
    """Providers available even when ``budget.yaml`` has no ``providers:`` section."""
    return {LOCAL_PROVIDER: ProviderConfig(LOCAL_PROVIDER, DEFAULT_LOCAL_URL)}


def parse_providers(raw: Mapping[str, Any] | None) -> dict[str, ProviderConfig]:
    """Parse the ``providers:`` mapping of ``budget.yaml`` on top of the defaults."""
    providers = default_providers()
    for name, spec in (raw or {}).items():
        spec = spec or {}
        kind = spec.get("kind", OPENAI_COMPATIBLE)
        if kind != OPENAI_COMPATIBLE:
            raise ValueError(f"provider {name!r}: unsupported kind {kind!r}")
        default_url = DEFAULT_LOCAL_URL if name == LOCAL_PROVIDER else None
        base_url = spec.get("base_url", default_url)
        if not base_url:
            raise ValueError(f"provider {name!r}: base_url is required")
        providers[name] = ProviderConfig(
            name=name,
            base_url=str(base_url),
            kind=kind,
            priced=bool(spec.get("priced", False)),
            timeout_s=float(spec.get("timeout_s", DEFAULT_TIMEOUT_S)),
            api_key_env=spec.get("api_key_env"),
        )
    return providers


def openai_compatible_call(
    config: ProviderConfig,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    system: str | None,
    params: dict[str, Any],
    *,
    http_client: httpx.Client | None = None,
) -> tuple[str, int, int, bool]:
    """POST ``/chat/completions``; return ``(text, prompt_tokens, completion_tokens, usage_ok)``.

    Missing ``usage`` in the reply yields zero counts and ``usage_ok=False``.
    """
    import httpx

    base_url = config.resolved_base_url()
    url = f"{base_url}/chat/completions"
    msgs = ([{"role": "system", "content": system}] if system is not None else []) + messages
    body: dict[str, Any] = {**params, "model": model, "messages": msgs, "max_tokens": max_tokens}
    headers = {"Content-Type": "application/json"}
    key = config.api_key()
    if key:
        headers["Authorization"] = f"Bearer {key}"

    hint = (
        f"is the server running? Start Ollama (`ollama serve`), llama.cpp (`llama-server`) or "
        f"vLLM (`vllm serve <model>`), or set {LOCAL_URL_ENV} to the right base URL"
    )
    client = http_client or httpx.Client(timeout=config.timeout_s)
    try:
        resp = client.post(url, json=body, headers=headers, timeout=config.timeout_s)
    except httpx.TimeoutException as exc:
        raise LocalLLMUnavailableError(
            f"provider {config.name!r}: request to {url} timed out after {config.timeout_s:g}s "
            f"({exc.__class__.__name__}); raise timeout_s in budget.yaml or {hint}"
        ) from exc
    except httpx.TransportError as exc:
        raise LocalLLMUnavailableError(
            f"provider {config.name!r}: cannot reach OpenAI-compatible server at {base_url} "
            f"({exc.__class__.__name__}: {exc}); {hint}"
        ) from exc
    finally:
        if http_client is None:
            client.close()

    if resp.status_code >= 400:
        raise LocalLLMError(
            f"provider {config.name!r}: {url} returned HTTP {resp.status_code}: {resp.text[:500]}"
        )
    try:
        data = resp.json()
        text = data["choices"][0]["message"]["content"] or ""
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise LocalLLMError(
            f"provider {config.name!r}: unexpected response from {url}: {resp.text[:500]}"
        ) from exc

    usage = data.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
        return text, prompt_tokens, completion_tokens, True
    return text, 0, 0, False
