"""LLM access (hosted or local OpenAI-compatible) with per-phase budget caps and a disk cache."""

from terser.llm.budget import Budget, BudgetExceededError, Ledger, Usage
from terser.llm.cache import ResponseCache, cache_key
from terser.llm.client import LLMClient, LLMResponse, ProviderTransport, Transport, TransportResult
from terser.llm.providers import (
    LocalLLMError,
    LocalLLMUnavailableError,
    ProviderConfig,
    parse_providers,
)

__all__ = [
    "Budget",
    "BudgetExceededError",
    "LLMClient",
    "LLMResponse",
    "Ledger",
    "LocalLLMError",
    "LocalLLMUnavailableError",
    "ProviderConfig",
    "ProviderTransport",
    "ResponseCache",
    "Transport",
    "TransportResult",
    "Usage",
    "cache_key",
    "parse_providers",
]
