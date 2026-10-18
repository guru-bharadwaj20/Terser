"""Per-request net-savings accounting (plan row **11.3.6**, headline criterion S4).

``net_savings = price_T * (tokens_before - tokens_after) - rewriter_cost``

* ``price_T`` is the input price per token of target ``T`` (configured per million tokens in
  ``configs/serving/default.yaml``; the values there are deployment inputs, not measurements).
* ``rewriter_cost`` is what producing the rewrite cost: a fixed per-request cost for the CPU fast
  path, plus, for the quality path, the rewriter's own input and output tokens at the rewriter's
  per-token price.

A fallback is charged for the path that produced its result: the fast path's per-request cost,
and nothing for the quality call it abandoned (a timed-out or failed call reports no token usage,
so that work is not measured). A fallback that returns the original prompt therefore saves
nothing and costs the fast-path fee, so its net savings are at most zero.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

__all__ = ["Pricing", "Savings", "net_savings"]


@dataclass(frozen=True)
class Pricing:
    """Prices in USD per million tokens, and fixed per-request path costs in USD."""

    target_input_per_mtok: Mapping[str, float]
    rewriter_input_per_mtok: float = 0.0
    rewriter_output_per_mtok: float = 0.0
    fast_path_per_request: float = 0.0

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> Pricing:
        return cls(
            target_input_per_mtok=dict(cfg.get("target_input_per_mtok", {})),
            rewriter_input_per_mtok=float(cfg.get("rewriter_input_per_mtok", 0.0)),
            rewriter_output_per_mtok=float(cfg.get("rewriter_output_per_mtok", 0.0)),
            fast_path_per_request=float(cfg.get("fast_path_per_request", 0.0)),
        )

    def target_price(self, target: str) -> float:
        """USD per token for ``target``; ``KeyError`` if it has no price."""
        if target not in self.target_input_per_mtok:
            raise KeyError(f"no input price configured for target {target!r}")
        return float(self.target_input_per_mtok[target]) / 1e6


@dataclass(frozen=True)
class Savings:
    """Accounting for one request (USD)."""

    tokens_saved: int
    gross_savings: float
    rewriter_cost: float
    net_savings: float

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def net_savings(
    pricing: Pricing,
    target: str,
    tokens_before: int,
    tokens_after: int,
    *,
    path: str,
    rewriter_tokens_in: int = 0,
    rewriter_tokens_out: int = 0,
) -> Savings:
    """``price_T * (before - after) - rewriter_cost`` for one request on ``path``."""
    if tokens_before < 0 or tokens_after < 0:
        raise ValueError("token counts must be non-negative")
    saved = tokens_before - tokens_after
    gross = pricing.target_price(target) * saved
    cost = pricing.fast_path_per_request
    if path == "quality":
        cost += (
            rewriter_tokens_in * pricing.rewriter_input_per_mtok
            + rewriter_tokens_out * pricing.rewriter_output_per_mtok
        ) / 1e6
    return Savings(saved, gross, cost, gross - cost)
