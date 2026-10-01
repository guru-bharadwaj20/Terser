"""Per-phase API budget: YAML caps, a persisted JSON spend ledger and pre-call checks."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from terser.llm.providers import ProviderConfig, default_providers, parse_providers

DEFAULT_LEDGER = ".cache/llm_ledger.json"
DEFAULT_CACHE_DIR = ".cache/llm_responses/"


class BudgetExceededError(RuntimeError):
    """Raised before an API call that would exceed (or cannot be verified against) a cap."""


def atomic_write_json(path: Path, data: Any, compact: bool = False) -> None:
    """Write JSON to ``path`` via a temp file in the same dir + ``os.replace``.

    ``compact`` drops indentation so the C encoder is used (``indent`` forces the pure-Python
    one, ~10x slower); use it for large, frequently rewritten files such as the ledger.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            if compact:
                fh.write(json.dumps(data, sort_keys=True, separators=(",", ":")))
            else:
                json.dump(data, fh, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def project_root(config_path: Path) -> Path:
    """Project root: env ``TERSER_ROOT``, else the parent of the ``configs/`` dir."""
    env = os.environ.get("TERSER_ROOT")
    if env:
        return Path(env)
    parent = config_path.resolve().parent
    return parent.parent if parent.name == "configs" else parent


def approx_tokens(text: str) -> int:
    """Rough token estimate (~4 chars per token), used when the caller gives no count."""
    return max(1, -(-len(text) // 4))


@dataclass(frozen=True)
class Price:
    input_per_mtok: float | None
    output_per_mtok: float | None

    @property
    def known(self) -> bool:
        return self.input_per_mtok is not None and self.output_per_mtok is not None

    def cost(self, input_tokens: int, output_tokens: int) -> float | None:
        if self.input_per_mtok is None or self.output_per_mtok is None:
            return None
        return (input_tokens * self.input_per_mtok + output_tokens * self.output_per_mtok) / 1e6


@dataclass(frozen=True)
class Usage:
    """Token usage of one (estimated or actual) call."""

    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cache_key: str | None = None
    usage_reported: bool = True


class Ledger:
    """JSON ledger of spend per phase plus per-call entries; every mutation is atomic."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._stamp: tuple[int, int] | None = None
        self.data: dict[str, Any] = self._read()

    def _file_stamp(self) -> tuple[int, int] | None:
        try:
            st = self.path.stat()
        except FileNotFoundError:
            return None
        return st.st_mtime_ns, st.st_size

    def _read(self) -> dict[str, Any]:
        """Return the on-disk ledger, re-parsing only if the file changed since the last read
        or write by this object (other processes may append to the same ledger)."""
        stamp = self._file_stamp()
        if stamp is not None and stamp == self._stamp and hasattr(self, "data"):
            return self.data
        self._stamp = stamp
        if stamp is not None:
            return json.loads(self.path.read_text(encoding="utf-8"))
        return {"phases": {}}

    def reload(self) -> None:
        with self._lock:
            self.data = self._read()

    def spent(self, phase: str) -> dict[str, Any]:
        return self.data["phases"].get(phase, {"tokens": 0, "usd": 0.0, "calls": []})

    def add(self, phase: str, usage: Usage, usd: float | None) -> dict[str, Any]:
        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "provider": usage.provider,
            "model": usage.model,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "usd": usd,
            "cache_key": usage.cache_key,
            "usage_reported": usage.usage_reported,
        }
        with self._lock:
            data = self._read()  # merge with whatever is on disk now
            rec = data["phases"].setdefault(phase, {"tokens": 0, "usd": 0.0, "calls": []})
            rec["tokens"] += usage.input_tokens + usage.output_tokens
            rec["usd"] += usd or 0.0
            rec["calls"].append(entry)
            try:
                atomic_write_json(self.path, data, compact=True)
            except BaseException:
                self._stamp = None  # in-memory data now differs from disk: re-read next time
                raise
            self._stamp = self._file_stamp()
            self.data = data
        return entry


@dataclass
class Budget:
    """Caps for one phase, price table and the ledger they are checked against."""

    phase: str
    max_usd: float
    max_tokens: int
    allow_unpriced: bool
    prices: dict[str, Price]
    ledger: Ledger
    cache_dir: Path
    config_path: Path | None = field(default=None)
    providers: dict[str, ProviderConfig] = field(default_factory=default_providers)

    @classmethod
    def load(
        cls,
        path: str | Path,
        phase: str,
        *,
        ledger_path: str | Path | None = None,
        cache_dir: str | Path | None = None,
    ) -> Budget:
        path = Path(path)
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        phases = cfg.get("phases") or {}
        if phase not in phases:
            raise KeyError(f"phase {phase!r} not defined in {path}")
        p = phases[phase] or {}
        root = project_root(path)

        def resolve(value: str | Path) -> Path:
            value = Path(value)
            return value if value.is_absolute() else root / value

        prices = {
            m: Price(v.get("input_per_mtok"), v.get("output_per_mtok"))
            for m, v in (cfg.get("prices") or {}).items()
        }
        return cls(
            phase=phase,
            max_usd=float(p.get("max_usd", 0)),
            max_tokens=int(p.get("max_tokens", 0)),
            allow_unpriced=bool(p.get("allow_unpriced", False)),
            prices=prices,
            ledger=Ledger(resolve(ledger_path or cfg.get("ledger_path", DEFAULT_LEDGER))),
            cache_dir=resolve(cache_dir or cfg.get("cache_dir", DEFAULT_CACHE_DIR)),
            config_path=path,
            providers=parse_providers(cfg.get("providers")),
        )

    def price(self, model: str) -> Price:
        return self.prices.get(model, Price(None, None))

    def is_free(self, provider: str) -> bool:
        """True for self-hosted providers (``priced: false``): cost is $0 by definition."""
        cfg = self.providers.get(provider)
        return cfg is not None and not cfg.priced

    def cost(self, usage: Usage) -> float | None:
        """USD cost of ``usage``: 0.0 for free providers, None if the model is unpriced."""
        if self.is_free(usage.provider):
            return 0.0
        return self.price(usage.model).cost(usage.input_tokens, usage.output_tokens)

    def check(self, estimate: Usage) -> float | None:
        """Raise ``BudgetExceededError`` if ``estimate`` would break a cap; return est. USD.

        Free (self-hosted) providers skip the USD cap and ``allow_unpriced`` rules; the token cap
        still applies because it bounds compute time.
        """
        self.ledger.reload()
        spent = self.ledger.spent(self.phase)
        tokens = estimate.input_tokens + estimate.output_tokens
        free = self.is_free(estimate.provider)
        if self.max_tokens <= 0 or (not free and self.max_usd <= 0):
            raise BudgetExceededError(f"{self.phase}: API calls are not allowed (cap is 0)")
        if spent["tokens"] + tokens > self.max_tokens:
            raise BudgetExceededError(
                f"{self.phase}: token cap {self.max_tokens} would be exceeded "
                f"(spent {spent['tokens']} + estimate {tokens})"
            )
        usd = self.cost(estimate)
        if free:
            return usd
        if usd is None:
            if not self.allow_unpriced:
                raise BudgetExceededError(
                    f"{self.phase}: model {estimate.model!r} has no price in budget.yaml; "
                    "USD cap cannot be verified (set allow_unpriced: true to override)"
                )
        elif spent["usd"] + usd > self.max_usd:
            raise BudgetExceededError(
                f"{self.phase}: USD cap {self.max_usd} would be exceeded "
                f"(spent {spent['usd']:.4f} + estimate {usd:.4f})"
            )
        return usd

    def record(self, actual: Usage) -> float | None:
        """Record an actual call's usage in the ledger; return its USD cost (None if unpriced)."""
        usd = self.cost(actual)
        self.ledger.add(self.phase, actual, usd)
        return usd
