"""LRU response cache keyed on a normalised prompt hash and the target (plan row **11.3.4**).

Normalisation (:func:`normalise`) is deliberately conservative: Unicode NFC, ``\\r\\n`` -> ``\\n``,
trailing whitespace stripped per line and leading/trailing blank lines removed. Case and inner
spacing are *kept*, because they can change the token count on the target and, inside code
blocks, the meaning. The key is ``sha256(target \\x00 mode \\x00 normalised prompt)``, so raw text
is never stored as a key.
"""

from __future__ import annotations

import hashlib
import threading
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass
from typing import Generic, TypeVar

__all__ = ["CacheStats", "LRUCache", "cache_key", "normalise"]

V = TypeVar("V")


def normalise(prompt: str) -> str:
    """Canonical form of a prompt for cache lookup."""
    text = unicodedata.normalize("NFC", prompt).replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.split("\n")).strip("\n").strip()


def cache_key(prompt: str, target: str, mode: str = "") -> str:
    """Hex sha256 of target, mode and the normalised prompt."""
    return hashlib.sha256(f"{target}\x00{mode}\x00{normalise(prompt)}".encode()).hexdigest()


@dataclass
class CacheStats:
    """Hit/miss counters."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


class LRUCache(Generic[V]):
    """Thread-safe bounded LRU map."""

    def __init__(self, maxsize: int = 10_000) -> None:
        if maxsize < 1:
            raise ValueError("maxsize must be >= 1")
        self.maxsize = maxsize
        self._data: OrderedDict[str, V] = OrderedDict()
        self._lock = threading.Lock()
        self.stats = CacheStats()

    def get(self, key: str) -> V | None:
        """Value for ``key`` (marking it most recently used) or ``None``."""
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                self.stats.hits += 1
                return self._data[key]
            self.stats.misses += 1
            return None

    def put(self, key: str, value: V) -> None:
        """Insert or refresh ``key``; evict the least recently used entry when full."""
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)
                self.stats.evictions += 1

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
