"""Cached prompt embeddings with an id index (plan row **6.1**).

The cache is two files: ``<name>.npy`` (an (n, d) float32 matrix) and ``<name>.ids.json``
(the prompt id of each row). :meth:`EmbeddingCache.get` returns rows in the order of the ids
asked for and raises ``KeyError`` naming any id that is not cached, so a stale cache never
silently drops prompts.

The encoder is pluggable: :func:`bge_small_encoder` wraps ``BAAI/bge-small-en-v1.5`` (384-d,
L2-normalised) through ``sentence-transformers``, imported lazily; tests use a fake encoder.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

__all__ = ["BGE_SMALL", "EmbeddingCache", "Encoder", "bge_small_encoder"]

BGE_SMALL = "BAAI/bge-small-en-v1.5"
Encoder = Callable[[Sequence[str]], npt.NDArray[np.float32]]


def bge_small_encoder(device: str = "cpu", batch_size: int = 64) -> Encoder:
    """bge-small 384-d encoder with normalised outputs (needs ``sentence-transformers``)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(BGE_SMALL, device=device)

    def encode(texts: Sequence[str]) -> npt.NDArray[np.float32]:
        out = model.encode(list(texts), batch_size=batch_size, normalize_embeddings=True)
        return np.asarray(out, dtype=np.float32)

    return encode


@dataclass
class EmbeddingCache:
    """An (n, d) matrix plus the id of each row."""

    ids: list[str]
    matrix: npt.NDArray[np.float32]

    def __post_init__(self) -> None:
        if self.matrix.ndim != 2 or len(self.ids) != self.matrix.shape[0]:
            raise ValueError(f"{len(self.ids)} ids for a matrix of shape {self.matrix.shape}")
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("duplicate ids in embedding cache")
        self._index = {pid: i for i, pid in enumerate(self.ids)}

    @classmethod
    def build(cls, ids: Sequence[str], texts: Sequence[str], encoder: Encoder) -> EmbeddingCache:
        """Encode ``texts`` and index them by ``ids``."""
        if len(ids) != len(texts):
            raise ValueError("ids and texts differ in length")
        return cls(list(ids), np.asarray(encoder(texts), dtype=np.float32))

    def get(self, ids: Sequence[str]) -> npt.NDArray[np.float32]:
        """Rows for ``ids`` in that order; ``KeyError`` lists every missing id."""
        missing = [pid for pid in ids if pid not in self._index]
        if missing:
            raise KeyError(f"{len(missing)} id(s) not in embedding cache: {missing[:10]}")
        return self.matrix[[self._index[pid] for pid in ids]]

    def save(self, directory: Path, name: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / f"{name}.npy", self.matrix)
        (directory / f"{name}.ids.json").write_text(json.dumps(self.ids), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path, name: str) -> EmbeddingCache:
        ids = json.loads((directory / f"{name}.ids.json").read_text(encoding="utf-8"))
        return cls(list(ids), np.load(directory / f"{name}.npy"))
