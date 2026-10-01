"""Pair embedding input for the MLP quality gate (plan row **4.1.2**).

For an original prompt with embedding ``e_o`` and a rewrite with ``e_r`` (384-d bge-small), the
pair feature is ``[e_o, e_r, |e_o - e_r|, e_o * e_r]`` (1536-d). The absolute difference and the
element-wise product make the representation symmetric-aware and expose per-dimension agreement,
the standard sentence-pair recipe (InferSent / SBERT).

:func:`build_cache` embeds every pair through an injected ``embed`` callable (so tests need no
model download), writes ``data/features/pair_emb.npy`` and an id index
``data/features/pair_emb_ids.json`` whose ``i``-th entry is the pair id of row ``i``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt

__all__ = ["EMB_DIM", "PAIR_DIM", "build_cache", "load_cache", "pair_features"]

FloatArray = npt.NDArray[np.float64]
EMB_DIM = 384
PAIR_DIM = 4 * EMB_DIM
Embed = Callable[[Sequence[str]], npt.ArrayLike]


def pair_features(e_o: npt.ArrayLike, e_r: npt.ArrayLike) -> FloatArray:
    """Row-wise ``[e_o, e_r, |e_o - e_r|, e_o * e_r]``; inputs shaped ``(n, d)`` or ``(d,)``."""
    a = np.asarray(e_o, dtype=np.float64)
    b = np.asarray(e_r, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"embedding shapes differ: {a.shape} vs {b.shape}")
    return np.concatenate([a, b, np.abs(a - b), a * b], axis=-1)


def build_cache(
    ids: Sequence[str],
    originals: Sequence[str],
    rewrites: Sequence[str],
    embed: Embed,
    out_dir: Path,
    batch_size: int = 256,
) -> Path:
    """Embed all pairs and write ``pair_emb.npy`` + ``pair_emb_ids.json``; returns the npy path."""
    if not len(ids) == len(originals) == len(rewrites):
        raise ValueError("ids, originals and rewrites must have equal length")
    if len(set(ids)) != len(ids):
        raise ValueError("pair ids must be unique")
    chunks = []
    for i in range(0, len(ids), batch_size):
        e_o = np.asarray(embed(originals[i : i + batch_size]), dtype=np.float64)
        e_r = np.asarray(embed(rewrites[i : i + batch_size]), dtype=np.float64)
        chunks.append(pair_features(e_o, e_r))
    mat = np.concatenate(chunks) if chunks else np.empty((0, PAIR_DIM))
    out_dir.mkdir(parents=True, exist_ok=True)
    npy = out_dir / "pair_emb.npy"
    np.save(npy, mat.astype(np.float32))
    (out_dir / "pair_emb_ids.json").write_text(json.dumps(list(ids)) + "\n", encoding="utf-8")
    return npy


def load_cache(out_dir: Path) -> tuple[npt.NDArray[np.float32], list[str]]:
    """Read the matrix and its id index back; checks they align."""
    mat = np.load(out_dir / "pair_emb.npy")
    ids = json.loads((out_dir / "pair_emb_ids.json").read_text(encoding="utf-8"))
    if len(ids) != len(mat):
        raise ValueError(f"index has {len(ids)} ids but matrix has {len(mat)} rows")
    return mat, [str(i) for i in ids]
