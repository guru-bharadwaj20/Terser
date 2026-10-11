"""Prompt embeddings for clustering (plan row **6.1**).

Encodes every train prompt with bge-small (384-d, normalised) and saves an
:class:`~terser.features.embeddings.EmbeddingCache` (matrix + id index).

Entry point::

    python -m terser.features.embed_prompts --device cuda --out data/processed/embeddings
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from terser.features.embeddings import EmbeddingCache, bge_small_encoder

__all__ = ["main"]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the row 6.1 prompt embedding cache.")
    ap.add_argument("--split", default="data/processed/splits/train.jsonl")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--out", default="data/processed/embeddings")
    ap.add_argument("--name", default="train_bge_small")
    args = ap.parse_args(argv)

    rows = [json.loads(x) for x in Path(args.split).read_text(encoding="utf-8").splitlines() if x]
    cache = EmbeddingCache.build(
        [r["id"] for r in rows],
        [r["prompt"] for r in rows],
        bge_small_encoder(device=args.device, batch_size=args.batch_size),
    )
    cache.save(Path(args.out), args.name)
    print(json.dumps({"n": len(rows), "out": f"{args.out}/{args.name}"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
