"""Phase 4 / 5 inputs from the row 3.1.3 pair table (rows **4.1.2** and the table contract).

* ``data/processed/p4_pairs.parquet`` - :mod:`terser.phase4.tables` contract: ``id``, ``group``
  (source prompt), ``category``, ``label`` (1 = degraded) and every pair / prompt feature.
* ``data/features/pair_emb.npy`` + ``pair_emb_ids.json`` - ``[e_o, e_r, |e_o - e_r|, e_o * e_r]``
  bge-small pair embeddings (1536-d), rows aligned with ``p4_pairs.parquet``.

CLI::

    python -m terser.phase4.build_inputs
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from terser.features.embeddings import bge_small_encoder
from terser.features.pair_embedding import build_cache

__all__ = ["main"]

DROP = ("prompt_id", "savings_pct", "overhead_pct", "stand_in")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the Phase 4 / 5 input tables.")
    ap.add_argument("--pairs", type=Path, default=Path("data/processed/pair_features.parquet"))
    ap.add_argument("--prompts", type=Path, default=Path("data/interim/label_prompts.jsonl"))
    ap.add_argument("--candidates", type=Path, default=Path("data/interim/candidates.jsonl"))
    ap.add_argument("--labels", type=Path, default=Path("data/processed/labels.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/processed/p4_pairs.parquet"))
    ap.add_argument("--emb-dir", type=Path, default=Path("data/features"))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    pf = pd.read_parquet(args.pairs)
    df = pf.rename(columns={"prompt_id": "group"}).assign(
        label=(pf.label == "degraded").astype(int)
    )
    df = df.drop(columns=[c for c in DROP if c in df.columns])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)

    def rows(path: Path) -> list[dict[str, object]]:
        return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]

    prompt = {p["id"]: p["prompt"] for p in rows(args.prompts)}
    text = {c["candidate_id"]: c["text"] for c in rows(args.candidates)}
    lab = {r["id"]: r for r in rows(args.labels)}
    ids = pf["id"].astype(str).tolist()
    originals = [str(prompt[lab[i]["prompt_id"]]) for i in ids]
    rewrites = [str(text[lab[i]["candidate_id"]]) for i in ids]
    build_cache(ids, originals, rewrites, bge_small_encoder(device=args.device), args.emb_dir)
    print(json.dumps({"rows": len(df), "features": df.shape[1] - 4, "pair_emb": len(ids)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
