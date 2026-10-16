"""Build the Phase 3 feature tables from the Phase 2 judge labels (rows **3.1.2**, **3.1.3**).

* ``data/processed/pair_features.parquet`` - one row per labelled (original, candidate, target)
  train record: :data:`~terser.features.pair.PAIR_FEATURES`, ``target_phi`` (which target model
  answered), prompt-level ``prompt_*`` features, ``overhead_pct`` (quality-path rewrite cost,
  :func:`quality_overhead_pct`) and ``savings_pct`` (mean oracle savings of the
  prompt over targets, in percent) for the KNN savings regression.
* ``data/processed/span_features.parquet`` - token-level KEEP / DROP / REPLACE table for each
  train prompt against its Qwen2.5 oracle rewrite (``span_feature_table``).

Costs use the Qwen2.5 tokenizer, surprisal GPT-2 small, POS spaCy ``en_core_web_sm`` and the
embedding cosine bge-small. Train split only: the val and test splits are never read.

CLI::

    python -m terser.phase3.build_features
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Sequence
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from terser.features.alignment import tokenize
from terser.features.embeddings import bge_small_encoder
from terser.features.lexicons import is_constraint, is_filler, is_negation, is_number, is_stopword
from terser.features.pair import pair_feature_table
from terser.features.span import gpt2_surprisal, spacy_pos, span_feature_table
from terser.tokenizers.count import count_tokens

__all__ = ["has_code", "main", "prompt_features", "quality_overhead_pct"]


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


CODE_RE = re.compile(
    r"```|^\s{4}\S|\bdef \w+\s*\(|\bclass \w+\s*[:(]|\bfunction\s*\w*\s*\(|#include\b|=>"
    r"|\bimport \w+",
    re.M,
)


def has_code(text: str) -> bool:
    """Code fence, indented block, or a definition / import pattern (not just the word "def")."""
    return bool(CODE_RE.search(text))


def quality_overhead_pct(
    config: Path = Path("configs/serving/default.yaml"), target: str = "gpt-4o"
) -> float:
    """Rewrite overhead per prompt, in percent of the prompt's target cost (row 3.5.4).

    The quality-path rewriter reads the prompt and writes at most as many tokens, so per input
    token it costs ``rewriter_input + rewriter_output`` against the target's input price
    (prices from the serving config; 6% for ``gpt-4o`` at the configured prices).
    """
    import yaml

    pr = yaml.safe_load(config.read_text(encoding="utf-8"))["pricing"]
    rewrite = pr["rewriter_input_per_mtok"] + pr["rewriter_output_per_mtok"]
    return 100.0 * rewrite / pr["target_input_per_mtok"][target]


def prompt_features(text: str, n_tokens: int, bits: Sequence[float]) -> dict[str, float]:
    """``prompt_*`` features of the original prompt alone (no rewrite, so no savings leak)."""
    toks = tokenize(text)
    n = max(len(toks), 1)
    return {
        "prompt_tokens": float(n_tokens),
        "prompt_log_tokens": float(np.log1p(n_tokens)),
        "prompt_words": float(len(toks)),
        "prompt_stopword_ratio": sum(map(is_stopword, toks)) / n,
        "prompt_filler": float(sum(map(is_filler, toks))),
        "prompt_filler_ratio": sum(map(is_filler, toks)) / n,
        "prompt_negations": float(sum(map(is_negation, toks))),
        "prompt_numbers": float(sum(map(is_number, toks))),
        "prompt_constraints": float(sum(map(is_constraint, toks))),
        "prompt_mean_si": float(np.mean(bits)) if len(bits) else 0.0,
        "prompt_has_code": float(has_code(text)),
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the Phase 3 pair and span feature tables.")
    ap.add_argument("--prompts", type=Path, default=Path("data/interim/label_prompts.jsonl"))
    ap.add_argument("--candidates", type=Path, default=Path("data/interim/candidates.jsonl"))
    ap.add_argument("--labels", type=Path, default=Path("data/processed/labels.jsonl"))
    ap.add_argument("--oracle", type=Path, default=Path("data/processed/oracle.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--target", default="qwen2.5", help="tokenizer for costs")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    prompts = {p["id"]: p for p in _jsonl(args.prompts) if p.get("split") == "train"}
    cands = {c["candidate_id"]: c for c in _jsonl(args.candidates)}
    oracle = pd.read_csv(args.oracle)
    oracle = oracle[~oracle.stand_in & oracle.prompt_id.isin(prompts)]
    savings = (oracle.groupby("prompt_id").savings.mean() * 100).to_dict()

    surprisal = gpt2_surprisal("gpt2")

    @cache
    def cost(text: str) -> int:
        return count_tokens(text, args.target)

    @cache
    def bits(toks: tuple[str, ...]) -> list[float]:
        return surprisal(list(toks)) if toks else []

    def si(toks: Sequence[str]) -> list[float]:
        return bits(tuple(toks))

    pairs = []
    for r in _jsonl(args.labels):
        p = prompts.get(r["prompt_id"])
        c = cands.get(r["candidate_id"])
        if p is None or c is None or r.get("stand_in") or c.get("rejected"):
            continue
        pairs.append(
            {
                "id": r["id"],
                "prompt_id": r["prompt_id"],
                "category": p["category"],
                "original": p["prompt"],
                "rewrite": c["text"],
                "label": r["label"],
                "target": r["target"],
            }
        )
    # Embed every distinct text once, in one batched pass.
    texts = sorted({t for q in pairs for t in (q["original"], q["rewrite"])})
    vecs = dict(zip(texts, bge_small_encoder(device=args.device)(texts), strict=True))
    pf = pair_feature_table(pairs, cost, si, lambda ts: np.stack([vecs[t] for t in ts]))
    pf["target_phi"] = [float("phi" in q["target"].lower()) for q in pairs]
    pmeta = {
        pid: prompt_features(p["prompt"], cost(p["prompt"]), si(tokenize(p["prompt"])))
        for pid, p in prompts.items()
    }
    pdf = pd.DataFrame.from_dict(pmeta, orient="index")
    pf = pf.join(pdf, on="prompt_id")
    pf["savings_pct"] = pf.prompt_id.map(savings).astype(float)
    pf["overhead_pct"] = quality_overhead_pct()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pf.to_parquet(args.out_dir / "pair_features.parquet", index=False)

    best = oracle[oracle.target == args.target].set_index("prompt_id").oracle_candidate_id
    span_pairs = [
        {
            "prompt_id": pid,
            "category": p["category"],
            "original": p["prompt"],
            "rewrite": cands[best[pid]]["text"] if best.get(pid) in cands else p["prompt"],
        }
        for pid, p in prompts.items()
        if pid in best
    ]
    sf = span_feature_table(span_pairs, lambda t: count_tokens(t, args.target), si, spacy_pos())
    sf.to_parquet(args.out_dir / "span_features.parquet", index=False)
    print(
        json.dumps(
            {
                "pairs": len(pf),
                "pair_prompts": int(pf.prompt_id.nunique()),
                "degraded_rate": float((pf.label == "degraded").mean()),
                "span_rows": len(sf),
                "span_prompts": len(span_pairs),
                "span_labels": sf.label.value_counts().to_dict(),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
