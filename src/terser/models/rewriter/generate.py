"""Greedy rewrites from a trained SFT rewriter (val report input for rows **8.3.3** / **8.3.4**).

Input format matches :mod:`terser.models.rewriter.train`: prompt tokens, then the separator
token, then the model decodes the rewrite up to EOS. Every rewrite goes through the 8.3.7
:func:`~terser.models.rewriter.safeguards.safeguard`, so the output is never longer than the
prompt on the base tokenizer.

Entry point::

    python -m terser.models.rewriter.generate --base Qwen/Qwen2.5-0.5B-Instruct \
        --model models/rewriter/sft-qwen0.5b-full \
        --out data/interim/rewrites/sft-qwen0.5b-full.val.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from terser.models.rewriter.safeguards import safeguard

__all__ = ["main"]

SEP_TEXT = "\n### Rewrite:\n"


def _load(
    base: str, model_dir: str, quant: str | None = None
) -> tuple[Any, Any]:  # pragma: no cover - GPU only
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(base)
    model: Any
    if quant == "nf4":  # QLoRA rung: the adapter runs on the same NF4 base it was trained on
        from peft import PeftModel
        from transformers import BitsAndBytesConfig

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base, quantization_config=bnb, torch_dtype=torch.bfloat16, device_map={"": 0}
        )
        return PeftModel.from_pretrained(model, model_dir).eval(), tok
    if (Path(model_dir) / "adapter_config.json").exists():
        from peft import PeftModel

        model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(model, model_dir).merge_and_unload()
    else:
        model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16)
    return model.to("cuda").eval(), tok


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - GPU only
    ap = argparse.ArgumentParser(description="Generate SFT rewriter outputs for a split.")
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/processed/sft/val.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--quant", choices=["nf4"], default=None, help="4-bit base (QLoRA rung)")
    args = ap.parse_args(argv)

    import torch

    model, tok = _load(args.base, args.model, args.quant)
    sep = tok(SEP_TEXT, add_special_tokens=False)["input_ids"][-1]
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    rows = [json.loads(x) for x in Path(args.data).read_text(encoding="utf-8").splitlines() if x]

    def count(text: str) -> int:
        return len(tok(text, add_special_tokens=False)["input_ids"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for i in range(0, len(rows), args.batch_size):
            chunk = rows[i : i + args.batch_size]
            seqs = [tok(r["prompt"], add_special_tokens=False)["input_ids"] + [sep] for r in chunk]
            width = max(map(len, seqs))
            ids = torch.tensor([[pad] * (width - len(s)) + s for s in seqs], device="cuda")
            mask = torch.tensor(
                [[0] * (width - len(s)) + [1] * len(s) for s in seqs], device="cuda"
            )
            t0 = time.perf_counter()
            with torch.no_grad():
                gen = model.generate(
                    input_ids=ids,
                    attention_mask=mask,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    eos_token_id=tok.eos_token_id,
                    pad_token_id=pad,
                )
            dt = (time.perf_counter() - t0) / len(chunk)
            for r, g in zip(chunk, gen[:, width:], strict=True):
                raw = tok.decode(g, skip_special_tokens=True).strip()
                text, fell_back = safeguard(r["prompt"], raw, count)
                n0, n1 = count(r["prompt"]), count(text)
                fh.write(
                    json.dumps(
                        {
                            "id": r["id"],
                            "category": r["category"],
                            "prompt": r["prompt"],
                            "raw_rewrite": raw,
                            "rewrite": text,
                            "fell_back": fell_back,
                            "savings": 1 - n1 / n0,
                            "oracle_savings": r["savings"],
                            "latency_s": dt,
                            "model": args.model,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    print(json.dumps({"n": len(rows), "out": str(out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
