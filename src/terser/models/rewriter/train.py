"""Supervised fine-tuning of the generative rewriter (plan rows **8.3.2**-**8.3.4**).

Data: JSONL of ``{"prompt": casual, "rewrite": oracle}`` (``efficient`` is accepted as an alias
for ``rewrite``, which is the field name in ``tests/fixtures/casual_prompts.jsonl``). Each example
is ``<prompt> SEP <rewrite> EOS``; the loss is next-token cross-entropy on the rewrite tokens only
(prompt positions are labelled ``-100``).

Models:

* ``model: tiny`` - :class:`TinyCausalLM`, a 2-layer causal transformer over bytes, built here.
  Used by the CPU smoke test (row 8.3.2); never reported.
* any Hugging Face repo id (``Qwen/Qwen2.5-0.5B-Instruct``, ...) - loaded with ``transformers``;
  full fine-tune or LoRA via ``peft`` (``lora: {r, alpha, dropout}``), bf16 and gradient
  checkpointing on the GPU rungs (8.3.3, 8.3.4, queued in ``gpu_queue/``).

Entry point: ``python -m terser.models.rewriter.train experiment=sft_tiny`` (or
``sft_qwen05b_full`` / ``sft_qwen15b_lora`` with ``hardware=rtx4090``).
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from omegaconf import DictConfig, OmegaConf

from terser.config import hydra_main

if TYPE_CHECKING:
    import torch
    from torch import nn

__all__ = [
    "ByteTokenizer",
    "Pair",
    "build_tiny_lm",
    "encode_pair",
    "load_pairs",
    "run",
    "train_sft",
]

log = logging.getLogger(__name__)
IGNORE = -100


@dataclass(frozen=True)
class Pair:
    prompt: str
    rewrite: str


def load_pairs(path: Path, limit: int | None = None) -> list[Pair]:
    """Read ``prompt`` / ``rewrite`` (or ``efficient``) pairs from JSONL."""
    pairs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rewrite = row.get("rewrite", row.get("efficient"))
        if rewrite is None:
            raise KeyError(f"row {row.get('id', '?')} has no 'rewrite' or 'efficient' field")
        pairs.append(Pair(row["prompt"], rewrite))
    return pairs[:limit] if limit else pairs


class ByteTokenizer:
    """UTF-8 bytes 0..255 plus SEP = 256, EOS = 257, PAD = 258."""

    sep_id, eos_id, pad_id = 256, 257, 258
    vocab_size = 259

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: Sequence[int]) -> str:
        return bytes(i for i in ids if i < 256).decode("utf-8", errors="replace")


def encode_pair(
    prompt_ids: Sequence[int], rewrite_ids: Sequence[int], sep: int, eos: int, max_len: int
) -> tuple[list[int], list[int]]:
    """``(input_ids, labels)``; labels are ``-100`` on the prompt and separator.

    The prompt is truncated from the left so the full rewrite always fits when possible.
    """
    target = list(rewrite_ids)[: max_len - 2] + [eos]
    room = max_len - len(target) - 1
    prompt = list(prompt_ids)[-room:] if room > 0 else []
    ids = prompt + [sep] + target
    labels = [IGNORE] * (len(prompt) + 1) + target
    return ids, labels


def build_tiny_lm(vocab_size: int, d_model: int = 64, n_layers: int = 2, n_heads: int = 4,
                  max_len: int = 512) -> nn.Module:  # fmt: skip
    """A small causal transformer LM: ``forward(input_ids) -> logits (B, L, V)``."""
    import torch
    from torch import nn

    class TinyCausalLM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.tok = nn.Embedding(vocab_size, d_model)
            self.pos = nn.Embedding(max_len, d_model)
            layer = nn.TransformerEncoderLayer(
                d_model, n_heads, 4 * d_model, dropout=0.0, batch_first=True, norm_first=True
            )
            self.blocks = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size)

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            n = input_ids.shape[1]
            pos = torch.arange(n, device=input_ids.device)
            h = self.tok(input_ids) + self.pos(pos)[None]
            causal = nn.Transformer.generate_square_subsequent_mask(n, device=input_ids.device)
            h = self.blocks(h, mask=causal, is_causal=True)
            out: torch.Tensor = self.head(self.norm(h))
            return out

    return TinyCausalLM()


def _batches(
    examples: list[tuple[list[int], list[int]]], batch_size: int, pad: int, step: int
) -> tuple[list[list[int]], list[list[int]]]:
    start = (step * batch_size) % len(examples)
    chunk = [examples[(start + i) % len(examples)] for i in range(batch_size)]
    width = max(len(ids) for ids, _ in chunk)
    ids = [x + [pad] * (width - len(x)) for x, _ in chunk]
    labels = [y + [IGNORE] * (width - len(y)) for _, y in chunk]
    return ids, labels


def train_sft(
    model: nn.Module,
    examples: list[tuple[list[int], list[int]]],
    *,
    steps: int,
    lr: float,
    batch_size: int,
    pad_id: int,
    device: str = "cpu",
    hf: bool = False,
    seed: int = 0,
) -> list[float]:
    """AdamW on the masked next-token loss; returns the per-step loss history."""
    import torch

    if not examples:
        raise ValueError("no training examples")
    torch.manual_seed(seed)
    model.to(device).train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    history: list[float] = []
    for step in range(steps):
        ids_l, labels_l = _batches(examples, batch_size, pad_id, step)
        ids = torch.tensor(ids_l, device=device)
        labels = torch.tensor(labels_l, device=device)
        if hf:
            logits = model(input_ids=ids, attention_mask=(labels != IGNORE) | (ids != pad_id))
            logits = logits.logits
        else:
            logits = model(ids)
        # Shift: position t predicts token t+1.
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
            labels[:, 1:].reshape(-1),
            ignore_index=IGNORE,
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        history.append(float(loss.item()))
        if not math.isfinite(history[-1]):
            raise FloatingPointError(f"non-finite loss at step {step}")
    return history


def _load_hf(p: DictConfig, device: Any) -> tuple[Any, Any]:  # pragma: no cover - GPU rungs
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if device.precision == "bf16" else torch.float32
    tok = AutoTokenizer.from_pretrained(p.model)
    model = AutoModelForCausalLM.from_pretrained(p.model, torch_dtype=dtype)
    if p.get("grad_checkpointing", False):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    if p.get("lora"):
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(
            model,
            LoraConfig(
                r=p.lora.r,
                lora_alpha=p.lora.alpha,
                lora_dropout=p.lora.get("dropout", 0.05),
                task_type="CAUSAL_LM",
                target_modules="all-linear",
            ),
        )
    return model, tok


def run(cfg: DictConfig) -> dict[str, Any]:
    """Train per ``cfg.experiment.params``; writes ``loss.json`` (+ weights) to ``output_dir``."""
    from terser.hardware import get_device
    from terser.utils.seed import set_seed

    set_seed(int(cfg.seed))
    p = cfg.experiment.params
    device = get_device(cfg)
    pairs = load_pairs(Path(p.data), p.get("limit"))
    if p.model == "tiny":
        btok = ByteTokenizer()
        model: Any = build_tiny_lm(btok.vocab_size, **OmegaConf.to_container(p.tiny))  # type: ignore[arg-type]
        enc, sep, eos, pad, hf = btok.encode, btok.sep_id, btok.eos_id, btok.pad_id, False
    else:  # pragma: no cover - needs the gpu extra and a GPU worker
        model, tok = _load_hf(p, device)

        def enc(text: str) -> list[int]:
            return list(tok(text, add_special_tokens=False)["input_ids"])

        sep = enc("\n### Rewrite:\n")[-1]
        eos = tok.eos_token_id
        pad = tok.pad_token_id if tok.pad_token_id is not None else eos
        hf = True
    examples = [encode_pair(enc(x.prompt), enc(x.rewrite), sep, eos, p.max_len) for x in pairs]
    history = train_sft(
        model,
        examples,
        steps=p.steps,
        lr=p.lr,
        batch_size=p.batch_size,
        pad_id=pad,
        device=str(device),
        hf=hf,
        seed=int(cfg.seed),
    )
    out = Path(p.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = {"model": p.model, "steps": p.steps, "n_pairs": len(pairs), "loss": history}
    (out / "loss.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if p.get("save_weights", False):  # pragma: no cover - GPU rungs
        if hf:
            model.save_pretrained(out)
        else:
            import torch

            torch.save(model.state_dict(), out / "model.pt")
    log.info(
        "trained %s for %d steps: loss %.3f -> %.3f", p.model, p.steps, history[0], history[-1]
    )
    return result


@hydra_main()
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
