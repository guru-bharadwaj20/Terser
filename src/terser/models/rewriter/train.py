"""Supervised fine-tuning of the generative rewriter (plan rows **8.3.2**-**8.3.4**).

Data: JSONL of ``{"prompt": casual, "rewrite": oracle}`` (``efficient`` is accepted as an alias
for ``rewrite``, which is the field name in ``data/samples/casual_prompts.jsonl``). Each example
is ``<prompt> SEP <rewrite> EOS``; the loss is next-token cross-entropy on the rewrite tokens only
(prompt positions are labelled ``-100``).

Models:

* ``model: tiny`` - :class:`TinyCausalLM`, a 2-layer causal transformer over bytes, built here.
  Used by the CPU smoke test (row 8.3.2); never reported.
* any Hugging Face repo id (``Qwen/Qwen2.5-0.5B-Instruct``, ...) - loaded with ``transformers``;
  full fine-tune or LoRA via ``peft`` (``lora: {r, alpha, dropout}``), bf16 and gradient
  checkpointing on the GPU rungs (8.3.3, 8.3.4, queued in ``gpu_queue/``). ``quant: nf4`` loads
  the base in 4-bit NF4 with double quantisation through ``bitsandbytes`` (QLoRA, the 7B rung).

Learning-rate schedule (row 10.7): ``schedule: constant`` (the default) keeps the rate fixed;
``schedule: linear`` warms up linearly over ``warmup_steps`` (or ``warmup_ratio`` x ``steps``)
and then decays linearly towards 0 at ``steps`` (see :func:`lr_factor`). With the default
(constant, no warmup) no scheduler is built, so existing configs train exactly as before.

Validation hook (row 10.7): with ``val_data`` and ``eval_every`` set, the masked loss on the
validation pairs (:func:`eval_loss`) is computed every ``eval_every`` steps and after the last
step, written to ``loss.json`` as ``val_loss`` and passed to ``run(cfg, on_eval=...)``'s
callback as ``on_eval(steps_done, loss)``. The HPO pruner uses this callback, and it may raise to
stop training.

Entry point: ``python -m terser.models.rewriter.train experiment=sft_tiny`` (or
``sft_qwen05b_full`` / ``sft_qwen15b_lora`` with ``hardware=rtx4090``).
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from omegaconf import DictConfig, OmegaConf

from terser.config import hydra_main

if TYPE_CHECKING:
    import torch
    from torch import nn

__all__ = [
    "SCHEDULES",
    "ByteTokenizer",
    "Pair",
    "build_tiny_lm",
    "encode_pair",
    "eval_loss",
    "load_pairs",
    "lr_factor",
    "run",
    "train_sft",
]

log = logging.getLogger(__name__)
IGNORE = -100
SCHEDULES: tuple[str, ...] = ("constant", "linear")
Examples = list[tuple[list[int], list[int]]]


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
    prompt_ids: Sequence[int],
    rewrite_ids: Sequence[int],
    sep: int | Sequence[int],
    eos: int,
    max_len: int,
) -> tuple[list[int], list[int]]:
    """``(input_ids, labels)``; labels are ``-100`` on the prompt and separator.

    ``sep`` is the separator's token ids (an int for a one-token separator). The prompt is
    truncated from the left so the full rewrite fits when possible, but keeps at least half of
    ``max_len`` (or all of itself if shorter): a rewrite that cannot fit beside it is truncated
    *without* EOS, so the model never learns to stop in the middle of a rewrite.
    """
    sep_ids = [sep] if isinstance(sep, int) else list(sep)
    prompt_all = list(prompt_ids)
    budget = max_len - len(sep_ids)
    keep_prompt = min(len(prompt_all), budget // 2)
    rewrite = list(rewrite_ids)
    if len(rewrite) + 1 <= budget - keep_prompt:
        target = rewrite + [eos]
    else:
        target = rewrite[: budget - keep_prompt]
    room = budget - len(target)
    prompt = prompt_all[-room:] if room > 0 else []
    ids = prompt + sep_ids + target
    labels = [IGNORE] * (len(prompt) + len(sep_ids)) + target
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
    examples: Examples, batch_size: int, pad: int, step: int
) -> tuple[list[list[int]], list[list[int]]]:
    start = (step * batch_size) % len(examples)
    chunk = [examples[(start + i) % len(examples)] for i in range(batch_size)]
    return _pad(chunk, pad)


def _pad(chunk: Examples, pad: int) -> tuple[list[list[int]], list[list[int]]]:
    width = max(len(ids) for ids, _ in chunk)
    ids = [x + [pad] * (width - len(x)) for x, _ in chunk]
    labels = [y + [IGNORE] * (width - len(y)) for _, y in chunk]
    return ids, labels


def _loss(
    model: nn.Module, ids_l: list[list[int]], labels_l: list[list[int]], pad_id: int,
    device: str, hf: bool, reduction: str = "mean",
) -> torch.Tensor:  # fmt: skip
    """Masked next-token cross-entropy of one padded batch."""
    import torch

    ids = torch.tensor(ids_l, device=device)
    labels = torch.tensor(labels_l, device=device)
    if hf:
        logits = model(input_ids=ids, attention_mask=(labels != IGNORE) | (ids != pad_id))
        logits = logits.logits
    else:
        logits = model(ids)
    # Shift: position t predicts token t+1.
    out: torch.Tensor = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
        labels[:, 1:].reshape(-1),
        ignore_index=IGNORE,
        reduction=reduction,
    )
    return out


def lr_factor(step: int, steps: int, warmup_steps: int = 0, schedule: str = "constant") -> float:
    """Multiplier of the base learning rate for optimiser step ``step`` (0-based).

    Warmup: ``(step + 1) / warmup_steps`` while ``step < warmup_steps``. After it ``constant``
    stays at 1 and ``linear`` decays as ``(steps - step) / (steps - warmup_steps)``: 1 on the
    first step after warmup, ``1 / (steps - warmup_steps)`` on the last.
    """
    if schedule not in SCHEDULES:
        raise ValueError(f"unknown schedule {schedule!r}; choose from {SCHEDULES}")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be >= 0")
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    if schedule == "constant":
        return 1.0
    return max(0.0, (steps - step) / max(1, steps - warmup_steps))


def eval_loss(
    model: nn.Module,
    examples: Examples,
    *,
    batch_size: int,
    pad_id: int,
    device: str = "cpu",
    hf: bool = False,
) -> float:
    """Token-weighted masked loss over ``examples`` (each once, in order), in eval mode."""
    import torch

    if not examples:
        raise ValueError("no eval examples")
    was_training = model.training
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(examples), batch_size):
            chunk = examples[i : i + batch_size]
            ids, labels = _pad(chunk, pad_id)
            total += float(_loss(model, ids, labels, pad_id, device, hf, "sum").item())
            n += sum(1 for _, y in chunk for t in y[1:] if t != IGNORE)
    if was_training:
        model.train()
    return total / max(n, 1)


def train_sft(
    model: nn.Module,
    examples: Examples,
    *,
    steps: int,
    lr: float,
    batch_size: int,
    pad_id: int,
    device: str = "cpu",
    hf: bool = False,
    seed: int = 0,
    schedule: str = "constant",
    warmup_steps: int = 0,
    eval_examples: Examples | None = None,
    eval_every: int = 0,
    on_eval: Callable[[int, float], None] | None = None,
) -> list[float]:
    """AdamW on the masked next-token loss; returns the per-step loss history.

    ``schedule`` / ``warmup_steps``: see :func:`lr_factor`; the default (constant, 0) builds no
    scheduler. With ``eval_examples`` and ``eval_every > 0``, :func:`eval_loss` runs after every
    ``eval_every`` steps and after the last one, and ``on_eval(steps_done, loss)`` receives it;
    the callback may raise (e.g. ``optuna.TrialPruned``) to stop training.
    """
    import torch

    if not examples:
        raise ValueError("no training examples")
    lr_factor(0, steps, warmup_steps, schedule)  # validates schedule and warmup
    torch.manual_seed(seed)
    if not getattr(model, "hf_device_map", None):  # a 4-bit base is placed at load time
        model.to(device)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr)
    sched = None
    if schedule != "constant" or warmup_steps > 0:
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: lr_factor(s, steps, warmup_steps, schedule)
        )
    evaluate = bool(eval_examples) and eval_every > 0
    history: list[float] = []
    for step in range(steps):
        ids_l, labels_l = _batches(examples, batch_size, pad_id, step)
        loss = _loss(model, ids_l, labels_l, pad_id, device, hf)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if sched is not None:
            sched.step()
        history.append(float(loss.item()))
        if not math.isfinite(history[-1]):
            raise FloatingPointError(f"non-finite loss at step {step}")
        done = step + 1
        if evaluate and eval_examples and (done % eval_every == 0 or done == steps):
            v = eval_loss(
                model, eval_examples, batch_size=batch_size, pad_id=pad_id, device=device, hf=hf
            )
            if on_eval is not None:
                on_eval(done, v)
    return history


def _load_hf(p: DictConfig, device: Any) -> tuple[Any, Any]:  # pragma: no cover - GPU rungs
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if device.precision == "bf16" else torch.float32
    tok = AutoTokenizer.from_pretrained(p.model)
    model: Any
    if p.get("quant") == "nf4":  # QLoRA (row 8.3.4, 7B rung): frozen NF4 base, bf16 adapters
        from peft import prepare_model_for_kbit_training
        from transformers import BitsAndBytesConfig

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        model = AutoModelForCausalLM.from_pretrained(
            p.model, quantization_config=bnb, torch_dtype=dtype, device_map={"": 0}
        )
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=p.get("grad_checkpointing", False)
        )
    else:
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


def _warmup_steps(p: DictConfig) -> int:
    if p.get("warmup_steps") is not None:
        return int(p.warmup_steps)
    return round(float(p.get("warmup_ratio") or 0.0) * int(p.steps))


def run(cfg: DictConfig, on_eval: Callable[[int, float], None] | None = None) -> dict[str, Any]:
    """Train per ``cfg.experiment.params``; writes ``loss.json`` (+ weights) to ``output_dir``.

    Optional params (row 10.7): ``schedule``, ``warmup_steps`` / ``warmup_ratio``, ``val_data``
    and ``eval_every``. ``on_eval(steps_done, val_loss)`` is called at every evaluation.
    """
    from terser.hardware import get_device
    from terser.utils.seed import set_seed

    set_seed(int(cfg.seed))
    p = cfg.experiment.params
    device = get_device(cfg)
    pairs = load_pairs(Path(p.data), p.get("limit"))
    if p.model == "tiny":
        btok = ByteTokenizer()
        model: Any = build_tiny_lm(btok.vocab_size, **OmegaConf.to_container(p.tiny))  # type: ignore[arg-type]
        sep: int | Sequence[int]
        enc, sep, eos, pad, hf = btok.encode, btok.sep_id, btok.eos_id, btok.pad_id, False
    else:  # pragma: no cover - needs the gpu extra and a GPU worker
        model, tok = _load_hf(p, device)

        def enc(text: str) -> list[int]:
            return list(tok(text, add_special_tokens=False)["input_ids"])

        # The whole separator, as the served SFT format sends it (a single trailing token is
        # common, so the prompt/rewrite boundary would be ambiguous).
        sep = enc("\n### Rewrite:\n")
        eos = int(tok.eos_token_id)
        pad = int(tok.pad_token_id) if tok.pad_token_id is not None else eos
        hf = True
    examples = [encode_pair(enc(x.prompt), enc(x.rewrite), sep, eos, p.max_len) for x in pairs]
    val_examples: Examples | None = None
    if p.get("val_data"):
        val_pairs = load_pairs(Path(p.val_data))
        val_examples = [
            encode_pair(enc(x.prompt), enc(x.rewrite), sep, eos, p.max_len) for x in val_pairs
        ]
    val_curve: list[list[float]] = []

    def record(step: int, loss: float) -> None:
        val_curve.append([step, loss])
        log.info("step %d val loss %.4f", step, loss)
        if on_eval is not None:
            on_eval(step, loss)

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
        schedule=str(p.get("schedule") or "constant"),
        warmup_steps=_warmup_steps(p),
        eval_examples=val_examples,
        eval_every=int(p.get("eval_every") or 0),
        on_eval=record,
    )
    out = Path(p.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "model": p.model,
        "steps": p.steps,
        "n_pairs": len(pairs),
        "loss": history,
    }
    if val_examples is not None:
        result["val_loss"] = val_curve
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
