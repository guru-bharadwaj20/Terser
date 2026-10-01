"""Training memory and step-time measurement (row 7.7).

``python -m terser.hardware.train_bench --model Qwen/Qwen2.5-0.5B-Instruct --mode full`` runs a few
optimizer steps on random token ids (sequence length 1024, micro-batch 4, gradient checkpointing)
and reports peak VRAM and mean step time. ``--mode lora`` trains LoRA adapters (r=16 on every
linear projection) over a frozen bf16 base. Full fine-tuning keeps fp32 master weights under bf16
autocast by default; ``--weights-dtype bf16`` measures pure-bf16 weights and AdamW state instead.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["TrainResult", "lora_target_modules", "main", "run_name"]

LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


@dataclass
class TrainResult:
    """Measured peak memory and step times for one configuration."""

    model: str
    mode: str
    weights_dtype: str
    seq_len: int
    micro_batch: int
    trainable_params: int
    total_params: int
    peak_allocated_gib: float
    peak_reserved_gib: float
    step_seconds: list[float] = field(default_factory=list)

    @property
    def mean_step_seconds(self) -> float:
        return statistics.fmean(self.step_seconds) if self.step_seconds else 0.0

    @property
    def tokens_per_s(self) -> float:
        mean = self.mean_step_seconds
        return self.seq_len * self.micro_batch / mean if mean > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["mean_step_seconds"] = round(self.mean_step_seconds, 4)
        out["tokens_per_s"] = round(self.tokens_per_s, 1)
        return out


def lora_target_modules() -> list[str]:
    """Projection names LoRA adapts in Qwen2-style decoder blocks."""
    return list(LORA_TARGETS)


def run_name(model: str, mode: str, weights_dtype: str) -> str:
    """``7-trainbench-<model>-<variant>-0`` following the repo's run-name pattern."""
    short = model.split("/")[-1].lower().replace("-instruct", "").replace("-", "")
    variant = "lora-r16" if mode == "lora" else f"full-{weights_dtype}"
    return f"7-trainbench-{short}-{variant}-0"


def run(args: argparse.Namespace) -> TrainResult:
    import torch
    from transformers import AutoModelForCausalLM

    torch.manual_seed(0)
    device = torch.device("cuda")
    weights_dtype = torch.float32 if args.weights_dtype == "fp32" else torch.bfloat16
    if args.mode == "lora":
        weights_dtype = torch.bfloat16
    model: Any = AutoModelForCausalLM.from_pretrained(args.model, dtype=weights_dtype)
    model = model.to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if args.mode == "lora":
        from peft import LoraConfig, get_peft_model

        model.enable_input_require_grads()
        config = LoraConfig(
            r=args.lora_r,
            lora_alpha=2 * args.lora_r,
            lora_dropout=0.0,
            target_modules=lora_target_modules(),
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-5, fused=True)
    vocab = int(model.config.vocab_size)

    torch.cuda.reset_peak_memory_stats()
    times: list[float] = []
    for step in range(args.warmup + args.steps):
        ids = torch.randint(0, vocab, (args.micro_batch, args.seq_len), device=device)
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        if step >= args.warmup:
            times.append(time.perf_counter() - start)

    return TrainResult(
        model=args.model,
        mode=args.mode,
        weights_dtype="bf16" if weights_dtype == torch.bfloat16 else "fp32",
        seq_len=args.seq_len,
        micro_batch=args.micro_batch,
        trainable_params=sum(p.numel() for p in params),
        total_params=sum(p.numel() for p in model.parameters()),
        peak_allocated_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2),
        peak_reserved_gib=round(torch.cuda.max_memory_reserved() / 2**30, 2),
        step_seconds=[round(t, 4) for t in times],
    )


def _log_mlflow(result: TrainResult) -> None:
    import mlflow

    from terser.tracking.runs import start_run

    with start_run(
        phase=7,
        row="7.7",
        task="trainbench",
        run_name=run_name(result.model, result.mode, result.weights_dtype),
        hardware="rtx4500ada",
        params={
            "model": result.model,
            "mode": result.mode,
            "weights_dtype": result.weights_dtype,
            "seq_len": result.seq_len,
            "micro_batch": result.micro_batch,
            "gradient_checkpointing": True,
        },
    ):
        mlflow.log_metric("peak_allocated_gib", result.peak_allocated_gib)
        mlflow.log_metric("peak_reserved_gib", result.peak_reserved_gib)
        mlflow.log_metric("mean_step_seconds", result.mean_step_seconds)
        mlflow.log_metric("tokens_per_s", result.tokens_per_s)
        mlflow.log_metric("trainable_params", result.trainable_params)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=("full", "lora"), required=True)
    parser.add_argument("--weights-dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args(argv)

    result = run(args)
    text = json.dumps(result.to_dict(), indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    if not args.no_mlflow:
        _log_mlflow(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
