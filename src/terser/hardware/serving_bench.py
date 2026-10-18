"""vLLM smoke test and serving measurement (rows 7.5 and 7.6).

``python -m terser.hardware.serving_bench --model Qwen/Qwen2.5-0.5B-Instruct --smoke`` generates 32
tokens and prints them (row 7.5). Without ``--smoke`` the model is loaded with
``gpu_memory_utilization`` and ``max_model_len`` as given and decode throughput is measured at each
batch size (row 7.6). Peak device VRAM is sampled through NVML for the whole process lifetime, so it
includes vLLM's engine-core subprocess. Results are printed as JSON, optionally written to
``--output`` and logged to MLflow.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["BatchResult", "PeakVramSampler", "ServingResult", "main", "run_name", "tokens_per_s"]

PROMPT = "Write a detailed, step-by-step explanation of how a hash map handles collisions."


@dataclass(frozen=True)
class BatchResult:
    """Decode throughput for one batch size."""

    batch_size: int
    output_tokens: int
    seconds: float

    @property
    def tokens_per_s(self) -> float:
        return tokens_per_s(self.output_tokens, self.seconds)


@dataclass
class ServingResult:
    """Everything one invocation measured."""

    model: str
    quantization: str | None
    dtype: str
    gpu_memory_utilization: float
    max_model_len: int
    baseline_vram_mib: float
    peak_vram_mib: float
    load_seconds: float
    engine_args: dict[str, int] = field(default_factory=dict)
    batches: list[BatchResult] = field(default_factory=list)
    sample_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["batches"] = [
            {**asdict(b), "tokens_per_s": round(b.tokens_per_s, 2)} for b in self.batches
        ]
        return out


def tokens_per_s(tokens: int, seconds: float) -> float:
    """Aggregate throughput; 0.0 for a non-positive duration."""
    return tokens / seconds if seconds > 0 else 0.0


def run_name(model: str, quantization: str | None, smoke: bool) -> str:
    """``7-<task>-<model>-<variant>-0`` following the repo's run-name pattern."""
    short = model.split("/")[-1].lower().replace("-instruct", "").replace("-", "")
    task = "vllmsmoke" if smoke else "servingbench"
    return f"7-{task}-{short}-{quantization or 'bf16'}-0"


class PeakVramSampler:
    """Poll used memory (MiB) via NVML in a background thread, summed over every visible GPU.

    vLLM with tensor parallelism spreads the model over several GPUs, so device 0 alone
    under-reports the peak. NVML is shut down on exit.
    """

    def __init__(self, index: int = -1, interval_s: float = 0.05) -> None:
        """``index=-1`` samples every GPU; a device index samples that one only."""
        import pynvml

        self._nvml = pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        indices = range(count) if index < 0 else [index]
        self._handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in indices]
        self._interval = interval_s
        self._stop = threading.Event()
        self.peak_mib = self.used_mib()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def used_mib(self) -> float:
        used = sum(self._nvml.nvmlDeviceGetMemoryInfo(h).used for h in self._handles)
        return float(used) / 2**20

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak_mib = max(self.peak_mib, self.used_mib())
            time.sleep(self._interval)

    def __enter__(self) -> PeakVramSampler:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()
        self.peak_mib = max(self.peak_mib, self.used_mib())
        self._nvml.nvmlShutdown()


def _measure(llm: Any, batch_size: int, max_tokens: int) -> BatchResult:
    from vllm import SamplingParams

    params = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
    messages = [[{"role": "user", "content": f"{PROMPT} (variant {i})"}] for i in range(batch_size)]
    start = time.perf_counter()
    outputs = llm.chat(messages, params, use_tqdm=False)
    seconds = time.perf_counter() - start
    tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    return BatchResult(batch_size=batch_size, output_tokens=tokens, seconds=seconds)


def run(args: argparse.Namespace) -> ServingResult:
    # FlashInfer's sampler JIT-compiles with nvcc, which the WSL2 GPU host does not have.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm import LLM, SamplingParams

    engine_args = {
        k: v
        for k, v in (
            ("max_num_batched_tokens", args.max_num_batched_tokens),
            ("max_num_seqs", args.max_num_seqs),
        )
        if v is not None
    }
    with PeakVramSampler() as sampler:
        baseline = sampler.used_mib()
        start = time.perf_counter()
        llm = LLM(
            model=args.model,
            dtype=args.dtype,
            quantization=args.quantization,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            seed=0,
            **engine_args,
        )
        load_seconds = time.perf_counter() - start
        result = ServingResult(
            model=args.model,
            quantization=args.quantization,
            dtype=args.dtype,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            baseline_vram_mib=round(baseline, 1),
            peak_vram_mib=0.0,
            load_seconds=round(load_seconds, 1),
            engine_args=engine_args,
        )
        if args.smoke:
            params = SamplingParams(temperature=0.0, max_tokens=32, ignore_eos=True)
            start = time.perf_counter()
            hello: list[Any] = [{"role": "user", "content": "Say hello in one sentence."}]
            out = llm.chat(hello, params)
            seconds = time.perf_counter() - start
            result.batches.append(BatchResult(1, len(out[0].outputs[0].token_ids), seconds))
            # ignore_eos runs past the assistant turn (the chat template leaks into the text), so
            # the recorded sample is a separate generation that stops at EOS.
            stop = SamplingParams(temperature=0.0, max_tokens=32)
            result.sample_text = llm.chat(hello, stop)[0].outputs[0].text
        else:
            _measure(llm, 1, 16)  # warm-up (CUDA graphs, caches)
            for bs in args.batch_sizes:
                result.batches.append(_measure(llm, bs, args.max_tokens))
    result.peak_vram_mib = round(sampler.peak_mib, 1)
    return result


def _log_mlflow(result: ServingResult, smoke: bool) -> None:
    import mlflow

    from terser.hardware.tags import hardware_tag
    from terser.tracking.runs import start_run

    with start_run(
        phase=7,
        row="7.5" if smoke else "7.6",
        task="vllmsmoke" if smoke else "servingbench",
        run_name=run_name(result.model, result.quantization, smoke),
        hardware=hardware_tag(),
        params={
            "model": result.model,
            "quantization": result.quantization or "none",
            "dtype": result.dtype,
            "gpu_memory_utilization": result.gpu_memory_utilization,
            "max_model_len": result.max_model_len,
            **result.engine_args,
        },
    ):
        mlflow.log_metric("peak_vram_mib", result.peak_vram_mib)
        mlflow.log_metric("baseline_vram_mib", result.baseline_vram_mib)
        mlflow.log_metric("load_seconds", result.load_seconds)
        tokens = 0
        for b in result.batches:
            tokens += b.output_tokens
            if b.seconds > 0:
                mlflow.log_metric(f"tokens_per_s_bs{b.batch_size}", b.tokens_per_s)
        mlflow.log_metric("generated_tokens", tokens)
        if result.sample_text:
            mlflow.log_text(result.sample_text, "sample.txt")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-batched-tokens", type=int, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 16])
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--smoke", action="store_true", help="row 7.5: generate 32 tokens")
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
        _log_mlflow(result, args.smoke)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
