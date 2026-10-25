"""Row 7.6 follow-up: peak VRAM for bge-small encoding and a TextCNN training step.

Run on a CUDA host with the repo's ``src`` on PYTHONPATH:
    python docs/checkpoints/C7/bench_small.py --out small.json
Peak memory is the torch allocator's ``max_memory_allocated`` / ``max_memory_reserved``.
"""

from __future__ import annotations

import argparse
import json
import time

import torch
from transformers import AutoModel, AutoTokenizer

from terser.models.neural.textcnn import TextCNNConfig, build_textcnn

BGE = "BAAI/bge-small-en-v1.5"


def _reset() -> None:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def _peaks() -> dict[str, float]:
    return {
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3),
    }


def bench_bge(dtype: torch.dtype, batch: int, seq: int, steps: int) -> dict:
    _reset()
    tok = AutoTokenizer.from_pretrained(BGE)
    model = AutoModel.from_pretrained(BGE, dtype=dtype).cuda().eval()
    text = " ".join(["prompt"] * (seq * 2))
    enc = tok([text] * batch, truncation=True, max_length=seq, padding="max_length", return_tensors="pt")
    enc = {k: v.cuda() for k, v in enc.items()}
    times = []
    with torch.inference_mode():
        for _ in range(steps):
            torch.cuda.synchronize()
            t = time.perf_counter()
            model(**enc).last_hidden_state[:, 0]
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t)
    out = {"workload": f"bge-small encode {str(dtype).split('.')[-1]}", "batch": batch, "seq_len": seq}
    out.update(_peaks())
    out["texts_per_s"] = round(batch / (sum(times[1:]) / len(times[1:])), 1)
    del model
    return out


def bench_textcnn(batch: int, seq: int, steps: int) -> dict:
    _reset()
    model = build_textcnn(TextCNNConfig(pooling="concat")).cuda().train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(batch, seq, 384, device="cuda")
    mask = torch.ones(batch, seq, dtype=torch.bool, device="cuda")
    times = []
    for _ in range(steps):
        torch.cuda.synchronize()
        t = time.perf_counter()
        out = model(x, mask)
        loss = sum(o.float().mean() for o in (out if isinstance(out, (tuple, list)) else [out]))
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
    out = {"workload": "TextCNN train step fp32 (concat pooling)", "batch": batch, "seq_len": seq}
    out.update(_peaks())
    out["mean_step_seconds"] = round(sum(times[1:]) / len(times[1:]), 4)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=11)
    a = p.parse_args()
    torch.manual_seed(0)
    results = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "seed": 0,
        "runs": [
            bench_bge(torch.float32, 256, 512, a.steps),
            bench_bge(torch.float16, 256, 512, a.steps),
            bench_textcnn(64, 512, a.steps),
        ],
    }
    with open(a.out, "w") as f:
        json.dump(results, f, indent=1)
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
