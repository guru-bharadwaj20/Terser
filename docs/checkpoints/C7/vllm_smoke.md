# Row 7.5: vLLM smoke test

```bash
python -m terser.hardware.serving_bench --model Qwen/Qwen2.5-0.5B-Instruct --smoke \
  --output docs/checkpoints/C7/vllm_smoke.json
```

2026-10-01, RTX 4500 Ada workstation (WSL2), vLLM 0.30.0, torch 2.13.0+cu130. MLflow run
`7-vllmsmoke-qwen2.50.5b-bf16-0` (experiment `phase7`, local `./mlruns`). Raw result:
[vllm_smoke.json](vllm_smoke.json).

| Field | Value |
| :--- | :--- |
| Model | Qwen/Qwen2.5-0.5B-Instruct, bf16, `gpu_memory_utilization=0.90`, max len 4096 |
| Generated tokens | **32** (greedy, `ignore_eos`) in 0.49 s |
| Load time | 31.7 s (weights 0.93 GiB, KV cache 19.84 GiB) |
| Output | `Hello! It's a pleasure to meet you.\nHuman: What is the difference between a function and a method in Python?\nuser` |

The text runs past the assistant turn because `ignore_eos` forces all 32 tokens, so the chat
template (`Human:` … `user`) leaks into it; this sample is not a usable output. `serving_bench`
now records `sample_text` from a second generation that stops at EOS; the JSON above predates
that change.

Environment note: vLLM's FlashInfer sampler JIT-compiles with `nvcc`, which this WSL2 host does not
have (no CUDA toolkit, only the driver), so engine start-up failed until
`VLLM_USE_FLASHINFER_SAMPLER=0` was set. `serving_bench` sets that default itself; a bare
`vllm serve` on this host needs the variable exported.
