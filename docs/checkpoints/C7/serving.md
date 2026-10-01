# Row 7.6: serving measurement

RTX 4500 Ada workstation (WSL2), 2026-10-01, vLLM 0.30.0 offline engine through
`python -m terser.hardware.serving_bench`. All runs: `gpu_memory_utilization=0.90`, max model length
4096, greedy decoding, 256 output tokens per request (`ignore_eos`), one warm-up request first.
Peak VRAM is the device-wide NVML maximum over the whole process; "net" subtracts the
Windows desktop's baseline usage measured before the engine started. MLflow experiment `phase7`
(local `./mlruns`), runs `7-servingbench-*`.

| Model | Precision | Engine overrides | Weights | KV cache | Peak VRAM (device) | Peak VRAM (net) | tok/s, batch 1 | tok/s, batch 16 (aggregate) | Load |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen/Qwen2.5-7B-Instruct ¹ | bf16 | none | 14.29 GiB | 5.54 GiB (103,760 tok) | 21.65 GiB | 20.79 GiB | 24.1 | 376.2 | 64 s |
| Qwen/Qwen2.5-32B-Instruct-AWQ | AWQ int4 (`awq_marlin`), fp16 act. | `max_num_batched_tokens=2048`, `max_num_seqs=64` | 18.14 GiB | 1.46 GiB (5,968 tok) | 23.11 GiB | 22.15 GiB | 18.5 | 281.1 | 75 s |
| Qwen/Qwen2.5-32B-Instruct-AWQ | AWQ int4, vLLM defaults | none (`max_num_batched_tokens=8192`) | 18.14 GiB | 0.77 GiB | — | — | **fails to start** | — | — |

Raw results: [7B](serving_qwen2.5-7b_bf16.json), [32B AWQ](serving_qwen2.5-32b_awq.json).

¹ **Substitution.** The plan names `meta-llama/Llama-3.1-8B-Instruct`. That repository is gated and
this machine has no Hugging Face token, so the download fails with `GatedRepoError`.
Qwen2.5-7B-Instruct, the plan's other target model, was measured in its place. Llama-3.1-8B bf16
(~15 GiB weights) remains **unmeasured** on this card.

Findings:

- **The 32B judge does not fit at vLLM defaults.** At `gpu_memory_utilization=0.90` and max length
  4096, the profiling pass for the default 8192-token prefill chunk leaves 0.77 GiB for KV cache.
  One 4096-token sequence needs 1.0 GiB, so the engine refuses to start. Capping the prefill chunk
  at 2048 tokens and the batch at 64 sequences frees enough activation memory for 1.46 GiB of KV
  cache. That covers 5,968 tokens, about 1.5 full-length sequences or about 16 short judge calls at
  once. Judge throughput is therefore KV-bound. Every `vllm serve` of the 32B judge on a 24 GB card
  needs `--max-num-batched-tokens 2048` (or lower).
- The 32B judge's net peak (22.15 GiB) is just over the 22 GB single-job ceiling
  (`max_vram_gb` in `configs/hardware/rtx4500ada.yaml`). The overshoot comes from
  `gpu_memory_utilization=0.90` of 23.99 GiB plus the parent process's CUDA context. The judge has
  the card to itself (exclusive slot), so this is a bookkeeping issue for C7.4 rather than a
  blocker. It is recorded here, not resolved.
- The 7B target leaves 5.5 GiB of KV cache (25 concurrent full-length sequences), which is ample
  for pilot labelling.
