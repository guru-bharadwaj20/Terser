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

## C7.4 follow-up: worker-1 RTX 4090, 2026-10-03

**FAIL for the queued configuration; no checkpoint status change.** Driver 595.84,
Python 3.11.17, vLLM 0.30.0, Torch 2.13.0+cu130, CUDA runtime 13.0; compiler
NVCC/CRT/NVVM pinned to 13.0.88. Device total: 24,564 MiB; idle desktop baseline: 43 MiB.
The previously unverified `gpu_queue/p2-candidates-qwen32b.yaml` was tested with
`awq_marlin`, length 4096, utilization 0.90, batched tokens 1024, sequences 64 and
KV allocation 1,073,741,824 bytes.

That exact KV allocation supports only 4,080 tokens on this build, so the server
refused startup. Startup-only sampled maximum: **22,969 MiB device-wide = 22.430664 GiB;
22.388672 GiB net**. This provides no serving throughput for the original configuration.
Tracking run `fae1068486234367b9ac69216905d711`; [failure measurements](../../../data/raw/p2_smoke_20261003/startup_failures.json).

A smoke-only increase of one 4 MiB block, to 1,077,936,128 bytes, advertised 4,112
cache tokens and served the 20-prompt Phase 2 smoke. With all other limits above,
the working judge's sampled peak remained **22,969 MiB device / 22.388672 GiB net**,
exceeding the queue's numeric 22 GiB ceiling by **0.388672 GiB**. Consequently the
21.8 GiB queue estimate is disproven here, and the original queue file remains unchanged.
Do not treat the earlier workstation overshoot as harmless bookkeeping for this queue.

Working judge: 494 calls, 22,698 generated tokens, 264.625 seconds of stage wall time,
85.77 aggregate generated tok/s excluding startup. Tracking run
`15c1a4b59b604a85bd80c54ce06fef56`, with `hardware=rtx4090`, `host=worker-1`, seed 42.
Final judge startup took 30.045 seconds with existing compiler caches. Memory sampling
requested 200 ms; peaks are device-wide sampled maxima, with the measured baseline
subtracted for net figures. They are not continuous absolute maxima.

Evidence: [VRAM and run summary](../../../data/raw/p2_smoke_20261003/vram_summary.json),
[CSV samples](../../../data/raw/p2_smoke_20261003/server_logs/judge32final_vram.csv),
[exact queued startup output](../../../data/raw/p2_smoke_20261003/server_logs/candidates32_server.log),
[smoke report](../../../reports/p2_gpu_smoke.md), and
[proposed full-scale plan](../../../reports/p2_full_scale_plan.md).
Source SHA `7bfce9f364d02ac0ff52eebf3e507b6fdfa5812d`, dirty tree; executed source
archive SHA256 `9fbe0e454cf6394a5933fda0215c6b09c9596e6ba500545e384de6c1f11e1476`.
Input/data DVC hashes are recorded in the smoke report. Worker-2 serving remains
unmeasured. Resolve the memory cap and judge position inconsistency before a full run.
All started servers were shut down; no full-scale run began.
