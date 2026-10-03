# Checkpoint C7 — GPU Ready

| Field | Value |
| :--- | :--- |
| Date | 2026-10-01 |
| Git SHA at measurement | `8c58189` (rows 7.1–7.9: `27ba561` … `8c58189`) |
| Machine | RTX 4500 Ada workstation, WSL2 ([hardware.md § 1.2](../../hardware.md#12-rtx-4500-ada-workstation-wsl2)). The RTX 4090 workers were not used |
| Data | None. Phase 7 uses dummy tokens and fixed prompts only. DVC was not pulled (remote unreachable, see row 7.3) |
| Seeds | 0 (vLLM engine seed, torch manual seed) |
| MLflow | Experiment `phase7` in local `./mlruns` (SQLite), runs `7-vllmsmoke-*`, `7-servingbench-*`, `7-trainbench-*` |

## Checks

| # | Check | Target | Measured | Result |
| :---: | :--- | :--- | :--- | :---: |
| C7.1 🚧 | Environment | Verification script all green on ≥ 1 worker | `python -m terser.hardware.verify`: capability 8.9, 23.99 GiB, bf16 114.9 TFLOP/s, bf16-vs-fp32 max relative error 4.83e-3. **All green** ([hardware.md](hardware.md)). Measured on the RTX 4500 Ada, which ADR-0004 accepts as any 24 GB Ada-class card | ✅ |
| C7.2 | Memory plan | VRAM table replaced with measured values | The rows the plan asks Phase 7 to measure (7.6, 7.7) are replaced: the 32B AWQ judge, the Qwen2.5-7B target, 0.5B full FT and 1.5B LoRA. **Still estimates:** the revised 32B queue configuration (21.8, UNVERIFIED; the measured 22.15 GiB peak applies only to the original configuration), Llama-3.1-8B bf16 target (gated, no HF token), Llama-3.1-8B AWQ small judge, Qwen2.5-14B-AWQ fallback judge, 3B LoRA, 7B QLoRA, GRPO, embeddings, TextCNN, production serving | ❌ partial |
| C7.4 | Queue | Dry-run orders all jobs without dependency errors; every declared `est_vram_gb` ≤ the configured 22 GB ceiling, with hardware evidence before a full pass | Local CPU-only validation on 2026-10-03: `validate --dir gpu_queue` → `queue OK`; `dry-run --dir gpu_queue` orders all three jobs without dependency errors. Declared estimates 21.8, 17 and 13 all pass `validate()` against 22. **The revised 32B configuration is an UNVERIFIED estimate pending RTX 4090 hardware re-measurement; C7.4 is not a full, confident pass** | ❌ partial |

## Queue verification (2026-10-03, CPU only)

The project was installed with `pip install -q -e .` in an activated Python venv. Both commands
below exited 0. Their real output is reproduced verbatim; no GPU workload was run. Dry-run
orders checkpoint-gated jobs but does not prove their gates have passed or execute their commands.

```text
$ python -m terser.gpu_queue validate --dir gpu_queue
queue OK
$ python -m terser.gpu_queue dry-run --dir gpu_queue
#  id                     phase  prio  VRAM GB  hours  exclusive  gates
-----------------------------------------------------------------------
1  p2-candidates-qwen32b  2      1     21.8     3      yes        -
2  sft-qwen0.5b-full      8      2     17       6      yes        C7.1, C2.3
3  sft-qwen1.5b-lora      8      2     13       8      yes        C7.1, C2.3

Total estimated GPU-hours: 17
```

### Revised 32B configuration: derived ESTIMATE, UNVERIFIED

The only measured baseline is [serving.md](serving.md): 18.14 GiB weights, 1.46 GiB KV,
22.15 GiB net peak (23.11 GiB device peak), at utilization 0.90, max length 4096,
`max_num_batched_tokens=2048` and `max_num_seqs=64`. It exceeds the configured ceiling;
an exclusive slot does not resolve that breach. The old queue declaration of 21 was incorrect,
and its command omitted the measured batching limits.

All following budget numbers are **derivations/estimates, not measurements**. The queue compares
plain numeric values against `max_vram_gb: 22`; this calculation follows the existing checkpoint's
GiB convention despite the field/CLI's GB label. It does not convert GiB to decimal GB or change
the ceiling. Hardware verification must confirm actual usable headroom and the unit convention.

- Fixed weights leave `22 - 18.14 = 3.86 GiB`; this is not all usable for KV.
- Derived residual overhead: `22.15 - 18.14 - 1.46 = 2.55 GiB`.
- Assuming that residual persists, KV budget under the ceiling is
  `22 - 18.14 - 2.55 = 1.31 GiB`.
- Choose an explicit KV cap of `1.00 GiB = 1 * 2^30 = 1073741824 bytes`, below that budget.
  The recorded baseline says one 4096-token sequence needs 1.0 GiB; startup and exact block
  capacity with this cap remain unverified. The
  [vLLM cache documentation](https://docs.vllm.ai/en/v0.18.2/api/vllm/config/cache/#vllm.config.cache.CacheConfig.kv_cache_memory_bytes)
  specifies that an explicit KV cap overrides utilization-based KV allocation; retaining 0.90
  does not make it a peak cap.
- The revised `--max-num-batched-tokens 1024` is derived as `2048 / 2`, intended to reduce
  activation pressure; keep `--max-num-seqs 64` and `--max-model-len 4096`. No linear KV scaling
  or activation-memory saving is assumed or credited in this estimate.
- Estimated net peak: `18.14 + 1.00 + 2.55 = 21.69 GiB`. Round upward to `est_vram_gb: 21.8`,
  adding `21.8 - 21.69 = 0.11 GiB` allowance. Declared margin is `22 - 21.8 = 0.2 GiB`
  (unrounded estimated margin `22 - 21.69 = 0.31 GiB`). Residual stability and actual margin
  are unverified, especially on the RTX 4090 workers.

## Follow-up

**Follow-up: planned RTX 4090 verification task (worker-1 or worker-2).** Before this job is
trusted, confirm the installed vLLM accepts the revised flags, starts with sufficient KV blocks
for max length 4096, and completes representative serving/candidate-generation load. Record
baseline, device-wide and net peak VRAM in `docs/checkpoints/C7/`, verify the 22 ceiling and real
margin, then replace the estimate with the observed result and reassess C7.4. If startup fails
or the peak breaches the ceiling, revise and re-measure; do not mark C7.4 passed on these CPU checks.

## Findings

1. **The 32B AWQ judge only just fits.** At `gpu_memory_utilization=0.90` and max length 4096 it
   fails to start with vLLM defaults: 0.77 GiB of KV cache is available and 1.0 GiB is needed. With
   `max_num_batched_tokens=2048` it starts with 1.46 GiB of KV cache (5,968 tokens) and peaks at
   22.15 GiB net, just over the 22 GB single-job ceiling ([hardware.md § 1](../../hardware.md#1-resources)). See [serving.md](serving.md).
2. **Training has headroom.** 0.5B full FT peaks at 16.3 GiB reserved (fp32 master weights) or
   11.8 GiB (pure bf16); 1.5B LoRA peaks at 12.2 GiB. See [training.md](training.md).
3. **WSL2 specifics.** There is no CUDA toolkit or `nvcc`, so vLLM needs
   `VLLM_USE_FLASHINFER_SAMPLER=0`. The Windows desktop holds about 0.5–1 GB of VRAM before any job
   starts.

## Deviations

- Rows 7.1–7.3 name the RTX 4090 workers; they were satisfied on the RTX 4500 Ada (ADR-0004).
- 7.3: no `dvc pull` (remote path is on another machine), and MLflow is local, not a shared server.
- 7.6: Qwen2.5-7B-Instruct was measured in place of the gated Llama-3.1-8B-Instruct. The 32B judge
  was measured with `max_num_batched_tokens=2048`; the revised queued configuration is unverified.

## Decision

Not taken. All Phase 7 task rows are ✅, but C7.2 and C7.4 are open, and a reviewer who did not do
the work has not re-run a headline number (checkpoint protocol step 3).
