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
| C7.2 | Memory plan | VRAM table replaced with measured values | The rows the plan asks Phase 7 to measure (7.6, 7.7) are replaced: the 32B AWQ judge, the Qwen2.5-7B target, 0.5B full FT and 1.5B LoRA. **Still estimates:** Llama-3.1-8B bf16 target (gated, no HF token), Llama-3.1-8B AWQ small judge, Qwen2.5-14B-AWQ fallback judge, 3B LoRA, 7B QLoRA, GRPO, embeddings, TextCNN, production serving | ❌ partial |
| C7.4 | Queue | Dry-run orders all jobs without dependency errors; every job's `est_vram_gb` ≤ 22 GB measured headroom | `python -m terser.gpu_queue dry-run` → "queue empty": `gpu_queue/` holds no job specs yet, so nothing is ordered or checked against the measured headroom. A vacuous pass is not counted | ❌ |

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
  needs `max_num_batched_tokens=2048`.

## Decision

Not taken. All Phase 7 task rows are ✅, but C7.2 and C7.4 are open, and a reviewer who did not do
the work has not re-run a headline number (checkpoint protocol step 3).
