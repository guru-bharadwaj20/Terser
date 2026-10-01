# Row 7.7: training measurement

RTX 4500 Ada workstation (WSL2), 2026-10-01, torch 2.13.0+cu130, transformers 5.18.0, peft 0.21.1,
through `python -m terser.hardware.train_bench`. All runs use sequence length 1024, micro-batch 4,
gradient checkpointing (non-reentrant), bf16 autocast, fused AdamW, and random token ids as dummy
data. Each run does 3 warm-up steps and then 10 timed steps (forward, backward and optimizer step).
Peak memory is from the torch allocator (`max_memory_allocated` / `max_memory_reserved`). MLflow
experiment `phase7`, runs `7-trainbench-*`.

| Model | Mode | Weights / optimizer state | Trainable params | Peak allocated | Peak reserved | Step time (mean) | Tokens/s |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-0.5B-Instruct | full FT | fp32 master + bf16 autocast | 494.0 M | 13.11 GiB | **16.32 GiB** | 0.437 s | 9,383 |
| Qwen2.5-0.5B-Instruct | full FT | pure bf16 | 494.0 M | 9.94 GiB | 11.77 GiB | 0.368 s | 11,129 |
| Qwen2.5-1.5B-Instruct | LoRA r=16, α=32, all 7 projections | bf16 base, bf16 adapters | 18.5 M | 10.41 GiB | **12.19 GiB** | 1.106 s | 3,705 |

Raw results: [0.5B full fp32](train_qwen2.5-0.5b_full_fp32.json),
[0.5B full bf16](train_qwen2.5-0.5b_full_bf16.json), [1.5B LoRA](train_qwen2.5-1.5b_lora_r16.json).

Notes:

- The plan's estimates (0.5B full FT ~12–16 GB, 1.5B LoRA ~8–16 GB) hold. Both fit well under the
  22 GB ceiling, which leaves room for micro-batch 8 or sequence length 2048 on the 0.5B model.
- A large share of each peak is the logits: 4 × 1024 × 151,936 vocabulary entries are about
  2.3 GiB in fp32 before cross-entropy. Chunked or fused cross-entropy would cut this.
- Step time is for random tokens. Real SFT batches with padding or packing will differ slightly.
  Sustained-throughput drift was not measured.
