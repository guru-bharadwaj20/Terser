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
| Qwen2.5-3B-Instruct | LoRA r=16, α=32, all 7 projections | bf16 base, bf16 adapters | 29.9 M | 13.75 GiB | **15.98 GiB** | 0.963 s | 4,253 |
| Qwen2.5-7B-Instruct | QLoRA r=16, α=32, all 7 projections | NF4 base, double quant, bf16 compute | 40.4 M | 17.25 GiB | **18.92 GiB** | 2.260 s | 1,813 |

The 3B LoRA and 7B QLoRA rows were measured on the RTX 4090 `worker-2` (driver 580.178.04, torch
2.13.0+cu130, peft 0.21.2, bitsandbytes 0.50.2) on 2026-10-25 with the same settings, through
`train_bench --mode lora` and the new `--mode qlora`; no other process used the card during either run.

Raw results: [3B LoRA](train_qwen2.5-3b_lora_r16.json), [7B QLoRA](train_qwen2.5-7b_qlora_nf4.json),
[0.5B full fp32](train_qwen2.5-0.5b_full_fp32.json),
[0.5B full bf16](train_qwen2.5-0.5b_full_bf16.json), [1.5B LoRA](train_qwen2.5-1.5b_lora_r16.json).

Notes:

- The plan's estimates (0.5B full FT ~12–16 GB, 1.5B LoRA ~8–16 GB) hold. Both fit well under the
  22 GB ceiling, which leaves room for micro-batch 8 or sequence length 2048 on the 0.5B model.
- A large share of each peak is the logits: 4 × 1024 × 151,936 vocabulary entries are about
  2.3 GiB in fp32 before cross-entropy. Chunked or fused cross-entropy would cut this.
- Step time is for random tokens. Real SFT batches with padding or packing will differ slightly.
  Sustained-throughput drift was not measured.
- 3B LoRA (15.98 GiB) sits at the top of the plan's ~8–16 GB range; 7B QLoRA (18.92 GiB) is above
  its ~10–18 GB range at seq 1024, mb 4. Both are under the 22 GB ceiling.
