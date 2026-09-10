# Hardware Profile and Compute Allocation

Plan row: **0.1.6**. Source of truth: [CONTRIBUTING.md § Compute Allocation](../CONTRIBUTING.md#compute-allocation-cpu-api-and-the-rtx-4500-ada).
If this document and CONTRIBUTING.md disagree, CONTRIBUTING.md wins and this file is corrected.

---

## 1. Resources

| Resource | Details | Used for | Status |
| :--- | :--- | :--- | :---: |
| Dev laptop (Windows 11) | CPU only, no CUDA device | Classical ML (Phases 3–6), tokenization, DVC pipelines, CI, CPU smoke tests, fast-path serving | Available |
| NVIDIA RTX 4500 Ada Generation | 24 GB GDDR6 ECC · Ada Lovelace · compute capability 8.9 · bf16 supported · ~210 W board power · attached later via remote access (row 7.1) | vLLM serving of open targets and local judge, rewriter SFT/RL, neural HPO, quality-path serving | ❌ not connected |
| Hosted LLM APIs (Claude, GPT) | Rate-limited, budget-capped per phase in `configs/budget.yaml`; responses disk-cached | Casualisation, candidate generation, judging, API-only targets, Claude token counting | Available (budget-capped) |

Consequences of the profile:

- bf16 is the default mixed-precision dtype on the GPU. fp16 loss scaling is not needed.
- ECC memory means VRAM available to CUDA is slightly below 24 GB. Row 7.10 uses **22 GB** as the
  usable ceiling for any single job.
- Until Phase 7 is ✅, nothing in the repository may *require* a GPU to pass CI.

---

## 2. Device selection in code

No module hard-codes `"cuda"`. Device and precision come from the Hydra `hardware` config group and are
resolved by `terser.hardware.get_device(cfg)` (row 0.1.5).

| Config | Intended machine | Behaviour |
| :--- | :--- | :--- |
| `configs/hardware/cpu.yaml` | Dev laptop, CI | Forces CPU, fp32 |
| `configs/hardware/rtx4500ada.yaml` | GPU machine | Requires a CUDA device; bf16 enabled. On a machine without CUDA, `get_device` raises with a clear message instead of silently falling back |
| `configs/hardware/auto.yaml` | Either | Uses CUDA if available, otherwise CPU |

```powershell
python -m terser.x experiment=<name> hardware=cpu          # laptop
python -m terser.x experiment=<name> hardware=rtx4500ada   # GPU machine
python -m terser.x experiment=<name> hardware=auto         # portable default
```

The `hardware` value is logged as an MLflow tag (`cpu` or `rtx4500ada`) on every run.

---

## 3. Compute tags

Every task row in CONTRIBUTING.md carries one or more tags.

| Tag | Meaning | What a contributor does now |
| :--- | :--- | :--- |
| `CPU` | Runs on the dev laptop | Start immediately |
| `API` | Needs a hosted LLM API | Start immediately; stay within the phase cap in `configs/budget.yaml` |
| `GPU` | Needs the RTX 4500 Ada | Write the code, smoke-test on CPU with `configs/model/tiny.yaml`, add a `gpu_queue/` job. The row stays ❌ until the job has run on the card |
| `CPU→GPU` | Reduced scale on CPU now, full scale once the GPU is connected | Run the CPU version, record it as reduced-scale, and queue the full-scale job |

---

## 4. GPU job queue

GPU work is never a blocker. Until the card is connected, GPU jobs are written as YAML specs in
`gpu_queue/` and managed by `python -m terser.gpu_queue {validate,dry-run,run}` (row 0.2.10).

### 4.1 Job spec format

```yaml
# gpu_queue/sft-qwen0.5b-full.yaml
id: sft-qwen0.5b-full
phase: 8
depends_on: [C7.1, C2.3]
command: python -m terser.models.rewriter.train experiment=sft_qwen05b_full hardware=rtx4500ada
est_vram_gb: 14
est_hours: 6
exclusive: true            # nothing else on the GPU while this runs
priority: 2                # 1 = highest
cpu_smoke_test: tests/rewriter/test_train_tiny.py
outputs: [models/rewriter/sft-qwen0.5b-full/, mlruns/]
```

| Field | Type | Required | Meaning |
| :--- | :--- | :---: | :--- |
| `id` | string | yes | Unique job id; equals the file name without `.yaml` (naming in [conventions.md](conventions.md)) |
| `phase` | int | yes | Plan phase the job belongs to |
| `depends_on` | list[string] | yes (may be empty) | Checkpoint ids (e.g. `C7.1`) or other job ids that must be complete first |
| `command` | string | yes | Exact command; must pass `hardware=rtx4500ada` |
| `est_vram_gb` | number | yes | Peak VRAM estimate; replaced by the measured value after Phase 7 |
| `est_hours` | number | yes | Wall-clock estimate |
| `exclusive` | bool | yes | `true` ⇒ no other process may use the card while the job runs |
| `priority` | int ≥ 1 | yes | 1 = highest |
| `cpu_smoke_test` | path | yes | Pytest file that runs the same code path on CPU with a tiny config |
| `outputs` | list[path] | yes | Directories or files the job produces |

### 4.2 Queue rules

1. A job is queued only if its `cpu_smoke_test` passes in CI.
2. `exclusive` jobs never share the card.
3. **vLLM serving and training never run at the same time on the 24 GB card.** vLLM pre-allocates
   `gpu_memory_utilization` (0.90) of VRAM, which leaves no room for a training process. When a
   workload needs both (e.g. GRPO in 9.3 needs target outputs), target calls go to a hosted API or to a
   vLLM server in a separate, time-shared slot.
4. The queue drains by `priority` first, then by `depends_on` order. A job whose dependencies are not
   complete is skipped, not failed.
5. After Phase 7, every job must satisfy `est_vram_gb ≤ 22` using measured values (row 7.10).
6. Each job logs GPU-hours to MLflow (see Experiment standards in CONTRIBUTING.md).

---

## 5. VRAM budget

> **Planning estimates only.** Every value below is derived from parameter counts and typical
> overheads, not measured on this card. Phase 7 rows **7.6** (serving) and **7.7** (training)
> replace them with measurements, and checkpoint **C7.2** requires this table to be updated.

| Workload | Phase | Compute | Precision / method | Est. VRAM | Notes |
| :--- | :---: | :--- | :--- | :--- | :--- |
| Token counting (tiktoken, HF tokenizers) | 0+ | `CPU` | — | 0 | Ground truth, never approximated |
| Claude token counts | 0+ | `API` | token-counting endpoint | 0 | Disk-cached |
| Candidate rewrite generation, judging | 2+ | `API` | — | 0 | Strong LLM, position-swapped judge |
| Open target LLM outputs (Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct) | 2, 8+ | `CPU→GPU` | vLLM bf16, `gpu_memory_utilization=0.90` | ~16 GB weights + KV cache | Before GPU: hosted API for the same models, or a labelled 1.5B stand-in |
| Local judge (8B) | 8 | `GPU` | vLLM AWQ int4 | ~6 GB + KV | Only after agreeing with the API judge (C8.7) |
| Sentence embeddings (bge-small / e5-base) | 3+ | `CPU→GPU` | fp32 / fp16 | < 2 GB | CPU fine up to ~50k texts |
| All classical models (ID3, LR, KNN, SVM, NB, GMM, HMM, RF, GBM, k-means) | 3–6 | `CPU` | — | 0 | Never on GPU |
| Perceptron / MLP / optimizer study | 4 | `CPU` | fp32 | 0 | Small models |
| TextCNN span scorer | 8 | `CPU→GPU` | fp32 / bf16 | < 2 GB | GPU only for sweeps |
| Rewriter SFT: Qwen2.5-0.5B full fine-tune | 8 | `GPU` | bf16 + AdamW, grad checkpointing | ~12–16 GB | ~16 bytes/param for weights + grads + Adam (~8 GB) + activations |
| Rewriter SFT: Qwen2.5-1.5B / 3B LoRA | 8 | `GPU` | bf16 base, LoRA r=16 | ~8–16 GB | Main candidates |
| Rewriter SFT: 7B QLoRA | 8 | `GPU` | NF4 base + LoRA | ~10–18 GB at seq ≤ 1024 | Only if smaller rungs fail C8.3 |
| GRPO rewriter policy | 9 | `GPU` | 0.5B–1.5B LoRA | ~12–20 GB | Target calls via API or time-shared vLLM, never colocated |
| Tabular Q-learning | 9 | `CPU` | — | 0 | |
| Bayesian HPO of rewriter / TextCNN | 10 | `GPU` | Optuna, 1 trial at a time | per trial | Median pruner |
| Production serving | 11 | `GPU` (+ `CPU` fast path) | vLLM / ONNX Runtime | 2–6 GB | Fast path must also run on CPU |

### 5.1 Measurement plan (Phase 7)

| Row | Configuration measured | Values recorded |
| :---: | :--- | :--- |
| 7.6 | Llama-3.1-8B-Instruct, bf16, `gpu_memory_utilization=0.90`, max len 4096 | Peak VRAM; tokens/s at batch 1 and 16 |
| 7.7 | Qwen2.5-0.5B full FT and Qwen2.5-1.5B LoRA, seq 1024, micro-batch 4, grad checkpointing | Peak VRAM; step time |
| 7.8 | 30-minute sustained training step | Throughput drop (target < 5%) |

When measurements land, replace the "Est. VRAM" column with measured values, keep the estimate in a
note for comparison, and update the same table in CONTRIBUTING.md in the same commit.

---

## 6. Weekly GPU schedule (once connected)

| Slot | Use | Constraint |
| :--- | :--- | :--- |
| Night (long, exclusive) | SFT / RL training from `gpu_queue/` | `exclusive: true`; no vLLM |
| Day block A | vLLM serving of target / judge models for data generation and evaluation | No training |
| Day block B | Short sweeps (TextCNN, HPO trials), latency benchmarks | One trial at a time |
| Weekend | Full evaluation re-runs, checkpoint reports | |

Slots are separated in time so that rule 4.2.3 (no colocated serving and training) holds by construction.

---

## 7. Fallback: GPU still not connected by week 13

The target for Phase 7 is week 13 at the latest. If the card is not connected by then:

1. Pull forward the CPU halves of:
   - Phase 9: tabular Q-learning (9.1, 9.2).
   - Phase 10: classical HPO studies (10.1–10.5, 10.8).
   - Phase 11: fast-path serving, CI/CD, monitoring, orchestration (11.1, 11.2, 11.3.1–11.3.2, 11.3.4–11.3.6, 11.4, 11.5, 11.6).
2. Phases 8 and 9's neural parts shift right by the length of the delay.
3. Continue writing GPU code with CPU smoke tests and queued jobs, so the queue is ready to drain on day one.
4. **The schedule slips. Checkpoint targets are not lowered.** Changing a target requires an ADR
   (see Gate rule in CONTRIBUTING.md).

This risk is tracked as a register entry in [risks.md](risks.md).
