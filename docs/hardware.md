# Hardware Profile and Compute Allocation

Plan row: **0.1.6**. Source of truth: [CONTRIBUTING.md § Compute Allocation](../CONTRIBUTING.md#compute-allocation-cpu-local-open-models-and-the-rtx-4500-ada).
If this document and CONTRIBUTING.md disagree, CONTRIBUTING.md wins and this file is corrected.

---

## 1. Resources

| Resource | Owner | Details | Used for | Status |
| :--- | :--- | :--- | :--- | :---: |
| Dev laptop (Windows 11) | Guru Bharadwaj | CPU only, no CUDA device | Classical ML (Phases 3–6), tokenization, DVC pipelines, CI, CPU smoke tests, fast-path serving | Available |
| Dev workstation (Fedora 44) | Vivian Sobers | AMD Ryzen AI 7 350 (16 threads) · 14 GB RAM · 475 GB disk · no NVIDIA GPU · Python 3.14 system interpreter | Same as above | Available |
| NVIDIA RTX 4500 Ada Generation | Guru Bharadwaj | 24 GB GDDR6 ECC · Ada Lovelace · compute capability 8.9 · bf16 supported · ~210 W board power · attached later via remote access (row 7.1) | vLLM serving of all real LLM work (casualisation, candidate generation, judges, open targets), rewriter SFT/RL, neural HPO, quality-path serving | ❌ not connected |
| 2 × NVIDIA GeForce RTX 4090 | Vivian Sobers | `worker-1` / `worker-2`, see [§ 1.1](#11-vivians-gpu-workers) · each 24 GB · Ada Lovelace · compute capability 8.9 · bf16 supported | Same as the RTX 4500 Ada row | ✅ available |
| Local open-model serving | both | OpenAI-compatible servers only, reached through `terser.llm` provider `local`: Ollama / llama.cpp on a dev machine for tiny models (Qwen2.5-0.5B/1.5B, smoke tests), vLLM or Ollama on a GPU host for all real generation and judging. Per-phase token caps in `configs/budget.yaml` bound compute; responses disk-cached. **No hosted API keys are available or assumed** | CPU smoke tests (dev machines); every output that feeds labels, metrics or reports (GPU) | CPU: available · GPU: ✅ on the 4090s |


Consequences of the profile:

- bf16 is the default mixed-precision dtype on the GPU. fp16 loss scaling is not needed.
- ECC memory means VRAM available to CUDA is slightly below 24 GB. Row 7.10 uses **22 GB** as the
  usable ceiling for any single job.
- Until Phase 7 is ✅, nothing in the repository may *require* a GPU to pass CI.
- **The GPU is on the critical path.** With no hosted APIs, every labelled example (target outputs,
  candidates, judge verdicts) is produced on the RTX 4500 Ada, so Phase 2 cannot finish before Phase 7.
  Before the card is connected, Phase 2 stages are only smoke-tested end to end with the CPU stand-in
  model (row 2.2.4).

### 1.1 Vivian's GPU workers

Two shared lab machines administered by Vivian Sobers. Both were verified reachable and idle on
2026-09-10; they are shared with other users' work, so availability is not guaranteed at any moment.

| | `worker-1` | `worker-2` |
| :--- | :--- | :--- |
| Address | `ccbd@10.10.3.83` | `ccbd@10.10.3.50` |
| GPU | RTX 4090, 24564 MiB, cap 8.9 | RTX 4090, 24564 MiB, cap 8.9 |
| Driver | 595.84 | 580.173.02 |
| CPU / RAM | i9-13900, 32 threads, 125 GB | i9-13900, 26 threads, 125 GB |
| OS | Ubuntu 22.04.5 | Ubuntu 22.04.5 |
| Disk free | 235 GB of 938 GB | 141 GB of 938 GB |
| torch | 2.8.0+cu128 | 2.7.0+cu126 |
| vLLM | 0.28.0 in `~/vllm-env` (torch 2.13.0+cu130) | not installed |
| Ollama | installed | running, 12 models incl. `qwen2.5:32b` |
| Other | transformers 4.57.6, peft, trl, accelerate, datasets; 39 GB HF cache | transformers 4.46.3, peft, accelerate, mlflow 3.5.0; 19 GB HF cache |

Access and serving notes:

- SSH with password authentication as user `ccbd`. **Credentials are provisioned out-of-band and are
  never committed** (same rule as `.env`, row 7.1).
- Ollama binds to `127.0.0.1` on both hosts, so nothing is reachable directly from a dev machine.
  Forward the port and point the client at the tunnel:

  ```bash
  ssh -f -N -L 127.0.0.1:11434:127.0.0.1:11434 ccbd@10.10.3.50
  export TERSER_LOCAL_LLM_URL=http://localhost:11434/v1
  ```

  This path was exercised end to end on 2026-09-10: a rewrite request through the tunnel to
  `qwen2.5:32b` returned in about 20 s with `usage` counts populated, so `terser.llm` provider
  `local` needs no code change to use these hosts.
- System Python is 3.10.12 on both, below the project's `requires-python >= 3.11`. Use an explicit
  interpreter: `worker-1` has 3.11.13 / 3.12.11 / 3.13.7 via `uv`, `worker-2` has
  `/usr/bin/python3.11`.
- The two hosts have drifted (different drivers, different torch builds, vLLM on `worker-1` only).
  Any experiment reported from these machines must record which host produced it. This is risk
  **R14** in [risks.md](risks.md), now observed rather than hypothetical.
- `configs/hardware/rtx4090.yaml` selects either card; the queue runner is single-host, so jobs are
  currently placed on a host by hand (see [§ 4.2](#42-queue-rules)).

Consequences of the profile:

- bf16 is the default mixed-precision dtype on the GPU. fp16 loss scaling is not needed.
- ECC memory means VRAM available to CUDA is slightly below 24 GB. Row 7.10 uses **22 GB** as the
  usable ceiling for any single job.
- Until Phase 7 is ✅, nothing in the repository may *require* a GPU to pass CI.
- **The GPU is on the critical path.** With no hosted APIs, every labelled example (target outputs,
  candidates, judge verdicts) is produced on the RTX 4500 Ada, so Phase 2 cannot finish before Phase 7.
  Before the card is connected, Phase 2 stages are only smoke-tested end to end with the CPU stand-in
  model (row 2.2.4).

---

## 2. Device selection in code

No module hard-codes `"cuda"`. Device and precision come from the Hydra `hardware` config group and are
resolved by `terser.hardware.get_device(cfg)` (row 0.1.5).

| Config | Intended machine | Behaviour |
| :--- | :--- | :--- |
| `configs/hardware/cpu.yaml` | Dev laptop or workstation, CI | Forces CPU, fp32 |
| `configs/hardware/rtx4500ada.yaml` | RTX 4500 Ada machine | Requires a CUDA device; bf16 enabled. On a machine without CUDA, `get_device` raises with a clear message instead of silently falling back |
| `configs/hardware/rtx4090.yaml` | `worker-1` / `worker-2` | Same contract as the RTX 4500 Ada profile: requires CUDA, bf16, 22 GB usable ceiling |
| `configs/hardware/auto.yaml` | Any | Uses CUDA if available, otherwise CPU |


```powershell
python -m terser.x experiment=<name> hardware=cpu          # laptop or workstation
python -m terser.x experiment=<name> hardware=rtx4500ada   # RTX 4500 Ada machine
python -m terser.x experiment=<name> hardware=rtx4090      # worker-1 / worker-2
python -m terser.x experiment=<name> hardware=auto         # portable default
```

The `hardware` value is logged as an MLflow tag (`cpu`, `rtx4500ada` or `rtx4090`) on every run.
Runs on an RTX 4090 also set a `host` tag (`worker-1` or `worker-2`), because the two workers have
different driver and torch versions.


---

## 3. Compute tags

Every task row in CONTRIBUTING.md carries one or more tags.

| Tag | Meaning | What a contributor does now |
| :--- | :--- | :--- |
| `CPU` | Runs on either dev machine | Start immediately |
| `LLM` | Needs text generation from a local open-source model behind an OpenAI-compatible server (`terser.llm` provider `local`). No paid APIs | Build the code now and smoke-test it on CPU with a tiny model (Qwen2.5-0.5B/1.5B via Ollama or llama.cpp), within the phase token cap in `configs/budget.yaml`. Results that feed labels, metrics or reports come only from GPU-served models (an RTX 4090 worker, or the RTX 4500 Ada) |
| `GPU` | Needs a 24 GB Ada-class card: `worker-1`, `worker-2`, or the RTX 4500 Ada once connected | Write the code, smoke-test on CPU with `configs/model/tiny.yaml`, add a `gpu_queue/` job. The row stays ❌ until the job has run on a card |
| `CPU→GPU` | Reduced scale on CPU now, full scale on a GPU host | Run the CPU version, record it as reduced-scale, and queue the full-scale job |


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
   workload needs both (e.g. GRPO in 9.3 needs target outputs), target and judge calls go to a vLLM
   server in a separate, time-shared slot.
4. The queue drains by `priority` first, then by `depends_on` order. A job whose dependencies are not
   complete is skipped, not failed.
5. After Phase 7, every job must satisfy `est_vram_gb ≤ 22` using measured values (row 7.10).
6. Each job logs GPU-hours to MLflow (see Experiment standards in CONTRIBUTING.md).
7. **The runner is single-host.** `terser.gpu_queue` validates and drains one queue on the machine it
   runs on, and the job spec has no host field, so a job cannot be addressed to `worker-1` or
   `worker-2`. With two cards available, run one queue per host (`--dir gpu_queue/worker-1`, and so
   on) and record the host in the run's `host` tag. Giving the spec a real `host` field, so both
   cards drain from one queue, is open design work rather than something the current code supports.
8. The validator's VRAM ceiling is read from `configs/hardware/rtx4500ada.yaml` only, which is
   hardcoded in `terser.gpu_queue.queue`. Both profiles set `max_vram_gb: 22`, so jobs aimed at a
   4090 are still checked against the right number today. If the two ever diverge, that path has to
   become config-driven.



---

## 5. VRAM budget

> **Planning estimates only.** Every value below is derived from parameter counts and typical
> overheads, not measured on this card. Phase 7 rows **7.6** (serving) and **7.7** (training)
> replace them with measurements, and checkpoint **C7.2** requires this table to be updated.

| Workload | Phase | Compute | Precision / method | Est. VRAM | Notes |
| :--- | :---: | :--- | :--- | :--- | :--- |
| Token counting (tiktoken, HF tokenizers) | 0+ | `CPU` | — | 0 | Ground truth, never approximated |
| Claude token counts (optional) | 0+ | `CPU` | token-counting endpoint, only if an Anthropic key is ever added | 0 | Never required for any reported number |
| CPU smoke-test LLM (Qwen2.5-0.5B / 1.5B, GGUF Q4 via Ollama or llama.cpp) | 1+ | `CPU` | 4-bit | ~1–2 GB RAM | Pipeline tests only (a few tokens/s on the i3 laptop); outputs tagged `stand_in: true`, never used for labels |
| Candidate generation + pairwise judge (Qwen2.5-32B-Instruct-AWQ; fallback Qwen2.5-14B-Instruct-AWQ) | 2+ | `GPU` | vLLM AWQ int4 | ~19 GB weights (32B) / ~9 GB (14B) + KV cache | Exclusive slot, never colocated with target serving; 32B fit at max len 4096 is an estimate verified in 7.6 |
| Target LLM outputs (Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct) | 2, 8+ | `GPU` | vLLM bf16, `gpu_memory_utilization=0.90` | ~16 GB weights + KV cache | Before GPU: only the CPU smoke-test stand-in above |
| Small judge (Llama-3.1-8B-Instruct AWQ) | 2, 8+ | `GPU` | vLLM AWQ int4 | ~6 GB + KV | Self-preference control (2.2.5) and fast audits (8.4); replaces the 32B judge only after C8.7 |
| Sentence embeddings (bge-small / e5-base) | 3+ | `CPU→GPU` | fp32 / fp16 | < 2 GB | CPU fine up to ~50k texts |
| All classical models (ID3, LR, KNN, SVM, NB, GMM, HMM, RF, GBM, k-means) | 3–6 | `CPU` | — | 0 | Never on GPU |
| Perceptron / MLP / optimizer study | 4 | `CPU` | fp32 | 0 | Small models |
| TextCNN span scorer | 8 | `CPU→GPU` | fp32 / bf16 | < 2 GB | GPU only for sweeps |
| Rewriter SFT: Qwen2.5-0.5B full fine-tune | 8 | `GPU` | bf16 + AdamW, grad checkpointing | ~12–16 GB | ~16 bytes/param for weights + grads + Adam (~8 GB) + activations |
| Rewriter SFT: Qwen2.5-1.5B / 3B LoRA | 8 | `GPU` | bf16 base, LoRA r=16 | ~8–16 GB | Main candidates |
| Rewriter SFT: 7B QLoRA | 8 | `GPU` | NF4 base + LoRA | ~10–18 GB at seq ≤ 1024 | Only if smaller rungs fail C8.3 |
| GRPO rewriter policy | 9 | `GPU` | 0.5B–1.5B LoRA | ~12–20 GB | Target and judge calls go to a time-shared vLLM slot, never colocated with training |
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

## 7. Fallback: GPU still not connected by week 6

The target for Phase 7 is week 13 at the latest, but all labelled data waits for the GPU, so the
fallback starts earlier. Until the card is connected, complete the CPU rows of Phase 1, build and
smoke-test every Phase 2 stage with the CPU stand-in model (2.2.4), and write Phases 3–6 code against
the smoke-test data. If the card is still not connected by week 6:

1. Pull forward the CPU halves of:
   - Phase 9: tabular Q-learning (9.1, 9.2).
   - Phase 10: classical HPO studies (10.1–10.5, 10.8).
   - Phase 11: fast-path serving, CI/CD, monitoring, orchestration (11.1, 11.2, 11.3.1–11.3.2, 11.3.4–11.3.6, 11.4, 11.5, 11.6).
2. Phase 2 labelling and everything that depends on its labels (see the dependency map in
   CONTRIBUTING.md), including the neural parts of Phases 8 and 9, shift right by the length of the delay.
3. Continue writing GPU code with CPU smoke tests and queued jobs, so the queue is ready to drain on day one.
4. **The schedule slips. Checkpoint targets are not lowered.** Changing a target requires an ADR
   (see Gate rule in CONTRIBUTING.md).

This risk is tracked as a register entry in [risks.md](risks.md).
