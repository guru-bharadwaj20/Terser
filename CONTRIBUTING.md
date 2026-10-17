# Terser: Master Project Plan and Contribution Guide

**Rewriting casual prompts into token-efficient prompts that get the same output quality**

> Terser sits between a user and an LLM. It takes a prompt written the way people talk
> (filler, hedging, restated requirements) and rewrites it for a specific target model so that
> the prompt costs **fewer tokens on that model's tokenizer** while the model's answer keeps
> **the same quality and depth**. Every claim of savings is paired with a measured
> quality-preservation number. Savings alone never count as success.

---

## Legend

| Symbol | Meaning |
| :---: | :--- |
| ✅ | Done: implemented, tested, and committed |
| ❌ | Not done: pending or in progress |
| 🚧 | **Hard gate**: failing it blocks every dependent phase, with no conditional pass |

**Status rule:** a row flips from ❌ to ✅ only when its *Definition of Done* is satisfied **and**
the artifact exists in the repository (code + test + logged experiment or report).

**Gate rule:** a phase is complete only when every task row is ✅ **and** its checkpoint table
is ✅. A checkpoint target may only be changed by an ADR in `docs/decisions/` that explains why
the original target was wrong. "We did not reach it" is not a valid reason.

**Scope rule:** a row exists only if the syllabus requires it **or** the product or its evaluation
needs it. The last scope review is [ADR-0004](docs/decisions/ADR-0004-plan-revision.md); removed row
ids are never reused. Rows under [Stretch](#stretch-goals-never-block-a-checkpoint) never block a
checkpoint.

---

## Table of Contents

| # | Phase | Theme | Syllabus Unit | Weeks | Owner |
| :---: | :--- | :--- | :--- | :---: | :--- |
| 0 | Foundations and Repository Scaffolding | Infra | Unit 4 (MLOps basics) | 1 | Guru |
| 1 | Problem Framing and Concept Learning | Framing | Unit 1 | 2–3 | Guru |
| 2 | Data Generation Pipeline and Judge Validation | Data | — | 3–6 | Vivian |
| 3 | Metrics, Cross-Validation, Decision Trees, Logistic Regression, KNN | Modeling | Unit 1 | 6–8 | Vivian |
| 4 | Artificial Neural Networks and SVMs | Modeling | Unit 2 | 8–10 | Vivian |
| 5 | Boosting, Bayesian Learning, GMM+EM, HMM | Modeling | Unit 3 | 10–12 | Vivian |
| 6 | Unsupervised Structure of Prompts | Modeling | Unit 4 | 12–13 | Vivian |
| 7 | GPU Onboarding (RTX 4090 workers) | Infra | — | 3–4 | Vivian |
| 8 | Deep Learning: CNN Span Scorer and Generative Rewriter | Deep Learning | Unit 4 | 14–17 | Vivian |
| 9 | Reinforcement Learning: Tabular Q-Learning over Operators | RL | Unit 4 | 17–18 | Vivian |
| 10 | Hyperparameter Optimisation | Tuning | Unit 4 | 18–20 | Vivian |
| 11 | MLOps: Pipelines, Serving, CI/CD/CT, Monitoring | MLOps | Unit 4 | 20–23 | Vivian |
| 12 | Final Evaluation, Ablations and Benchmark Release | Science | All | 24 | Vivian |
| 13 | Documentation, Report and Viva Preparation | Delivery | All | 23–24 | Guru |

Supporting sections: [Compute Allocation](#compute-allocation-cpu-local-open-models-and-the-rtx-4500-ada) ·
[Timeline](#timeline) · [Cross-Phase Dependency Map](#cross-phase-dependency-map) ·
[Syllabus Coverage Matrix](#syllabus-coverage-matrix) · [Headline Success Criteria](#headline-success-criteria) ·
[Overall Progress](#overall-progress) · [Stretch Goals](#stretch-goals-never-block-a-checkpoint) ·
[How to Contribute](#how-to-contribute)

---

## Compute Allocation: CPU, Local Open Models, and the RTX 4500 Ada

### Hardware profile

| Resource | Owner | Details | Status |
| :--- | :--- | :--- | :---: |
| Dev laptop (Windows 11) | Guru Bharadwaj | CPU only; all classical ML, tokenization, pipelines, CI | Available |
| Dev workstation (Fedora 44) | Vivian Sobers | AMD Ryzen AI 7 350, 16 threads, 14 GB RAM, no NVIDIA GPU; classical ML, tokenization, pipelines, CI | Available |
| **NVIDIA RTX 4500 Ada Generation** | Guru Bharadwaj | 24 GB GDDR6 ECC, Ada Lovelace (compute capability 8.9), bf16 supported, ~210 W. WSL2 workstation ([hardware.md § 1.2](docs/hardware.md#12-rtx-4500-ada-workstation-wsl2)) | ✅ connected 2026-10-01 |
| **2 × NVIDIA GeForce RTX 4090** | Vivian Sobers | `worker-1` (10.10.3.83) and `worker-2` (10.10.3.50). Each 24 GB, Ada Lovelace (compute capability 8.9), bf16 supported. Ubuntu 22.04.5, i9-13900, 125 GB RAM. Shared lab machines reached over SSH | ✅ available |
| Local open-model serving | both | OpenAI-compatible servers only: Ollama / llama.cpp on a dev machine for tiny models (Qwen2.5-0.5B/1.5B, smoke tests), vLLM or Ollama on an RTX 4090 (and later the RTX 4500 Ada) for all real generation and judging. **No hosted API keys are available or assumed** | CPU: available · GPU: ✅ on the 4090s |

The two RTX 4090s have the same compute capability (8.9), VRAM (24 GB) and bf16 support as the
RTX 4500 Ada, so the precision and VRAM planning below applies to them unchanged. Details, access
and measurements: [docs/hardware.md § 1](docs/hardware.md#1-resources).

Since [ADR-0004](docs/decisions/ADR-0004-plan-revision.md) the RTX 4090 workers are the primary GPUs
and the RTX 4500 Ada is optional extra capacity. Wherever a row below names the RTX 4500 Ada, any
24 GB Ada-class card satisfies it; record the card used in the `hardware` tag.

### GPU job queue

GPU work is written, unit-tested on CPU with `configs/model/tiny.yaml`, and
queued as a job spec in `gpu_queue/`:

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

Queue rules: a job is queued only if its `cpu_smoke_test` passes in CI · `exclusive` jobs never share
the card · **vLLM serving and training never run at the same time on 24 GB** · the queue drains by
`priority`, then `depends_on`.

### VRAM budget (planning estimates, replaced by measurements in Phase 7)

| Workload | Phase | Compute | Precision / method | Est. VRAM | Notes |
| :--- | :---: | :--- | :--- | :--- | :--- |
| Token counting (tiktoken, HF tokenizers) | 0+ | `CPU` | — | 0 | Ground truth, never approximated |
| Claude token counts (optional) | 0+ | `CPU` | token-counting endpoint, only if an Anthropic key is ever added | 0 | Never required for any reported number |
| CPU smoke-test LLM (Qwen2.5-0.5B / 1.5B, GGUF Q4 via Ollama or llama.cpp) | 1+ | `CPU` | 4-bit | ~1–2 GB RAM | Pipeline tests only (a few tokens/s on the i3 laptop); outputs tagged `stand_in: true`, never used for labels |
| Candidate generation + pairwise judge (Qwen2.5-32B-Instruct-AWQ; fallback Qwen2.5-14B-Instruct-AWQ) | 2+ | `GPU` | vLLM AWQ int4 | **Measured (7.6), 32B:** 18.14 GiB weights + 1.46 GiB KV, peak 22.15 GiB net · 14B: ~9 GB (est.) | Exclusive slot. At max len 4096 the 32B fits **only** with `max_num_batched_tokens=2048` (fails at vLLM defaults); KV holds 5,968 tokens, 18.5 / 281 tok/s at batch 1 / 16. Was est. ~19 GB weights. [C7/serving.md](docs/checkpoints/C7/serving.md) |
| Target LLM outputs (Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct) | 2, 8+ | `GPU` | vLLM bf16, `gpu_memory_utilization=0.90` | **Measured (7.6), Qwen2.5-7B:** 14.29 GiB weights + 5.54 GiB KV, peak 20.79 GiB net · Llama-3.1-8B: not measured (gated) | 24.1 / 376 tok/s at batch 1 / 16. Was est. ~16 GB weights. [C7/serving.md](docs/checkpoints/C7/serving.md) |
| Second judge (Llama-3.1-8B-Instruct AWQ) | 2 | `GPU` | vLLM AWQ int4 | ~6 GB + KV | Self-preference control (2.2.5) only |
| Sentence embeddings (bge-small / e5-base) | 3+ | `CPU→GPU` | fp32 / fp16 | < 2 GB | CPU fine up to ~50k texts |
| All classical models (ID3, LR, KNN, SVM, NB, GMM, HMM, RF, GBM, k-means) | 3–6 | `CPU` | — | 0 | Never on GPU |
| Perceptron / MLP / optimizer study | 4 | `CPU` | fp32 | 0 | Small models |
| TextCNN span scorer | 8 | `CPU→GPU` | fp32 / bf16 | < 2 GB | GPU only for sweeps |
| Rewriter SFT: Qwen2.5-0.5B full fine-tune | 8 | `GPU` | bf16 + AdamW, grad checkpointing | **Measured (7.7):** 16.32 GiB reserved fp32-master / 11.77 GiB pure bf16 (seq 1024, mb 4); 0.44 / 0.37 s per step | ~16 bytes/param for weights+grads+Adam (~8 GB) + activations; was est. ~12–16 GB. [C7/training.md](docs/checkpoints/C7/training.md) |
| Rewriter SFT: Qwen2.5-1.5B / 3B LoRA | 8 | `GPU` | bf16 base, LoRA r=16 | **Measured (7.7), 1.5B:** 12.19 GiB reserved (seq 1024, mb 4), 1.11 s per step · 3B: ~8–16 GB (est.) | Main candidates; 1.5B was est. ~8–16 GB 
| Rewriter SFT: 7B QLoRA | 8 | `GPU` | NF4 base + LoRA | ~10–18 GB at seq ≤ 1024 | Only if smaller rungs fail C8.3 |
| GRPO rewriter policy (stretch) | 9 | `GPU` | 0.5B–1.5B LoRA | ~12–20 GB | Only if the stretch rows are attempted; never colocated with serving |
| Tabular Q-learning | 9 | `CPU` | — | 0 | |
| Bayesian HPO of the rewriter | 10 | `GPU` | Optuna, 1 trial at a time | per trial | Median pruner |
| Production serving | 11 | `GPU` (+ `CPU` fast path) | vLLM / ONNX Runtime | 2–6 GB | Fast path must also run on CPU |

### Weekly GPU schedule

| Slot | Use |
| :--- | :--- |
| Night (long, exclusive) | SFT / RL training from `gpu_queue/` |
| Day block A | vLLM serving of target / judge models for data generation and evaluation |
| Day block B | Short sweeps (TextCNN, HPO trials), latency benchmarks |
| Weekend | Full evaluation re-runs, checkpoint reports |

> **GPU time is still the critical resource.** Without hosted APIs, every labelled example (target
> outputs, candidates, judge verdicts) is produced on a GPU worker, so Phase 2 cannot finish before
> Phase 7. The workers are shared lab machines: book the exclusive slots above, and keep writing
> Phases 3–6 code against the CPU smoke-test data while labelling runs. The schedule slips. The
> checkpoints are not lowered.

---

## Timeline

```
Week:      1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24
Phase 0   ███
Phase 1       ███████
Phase 2           ███████████████
Phase 3                       ███████████
Phase 4                               ███████████
Phase 5                                       ███████████
Phase 6                                               ███████
Phase 7           ███████
Phase 8                                                       ███████████████
Phase 9                                                                   ███████
Phase 10                                                                      ███████████
Phase 11                                                                              ███████████████
Phase 12                                                                                          ████
Phase 13                                                                                      ████████
```

---

# Phase 0 — Foundations and Repository Scaffolding

**Goal:** a reproducible, hardware-aware skeleton, so every later experiment is tracked, versioned
and comparable, and runs on CPU today and on the RTX 4500 Ada later without code changes.
**Weeks:** 1 · **Depends on:** — · **Unlocks:** everything

### 0.1 Environment

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 0.1.1 | Python environment | Python ≥ 3.11 venv, `pyproject.toml` with extras `dev` (CPU) and `gpu` (torch CUDA, vllm, peft, trl, bitsandbytes) | `pip install -e ".[dev]"` reproduces from scratch | ✅ |
| 0.1.2 | Core stack | numpy, pandas, scipy, scikit-learn, lightgbm, cvxopt, matplotlib | Import smoke test passes | ✅ |
| 0.1.3 | Tokenizer stack | `tiktoken` (`o200k_base`), HF `tokenizers` (Qwen2.5, Llama-3.1 via ungated mirror), optional Anthropic token counting | All four targets count a fixture string | ✅ |
| 0.1.4 | Determinism harness | `set_seed()` for random / numpy / torch / CUDA, `PYTHONHASHSEED` | Two runs of a dummy experiment give identical metrics | ✅ |
| 0.1.5 | Device abstraction | `terser.hardware.get_device(cfg)` with `device: auto`, no hard-coded `"cuda"` | `hardware=rtx4500ada` on CPU fails with a clear message; `auto` falls back to CPU | ✅ |
| 0.1.6 | Hardware profile doc | VRAM budget table above, GPU queue rules | `docs/hardware.md` written | ✅ |

### 0.2 Repository Layout

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 0.2.1 | Directory tree | `configs/ data/{raw,interim,processed,gold}/ docs/{decisions,checkpoints,derivations}/ notebooks/ src/terser/ gpu_queue/ tests/ reports/` | Tree created with `__init__.py` | ✅ |
| 0.2.2 | Package modules | `src/terser/{tokenizers,operators,concept,data,judge,features,models/{classical,neural,ensembles,clustering,rewriter,rl},eval,hpo,serving,monitoring,pipelines}` | Each imports cleanly | ✅ |
| 0.2.3 | Config system | Hydra configs incl. `configs/hardware/{cpu,rtx4500ada}.yaml`; no hard-coded paths or hyperparameters | Every script runs `python -m terser.x experiment=...` | ✅ |
| 0.2.4 | Experiment tracking | Local MLflow server + Model Registry; tags `phase`, `task`, `hardware`, `data_hash`, `git_sha` | Dummy run visible with params, metrics, artifacts | ✅ |
| 0.2.5 | Data versioning | DVC initialised with a local remote | `dvc status` clean | ✅ |
| 0.2.6 | Task runner | `tasks.ps1` (Windows) + `Makefile` mirror: `data`, `train-classical`, `eval`, `test` | All targets execute | ✅ |
| 0.2.7 | Pre-commit | black, ruff, mypy, nbstripout | Hook blocks a bad commit | ✅ |
| 0.2.8 | Test scaffold | pytest with markers `gpu`, `api`, `slow`; fixtures of 20 casual prompts | `pytest -m "not gpu and not api"` green | ✅ |
| 0.2.9 | CI | GitHub Actions: lint, types, CPU tests | CI green on a PR | ✅ |
| 0.2.10 | GPU queue | `gpu_queue/` schema + validator + `python -m terser.gpu_queue {validate,dry-run,run}` | Dry-run orders two sample jobs correctly | ✅ |

### 0.3 Project Governance

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 0.3.1 | Data card template | Source, licence, size, PII notes, bias notes | `docs/data_cards/_template.md` | ✅ |
| 0.3.2 | Model card template | Intended use, targets supported, metrics per category, limits | `docs/model_cards/_template.md` | ✅ |
| 0.3.3 | Naming convention | Runs `<phase>-<task>-<model>-<variant>-<seed>`; branches `phase<N>/<topic>` | `docs/conventions.md` | ✅ |
| 0.3.4 | ML lifecycle ADR | data → label → train → evaluate → register → deploy → monitor → retrain | `docs/decisions/ADR-0001-ml-lifecycle.md` | ✅ |
| 0.3.5 | Risk register | See [Risk Register](#risk-register) | `docs/risks.md` with mitigations | ✅ |
| 0.3.6 | LLM budget guard | `configs/budget.yaml` per phase; client refuses calls past cap (token caps bound local compute; USD caps only for priced providers); local OpenAI-compatible provider; disk cache keyed by model+params+prompt hash | Test: cap reached ⇒ call refused; cached call costs 0 | ✅ |

### Checkpoint C0 — Foundations Ready

| # | Check | Target | Status |
| :---: | :--- | :--- | :---: |
| C0.1 | CI | Green on `main` | ✅ |
| C0.2 | Tokenizer correctness | `count_tokens` equals each official tokenizer on a 200-string fixture set **exactly (100%)** | ✅ |
| C0.3 | Reproducibility | Same seed ⇒ identical metrics across two runs | ✅ |
| C0.4 | CPU fallback | Full test suite passes on a machine with no GPU | ✅ |

---

# Phase 1 — Problem Framing and Concept Learning (Unit 1)

**Goal:** frame compression as a set of learning problems, define interpretable rewrite
operators, and use **concept learning** to describe "a safe rewrite" before building any
statistical model.
**Weeks:** 2–3 · **Depends on:** 0 · **Unlocks:** 2

### 1.1 Machine Learning Framing

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 1.1.1 | Sub-problem taxonomy | Quality gate (binary classification) · span tagger (sequence labelling KEEP/DROP/REPLACE) · savings predictor (regression) · intent discovery (clustering) · rewriter (conditional generation) · rewrite policy (RL) | `docs/decisions/ADR-0002-problem-framing.md` reviewed | ❌ |
| 1.1.2 | Task categories | `code`, `fact`, `reason`, `explain`, `creative`, `transform`, each with its quality measure (unit tests / EM-F1 / answer accuracy / rubric judge / pairwise judge / rubric + constraint checklist) | Category spec in `docs/categories.md` | ✅ |
| 1.1.3 | Positive-class convention | Quality gate positive = **degraded**, so sensitivity = catching degradations, specificity = not rejecting good rewrites | Recorded in ADR-0002 and used by `terser.eval` | ✅ |

### 1.2 Rewrite Operators (the hypothesis-space building blocks)

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 1.2.1 | `F` drop filler | "hey so um, basically" → "" (conservative: "like" as comparison kept) | Unit tests incl. no-op on efficient prompts | ✅ |
| 1.2.2 | `H` drop hedging / politeness | "could you maybe please", "thanks in advance" → "" | Same | ✅ |
| 1.2.3 | `D` deduplicate restated requirements | "keep it short… keep it brief" → one sentence; never touches code blocks | Same | ✅ |
| 1.2.4 | `I` imperative conversion | "I was wondering if you could explain" → "Explain" | Same | ✅ |
| 1.2.5 | `L` list-ify constraints | ≥ 3 constraint sentences → bullet list | Same | ✅ |
| 1.2.6 | `C` compress meta-talk | "I'm going to give you some text and I want you to" → "" | Same | ✅ |
| 1.2.7 | `S` tokenizer-aware substitution | Phrase table ("in order to" → "to"); picks the alternative that is cheapest **on target T's tokenizer in context** | Test shows `S` choosing differently for two tokenizers on at least one fixture, or documents that it never does | ✅ |
| 1.2.8 | Canonical application order | `C H F I D S L`; `apply_ops(text, codes, target)` is order-independent in its input | Property test | ✅ |

### 1.3 Concept Learning

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 1.3.1 | Instance representation | Discrete attributes: `length{short,medium,long}`, `has_constraints`, `has_payload`, `tone_cue`, `filler_density{low,high}` + operator flags `op_F … op_S` | `terser.concept.attributes` with tests | ✅ |
| 1.3.2 | Hypotheses | Conjunctive constraint vectors with `?` (any) and `∅` (none), e.g. `<long, ?, no, no, high, op_F=yes, …>` ⇒ "preserved" | Documented in `docs/derivations/concept_learning.md` | ✅ |
| 1.3.3 | Find-S | Most specific hypothesis covering all positives | Reproduces Mitchell's EnjoySport result `<Sunny, Warm, ?, Strong, ?, ?>` | ✅ |
| 1.3.4 | Candidate-Elimination | S and G boundaries; collapse detection that reports the offending example and the examples it conflicts with | Reproduces EnjoySport S/G; version space of size 6 | ✅ |
| 1.3.5 | Version space enumeration | All hypotheses between S and G | Enumerator matches brute force on a 5-attribute toy space | ✅ |
| 1.3.6 | Pilot labels | **150 prompts (25/category)** × operator subsets (7 singletons + `FHC`, `FHCI`, `FHCID`, `FHCIDS`, all) → target output → position-swapped pilot judge → human review of every label | `data/pilot/labels.jsonl` + human-reviewed CSV | ❌ |
| 1.3.7 | Version space per category | Run Candidate-Elimination for each of the 6 categories on 1.3.6 | `reports/p1_version_spaces.md` with S, G, size or collapse diagnosis | ❌ |
| 1.3.8 | Inductive bias write-up | What the conjunctive language cannot express (e.g. "drop hedging *unless* tone matters in creative prompts") and why this motivates Phases 3–5 | Section in `reports/p1_version_spaces.md` | ✅ |

### 1.4 Pilot Tokenizer Economics

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 1.4.1 | Pilot prompt set | **500 casual prompts**: hand-written seeds plus casualisation of clean instructions (Qwen2.5 on vLLM; before the GPU, hand-written seeds only), balanced across categories | `data/pilot/prompts.jsonl` + data card | ✅ |
| 1.4.2 | Per-operator savings | Δtokens % for each operator and operator stack on gpt-4o (tokenizer only), qwen2.5, llama-3.1 | `reports/p1_tokenizer_study.{csv,md}` | ✅ |
| 1.4.3 | Operator applicability | Share of prompts each operator changes, per category | Table in report | ✅ |
| 1.4.4 | Cross-tokenizer agreement | Spearman ρ of per-(prompt, operator) savings for every tokenizer pair | Table in report | ✅ |

### Checkpoint C1 — Framing and Operators

| # | Check | Target | Status |
| :---: | :--- | :--- | :---: |
| C1.1 | ADR-0002 merged | Reviewed by ≥ 1 other contributor | ❌ |
| C1.2 | Operator safety | Every operator is a no-op on 50 already-efficient prompts (100%) | ✅ |
| C1.3 | Version space | Non-empty for **≥ 4 of 6** categories; each collapse explained by named conflicting examples | ❌ |
| C1.4 | Savings exist (non-blocking, [ADR-0004](docs/decisions/ADR-0004-plan-revision.md)) | Median best-single-operator savings **≥ 10%** on the pilot set. Measured 1.7%: recorded negative result; savings must come from generative rewriting, gated at C2.7 | ❌ |
| C1.5 | Tokenizer dependence | Spearman ρ < 0.9 for ≥ 1 tokenizer pair **or** documented finding that tokenizers agree (Phase 8 then ships a single target-agnostic rewriter) | ✅ |

---

# Phase 2 — Data Generation Pipeline and Judge Validation

**Goal:** a labelled dataset of (casual prompt, candidate rewrite, target, token counts,
preserved/degraded), produced by a judge that has been **validated against humans**. Nothing
downstream can be measured without it.
**Weeks:** 3–6 · **Depends on:** 1 · **Unlocks:** 3, 4, 5, 6, 8

### 2.1 Prompt Collection

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 2.1.1 | Source datasets | Permissively licensed instruction sets (e.g. Dolly, OpenAssistant), HumanEval/MBPP prompts, GSM8K problems | `data/SOURCES.md` + one data card per source | ✅ |
| 2.1.2 | Licence audit | Research use and redistribution terms; unknown ⇒ non-redistributable | `reports/license_audit.md` | ✅ |
| 2.1.3 | PII scrubbing | PII detector over all text before `data/interim/` | Scrub report; 0 detected PII in a 500-row manual sample | ❌ |
| 2.1.4 | Casualisation | LLM rewrites clean instructions in 5 styles: rambling, hedging, texting, over-polite, repetitive; clean version kept as reference | ≥ 6,000 casual prompts | ✅ |
| 2.1.5 | Deduplication | MinHash, Jaccard ≥ 0.8 removed; paraphrase groups kept whole | Duplicate report | ✅ |
| 2.1.6 | **Source-grouped splits** | Train 70 / val 15 / test 15 by source prompt; test frozen end of week 4 and hash-locked | Automated test: 0 source prompts shared across splits | ✅ |
| 2.1.7 | DVC stages | `collect → casualise → dedupe → split` | `dvc repro` rebuilds from cache | ✅ |
| 2.1.8 | **Real-prompt slice** | ~1,000 first-turn user prompts from a licence-audited public chat log (e.g. WildChat), PII-scrubbed, category-tagged. Never casualised and never used for training: headroom (2.4.2) and final test (12.3) only. Guards against rewriters that merely undo the verbosity 2.1.4 injected | Data card + licence row; frozen with the test split | ✅ |

### 2.2 Candidates and Target Outputs

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 2.2.1 | Candidate generation | 6 per prompt: 3 rewrites from Qwen2.5-32B-Instruct-AWQ at low/medium/high aggressiveness, 2 operator stacks, 1 LLMLingua-2 baseline | `candidates` stage output | ✅ |
| 2.2.2 | Token counts | Every prompt and candidate on every target tokenizer | Counts table, 0 missing | ✅ |
| 2.2.3 | Target outputs | Qwen2.5-7B-Instruct and Phi-3.5-mini-instruct ([ADR-0011](docs/decisions/ADR-0011-second-family-phi35.md)) on vLLM bf16, temperature 0 (creative: n = 3 samples) | `respond` stage output | ✅ |
| 2.2.4 | Pipeline smoke run (pre-GPU) | `respond → judge → label` on 50 prompts with Qwen2.5-0.5B/1.5B via Ollama on CPU, through the same `terser.llm` client | Stages run end to end; every output tagged `stand_in: true` and excluded from labels | ✅ |
| 2.2.5 | Judge self-preference control | Second judge Llama-3.1-8B-Instruct on 1,000 pairs; compare verdict agreement separately for Qwen-target and Llama-target outputs | Report: agreement gap between target families ≤ 5 pp, or mitigation documented |✅ |

### 2.3 Quality Judging

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 2.3.1 | Task metrics | `code`: unit-test pass rate · `fact`: EM / token F1 · `reason`: final-answer accuracy | `terser.judge.metrics` with tests | ✅ |
| 2.3.2 | Rubrics | Per-category YAML: coverage, correctness, depth, constraint adherence, tone | `src/terser/judge/rubrics/*.yaml` | ✅ |
| 2.3.3 | Pairwise judge with position swap | Qwen2.5-32B-Instruct-AWQ (fallback 14B-AWQ) on vLLM judges (original-output, rewrite-output) in both A/B orders | `judge` stage output |✅ |
| 2.3.4 | Label definition | `preserved` iff task metric not worse **and** judge says tie-or-rewrite-better in **both** orders | `label` stage + tests | ✅ |
| 2.3.5 | Human annotation | 2 annotators label **300 pairs** (50/category), blind to which is the rewrite | `data/gold/human_300.jsonl` + guideline doc | ❌ |
| 2.3.6 | Agreement analysis | Human–human κ, judge–human κ per category, position-bias rate | `reports/p2_judge_validation.md` | ❌ |

### 2.4 Headroom and Cost

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 2.4.1 | Oracle rewrite | Shortest candidate still `preserved`, per prompt and target | `oracle` table |✅ |
| 2.4.2 | Headroom report | Oracle savings per target × category; degraded share among candidates | `reports/p2_headroom.md` |✅ |

### Checkpoint C2 — Dataset, Judge, Headroom

| # | Check | Target | Status |
| :---: | :--- | :--- | :---: |
| C2.1 | Dataset size | ≥ 6,000 casual prompts, ≥ 800 per category, < 1% near-duplicates | ✅ |
| C2.2 | Leakage | 0 source prompts shared across splits | ✅ |
| C2.3 🚧 | Human–human agreement | Cohen's κ **≥ 0.60** | ❌ |
| C2.4 🚧 | Judge–human agreement | κ **≥ 0.60** overall and **≥ 0.50** in every category | ❌ |
| C2.5 🚧 | Judge as degradation detector | Sensitivity **≥ 0.85**, specificity **≥ 0.80** vs human labels | ❌ |
| C2.6 | Position bias | A/B-order disagreement **≤ 10%** | ❌ |
| C2.7 🚧 | **Go / no-go: oracle savings** | Median **≥ 25%** on ≥ 2 targets; ≥ 15% in ≥ 5 of 6 categories. Also reported (not gated) on the real-prompt slice 2.1.8. On failure: negative-result report and scope narrowing before continuing |✅ |
| C2.8 | Class balance | Degraded share among candidates between **15% and 60%** |✅ |
| C2.9 | Reproducibility | `dvc repro` gives an identical dataset hash from cached responses | ✅ |

---

# Phase 3 — Metrics, Cross-Validation, Decision Trees, Logistic Regression, KNN (Unit 1)

**Goal:** the evaluation backbone every later phase reuses, plus interpretable baselines for the
**quality gate**, **span tagger** and **savings predictor**, with full treatment of
bias–variance and overfitting.
**Weeks:** 6–8 · **Depends on:** 2 · **Unlocks:** 4, 5, 6

### 3.1 Features

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 3.1.1 | Alignment labels | Token-level KEEP/DROP/REPLACE from diff-aligning each prompt with its oracle rewrite | `features/alignment.py` + tests on 20 hand-aligned pairs | ✅ |
| 3.1.2 | Span features | Token cost on T, self-information under GPT-2 small, stopword ratio, relative position, POS, filler-lexicon hit, negation / number / constraint keyword flags | `features/span.py`; feature table in DVC | ✅ |
| 3.1.3 | Pair features | Token ratio, max self-information removed, # negations / numbers / constraint keywords removed, embedding cosine, category one-hot | `features/pair.py`; feature table in DVC | ✅ |
| 3.1.4 | Continuous-value handling | Standardisation for LR/KNN; C4.5 threshold search for trees; equal-frequency bins for ID3 | Documented per model | ✅ |

### 3.2 Evaluation Backbone

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 3.2.1 | Metric suite | Accuracy, Precision, Recall, **Sensitivity**, **Specificity**, F1, confusion matrix, per-category breakdown | `eval/metrics.py`, matches scikit-learn to 1e-9 | ✅ |
| 3.2.2 | ROC / AUC | ROC curves, AUC; PR curves and AP (degraded is the minority) | `eval/curves.py` + figure template | ✅ |
| 3.2.3 | Threshold rule | Maximise specificity subject to **sensitivity ≥ 0.95** | `eval/threshold.py` + test | ✅ |
| 3.2.4 | **Grouped stratified k-fold** | `StratifiedGroupKFold(k=5)`, group = source prompt, stratify = category × label | `eval/cv.py`; out-of-fold predictions as the shared unit | ✅ |
| 3.2.5 | Leakage demonstration | Naive KFold vs grouped CV on the same model | Gap recorded in `reports/p3_cv_leakage.md` | ✅ |
| 3.2.6 | Nested CV | Outer 5 / inner 3 for unbiased estimates while tuning | Used by 3.6 | ✅ |
| 3.2.7 | Leave-one-category-out | Generalisation to unseen task types | Per-category table | ✅ |
| 3.2.8 | Statistical comparison | McNemar on shared out-of-fold predictions; bootstrap CIs | `eval/significance.py` | ✅ |

### 3.3 Decision Trees

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 3.3.1 | **ID3 from scratch** | Entropy, information gain, recursive split on discretised span features | Matches a hand-computed toy example; within 2 F1 of scikit-learn `criterion="entropy"` when both trees are grown to full depth (equal-depth multiway trees have more capacity than binary ones) | ✅ |
| 3.3.2 | Continuous values | C4.5-style threshold search on midpoints between class changes | Test on a 1-D separable fixture | ✅ |
| 3.3.3 | Overfitting study | Train vs val F1 for depth 1–30 and min-samples-leaf sweep | `reports/figures/p3_tree_overfit.png` | ✅ |
| 3.3.4 | Overfitting solutions | Pre-pruning, **reduced-error pruning**, **rule post-pruning**, cost-complexity pruning | Comparison table: leaves, val F1, train–val gap | ✅ |
| 3.3.5 | Decision boundary (x–y) | Axis-aligned regions on `self_information` vs `token_cost` | `reports/figures/p3_tree_boundary.png` | ✅ |
| 3.3.6 | Hypothesis search and inductive bias | Greedy information-gain search (preference bias for short trees) vs Phase 1 conjunctive restriction bias | `reports/p3_tree_bias.md` | ✅ |
| 3.3.7 | Rule export | Pruned tree as human-readable rules ("drop is unsafe when a negation is removed and…") | `reports/p3_tree_rules.md` | ✅ |

### 3.4 Logistic Regression

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 3.4.1 | From-scratch batch GD | Sigmoid, log-loss, gradient | Coefficients within 1e-3 of scikit-learn | ✅ |
| 3.4.2 | Regularisation | L1 / L2 / elastic-net, C sweep | Grid results table | ✅ |
| 3.4.3 | Calibration | Reliability diagram, Brier score, ECE | `reports/figures/p3_lr_calibration.png` | ✅ |
| 3.4.5 | Coefficient interpretation | Odds ratios on standardised features | `reports/p3_lr_coefficients.md` | ✅ |

### 3.5 K-Nearest Neighbours

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 3.5.1 | KNN classification | Quality gate; k ∈ {1,3,5,7,11,15,25}; Euclidean / Manhattan / cosine; uniform / distance weighting | Grid table under 3.2.4 protocol | ✅ |
| 3.5.2 | **KNN regression** | Predict a prompt's achievable savings % from its k nearest training prompts | MAE, R² table | ✅ |
| 3.5.3 | KNN (and LR) decision boundary | k = 1, 5, 25 on 2-D PCA refit inside each fold; LR's linear boundary on the same plane as 3.3.5 | `reports/figures/p3_knn_boundaries.png` | ✅ |
| 3.5.4 | "Skip short prompts" rule | Use 3.5.2 to skip rewriting when predicted savings < overhead | Rule + expected net-savings curve | ✅ |

### 3.6 Bias–Variance and Grid Search

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 3.6.1 | **Bias–variance decomposition and trade-off** | 200 bootstrap resamples; squared loss (KNN regression), 0-1 loss decomposition (classifiers); tree depth ∈ {2,4,8,16,∞}, k ∈ {1,3,5,15,51}; identify and explain the minimum-error point | `reports/figures/p3_bias_variance.png` + `reports/p3_bias_variance.md` | ✅ |
| 3.6.3 | Learning curves | Train size 10% → 100% for all three models | Figure + diagnosis (high bias vs high variance) | ✅ |
| 3.6.4 | Grid search | Tree depth × min-samples; KNN k × metric × weighting; LR penalty × C, inside nested CV | Best params recorded in `configs/` | ✅ |
| 3.6.5 | Baselines | Majority class, stratified random, "always preserved" | Baseline row in every table | ✅ |

### Checkpoint C3 — Evaluation Backbone and Classical Baselines

| # | Check | Target | Status |
| :---: | :--- | :--- | :---: |
| C3.1 | Metric implementations | Match scikit-learn to 1e-9 on 20 random fixtures | ✅ |
| C3.2 | Leakage demonstration | Grouped-CV AUC lower than naive-KFold AUC; gap recorded | ✅ |
| C3.3 | Quality gate (best of Tree / LR / KNN) | ROC-AUC **≥ 0.75**; specificity **≥ 0.40** at sensitivity 0.95 | ✅ |
| C3.4 | Span tagger (best of Tree / LR) | Token macro-F1 **≥ 0.65** | ❌ |
| C3.5 | Savings predictor (KNN regression) | MAE **≤ 10 pp**, R² **≥ 0.3** | ❌ |
| C3.6 | Overfitting control | ≥ 1 pruning method closes the tree's train–val gap by **≥ 50%** | ✅ |
| C3.7 | Figures | Tree, LR, KNN (3 k values) boundaries on identical axes; bias–variance and learning curves present | ✅ |

---

# Phase 4 — Artificial Neural Networks and SVMs (Unit 2)

**Goal:** learned decision functions for the quality gate and span tagger, with from-scratch
implementations verified against libraries, plus controlled activation and optimizer studies.
**Weeks:** 8–10 · **Depends on:** 3 · **Unlocks:** 5, 8, 10

### 4.1 Perceptron and Multi-Layer Networks

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 4.1.1 | Perceptron from scratch | Step activation; filler vs non-filler spans | Converges on the linearly separable subset; fails to converge on the full set (logged) | ✅ |
| 4.1.2 | Pair embedding input | `[e_o, e_r, \|e_o − e_r\|, e_o ⊙ e_r]` from 384-d bge-small ⇒ **1536-d** | Cached `data/features/pair_emb.npy` + id index | ✅ |
| 4.1.3 | NumPy MLP with backprop | Forward, backward, update; no autograd | Gradient check vs finite differences, relative error **< 1e-6**; matches `torch.autograd` | ✅ |
| 4.1.4 | PyTorch MLP quality gate | 1536 → 256 → 64 → 1 | Scored under the 3.2.4 protocol | ✅ |
| 4.1.5 | Parameter calculation | 1536×256+256 = 393,472 · 256×64+64 = 16,448 · 64×1+1 = 65 · **total 409,985** | Equals `sum(p.numel())` in a test | ✅ |
| 4.1.7 | Backprop derivation | Chain rule through sigmoid output + BCE; hidden-layer recursion | `docs/derivations/backprop.md` | ✅ |

### 4.2 Activation Functions

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 4.2.1 | Activation study | Step (straight-through estimator), sigmoid (+ saturated-unit fraction), tanh, ReLU (+ dead-unit fraction): val AUC, epochs to plateau, per-layer gradient norm; 5 seeds each; mechanism tied to observed behaviour | `reports/p4_activations.md` | ✅ |

### 4.3 Optimizers

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 4.3.1 | Optimizer study | Full-batch GD, SGD (mini-batch 64), SGD + Momentum (β ∈ {0.9, 0.99}), Adagrad, Adam (β₁ 0.9, β₂ 0.999), each over a small lr grid, at **equal gradient-evaluation budget**, 5 seeds; convergence speed vs generalisation tied to each update rule | `reports/figures/p4_optimizers.png` + `reports/p4_optimizers.md` | ✅ |

### 4.4 Support Vector Machines

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 4.4.1 | Margin and primal | Margin = 2/‖w‖; hard- and soft-margin primal | `docs/derivations/svm.md` §1–2 | ✅ |
| 4.4.2 | Lagrangian dual | Lagrangian, KKT conditions, dual problem, recovering w and b from support vectors | `docs/derivations/svm.md` §3–4 | ✅ |
| 4.4.3 | Linear kernel derivation | K(x, z) = xᵀz substituted into the dual | `docs/derivations/svm.md` §5 | ✅ |
| 4.4.4 | **Dual solver from scratch** | QP via `cvxopt` on span features | Same support-vector set as scikit-learn; ‖w − w_sk‖/‖w_sk‖ **< 1e-3** | ✅ |
| 4.4.5 | Hard margin | Linearly separable filler subset | Support vectors plotted | ✅ |
| 4.4.6 | Soft margin | C ∈ {0.01, 0.1, 1, 10, 100}: margin width, # support vectors, val metrics | Table + figure | ✅ |
| 4.4.7 | Polynomial kernel | Degree 2, 3 × coef0 × C vs linear | Table: AUC, time, # SVs | ✅ |
| 4.4.8 | Decision surfaces | Linear vs polynomial on the Phase 3 2-D plane | `reports/figures/p4_svm_surfaces.png` | ✅ |

### Checkpoint C4 — Neural and Margin Models

| # | Check | Target | Status |
| :---: | :--- | :--- | :---: |
| C4.1 | Backprop correctness | Relative gradient error **< 1e-6** on all layers | ✅ |
| C4.2 | SVM dual correctness | Support-vector set equals scikit-learn's; weight error **< 1e-3** | ✅ |
| C4.3 | Studies complete | 4 activations × 5 seeds; 5 optimizers × 5 seeds at equal budget; written interpretation | ✅ |
| C4.4 | Better quality gate | Best MLP/SVM ROC-AUC **≥ C3.3 best + 0.03**, or documented reason (classical model stays champion) | ✅ |
| C4.5 | Worst category | AUC **≥ 0.70** in every category | ❌ |

---

# Phase 5 — Boosting, Bayesian Learning, GMM+EM, HMM (Unit 3)

**Goal:** the strongest tabular quality gate, Bayesian components for filler detection and judge
sanity-checking, and the **first end-to-end non-generative compressor** (an HMM tagger).
**Weeks:** 10–12 · **Depends on:** 3, 4 · **Unlocks:** 6, 8, 9, 11

### 5.1 Ensembles

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 5.1.1 | Combining weak learners | AdaBoost with depth-1 stumps from scratch; which pairs get the largest weights | Matches scikit-learn within 0.01 AUC; training error vs rounds plot; hard-example note | ✅ |
| 5.1.3 | Random Forest | n_estimators × max_features × bootstrap | Table; OOB vs CV comparison | ✅ |
| 5.1.5 | Gradient Boosting | LightGBM: lr × n_estimators × max_depth with early stopping on grouped validation | Table | ✅ |
| 5.1.6 | Feature importance | Permutation importance + SHAP per task category: what makes a rewrite unsafe | Top-10 features + `reports/figures/p5_shap.png` | ✅ |
| 5.1.7 | Feature feedback | ≥ 1 importance finding turned into a new feature or operator rule | PR linked | ✅ |

### 5.2 Bayesian Learning

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 5.2.1 | Bayes theorem | P(filler \| phrase, context) worked from corpus counts | `docs/derivations/bayes.md` | ✅ |
| 5.2.3 | Naive Bayes filler detector | Multinomial NB on 1–3-grams: MLE parameters with Laplace smoothing (and why smoothing is MAP); vs LR on the same features | PR curve + table | ✅ |
| 5.2.4 | Bayes optimal classifier | Phase 1 hypotheses as H; P(h \| D) from validation likelihood; posterior-weighted vote vs MAP hypothesis | `reports/p5_bayes_optimal.md` | ✅ |
| 5.2.5 | EM from scratch | E-step responsibilities, M-step means/variances/weights; log-likelihood monotone | Test asserts non-decreasing log-likelihood; matches scikit-learn GMM | ✅ |
| 5.2.6 | GMM on judge score deltas | 2-component GMM on (rewrite score − original score): unsupervised preserved/degraded split vs threshold labels | ARI reported; judge-threshold review issue opened if low | ✅ |

### 5.3 Hidden Markov Model Span Tagger

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 5.3.1 | Model definition | Hidden states {KEEP, DROP, REPLACE}; observations = discretised token class (filler-lexicon id × POS × cost bucket) | `models/classical/hmm.py` | ✅ |
| 5.3.2 | Supervised MLE | Initial, transition, emission matrices from alignment labels with smoothing | Rows sum to 1 (test) | ✅ |
| 5.3.3 | Forward algorithm | Sequence likelihood in log space | Matches brute-force enumeration on length-5 sequences | ✅ |
| 5.3.4 | Viterbi decoding | Most likely tag sequence → compressed prompt | Matches brute force on length-5 sequences | ✅ |
| 5.3.5 | Baum-Welch | EM on unlabelled casual prompts from supervised initialisation | F1 before vs after | ✅ |
| 5.3.6 | End-to-end HMM compressor | Viterbi tags → rewrite → Phase 2 judge on **500 val prompts** | `reports/p5_hmm_compressor.md` | ✅ |

### Checkpoint C5 — Ensembles, Bayes, HMM

| # | Check | Target | Status |
| :---: | :--- | :--- | :---: |
| C5.1 | Champion quality gate | ROC-AUC **≥ 0.85**; specificity **≥ 0.60** at sensitivity 0.95 | ❌ |
| C5.2 | RF OOB vs CV | OOB AUC within **0.02** of grouped-CV AUC | ❌ |
| C5.3 | Naive Bayes filler detector | Precision **≥ 0.85** at recall ≥ 0.70 | ❌ |
| C5.4 | Bayes optimal vs MAP | Bayes optimal **no worse** than MAP on validation accuracy | ❌ |
| C5.5 | GMM agreement | ARI **≥ 0.4** with judge labels, or open judge-threshold review | ✅ |
| C5.6 | First non-generative compressor | HMM (or tagger + gate) on 500 val prompts: median savings **≥ 15%**, preservation **≥ 90%** on ≥ 1 target, and beats best single operator at equal preservation | ❌ |

---

# Phase 6 — Unsupervised Structure of Prompts (Unit 4)

**Goal:** discover prompt intent families without labels, find clusters where compression is
dangerous, and use them to stratify training and evaluation.
**Weeks:** 12–13 · **Depends on:** 2, 5 · **Unlocks:** 8, 11, 12

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 6.1 | Prompt embeddings | bge-small 384-d for all train prompts | Cached matrix + id index; raises on missing id | ✅ |
| 6.2 | K-means | k-means++, k ∈ 4..30; elbow + silhouette | `reports/figures/p6_kmeans_k.png` | ✅ |
| 6.3 | Bisecting K-means | Same k range; SSE and cluster-size balance vs K-means | Comparison table | ✅ |
| 6.4 | Hierarchical vs non-hierarchical | Written comparison of assumptions, cost, output | `reports/p6_clustering.md` §1 | ✅ |
| 6.5 | Agglomerative clustering | Ward / average / complete linkage on a 2,000-prompt sample | Dendrograms | ✅ |
| 6.6 | Divisive clustering | Top-down recursive bisection; tree compared with 6.5 | Figure + cophenetic correlation | ✅ |
| 6.7 | **K-means as a special case of EM** | GMM-EM with shared spherical covariance σ²I; anneal σ → 0; responsibilities become hard | `docs/derivations/kmeans_em.md` + assignment agreement vs K-means | ✅ |
| 6.8 | Clusters vs categories | ARI, NMI, contingency table against the 6 hand-defined categories | Table | ✅ |
| 6.9 | Per-cluster risk | Oracle savings and degradation rate per cluster; high-risk clusters become eval slices in 12.3 | `reports/p6_cluster_risk.md` | ✅ |

### Checkpoint C6 — Structure Discovered and Used

| # | Check | Target | Status |
| :---: | :--- | :--- | :---: |
| C6.1 | Clustering quality | Silhouette **≥ 0.10** at the chosen k | ❌ |
| C6.2 | K-means ⇔ EM | Assignment agreement **≥ 98%** at the smallest σ | ✅ |
| C6.3 | Actionable finding | ≥ 1 high-risk cluster (degradation rate ≥ 1.5× average) tracked as an eval slice | ❌ |

---

# Phase 7 — GPU Onboarding (RTX 4090 Workers)

**Goal:** verify each GPU worker end to end and replace every VRAM estimate with a measurement,
before any queued GPU job runs. The RTX 4500 Ada, if connected later, repeats 7.2–7.4 only.
**Weeks:** 3–4 · **Depends on:** 0 · **Unlocks:** all LLM rows of 1–2, 8, 10 (rewriter), 11 (quality path)

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 7.1 | Remote access | SSH to `worker-1` / `worker-2`; repo cloned; `.env` provisioned out-of-band | Contributor can run `pytest` remotely | ✅ |
| 7.2 | Driver + CUDA | Driver and CUDA runtime compatible with the pinned PyTorch build on each worker | `nvidia-smi` shows a 24 GB card; versions recorded in `docs/hardware.md` | ✅ |
| 7.3 | GPU environment | Python ≥ 3.11 venv; `pip install -e ".[dev,gpu]"`; `dvc pull`; MLflow pointed at shared server | Import test for torch/vllm/peft/trl/bitsandbytes | ✅ |
| 7.4 | Verification script | `python -m terser.hardware.verify`: device name, memory, capability (8, 9), bf16 matmul benchmark + parity with fp32 | Report in `docs/checkpoints/C7/hardware.md` | ✅ |
| 7.5 | vLLM smoke test | Qwen2.5-0.5B-Instruct generates 32 tokens | Logged | ✅ |
| 7.6 | Serving measurement | Llama-3.1-8B-Instruct bf16 and Qwen2.5-32B-AWQ, `gpu_memory_utilization=0.90`, max len 4096: peak VRAM, tokens/s at batch 1 and 16 | VRAM table updated | ✅ |
| 7.7 | Training measurement | Qwen2.5-0.5B full FT and 1.5B LoRA, seq 1024, micro-batch 4, grad checkpointing: peak VRAM, step time | VRAM table updated | ✅ |
| 7.9 | GPU test suite | All `@pytest.mark.gpu` tests | Green | ✅ |

### Checkpoint C7 — GPU Ready 🚧

| # | Check | Target | Status |
| :---: | :--- | :--- | :---: |
| C7.1 🚧 | Environment | Verification script all green on ≥ 1 worker | ✅ |
| C7.2 | Memory plan | VRAM table in [Compute Allocation](#compute-allocation-cpu-local-open-models-and-the-rtx-4500-ada) partly measured; revised 32B queue configuration and remaining workloads still estimated ([C7 details](docs/checkpoints/C7/README.md)) | ❌ partial |
| C7.4 | Queue | Local dry-run orders all three jobs without dependency errors; declared estimates pass validation against the configured 22 GB ceiling. Revised 32B estimate is UNVERIFIED pending planned RTX 4090 re-measurement ([follow-up](docs/checkpoints/C7/README.md#follow-up)); not a full pass | ❌ partial |

---

# Phase 8 — Deep Learning: CNN Span Scorer and Generative Rewriter (Unit 4)

**Goal:** a convolutional span-importance scorer and a small fine-tuned generative rewriter that,
together with the quality gate and fallback, reach the main quality/savings gate.
**Weeks:** 14–17 · **Depends on:** 2, 5, 6, 7 · **Unlocks:** 10, 11

### 8.1 Deep Learning Foundations

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 8.1.1 | Introduction write-up | Representation learning; why learned features beat Phase 3 hand features for span importance; decoder-only transformer overview | `docs/derivations/deep_learning_intro.md` | ✅ |
| 8.1.2 | Convolution operation | Manual 1-D convolution (kernel, stride, padding, output length) | Equals `nn.Conv1d` on fixture input (test) | ✅ |

### 8.2 TextCNN Span Scorer

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 8.2.1 | Architecture | Token embeddings (d = 384) → parallel Conv1d k = 3 and k = 5, 128 filters each, 'same' padding → per-token KEEP head + pooled prompt-compressibility head | `models/neural/textcnn.py` | ✅ |
| 8.2.2 | **Parameter calculation** | Conv k=3: 384×3×128+128 = 147,584 · Conv k=5: 384×5×128+128 = 245,888 · token head 256+1 = 257 · prompt head 256+1 = 257 · **total 393,986** | Equals `sum(p.numel())` in a test; derivation in docs | ✅ |
| 8.2.3 | **Max vs average pooling** | Prompt head on global max, global avg, and concatenated (head becomes 512+1 = 513 params); 3 seeds each | Table + interpretation ("any droppable region" vs "how much is droppable") | ✅ |
| 8.2.4 | Comparison | TextCNN vs HMM vs GBM span taggers on identical folds | McNemar + table | ✅ |

### 8.3 Generative Rewriter (Supervised Fine-Tuning)

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 8.3.1 | SFT data | (casual prompt, oracle preserved rewrite) from train split; one target-agnostic rewriter (C1.5: tokenizers agree) | HF dataset in DVC | ✅ |
| 8.3.2 | Tiny-config smoke test | Full training loop on a tiny model on CPU | CI test passes; GPU jobs queued | ✅ |
| 8.3.3 | Rung 1: Qwen2.5-0.5B full FT | bf16, AdamW, grad checkpointing | Model registered; val report | ✅ |
| 8.3.4 | Rung 2: Qwen2.5-1.5B LoRA | r = 16, α = 32, bf16. Escalate to 3B LoRA, then 7B QLoRA (NF4), only if the previous rung fails C8.3 | Same, per rung trained | ✅ |
| 8.3.7 | Decoding safeguards | Never longer than input on T; fall back to original if longer | Property test | ✅ |
| 8.3.8 | Inference-time quality gate | Champion gate rejects risky rewrites ⇒ lighter rewrite or original | Integration test | ✅ |
| 8.3.9 | Baselines | Original · best operator · HMM compressor · LLMLingua-2 · zero-shot Qwen2.5-32B-Instruct-AWQ "rewrite concisely" | Same val table | ✅ |
| 8.3.10 | Failure analysis | 100 degraded cases hand-categorised: dropped constraint / tone / example / changed meaning / other | `reports/p8_failures.md` | ✅ |

### Checkpoint C8 — Deep Learning

| # | Check | Target | Status |
| :---: | :--- | :--- | :---: |
| C8.1 | TextCNN parameters | Matches documented count exactly | ✅ |
| C8.2 | TextCNN tagging | Token macro-F1 **≥ C3.4 best + 0.05**, or documented reason | ❌ |
| C8.3 🚧 | **Rewriter main gate (val split)** | Median savings **≥ 25%** on every target; preservation **≥ 92%** overall and **≥ 85%** per category (gate + fallback on) | ❌ |
| C8.4 | vs zero-shot large open LLM | At equal preservation (±1 pp): savings within **5 pp** of zero-shot Qwen2.5-32B-AWQ while **≥ 10× lower** rewrite latency on the same GPU | ❌ |
| C8.5 | vs LLMLingua-2 | Higher preservation at equal or better savings | ❌ |

---

# Phase 9 — Reinforcement Learning: Tabular Q-Learning over Operators (Unit 4)

**Goal:** treat rewriting as a sequential decision problem (which operator next, or stop) and learn
a policy tabularly, using the quality gate as a cheap reward model. Policy optimisation of the
neural rewriter is a [stretch goal](#stretch-goals-never-block-a-checkpoint).
**Weeks:** 17–18 · **Depends on:** 5 · **Unlocks:** 12

### 9.1 MDP Formulation

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 9.1.1 | RL derivation | Agent = rewriter, environment = draft + tokenizer + gate; policy π(a \| s), V^π, Q^π; Bellman expectation and optimality forms; Q-learning update derived from optimality | `docs/derivations/rl.md` | ✅ |
| 9.1.2 | Environment | States (remaining filler bucket, constraint count, length bucket, category, ops applied); actions `F H D I L C S` or `STOP` | `rl/env.py` with tests | ✅ |
| 9.1.4 | Rewards | On `STOP`: Δtokens% − λ·1[degraded] (λ = 100); per step −0.01 | Reward unit tests | ✅ |

### 9.2 Tabular Q-Learning

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 9.2.1 | Q-learning from scratch | ε-greedy; quality gate as cheap reward model (no LLM calls during learning) | Learning curve | ✅ |
| 9.2.2 | Hyperparameter sweep | α ∈ {0.05, 0.1, 0.3}, γ ∈ {0.9, 0.99}, ε schedule | Table | ✅ |
| 9.2.3 | Learned policy | Greedy operator order per category | `reports/p9_policy.md` | ✅ |
| 9.2.4 | Judge verification | Greedy policy on 300 val prompts with the real judge | Savings and preservation table | ✅ |

### Checkpoint C9 — Reinforcement Learning

| # | Check | Target | Status |
| :---: | :--- | :--- | :---: |
| C9.1 | Tabular convergence | Q-table change **< 1e-3** over last 1,000 episodes | ❌ |
| C9.2 | Tabular policy value | Judge-verified savings **≥** best fixed operator order at equal preservation | ❌ |

---

# Phase 10 — Hyperparameter Optimisation (Unit 4)

**Goal:** a formal comparison of grid, random and Bayesian optimisation, and tuned champions
for the gate and the rewriter.
**Weeks:** 18–20 (grid search already used in Phase 3) · **Depends on:** 3, 4, 5, 8 · **Unlocks:** 11, 12

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 10.1 | HPO module | `terser.hpo` wrapping grid / random / Optuna TPE / GP samplers; MLflow parent-child runs | Module + tests | ✅ |
| 10.2 | Method study | GBM quality gate: 60-point grid vs random, TPE and GP at 60 trials × 5 repeats. Random and TPE also on the Phase 4 MLP (depth 1–3, width {64..512}, dropout, lr, weight decay) | Best-so-far curves logged to MLflow | ✅ |
| 10.5 | Method comparison | Grid vs Random vs Bayesian at equal budget | `reports/figures/p10_hpo_methods.png` + report | ✅ |
| 10.7 | Rewriter LoRA tuning | Rank, α, lr, warmup, epochs; objective = savings s.t. preservation ≥ 0.92; median pruner | Tuned model registered | ❌ |
| 10.8 | Test-set isolation audit | Data-access logs show HPO touched only train/val | Automated check green | ✅ |

### Checkpoint C10 — Tuning

| # | Check | Target | Status |
| :---: | :--- | :--- | :---: |
| C10.1 | Method comparison | 3 methods × 5 repeats at equal budget, plot + table | ✅ |
| C10.2 | Gate improvement | Tuned AUC **≥ untuned + 0.01**, or documented plateau | ✅ |
| C10.3 | Rewriter improvement | Still passes C8.3 with **≥ 1 pp** more savings, or documented plateau | ❌ |
| C10.4 | No test leakage | 10.8 green | ✅ |

---

# Phase 11 — MLOps: Pipelines, Serving, CI/CD/CT, Monitoring (Unit 4)

**Goal:** Terser as a production system, with reproducible pipelines, versioned models, a
two-tier inference service with net-savings accounting, automated retraining, and drift and
quality monitoring.
**Weeks:** 20–23 · **Depends on:** 5, 6, 8 (10 for champions) · **Unlocks:** 12, 13

### 11.1 Lifecycle, Data and Feature Pipelines

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 11.1.1 | MLOps overview ADR | Lifecycle diagram + Terser-specific production challenges: judge cost, tokenizer drift, silent over-compression, latency budget, training/serving skew | `docs/decisions/ADR-0010-mlops.md` | ✅ |
| 11.1.2 | Full DVC pipeline | `collect → casualise → candidates → respond → judge → label → features → split → stratify` | `dvc repro` end to end | ✅ |
| 11.1.3 | Shared feature code | One `terser.features` used offline and online | Parity test: 100/100 prompts identical | ✅ |
| 11.1.4 | Data validation | pandera schemas: nulls, token-ratio ranges, category balance | Fails on a corrupted fixture | ✅ |

### 11.2 Experiment Tracking and Model Versioning

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 11.2.2 | Model registry | Stages `None → Staging → Production → Archived` for gate, tagger, rewriter | Registry populated | ✅ |
| 11.2.3 | Lineage | Each version records git SHA, DVC hash, config, checkpoint IDs passed | Any production model traceable | ✅ |

### 11.3 Deployment, Serving and Inference Systems

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 11.3.1 | API | FastAPI `POST /rewrite {prompt, target, mode: fast\|quality}` → `{rewrite, tokens_before, tokens_after, gate_score, fallback_used}` | OpenAPI docs + request/response examples | ✅ |
| 11.3.2 | Fast path | Span tagger + gate exported to ONNX Runtime | Runs with no GPU | ✅ |
| 11.3.3 | Quality path | Rewriter on vLLM with continuous batching + gate + fallback | Endpoint live on a GPU worker | ✅ |
| 11.3.4 | Caching | LRU keyed on normalised prompt hash + target | Warm-hit latency measured | ✅ |
| 11.3.5 | Timeout fallback | Quality path over budget ⇒ fast-path result | Fault-injection test | ✅ |
| 11.3.6 | Net-savings accounting | Per request: price_T × (tokens_before − tokens_after) − rewriter_cost | Logged per request | ✅ |
| 11.3.7 | Containers | `terser-cpu` and `terser-gpu` (CUDA base) images; docker-compose | `docker compose up` serves both paths | ✅ |

### 11.4 CI / CD / CT

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 11.4.1 | CI | Every PR: lint, types, unit tests, CPU smoke training, `dvc repro --dry`, feature parity | Median ≤ 15 min | ❌ |
| 11.4.2 | CD | Tag `v*`: build + scan images → push to GHCR → start the container in CI → smoke + latency tests | Release images published by the workflow | ❌ |
| 11.4.3 | CT | On drift alert or a new data batch: judge labels → retrain gate/tagger (CPU) → queue rewriter retrain (GPU) → evaluate → register as Staging if checkpoints pass | One full cycle completed | ✅ |
| 11.4.4 | Promotion guard | CT refuses promotion when a synthetic regression is injected | Test green | ✅ |

### 11.5 Monitoring, Drift and Observability

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 11.5.1 | Metrics | Prometheus: RPS, p50/p95/p99 latency per path, fallback rate, gate rejections, tokens saved, net savings, GPU util / VRAM | Exposed at `/metrics` | ✅ |
| 11.5.2 | Input drift | Evidently daily: embedding drift (PSI / MMD on PCA components), length and category distribution | Report generated | ✅ |
| 11.5.3 | Tokenizer drift | Re-count fixture set with current tokenizer versions; alert on any change | Alert fires on a pinned-version swap | ✅ |
| 11.5.4 | Quality canary | Nightly: a fixed 200-prompt canary set goes through the deployed service and the judge; rolling preservation | Metric on dashboard | ✅ |
| 11.5.5 | Dashboard and alerts | Grafana; alert at rolling preservation < 92% | Dashboard live | ✅ |
| 11.5.6 | Structured logging | Request IDs; hashes by default, raw text only if consented | Log schema doc | ✅ |

### 11.6 ML Orchestration

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 11.6.1 | Prefect flows | `data_flow`, `train_classical_flow`, `train_rewriter_flow` (submits to `gpu_queue`), `evaluate_flow`, `promote_flow`, `drift_flow` | Flows registered | ✅ |
| 11.6.2 | Reliability | Retries, cached LLM steps, failure notifications | Injected failure recovers | ✅ |

### Checkpoint C11 — Production Readiness

| # | Check | Target | Status |
| :---: | :--- | :--- | :---: |
| C11.1 | End-to-end orchestration | `data → train_classical → evaluate → promote` runs unattended on a fresh clone (CPU) | ✅ |
| C11.2 | Fast-path latency | p95 **≤ 150 ms** at 20 RPS | ✅ |
| C11.3 | Quality-path latency | p95 **≤ 400 ms** at 5 RPS, prompts ≤ 256 tokens, one 24 GB Ada GPU | ❌ |
| C11.4 | Throughput | Quality path ≥ 20 RPS sustained 30 min, no OOM | ❌ |
| C11.5 | Fallback correctness | Rewriter crash/timeout returns original prompt **100%** of the time | ✅ |
| C11.6 | Drift detection | Synthetic shift (only `code` prompts, or tokenizer version swap) alerts within one cycle | ✅ |
| C11.7 | CT guard | Synthetic regression correctly refused promotion | ✅ |

---

# Phase 12 — Final Evaluation, Ablations and Benchmark Release

**Goal:** a single, statistically rigorous answer to "does Terser save tokens without losing
quality?", on the untouched test split.
**Week:** 24 · **Depends on:** 8, 9, 10, 11 · **Unlocks:** 13

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 12.1 | Test-split unlock | Logged, one-time event | Access log entry | ❌ |
| 12.2 | Master results table | Every system (original, best operator, Q-learning policy, HMM, GBM tagger+gate, TextCNN+gate, SFT rewriter, LLMLingua-2, zero-shot LLM) × savings × preservation × net savings × latency | `reports/final_results.md` | ❌ |
| 12.3 | Stratified reporting | Per target × category × high-risk cluster, and separately on the real-prompt slice (2.1.8) | Tables | ❌ |
| 12.4 | Statistical rigour | 95% bootstrap CIs (1,000 resamples); ≥ 3 seeds; McNemar between top systems | Error bars on every headline number | ❌ |
| 12.6 | **Ablation matrix** | Remove gate / fallback / HPO one at a time; generative rewriter vs operator-only | 4-row ablation table | ❌ |
| 12.7 | Net savings by prompt length | Savings minus rewriter cost across length percentiles | Figure | ❌ |
| 12.8 | Error taxonomy | 200 degraded test cases categorised | `reports/final_errors.md` | ❌ |
| 12.9 | Qualitative gallery | Best and worst casual → rewrite → output pairs | `reports/gallery.md` | ❌ |
| 12.10 | Compute accounting | Generated tokens, GPU-hours, energy per phase; GPU-seconds per labelled pair (estimate vs actual) | Table | ❌ |
| 12.11 | Benchmark release | Prompts, candidates, labels, rubrics, eval code (licence permitting) | Release tag + data card | ❌ |
| 12.12 | Reproduction | Second contributor reproduces headline numbers from a fresh clone | Within CI bounds | ❌ |

---

# Phase 13 — Documentation, Report and Viva Preparation

**Goal:** everything needed to understand, reproduce, present and defend the project.
**Weeks:** 23–24 · **Depends on:** 12 · **Unlocks:** delivery

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 13.1 | README | One-paragraph pitch, architecture diagram (casual prompt → fast/quality path → gate → target LLM, with training loop), headline results table with CIs, `docker compose up` quickstart, links to report and benchmark. Start it now and update at every checkpoint | A reader understands the project in 60 seconds | ❌ |
| 13.2 | Syllabus mapping doc | Unit 1–4 → phase / file / figure links | `docs/syllabus_map.md` | ✅ |
| 13.4 | Model cards | Gate, tagger, TextCNN, rewriter | Complete | ✅ |
| 13.6 | Final report | Problem, related work (LLMLingua / LLMLingua-2, Selective Context), method, experiments, results, ablations, negative results (C1.4, C1.5), limitations (creative prompts, multi-turn context, tokenizer drift, judge dependence, synthetic prompts), future work | `docs/report.md` | ❌ |
| 13.8 | Demo script | Live casual prompt → rewrite → side-by-side outputs + token counter | Rehearsed; cached fallback if GPU unavailable | ✅ |
| 13.9 | Viva prep | ID3 gain, bias–variance, ROC, backprop, SVM dual, AdaBoost weights, EM, Viterbi, K-means ⇔ EM, Bellman / Q-update, CNN parameters, LoRA rank | `docs/viva.md` | ✅ |

---

## Cross-Phase Dependency Map

| Phase | Depends On | Unlocks |
| :--- | :--- | :--- |
| 0 | — | everything |
| 1 | 0 | 2 |
| 2 | 1, 7 (all LLM rows) | 3, 4, 5, 6, 8 |
| 3 | 2 | 4, 5, 6 |
| 4 | 3 | 5, 8, 10 |
| 5 | 3, 4 | 6, 8, 9, 11 |
| 6 | 2, 5 | 8, 11, 12 |
| 7 | 0 | 1 (LLM rows), 2, 8, 10, 11 |
| 8 | 2, 5, 6, 7 | 10, 11 |
| 9 | 5 | 12 |
| 10 | 3, 4, 5, 8 | 11, 12 |
| 11 | 5, 6, 8 (10 for champions) | 12, 13 |
| 12 | 8, 9, 10, 11 | 13 |
| 13 | 12 | delivery |

Hard gates on the critical path: **C7.1** (GPU environment, needed for all labelled data) → **C2.3–C2.5** (judge trustworthy) → **C2.7** (go/no-go) → **C8.3** (rewriter) → **S1–S5** (final).

---

## Syllabus Coverage Matrix

| Unit | Requirement | Where It Lives | Status |
| :--- | :--- | :--- | :---: |
| 1 | Introduction to ML, ML models | 1.1.1 (sub-problem taxonomy) | ❌ |
| 1 | Concept learning: hypotheses, version space | 1.3.2–1.3.7 | ❌ |
| 1 | Inductive bias | 1.3.8, 3.3.6 | ✅ |
| 1 | Accuracy, Precision, Recall, Sensitivity, Specificity | 3.2.1 | ✅ |
| 1 | ROC, AUC | 3.2.2, 3.2.3 | ✅ |
| 1 | Bias–variance decomposition and trade-off | 3.6.1 | ✅ |
| 1 | Cross-validation strategies | 3.2.4–3.2.7 | ✅ |
| 1 | Decision trees: ID3 | 3.3.1 | ✅ |
| 1 | Decision boundary for decision trees (x–y axis) | 3.3.5 | ✅ |
| 1 | Hypothesis search and inductive bias in trees | 3.3.6 | ✅ |
| 1 | Overfitting and solutions | 3.3.3, 3.3.4 | ✅ |
| 1 | Continuous values | 3.1.4, 3.3.2 | ✅ |
| 1 | Logistic regression | 3.4, 3.5.3 (boundary) | ✅ |
| 1 | KNN classification and regression | 3.5.1, 3.5.2 | ✅ |
| 1 | Decision boundary for KNN | 3.5.3 | ✅ |
| 2 | Perceptron | 4.1.1 | ✅ |
| 2 | Multi-layer networks and backpropagation | 4.1.2–4.1.5, 4.1.7 | ✅ |
| 2 | Activations: Step, Sigmoid, Tanh, ReLU | 4.2.1 | ✅ |
| 2 | Optimizers: GD, SGD, Momentum, Adagrad, Adam | 4.3.1 | ✅ |
| 2 | SVM margin maximisation, primal, Lagrangian dual, dual solution | 4.4.1, 4.4.2, 4.4.4 | ✅ |
| 2 | Hard margin and soft margin (classification) | 4.4.5, 4.4.6 | ✅ |
| 2 | Kernels: linear (derivation), polynomial | 4.4.3, 4.4.7 | ✅ |
| 3 | Combining weak learners | 5.1.1 | ✅ |
| 3 | Gradient boosting | 5.1.5, 5.1.6 | ✅ |
| 3 | Random forest | 5.1.3 | ✅ |
| 3 | Bayes theorem | 5.2.1 | ✅ |
| 3 | Maximum likelihood | 5.2.3, 5.3.2 | ✅ |
| 3 | Bayes optimal classifier | 5.2.4 | ✅ |
| 3 | Naive Bayes | 5.2.3 | ✅ |
| 3 | Expectation maximisation and GMM | 5.2.5, 5.2.6 | ✅ |
| 3 | Hidden Markov Models | 5.3 | ✅ |
| 4 | Hierarchical vs non-hierarchical clustering | 6.4 | ✅ |
| 4 | Agglomerative and divisive clustering | 6.5, 6.6 | ✅ |
| 4 | K-means, bisecting K-means | 6.2, 6.3 | ✅ |
| 4 | K-means as special case of EM | 6.7 | ✅ |
| 4 | RL: agent, environment, states, actions, rewards, policy, value functions | 9.1.1, 9.1.2, 9.1.4 | ✅ |
| 4 | Bellman equation and Q-learning | 9.1.1, 9.2 | ✅ |
| 4 | Introduction to deep learning | 8.1.1, 8.3 | ✅ |
| 4 | Convolution operation and CNNs | 8.1.2, 8.2.1 | ✅ |
| 4 | CNN parameter calculation, max and avg pooling | 8.2.2, 8.2.3 | ✅ |
| 4 | Hyperparameter tuning: Grid, Random, Bayesian optimisation | 3.6.4, 10.2, 10.5 | ✅ |
| 4 | MLOps overview, ML lifecycle and production challenges | 0.3.4, 11.1.1 | ✅ |
| 4 | Data and feature pipelines | 2.1.7, 11.1 | ✅ |
| 4 | Experiment tracking and model versioning | 0.2.4, 0.2.5, 11.2 | ✅ |
| 4 | Model deployment and serving | 11.3 | ✅ |
| 4 | CI/CD/CT for ML systems | 0.2.9, 11.4 | ❌ |
| 4 | Inference systems | 11.3.2–11.3.6 | ✅ |
| 4 | Monitoring, drift and observability | 11.5 | ✅ |
| 4 | ML orchestration | 11.6 | ✅ |

---

## Headline Success Criteria

Evaluated once, on the frozen **test** split, in Phase 12.

| # | Criterion | Target | Status |
| :---: | :--- | :--- | :---: |
| S1 | Median token savings on target tokenizers | ≥ 30% (95% CI lower bound ≥ 25%) | ❌ |
| S2 | Quality preservation, overall | ≥ 95% | ❌ |
| S3 | Quality preservation, worst task category | ≥ 90% | ❌ |
| S4 | Net savings after rewriter cost | Positive at the 25th-percentile prompt length for every target | ❌ |
| S5 | Per-target robustness | S1–S3 hold for **each** target model separately with the single target-agnostic rewriter | ❌ |
| S6 | Judge validity | Judge–human κ ≥ 0.60 (C2.4) holds on a fresh 100-pair test sample | ❌ |
| S7 | Fast-path latency | p95 ≤ 150 ms | ❌ |
| S8 | Quality-path latency on one 24 GB Ada GPU | p95 ≤ 400 ms | ❌ |
| S9 | Independent reproduction | Headline numbers reproduced from a fresh clone by a second contributor | ❌ |
| S10 | All four syllabus units demonstrably covered | Every row of the Syllabus Coverage Matrix ✅ | ❌ |

S1–S3 are also reported, not gated, on the real-prompt slice (2.1.8). A large gap between the
synthetic test split and the real slice is itself a headline finding.

Missing a target is reported, never hidden: the final report states which targets were met,
which were not, and why.

---

## Overall Progress

| Phase | Owner | Tasks | Checks | Done | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| 0 — Foundations | Guru | 22 | 4 | 26 | ✅ |
| 1 — Framing and Concept Learning | Guru | 23 | 5 | 22 | ❌ |
| 2 — Data Pipeline and Judge | Vivian | 21 | 9 | 23 | ❌ |
| 3 — Metrics, Trees, LR, KNN | Vivian | 31 | 7 | 36 | ❌ |
| 4 — ANN and SVM | Vivian | 16 | 5 | 20 | ❌ |
| 5 — Boosting, Bayes, GMM, HMM | Vivian | 16 | 6 | 17 | ❌ |
| 6 — Clustering | Vivian | 9 | 3 | 10 | ❌ |
| 7 — GPU Onboarding | Vivian | 8 | 3 | 9 | ❌ |
| 8 — CNN and Rewriter | Vivian | 14 | 5 | 15 | ❌ |
| 9 — Reinforcement Learning | Vivian | 7 | 2 | 7 | ❌ |
| 10 — HPO | Vivian | 5 | 4 | 7 | ❌ |
| 11 — MLOps | Vivian | 25 | 7 | 28 | ❌ |
| 12 — Final Evaluation | Vivian | 11 | — | 0 | ❌ |
| 13 — Documentation | Guru | 6 | — | 4 | ❌ |
| **Total** | | **214** | **60** | **224** | ❌ |

Stretch rows are not counted.

### Owners

| Owner | Hardware | Phases | Rows | Done | Remaining |
| :--- | :--- | :--- | :---: | :---: | :---: |
| Vivian | 2 × RTX 4090 (24 GB each) | 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 | 214 | 172 | 42 |
| Guru | 1 × RTX 4500 Ada (24 GB) | 0, 1, 13 | 60 | 52 | 8 |

Each phase has one owner, who runs it end to end, including its checkpoint.

---

## Stretch Goals (never block a checkpoint)

Attempt only after C8.3 passes and Phase 11 is on track. Outside the syllabus; valuable on a resume
only if they work, so each ends in a report whether or not it succeeds.

### S-9.3 Policy Optimisation of the Neural Rewriter (GRPO)

| # | Task | Detail | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :---: |
| 9.3.1 | GRPO setup | Start from C8.3 model; LoRA; group size 4–8; KL penalty to SFT reference | Tiny CPU smoke test + queued job | ❌ |
| 9.3.2 | Reward | `tokens_saved_pct × gate_pass` | Reward logged per step | ❌ |
| 9.3.3 | Reward-hacking audits | Every N steps sample rewrites → 32B judge in a time-shared vLLM slot, never colocated with training | Audit log | ❌ |
| 9.3.4 | Degenerate-output monitor | Empty / truncated / non-natural-language rate | Metric logged | ❌ |
| 9.3.5 | RL vs SFT comparison | Same val table as C8.3 | `reports/p9_rl_vs_sft.md` | ❌ |

| # | Check | Target | Status |
| :---: | :--- | :--- | :---: |
| C9.3 | No reward hacking | Gate precision on RL rewrites within **5 pp** of its precision on Phase 2 data | ❌ |
| C9.4 | No degenerate outputs | **< 1%** on 1,000 samples | ❌ |
| C9.5 | RL improves on SFT | Savings **≥ SFT + 3 pp**; preservation not lower by > 1 pp overall or > 2 pp in any category. Else SFT stays champion | ❌ |

### Other stretch items

- Connect the RTX 4500 Ada as a third GPU (repeat 7.2–7.4).
- A small judge (Llama-3.1-8B AWQ) replacing the 32B judge for audits, only with κ ≥ 0.70 against it.
- A hosted demo (e.g. a Hugging Face Space running the CPU fast path).

---

# How to Contribute

### Contributors and the compute they bring

| Contributor | Machines | Reviews |
| :--- | :--- | :--- |
| Guru Bharadwaj (`guru-bharadwaj20`) | Dev laptop (Windows 11, CPU only); NVIDIA RTX 4500 Ada workstation (WSL2), connected 2026-10-01 | Repository owner |
| Vivian Sobers (`vivianedwardbangalore@gmail.com`) | Dev workstation (Fedora 44, CPU only); `worker-1` and `worker-2`, one RTX 4090 each, administered by Vivian | GPU onboarding (Phase 7), serving and queue work |

Machine details, access and the measurements taken on them:
[docs/hardware.md § 1.1](docs/hardware.md#11-vivians-gpu-workers). Credentials for the GPU workers
are shared out-of-band and never committed.

### Getting started

```bash
git clone https://github.com/guru-bharadwaj20/Terser.git && cd Terser
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1    Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"            # CPU
pip install -e ".[dev,gpu]"        # on a GPU machine (RTX 4090 workers, or the RTX 4500 Ada)
cp .env.example .env               # HF_TOKEN, TERSER_LOCAL_LLM_URL (never commit)
pre-commit install
pytest -m "not gpu and not api"
```

`bash scripts/ci_local.sh` runs the same checks as the CI workflow (ruff, black, mypy, DVC dry
run, test-set audit, feature parity, tokenizer drift, CPU tests) when GitHub Actions is
unavailable or before a push.

On the RTX 4090 workers the system interpreter is Python 3.10, below this project's minimum.
Create the environment with an explicit 3.11 or newer interpreter (`uv venv --python 3.11` on
`worker-1`, `/usr/bin/python3.11 -m venv .venv` on `worker-2`).

### Picking up work

1. Choose a ❌ row whose phase dependencies are ✅ (see the dependency map).
2. Open an issue titled with the row id, e.g. `3.3.1 ID3 from scratch`, and assign yourself.
3. Branch `phase<N>/<row-id>-<topic>`, e.g. `phase3/3.3.1-id3`.
4. In the same PR that satisfies the Definition of Done, flip the row to ✅ and update
   [Overall Progress](#overall-progress). If the row maps to a syllabus requirement, also update the
   [Syllabus Coverage Matrix](#syllabus-coverage-matrix).
5. For `GPU` rows before Phase 7 is ✅: merge the code with its CPU smoke test and a `gpu_queue/` job.
   The row stays ❌ until the job has run on a GPU worker.

### Checkpoint protocol

1. When all task rows in a phase are ✅, open a `checkpoint` issue, e.g. `C3 — Classical baselines`.
2. Write `docs/checkpoints/C<N>/README.md`: each check, its **measured value**, pass/fail, MLflow
   run links, DVC data hash, git SHA, seeds, figures, deviations.
3. A reviewer **who did not do the work** re-runs at least one headline number.
4. Record the decision: `PASS`, `PASS WITH CONDITIONS` (conditions become issues due before the next
   checkpoint), or `FAIL`. 🚧 checks allow no conditional pass.
5. On `FAIL`, dependent phases do not start. Changing a target requires an ADR.

### Experiment standards

- Every run logs to MLflow with tags `phase`, `row`, `task`, `hardware` (`cpu` / `rtx4090` / `rtx4500ada`), `data_hash`, `git_sha`.
- Report mean ± std over ≥ 3 seeds (≥ 5 for Phase 4 studies).
- **Never tune on test.** Code touching `data/processed/test*` outside `terser.eval.final` fails CI.
- Every performance table has per-category rows. An aggregate alone is rejected in review.
- Every new model is compared to the current champion **and** the "original prompt" baseline
  (0% savings, 100% preservation), with McNemar or paired bootstrap for claimed wins.
- Log generated tokens and GPU-hours per experiment.
- Negative results are committed as reports. They are results.

### Coding standards

- Python ≥ 3.11, type hints throughout `src/`, `mypy src` clean (settings in `pyproject.toml`); `black` + `ruff` via pre-commit.
- The formatter is **black** (line length 100); ruff is the linter only. `make format` runs
  `ruff check --fix` then `black`. Do not run `ruff format`: its output differs from black's
  on some constructs, and CI runs `black --check .`.
- From-scratch implementations (ID3, LR, KNN, perceptron, backprop, SVM dual, AdaBoost, NB, EM/GMM,
  HMM forward/Viterbi/Baum-Welch, K-means, bisecting K-means, Q-learning) live in
  `src/terser/models/**/scratch_*.py`, each with a derivation link and a test against a library or
  hand-computed example.
- No hard-coded `"cuda"`: use `terser.hardware.get_device(cfg)`.
- Hydra configs only, no magic numbers. Notebooks are exploration only, never imported, outputs stripped.
- Pytest markers: `gpu`, `api`, `slow`. Coverage ≥ 80% for `tokenizers`, `eval`, `features`, `judge`.

### Commits and pull requests

Conventional Commits, referencing the row id:

```
feat(judge): 2.3.3 position-swapped pairwise judging
exp(phase4): 4.3.6 optimizer comparison at equal gradient budget
docs(derivations): 4.4.2 SVM Lagrangian dual
chore(gpu_queue): 8.3.4 add sft-qwen1.5b-lora job
```

PR checklist:

- [ ] Row id(s) and phase; status flipped to ✅ only if the Definition of Done is met
- [ ] Compute tag(s); for `GPU` rows, CPU smoke test + `gpu_queue/` job
- [ ] Tests added; `pytest -m "not gpu and not api"` passes
- [ ] MLflow links for every reported number; per-category results
- [ ] Generated tokens / GPU-hours for this PR's experiments
- [ ] No secrets, no raw personal data, no test-split access
- [ ] Docs / derivations / progress tables updated

Review: ≥ 1 approval. Changes to `judge/`, `eval/`, data splits or checkpoint targets need **2**. Rebase-merge into `main`.


### Data, privacy and LLM usage

- Only datasets whose licences permit research use (and redistribution, for the benchmark), recorded in `data/SOURCES.md`.
- No real user prompts without explicit consent. Production stores hashes and metrics by default; raw text only when opted in, retained ≤ 30 days.
- PII scrubbing before `data/interim/`. Secrets in `.env` only; CI uses repository secrets.
- All LLM calls go through `terser.llm` against local open-source models; per-phase token caps in `configs/budget.yaml` bound compute; all responses disk-cached.
- Training on model outputs must comply with each open model's licence (Llama 3.1 Community Licence, Qwen2.5 licence). Record the check in ADR-0003 before Phase 8.

### Risk Register

| Risk | Likelihood | Impact | Mitigation | Watched by |
| :--- | :---: | :---: | :--- | :--- |
| Judge disagrees with humans (open 32B judge weaker than frontier APIs) | Medium–High | Critical | Task metrics first, position swap, rubric iteration, self-preference control (2.2.5), GMM sanity check | C2.3–C2.5, C5.5 |
| Synthetic-prompt circularity (rewriter only undoes LLM-injected verbosity) | High | High | Real-prompt slice (2.1.8), S1–S3 reported on it | C2.7, 12.3 |
| Too little compression headroom | Low–Medium | Critical | Go/no-go with scope-narrowing plan | C2.7 |
| GPU contention on shared workers | Medium | High | Two workers, exclusive booked slots, GPU queue, response caching, CPU stand-in smoke runs (2.2.4) | C7 |
| 24 GB VRAM insufficient | Low | Medium | Model ladder 0.5B → 7B QLoRA, grad checkpointing, no colocated serving + training | C7.2, C8.3 |
| Silent over-compression (dropped constraints) | High | High | Constraint features, gate at sensitivity 0.95, fallback, error taxonomy, shadow judging | C8.3, C11.6 |
| Rewriter overhead wipes savings | Medium | High | Net-savings metric, CPU fast path, skip short prompts (3.5.4) | S4, C11.2 |
| Tokenizer drift | Medium | Medium | Tokenizer fixture alert, CT retraining | 11.5.3 |
| LLM compute overrun | Medium | Medium | Token caps, response caching | 12.10 |
| Paraphrase leakage across splits | Medium | High | Source-grouped splits, automated leakage tests | C2.2, C3.2 |

### Glossary

| Term | Meaning |
| :--- | :--- |
| Target (T) | The LLM receiving the rewritten prompt; determines the tokenizer |
| Casual prompt | Naturally phrased, possibly redundant user instruction |
| Candidate | A proposed rewrite of a casual prompt |
| Oracle rewrite | Shortest candidate still judged `preserved` |
| Preserved / Degraded | Label from task metrics + position-swapped pairwise judge |
| Quality gate | Classifier on (original, rewrite); positive class = degraded |
| Span tagger | Model labelling tokens KEEP / DROP / REPLACE |
| Fast path / Quality path | Tagger + gate (CPU-capable) vs generative rewriter + gate (GPU) |
| Net savings | Tokens saved on T, priced, minus the rewriter's own cost |
| Hard gate 🚧 | Checkpoint that blocks dependent work with no conditional pass |
| GPU queue | Job specs in `gpu_queue/` waiting for a GPU worker |
