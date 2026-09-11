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
| `CPU` | Runs on the dev laptop. Start immediately |
| `LLM` | Needs text generation from a **local open-source model** behind an OpenAI-compatible server (`terser.llm` provider `local`). No paid APIs. Results that feed labels, metrics or reports come only from GPU-served models (vLLM on the RTX 4500 Ada); on the CPU laptop these rows may only be smoke-tested with tiny models |
| `GPU` | Needs the **NVIDIA RTX 4500 Ada (24 GB)**. Not connected yet: write the code, smoke-test on CPU with a tiny config, and queue the job |
| `CPU→GPU` | Runs at reduced scale on CPU now, re-run at full scale once the GPU is connected |
| 🚧 | **Hard gate**: failing it blocks every dependent phase, with no conditional pass |

**Status rule:** a row flips from ❌ to ✅ only when its *Definition of Done* is satisfied **and**
the artifact exists in the repository (code + test + logged experiment or report).

**Gate rule:** a phase is complete only when every task row is ✅ **and** its checkpoint table
is ✅. A checkpoint target may only be changed by an ADR in `docs/decisions/` that explains why
the original target was wrong. "We did not reach it" is not a valid reason.

---

## Table of Contents

| # | Phase | Theme | Syllabus Unit | Weeks | Compute |
| :---: | :--- | :--- | :--- | :---: | :--- |
| 0 | Foundations and Repository Scaffolding | Infra | Unit 4 (MLOps basics) | 1 | `CPU` |
| 1 | Problem Framing and Concept Learning | Framing | Unit 1 | 2–3 | `CPU` `LLM` |
| 2 | Data Generation Pipeline and Judge Validation | Data | — | 3–6 | `LLM` `CPU→GPU` |
| 3 | Metrics, Cross-Validation, Decision Trees, Logistic Regression, KNN | Modeling | Unit 1 | 6–8 | `CPU` |
| 4 | Artificial Neural Networks and SVMs | Modeling | Unit 2 | 8–10 | `CPU` |
| 5 | Boosting, Bayesian Learning, GMM+EM, HMM | Modeling | Unit 3 | 10–12 | `CPU` `LLM` |
| 6 | Unsupervised Structure of Prompts | Modeling | Unit 4 | 12–13 | `CPU` |
| 7 | GPU Onboarding (RTX 4500 Ada) | Infra | — | as soon as connected (critical path) | `GPU` |
| 8 | Deep Learning: CNN Span Scorer and Generative Rewriter | Deep Learning | Unit 4 | 14–17 | `GPU` `LLM` |
| 9 | Reinforcement Learning with a Quality Gate | RL | Unit 4 | 17–19 | `CPU` `GPU` `LLM` |
| 10 | Hyperparameter Optimisation | Tuning | Unit 4 | 19–20 | `CPU` `GPU` |
| 11 | MLOps: Pipelines, Serving, CI/CD/CT, Monitoring | MLOps | Unit 4 | 20–23 | `CPU` `GPU` |
| 12 | Final Evaluation, Ablations and Benchmark Release | Science | All | 24 | all |
| 13 | Documentation, Report and Viva Preparation | Delivery | All | 23–24 | `CPU` |

Supporting sections: [Compute Allocation](#compute-allocation-cpu-local-open-models-and-the-rtx-4500-ada) ·
[Timeline](#timeline) · [Cross-Phase Dependency Map](#cross-phase-dependency-map) ·
[Syllabus Coverage Matrix](#syllabus-coverage-matrix) · [Headline Success Criteria](#headline-success-criteria) ·
[Overall Progress](#overall-progress) · [How to Contribute](#how-to-contribute)

---

## Compute Allocation: CPU, Local Open Models, and the RTX 4500 Ada

### Hardware profile

| Resource | Details | Status |
| :--- | :--- | :---: |
| Dev laptop (Windows 11) | CPU only; all classical ML, tokenization, pipelines, CI | Available |
| **NVIDIA RTX 4500 Ada Generation** | 24 GB GDDR6 ECC, Ada Lovelace (compute capability 8.9), bf16 supported, ~210 W. Attached later via remote access | ❌ not connected |
| Local open-model serving | OpenAI-compatible servers only: Ollama / llama.cpp on the laptop for tiny models (Qwen2.5-0.5B/1.5B, smoke tests), vLLM on the RTX 4500 Ada for all real generation and judging. **No hosted API keys are available or assumed** | CPU: available · GPU: ❌ |

### GPU job queue (until the card is connected)

GPU work is **never a blocker**. It is written, unit-tested on CPU with `configs/model/tiny.yaml`, and
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
| Candidate generation + pairwise judge (Qwen2.5-32B-Instruct-AWQ; fallback Qwen2.5-14B-Instruct-AWQ) | 2+ | `GPU` | vLLM AWQ int4 | ~19 GB weights (32B) / ~9 GB (14B) + KV cache | Exclusive slot, never colocated with target serving; 32B fit at max len 4096 is an estimate verified in 7.6 |
| Target LLM outputs (Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct) | 2, 8+ | `GPU` | vLLM bf16, `gpu_memory_utilization=0.90` | ~16 GB weights + KV cache | Before GPU: only the CPU smoke-test stand-in above |
| Small judge (Llama-3.1-8B-Instruct AWQ) | 2, 8+ | `GPU` | vLLM AWQ int4 | ~6 GB + KV | Self-preference control (2.2.5) and fast audits (8.4); replaces the 32B judge only after C8.7 |
| Sentence embeddings (bge-small / e5-base) | 3+ | `CPU→GPU` | fp32 / fp16 | < 2 GB | CPU fine up to ~50k texts |
| All classical models (ID3, LR, KNN, SVM, NB, GMM, HMM, RF, GBM, k-means) | 3–6 | `CPU` | — | 0 | Never on GPU |
| Perceptron / MLP / optimizer study | 4 | `CPU` | fp32 | 0 | Small models |
| TextCNN span scorer | 8 | `CPU→GPU` | fp32 / bf16 | < 2 GB | GPU only for sweeps |
| Rewriter SFT: Qwen2.5-0.5B full fine-tune | 8 | `GPU` | bf16 + AdamW, grad checkpointing | ~12–16 GB | ~16 bytes/param for weights+grads+Adam (~8 GB) + activations |
| Rewriter SFT: Qwen2.5-1.5B / 3B LoRA | 8 | `GPU` | bf16 base, LoRA r=16 | ~8–16 GB | Main candidates |
| Rewriter SFT: 7B QLoRA | 8 | `GPU` | NF4 base + LoRA | ~10–18 GB at seq ≤ 1024 | Only if smaller rungs fail C8.3 |
| GRPO rewriter policy | 9 | `GPU` | 0.5B–1.5B LoRA | ~12–20 GB | Target and judge calls go to a time-shared vLLM slot, never colocated with training |
| Tabular Q-learning | 9 | `CPU` | — | 0 | |
| Bayesian HPO of rewriter / TextCNN | 10 | `GPU` | Optuna, 1 trial at a time | per trial | Median pruner |
| Production serving | 11 | `GPU` (+ `CPU` fast path) | vLLM / ONNX Runtime | 2–6 GB | Fast path must also run on CPU |

### Weekly GPU schedule (once connected)

| Slot | Use |
| :--- | :--- |
| Night (long, exclusive) | SFT / RL training from `gpu_queue/` |
| Day block A | vLLM serving of target / judge models for data generation and evaluation |
| Day block B | Short sweeps (TextCNN, HPO trials), latency benchmarks |
| Weekend | Full evaluation re-runs, checkpoint reports |

> **The GPU is on the critical path.** Without hosted APIs, every labelled example (target outputs,
> candidates, judge verdicts) is produced on the RTX 4500 Ada, so Phase 2 cannot finish before
> Phase 7. Until the card is connected: complete the CPU rows of Phase 1 (operators, concept-learning
> code, tokenizer economics on hand-written prompts), build and smoke-test every Phase 2 stage end to
> end with the CPU stand-in model, and write Phases 3–6 code against the smoke-test data. **If the GPU
> is still not connected by week 6**, also pull forward the CPU halves of Phase 9 (tabular
> Q-learning), Phase 10 (classical HPO on synthetic data) and Phase 11 (fast-path serving, CI/CD,
> monitoring). The schedule slips. The checkpoints are not lowered.

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
Phase 7                                          (as soon as the GPU is connected: critical path for Phase 2)
Phase 8                                                       ███████████████
Phase 9                                                                   ███████████
Phase 10                                                                          ███████
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

| # | Task | Detail | HW | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| 0.1.1 | Python environment | Python ≥ 3.11 venv, `pyproject.toml` with extras `dev` (CPU) and `gpu` (torch CUDA, vllm, peft, trl, bitsandbytes) | `CPU` | `pip install -e ".[dev]"` reproduces from scratch | ✅ |
| 0.1.2 | Core stack | numpy, pandas, scipy, scikit-learn, lightgbm, cvxopt, matplotlib | `CPU` | Import smoke test passes | ✅ |
| 0.1.3 | Tokenizer stack | `tiktoken` (`o200k_base`), HF `tokenizers` (Qwen2.5, Llama-3.1 via ungated mirror), optional Anthropic token counting | `CPU` | All four targets count a fixture string | ✅ |
| 0.1.4 | Determinism harness | `set_seed()` for random / numpy / torch / CUDA, `PYTHONHASHSEED` | `CPU` | Two runs of a dummy experiment give identical metrics | ✅ |
| 0.1.5 | Device abstraction | `terser.hardware.get_device(cfg)` with `device: auto`, no hard-coded `"cuda"` | `CPU` | `hardware=rtx4500ada` on CPU fails with a clear message; `auto` falls back to CPU | ✅ |
| 0.1.6 | Hardware profile doc | VRAM budget table above, GPU queue rules | `CPU` | `docs/hardware.md` written | ✅ |

### 0.2 Repository Layout

| # | Task | Detail | HW | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| 0.2.1 | Directory tree | `configs/ data/{raw,interim,processed,gold}/ docs/{decisions,checkpoints,derivations}/ notebooks/ src/terser/ gpu_queue/ tests/ reports/` | `CPU` | Tree created with `__init__.py` | ✅ |
| 0.2.2 | Package modules | `src/terser/{tokenizers,operators,concept,data,judge,features,models/{classical,neural,ensembles,clustering,rewriter,rl},eval,hpo,serving,monitoring,pipelines}` | `CPU` | Each imports cleanly | ✅ |
| 0.2.3 | Config system | Hydra configs incl. `configs/hardware/{cpu,rtx4500ada}.yaml`; no hard-coded paths or hyperparameters | `CPU` | Every script runs `python -m terser.x experiment=...` | ✅ |
| 0.2.4 | Experiment tracking | Local MLflow server + Model Registry; tags `phase`, `task`, `hardware`, `data_hash`, `git_sha` | `CPU` | Dummy run visible with params, metrics, artifacts | ✅ |
| 0.2.5 | Data versioning | DVC initialised with a local remote | `CPU` | `dvc status` clean | ✅ |
| 0.2.6 | Task runner | `tasks.ps1` (Windows) + `Makefile` mirror: `data`, `train-classical`, `eval`, `test` | `CPU` | All targets execute | ✅ |
| 0.2.7 | Pre-commit | black, ruff, mypy, nbstripout | `CPU` | Hook blocks a bad commit | ✅ |
| 0.2.8 | Test scaffold | pytest with markers `gpu`, `api`, `slow`; fixtures of 20 casual prompts | `CPU` | `pytest -m "not gpu and not api"` green | ✅ |
| 0.2.9 | CI | GitHub Actions: lint, types, CPU tests | `CPU` | CI green on a PR | ✅ |
| 0.2.10 | GPU queue | `gpu_queue/` schema + validator + `python -m terser.gpu_queue {validate,dry-run,run}` | `CPU` | Dry-run orders two sample jobs correctly | ✅ |

### 0.3 Project Governance

| # | Task | Detail | HW | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| 0.3.1 | Data card template | Source, licence, size, PII notes, bias notes | `CPU` | `docs/data_cards/_template.md` | ✅ |
| 0.3.2 | Model card template | Intended use, targets supported, metrics per category, limits | `CPU` | `docs/model_cards/_template.md` | ✅ |
| 0.3.3 | Naming convention | Runs `<phase>-<task>-<model>-<variant>-<seed>`; branches `phase<N>/<topic>` | `CPU` | `docs/conventions.md` | ✅ |
| 0.3.4 | ML lifecycle ADR | data → label → train → evaluate → register → deploy → monitor → retrain | `CPU` | `docs/decisions/ADR-0001-ml-lifecycle.md` | ✅ |
| 0.3.5 | Risk register | See [Risk Register](#risk-register) | `CPU` | `docs/risks.md` with mitigations | ✅ |
| 0.3.6 | LLM budget guard | `configs/budget.yaml` per phase; client refuses calls past cap (token caps bound local compute; USD caps only for priced providers); local OpenAI-compatible provider; disk cache keyed by model+params+prompt hash | `CPU` `LLM` | Test: cap reached ⇒ call refused; cached call costs 0 | ✅ |

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

| # | Task | Detail | HW | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| 1.1.1 | Sub-problem taxonomy | Quality gate (binary classification) · span tagger (sequence labelling KEEP/DROP/REPLACE) · savings predictor (regression) · intent discovery (clustering) · rewriter (conditional generation) · rewrite policy (RL) | `CPU` | `docs/decisions/ADR-0002-problem-framing.md` reviewed | ❌ |
| 1.1.2 | Task categories | `code`, `fact`, `reason`, `explain`, `creative`, `transform`, each with its quality measure (unit tests / EM-F1 / answer accuracy / rubric judge / pairwise judge / rubric + constraint checklist) | `CPU` | Category spec in `docs/categories.md` | ✅ |
| 1.1.3 | Positive-class convention | Quality gate positive = **degraded**, so sensitivity = catching degradations, specificity = not rejecting good rewrites | `CPU` | Recorded in ADR-0002 and used by `terser.eval` | ✅ |

### 1.2 Rewrite Operators (the hypothesis-space building blocks)

| # | Task | Detail | HW | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| 1.2.1 | `F` drop filler | "hey so um, basically" → "" (conservative: "like" as comparison kept) | `CPU` | Unit tests incl. no-op on efficient prompts | ✅ |
| 1.2.2 | `H` drop hedging / politeness | "could you maybe please", "thanks in advance" → "" | `CPU` | Same | ✅ |
| 1.2.3 | `D` deduplicate restated requirements | "keep it short… keep it brief" → one sentence; never touches code blocks | `CPU` | Same | ✅ |
| 1.2.4 | `I` imperative conversion | "I was wondering if you could explain" → "Explain" | `CPU` | Same | ✅ |
| 1.2.5 | `L` list-ify constraints | ≥ 3 constraint sentences → bullet list | `CPU` | Same | ✅ |
| 1.2.6 | `C` compress meta-talk | "I'm going to give you some text and I want you to" → "" | `CPU` | Same | ✅ |
| 1.2.7 | `S` tokenizer-aware substitution | Phrase table ("in order to" → "to"); picks the alternative that is cheapest **on target T's tokenizer in context** | `CPU` | Test shows `S` choosing differently for two tokenizers on at least one fixture, or documents that it never does | ✅ |
| 1.2.8 | Canonical application order | `C H F I D S L`; `apply_ops(text, codes, target)` is order-independent in its input | `CPU` | Property test | ✅ |

### 1.3 Concept Learning

| # | Task | Detail | HW | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| 1.3.1 | Instance representation | Discrete attributes: `length{short,medium,long}`, `has_constraints`, `has_payload`, `tone_cue`, `filler_density{low,high}` + operator flags `op_F … op_S` | `CPU` | `terser.concept.attributes` with tests | ✅ |
| 1.3.2 | Hypotheses | Conjunctive constraint vectors with `?` (any) and `∅` (none), e.g. `<long, ?, no, no, high, op_F=yes, …>` ⇒ "preserved" | `CPU` | Documented in `docs/derivations/concept_learning.md` | ✅ |
| 1.3.3 | Find-S | Most specific hypothesis covering all positives | `CPU` | Reproduces Mitchell's EnjoySport result `<Sunny, Warm, ?, Strong, ?, ?>` | ✅ |
| 1.3.4 | Candidate-Elimination | S and G boundaries; collapse detection that reports the offending example and the examples it conflicts with | `CPU` | Reproduces EnjoySport S/G; version space of size 6 | ✅ |
| 1.3.5 | Version space enumeration | All hypotheses between S and G | `CPU` | Enumerator matches brute force on a 5-attribute toy space | ✅ |
| 1.3.6 | Pilot labels | **150 prompts (25/category)** × operator subsets (7 singletons + `FHC`, `FHCI`, `FHCID`, `FHCIDS`, all) → target output → position-swapped pilot judge → human review of every label | `LLM` | `data/pilot/labels.jsonl` + human-reviewed CSV | ❌ |
| 1.3.7 | Version space per category | Run Candidate-Elimination for each of the 6 categories on 1.3.6 | `CPU` | `reports/p1_version_spaces.md` with S, G, size or collapse diagnosis | ❌ |
| 1.3.8 | Inductive bias write-up | What the conjunctive language cannot express (e.g. "drop hedging *unless* tone matters in creative prompts") and why this motivates Phases 3–5 | `CPU` | Section in `reports/p1_version_spaces.md` | ✅ |

### 1.4 Pilot Tokenizer Economics

| # | Task | Detail | HW | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| 1.4.1 | Pilot prompt set | **500 casual prompts**: hand-written seeds plus casualisation of clean instructions (Qwen2.5 on vLLM; before the GPU, hand-written seeds only), balanced across categories | `CPU` `LLM` | `data/pilot/prompts.jsonl` + data card | ✅ |
| 1.4.2 | Per-operator savings | Δtokens % for each operator and operator stack on gpt-4o (tokenizer only), qwen2.5, llama-3.1 | `CPU` | `reports/p1_tokenizer_study.{csv,md}` | ✅ |
| 1.4.3 | Operator applicability | Share of prompts each operator changes, per category | `CPU` | Table in report | ✅ |
| 1.4.4 | Cross-tokenizer agreement | Spearman ρ of per-(prompt, operator) savings for every tokenizer pair | `CPU` | Table in report | ✅ |

### Checkpoint C1 — Framing and Operators

| # | Check | Target | Status |
| :---: | :--- | :--- | :---: |
| C1.1 | ADR-0002 merged | Reviewed by ≥ 1 other contributor | ❌ |
| C1.2 | Operator safety | Every operator is a no-op on 50 already-efficient prompts (100%) | ✅ |
| C1.3 | Version space | Non-empty for **≥ 4 of 6** categories; each collapse explained by named conflicting examples | ❌ |
| C1.4 | Savings exist | Median best-single-operator savings **≥ 10%** on the pilot set | ❌ |
| C1.5 | Tokenizer dependence | Spearman ρ < 0.9 for ≥ 1 tokenizer pair **or** documented finding that tokenizers agree (Phase 8 then ships a single target-agnostic rewriter) | ✅ |

---

# Phase 2 — Data Generation Pipeline and Judge Validation

**Goal:** a labelled dataset of (casual prompt, candidate rewrite, target, token counts,
preserved/degraded), produced by a judge that has been **validated against humans**. Nothing
downstream can be measured without it.
**Weeks:** 3–6 · **Depends on:** 1 · **Unlocks:** 3, 4, 5, 6, 8

### 2.1 Prompt Collection

| # | Task | Detail | HW | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| 2.1.1 | Source datasets | Permissively licensed instruction sets (e.g. Dolly, OpenAssistant), HumanEval/MBPP prompts, GSM8K problems | `CPU` | `data/SOURCES.md` + one data card per source | ✅ |
| 2.1.2 | Licence audit | Research use and redistribution terms; unknown ⇒ non-redistributable | `CPU` | `reports/license_audit.md` | ✅ |
| 2.1.3 | PII scrubbing | PII detector over all text before `data/interim/` | `CPU` | Scrub report; 0 detected PII in a 500-row manual sample | ❌ |
| 2.1.4 | Casualisation | LLM rewrites clean instructions in 5 styles: rambling, hedging, texting, over-polite, repetitive; clean version kept as reference | `LLM` | ≥ 6,000 casual prompts | ❌ |
| 2.1.5 | Deduplication | MinHash, Jaccard ≥ 0.8 removed; paraphrase groups kept whole | `CPU` | Duplicate report | ✅ |
| 2.1.6 | **Source-grouped splits** | Train 70 / val 15 / test 15 by source prompt; test frozen end of week 4 and hash-locked | `CPU` | Automated test: 0 source prompts shared across splits | ✅ |
| 2.1.7 | DVC stages | `collect → casualise → dedupe → split` | `CPU` | `dvc repro` rebuilds from cache | ❌ |

### 2.2 Candidates and Target Outputs

| # | Task | Detail | HW | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| 2.2.1 | Candidate generation | 6 per prompt: 3 rewrites from Qwen2.5-32B-Instruct-AWQ at low/medium/high aggressiveness, 2 operator stacks, 1 LLMLingua-2 baseline | `GPU` `LLM` `CPU` | `candidates` stage output | ❌ |
| 2.2.2 | Token counts | Every prompt and candidate on every target tokenizer | `CPU` | Counts table, 0 missing | ❌ |
| 2.2.3 | Target outputs | Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct on vLLM bf16, temperature 0 (creative: n = 3 samples) | `GPU` `LLM` | `respond` stage output | ❌ |
| 2.2.4 | Pipeline smoke run (pre-GPU) | `respond → judge → label` on 50 prompts with Qwen2.5-0.5B/1.5B via Ollama on CPU, through the same `terser.llm` client | `CPU` `LLM` | Stages run end to end; every output tagged `stand_in: true` and excluded from labels | ❌ |
| 2.2.5 | Judge self-preference control | Second judge Llama-3.1-8B-Instruct on 1,000 pairs; compare verdict agreement separately for Qwen-target and Llama-target outputs | `GPU` `LLM` | Report: agreement gap between target families ≤ 5 pp, or mitigation documented | ❌ |

### 2.3 Quality Judging

| # | Task | Detail | HW | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| 2.3.1 | Task metrics | `code`: unit-test pass rate · `fact`: EM / token F1 · `reason`: final-answer accuracy | `CPU` | `terser.judge.metrics` with tests | ✅ |
| 2.3.2 | Rubrics | Per-category YAML: coverage, correctness, depth, constraint adherence, tone | `CPU` | `src/terser/judge/rubrics/*.yaml` | ✅ |
| 2.3.3 | Pairwise judge with position swap | Qwen2.5-32B-Instruct-AWQ (fallback 14B-AWQ) on vLLM judges (original-output, rewrite-output) in both A/B orders | `GPU` `LLM` | `judge` stage output | ❌ |
| 2.3.4 | Label definition | `preserved` iff task metric not worse **and** judge says tie-or-rewrite-better in **both** orders | `CPU` | `label` stage + tests | ✅ |
| 2.3.5 | Human annotation | 2 annotators label **300 pairs** (50/category), blind to which is the rewrite | `CPU` | `data/gold/human_300.jsonl` + guideline doc | ❌ |
| 2.3.6 | Agreement analysis | Human–human κ, judge–human κ per category, position-bias rate | `CPU` | `reports/p2_judge_validation.md` | ❌ |

### 2.4 Headroom and Cost

| # | Task | Detail | HW | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| 2.4.1 | Oracle rewrite | Shortest candidate still `preserved`, per prompt and target | `CPU` | `oracle` table | ❌ |
| 2.4.2 | Headroom report | Oracle savings per target × category; degraded share among candidates | `CPU` | `reports/p2_headroom.md` | ❌ |
| 2.4.3 | Compute report | Generated tokens and GPU-hours per stage; GPU-seconds per labelled pair; estimate vs actual | `CPU` | `reports/p2_cost.md` | ❌ |

### Checkpoint C2 — Dataset, Judge, Headroom

| # | Check | Target | Status |
| :---: | :--- | :--- | :---: |
| C2.1 | Dataset size | ≥ 6,000 casual prompts, ≥ 800 per category, < 1% near-duplicates | ❌ |
| C2.2 | Leakage | 0 source prompts shared across splits | ❌ |
| C2.3 🚧 | Human–human agreement | Cohen's κ **≥ 0.60** | ❌ |
| C2.4 🚧 | Judge–human agreement | κ **≥ 0.60** overall and **≥ 0.50** in every category | ❌ |
| C2.5 🚧 | Judge as degradation detector | Sensitivity **≥ 0.85**, specificity **≥ 0.80** vs human labels | ❌ |
| C2.6 | Position bias | A/B-order disagreement **≤ 10%** | ❌ |
| C2.7 🚧 | **Go / no-go: oracle savings** | Median **≥ 25%** on ≥ 2 targets; ≥ 15% in ≥ 5 of 6 categories. On failure: negative-result report and scope narrowing before continuing | ❌ |
| C2.8 | Class balance | Degraded share among candidates between **15% and 60%** | ❌ |
| C2.9 | Reproducibility and cost | `dvc repro` gives an identical dataset hash from cached responses; spend within ±20% of estimate | ❌ |

---

# Phase 3 — Metrics, Cross-Validation, Decision Trees, Logistic Regression, KNN (Unit 1)

**Goal:** the evaluation backbone every later phase reuses, plus interpretable baselines for the
**quality gate**, **span tagger** and **savings predictor**, with full treatment of
bias–variance and overfitting.
**Weeks:** 6–8 · **Depends on:** 2 · **Unlocks:** 4, 5, 6

### 3.1 Features

| # | Task | Detail | HW | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| 3.1.1 | Alignment labels | Token-level KEEP/DROP/REPLACE from diff-aligning each prompt with its oracle rewrite | `CPU` | `features/alignment.py` + tests on 20 hand-aligned pairs | ❌ |
| 3.1.2 | Span features | Token cost on T, self-information under GPT-2 small, stopword ratio, relative position, POS, filler-lexicon hit, negation / number / constraint keyword flags | `CPU` | `features/span.py`; feature table in DVC | ❌ |
| 3.1.3 | Pair features | Token ratio, max self-information removed, # negations / numbers / constraint keywords removed, embedding cosine, category one-hot | `CPU` | `features/pair.py`; feature table in DVC | ❌ |
| 3.1.4 | Continuous-value handling | Standardisation for LR/KNN; C4.5 threshold search for trees; equal-frequency bins for ID3 | `CPU` | Documented per model | ❌ |

### 3.2 Evaluation Backbone

| # | Task | Detail | HW | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| 3.2.1 | Metric suite | Accuracy, Precision, Recall, **Sensitivity**, **Specificity**, F1, confusion matrix, per-category breakdown | `CPU` | `eval/metrics.py`, matches scikit-learn to 1e-9 | ❌ |
| 3.2.2 | ROC / AUC | ROC curves, AUC; PR curves and AP (degraded is the minority) | `CPU` | `eval/curves.py` + figure template | ❌ |
| 3.2.3 | Threshold rule | Maximise specificity subject to **sensitivity ≥ 0.95** | `CPU` | `eval/threshold.py` + test | ❌ |
| 3.2.4 | **Grouped stratified k-fold** | `StratifiedGroupKFold(k=5)`, group = source prompt, stratify = category × label | `CPU` | `eval/cv.py`; out-of-fold predictions as the shared unit | ❌ |
| 3.2.5 | Leakage demonstration | Naive KFold vs grouped CV on the same model | `CPU` | Gap recorded in `reports/p3_cv_leakage.md` | ❌ |
| 3.2.6 | Nested CV | Outer 5 / inner 3 for unbiased estimates while tuning | `CPU` | Used by 3.6 | ❌ |
| 3.2.7 | Leave-one-category-out | Generalisation to unseen task types | `CPU` | Per-category table | ❌ |
| 3.2.8 | Statistical comparison | McNemar on shared out-of-fold predictions; bootstrap CIs | `CPU` | `eval/significance.py` | ❌ |

### 3.3 Decision Trees

| # | Task | Detail | HW | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| 3.3.1 | **ID3 from scratch** | Entropy, information gain, recursive split on discretised span features | `CPU` | Matches a hand-computed toy example; within 2 F1 of scikit-learn `criterion="entropy"` at equal depth | ❌ |
| 3.3.2 | Continuous values | C4.5-style threshold search on midpoints between class changes | `CPU` | Test on a 1-D separable fixture | ❌ |
| 3.3.3 | Overfitting study | Train vs val F1 for depth 1–30 and min-samples-leaf sweep | `CPU` | `reports/figures/p3_tree_overfit.png` | ❌ |
| 3.3.4 | Overfitting solutions | Pre-pruning, **reduced-error pruning**, **rule post-pruning**, cost-complexity pruning | `CPU` | Comparison table: leaves, val F1, train–val gap | ❌ |
| 3.3.5 | Decision boundary (x–y) | Axis-aligned regions on `self_information` vs `token_cost` | `CPU` | `reports/figures/p3_tree_boundary.png` | ❌ |
| 3.3.6 | Hypothesis search and inductive bias | Greedy information-gain search (preference bias for short trees) vs Phase 1 conjunctive restriction bias | `CPU` | `reports/p3_tree_bias.md` | ❌ |
| 3.3.7 | Rule export | Pruned tree as human-readable rules ("drop is unsafe when a negation is removed and…") | `CPU` | `reports/p3_tree_rules.md` | ❌ |

### 3.4 Logistic Regression

| # | Task | Detail | HW | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| 3.4.1 | From-scratch batch GD | Sigmoid, log-loss, gradient | `CPU` | Coefficients within 1e-3 of scikit-learn | ❌ |
| 3.4.2 | Regularisation | L1 / L2 / elastic-net, C sweep | `CPU` | Grid results table | ❌ |
| 3.4.3 | Calibration | Reliability diagram, Brier score, ECE | `CPU` | `reports/figures/p3_lr_calibration.png` | ❌ |
| 3.4.4 | Linear decision boundary | Same 2-D plane as 3.3.5 | `CPU` | Figure | ❌ |
| 3.4.5 | Coefficient interpretation | Odds ratios on standardised features | `CPU` | `reports/p3_lr_coefficients.md` | ❌ |

### 3.5 K-Nearest Neighbours

| # | Task | Detail | HW | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| 3.5.1 | KNN classification | Quality gate; k ∈ {1,3,5,7,11,15,25}; Euclidean / Manhattan / cosine; uniform / distance weighting | `CPU` | Grid table under 3.2.4 protocol | ❌ |
| 3.5.2 | **KNN regression** | Predict a prompt's achievable savings % from its k nearest training prompts | `CPU` | MAE, R² table | ❌ |
| 3.5.3 | KNN decision boundary | k = 1, 5, 25 on 2-D PCA refit inside each fold | `CPU` | `reports/figures/p3_knn_boundaries.png` | ❌ |
| 3.5.4 | "Skip short prompts" rule | Use 3.5.2 to skip rewriting when predicted savings < overhead | `CPU` | Rule + expected net-savings curve | ❌ |

### 3.6 Bias–Variance and Grid Search

| # | Task | Detail | HW | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| 3.6.1 | **Bias–variance decomposition** | 200 bootstrap resamples; squared loss (KNN regression), 0-1 loss decomposition (classifiers); tree depth ∈ {2,4,8,16,∞}, k ∈ {1,3,5,15,51} | `CPU` | `reports/figures/p3_bias_variance.png` | ❌ |
| 3.6.2 | Bias–variance trade-off | Identify the minimum-error point and explain it | `CPU` | `reports/p3_bias_variance.md` | ❌ |
| 3.6.3 | Learning curves | Train size 10% → 100% for all three models | `CPU` | Figure + diagnosis (high bias vs high variance) | ❌ |
| 3.6.4 | Grid search | Tree depth × min-samples; KNN k × metric × weighting; LR penalty × C, inside nested CV | `CPU` | Best params recorded in `configs/` | ❌ |
| 3.6.5 | Baselines | Majority class, stratified random, "always preserved" | `CPU` | Baseline row in every table | ❌ |

### Checkpoint C3 — Evaluation Backbone and Classical Baselines

| # | Check | Target | Status |
| :---: | :--- | :--- | :---: |
| C3.1 | Metric implementations | Match scikit-learn to 1e-9 on 20 random fixtures | ❌ |
| C3.2 | Leakage demonstration | Grouped-CV AUC lower than naive-KFold AUC; gap recorded | ❌ |
| C3.3 | Quality gate (best of Tree / LR / KNN) | ROC-AUC **≥ 0.75**; specificity **≥ 0.40** at sensitivity 0.95 | ❌ |
| C3.4 | Span tagger (best of Tree / LR) | Token macro-F1 **≥ 0.65** | ❌ |
| C3.5 | Savings predictor (KNN regression) | MAE **≤ 10 pp**, R² **≥ 0.3** | ❌ |
| C3.6 | Overfitting control | ≥ 1 pruning method closes the tree's train–val gap by **≥ 50%** | ❌ |
| C3.7 | Figures | Tree, LR, KNN (3 k values) boundaries on identical axes; bias–variance and learning curves present | ❌ |

---

# Phase 4 — Artificial Neural Networks and SVMs (Unit 2)

**Goal:** learned decision functions for the quality gate and span tagger, with from-scratch
implementations verified against libraries, plus controlled activation and optimizer studies.
**Weeks:** 8–10 · **Depends on:** 3 · **Unlocks:** 5, 8, 10

### 4.1 Perceptron and Multi-Layer Networks

| # | Task | Detail | HW | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| 4.1.1 | Perceptron from scratch | Step activation; filler vs non-filler spans | `CPU` | Converges on the linearly separable subset; fails to converge on the full set (logged) | ❌ |
| 4.1.2 | Pair embedding input | `[e_o, e_r, \|e_o − e_r\|, e_o ⊙ e_r]` from 384-d bge-small ⇒ **1536-d** | `CPU` | Cached `data/features/pair_emb.npy` + id index | ❌ |
| 4.1.3 | NumPy MLP with backprop | Forward, backward, update; no autograd | `CPU` | Gradient check vs finite differences, relative error **< 1e-6**; matches `torch.autograd` | ❌ |
| 4.1.4 | PyTorch MLP quality gate | 1536 → 256 → 64 → 1 | `CPU` | Scored under the 3.2.4 protocol | ❌ |
| 4.1.5 | Parameter calculation | 1536×256+256 = 393,472 · 256×64+64 = 16,448 · 64×1+1 = 65 · **total 409,985** | `CPU` | Equals `sum(p.numel())` in a test | ❌ |
| 4.1.6 | Architecture search | 1–3 hidden layers × widths {64,128,256,512} | `CPU` | Table | ❌ |
| 4.1.7 | Backprop derivation | Chain rule through sigmoid output + BCE; hidden-layer recursion | `CPU` | `docs/derivations/backprop.md` | ❌ |

### 4.2 Activation Functions

| # | Task | Detail | HW | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| 4.2.1 | Step | Straight-through estimator for the gradient | `CPU` | Val AUC, epochs to plateau, per-layer gradient norm; 5 seeds | ❌ |
| 4.2.2 | Sigmoid | + fraction of saturated units | `CPU` | Same | ❌ |
| 4.2.3 | Tanh | Same measurements | `CPU` | Same | ❌ |
| 4.2.4 | ReLU | + fraction of dead units | `CPU` | Same | ❌ |
| 4.2.5 | Activation report | Mechanism tied to observed behaviour | `CPU` | `reports/p4_activations.md` | ❌ |

### 4.3 Optimizers (equal gradient-evaluation budget, 5 seeds)

| # | Task | Detail | HW | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| 4.3.1 | Full-batch GD | lr ∈ {0.1, 0.03, 0.01} | `CPU` | Loss curves mean ± std | ❌ |
| 4.3.2 | SGD (mini-batch 64) | lr ∈ {0.1, 0.03, 0.01} | `CPU` | Same | ❌ |
| 4.3.3 | SGD + Momentum | lr ∈ {0.03, 0.01}, β ∈ {0.9, 0.99} | `CPU` | Same | ❌ |
| 4.3.4 | Adagrad | lr ∈ {0.1, 0.01} | `CPU` | Same | ❌ |
| 4.3.5 | Adam | lr ∈ {1e-3, 3e-4}, β₁ = 0.9, β₂ = 0.999 | `CPU` | Same | ❌ |
| 4.3.6 | Optimizer report | Convergence speed vs generalisation, each tied to its update rule | `CPU` | `reports/figures/p4_optimizers.png` + `reports/p4_optimizers.md` | ❌ |

### 4.4 Support Vector Machines

| # | Task | Detail | HW | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| 4.4.1 | Margin and primal | Margin = 2/‖w‖; hard- and soft-margin primal | `CPU` | `docs/derivations/svm.md` §1–2 | ❌ |
| 4.4.2 | Lagrangian dual | Lagrangian, KKT conditions, dual problem, recovering w and b from support vectors | `CPU` | `docs/derivations/svm.md` §3–4 | ❌ |
| 4.4.3 | Linear kernel derivation | K(x, z) = xᵀz substituted into the dual | `CPU` | `docs/derivations/svm.md` §5 | ❌ |
| 4.4.4 | **Dual solver from scratch** | QP via `cvxopt` on span features | `CPU` | Same support-vector set as scikit-learn; ‖w − w_sk‖/‖w_sk‖ **< 1e-3** | ❌ |
| 4.4.5 | Hard margin | Linearly separable filler subset | `CPU` | Support vectors plotted | ❌ |
| 4.4.6 | Soft margin | C ∈ {0.01, 0.1, 1, 10, 100}: margin width, # support vectors, val metrics | `CPU` | Table + figure | ❌ |
| 4.4.7 | Polynomial kernel | Degree 2, 3 × coef0 × C vs linear | `CPU` | Table: AUC, time, # SVs | ❌ |
| 4.4.8 | Decision surfaces | Linear vs polynomial on the Phase 3 2-D plane | `CPU` | `reports/figures/p4_svm_surfaces.png` | ❌ |
| 4.4.9 | Support vector analysis | Which (original, rewrite) pairs become SVs; are they the ambiguous ones? | `CPU` | `reports/p4_support_vectors.md` | ❌ |

### 4.5 Random Search

| # | Task | Detail | HW | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| 4.5.1 | Random search | 60 trials over MLP width, depth, dropout, lr, weight decay | `CPU` | Trials logged to MLflow | ❌ |
| 4.5.2 | Random vs grid | Best AUC vs number of trials at equal budget | `CPU` | `reports/figures/p4_random_vs_grid.png` | ❌ |

### Checkpoint C4 — Neural and Margin Models

| # | Check | Target | Status |
| :---: | :--- | :--- | :---: |
| C4.1 | Backprop correctness | Relative gradient error **< 1e-6** on all layers | ❌ |
| C4.2 | SVM dual correctness | Support-vector set equals scikit-learn's; weight error **< 1e-3** | ❌ |
| C4.3 | Studies complete | 4 activations × 5 seeds; 5 optimizers × 5 seeds at equal budget; written interpretation | ❌ |
| C4.4 | Better quality gate | Best MLP/SVM ROC-AUC **≥ C3.3 best + 0.03**, or documented reason (classical model stays champion) | ❌ |
| C4.5 | Worst category | AUC **≥ 0.70** in every category | ❌ |

---

# Phase 5 — Boosting, Bayesian Learning, GMM+EM, HMM (Unit 3)

**Goal:** the strongest tabular quality gate, Bayesian components for filler detection and judge
sanity-checking, and the **first end-to-end non-generative compressor** (an HMM tagger).
**Weeks:** 10–12 · **Depends on:** 3, 4 · **Unlocks:** 6, 8, 9, 11

### 5.1 Ensembles

| # | Task | Detail | HW | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| 5.1.1 | Combining weak learners | AdaBoost with depth-1 stumps from scratch | `CPU` | Matches scikit-learn within 0.01 AUC; training error vs rounds plot | ❌ |
| 5.1.2 | Hard-example analysis | Which pairs AdaBoost upweights most | `CPU` | `reports/p5_adaboost_hard.md` | ❌ |
| 5.1.3 | Random Forest | n_estimators × max_features × bootstrap | `CPU` | Table; OOB vs CV comparison | ❌ |
| 5.1.4 | Feature importance | Permutation importance: what makes a rewrite unsafe | `CPU` | Top-10 features documented | ❌ |
| 5.1.5 | Gradient Boosting | LightGBM: lr × n_estimators × max_depth with early stopping on grouped validation | `CPU` | Table | ❌ |
| 5.1.6 | SHAP per category | Explanation of gate decisions per task category | `CPU` | `reports/figures/p5_shap.png` | ❌ |
| 5.1.7 | Feature feedback | ≥ 1 importance finding turned into a new feature or operator rule | `CPU` | PR linked | ❌ |

### 5.2 Bayesian Learning

| # | Task | Detail | HW | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| 5.2.1 | Bayes theorem | P(filler \| phrase, context) worked from corpus counts | `CPU` | `docs/derivations/bayes.md` | ❌ |
| 5.2.2 | Maximum likelihood: token costs | MLE of per-phrase token-cost distributions per tokenizer; used as prior by operator `S` | `CPU` | Fitted parameters + goodness-of-fit | ❌ |
| 5.2.3 | Naive Bayes filler detector | Multinomial NB on 1–3-grams, Laplace smoothing; vs LR on the same features | `CPU` | PR curve + table | ❌ |
| 5.2.4 | Bayes optimal classifier | Phase 1 hypotheses as H; P(h \| D) from validation likelihood; posterior-weighted vote vs MAP hypothesis | `CPU` | `reports/p5_bayes_optimal.md` | ❌ |
| 5.2.5 | EM from scratch | E-step responsibilities, M-step means/variances/weights; log-likelihood monotone | `CPU` | Test asserts non-decreasing log-likelihood; matches scikit-learn GMM | ❌ |
| 5.2.6 | GMM on judge score deltas | 2-component GMM on (rewrite score − original score): unsupervised preserved/degraded split vs threshold labels | `CPU` | ARI reported; judge-threshold review issue opened if low | ❌ |

### 5.3 Hidden Markov Model Span Tagger

| # | Task | Detail | HW | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| 5.3.1 | Model definition | Hidden states {KEEP, DROP, REPLACE}; observations = discretised token class (filler-lexicon id × POS × cost bucket) | `CPU` | `models/classical/hmm.py` | ❌ |
| 5.3.2 | Supervised MLE | Initial, transition, emission matrices from alignment labels with smoothing | `CPU` | Rows sum to 1 (test) | ❌ |
| 5.3.3 | Forward algorithm | Sequence likelihood in log space | `CPU` | Matches brute-force enumeration on length-5 sequences | ❌ |
| 5.3.4 | Viterbi decoding | Most likely tag sequence → compressed prompt | `CPU` | Matches brute force on length-5 sequences | ❌ |
| 5.3.5 | Baum-Welch | EM on unlabelled casual prompts from supervised initialisation | `CPU` | F1 before vs after | ❌ |
| 5.3.6 | End-to-end HMM compressor | Viterbi tags → rewrite → Phase 2 judge on **500 val prompts** | `LLM` | `reports/p5_hmm_compressor.md` | ❌ |

### Checkpoint C5 — Ensembles, Bayes, HMM

| # | Check | Target | Status |
| :---: | :--- | :--- | :---: |
| C5.1 | Champion quality gate | ROC-AUC **≥ 0.85**; specificity **≥ 0.60** at sensitivity 0.95 | ❌ |
| C5.2 | RF OOB vs CV | OOB AUC within **0.02** of grouped-CV AUC | ❌ |
| C5.3 | Naive Bayes filler detector | Precision **≥ 0.85** at recall ≥ 0.70 | ❌ |
| C5.4 | Bayes optimal vs MAP | Bayes optimal **no worse** than MAP on validation accuracy | ❌ |
| C5.5 | GMM agreement | ARI **≥ 0.4** with judge labels, or open judge-threshold review | ❌ |
| C5.6 | First non-generative compressor | HMM (or tagger + gate) on 500 val prompts: median savings **≥ 15%**, preservation **≥ 90%** on ≥ 1 target, and beats best single operator at equal preservation | ❌ |

---

# Phase 6 — Unsupervised Structure of Prompts (Unit 4)

**Goal:** discover prompt intent families without labels, find clusters where compression is
dangerous, and use them to stratify training and evaluation.
**Weeks:** 12–13 · **Depends on:** 2, 5 · **Unlocks:** 8, 11, 12

| # | Task | Detail | HW | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| 6.1 | Prompt embeddings | bge-small 384-d for all train prompts | `CPU` | Cached matrix + id index; raises on missing id | ❌ |
| 6.2 | K-means | k-means++, k ∈ 4..30; elbow + silhouette | `CPU` | `reports/figures/p6_kmeans_k.png` | ❌ |
| 6.3 | Bisecting K-means | Same k range; SSE and cluster-size balance vs K-means | `CPU` | Comparison table | ❌ |
| 6.4 | Hierarchical vs non-hierarchical | Written comparison of assumptions, cost, output | `CPU` | `reports/p6_clustering.md` §1 | ❌ |
| 6.5 | Agglomerative clustering | Ward / average / complete linkage on a 2,000-prompt sample | `CPU` | Dendrograms | ❌ |
| 6.6 | Divisive clustering | Top-down recursive bisection; tree compared with 6.5 | `CPU` | Figure + cophenetic correlation | ❌ |
| 6.7 | **K-means as a special case of EM** | GMM-EM with shared spherical covariance σ²I; anneal σ → 0; responsibilities become hard | `CPU` | `docs/derivations/kmeans_em.md` + assignment agreement vs K-means | ❌ |
| 6.8 | Clusters vs categories | ARI, NMI, contingency table against the 6 hand-defined categories | `CPU` | Table | ❌ |
| 6.9 | Per-cluster risk | Oracle savings and degradation rate per cluster | `CPU` | `reports/p6_cluster_risk.md` | ❌ |
| 6.10 | Stratification update | Later sampling stratified by category × cluster | `CPU` | DVC `stratify` stage; `dvc repro` passes | ❌ |

### Checkpoint C6 — Structure Discovered and Used

| # | Check | Target | Status |
| :---: | :--- | :--- | :---: |
| C6.1 | Clustering quality | Silhouette **≥ 0.10** at the chosen k | ❌ |
| C6.2 | K-means ⇔ EM | Assignment agreement **≥ 98%** at the smallest σ | ❌ |
| C6.3 | Actionable finding | ≥ 1 high-risk cluster (degradation rate ≥ 1.5× average) tracked as an eval slice | ❌ |

---

# Phase 7 — GPU Onboarding (RTX 4500 Ada)

**Goal:** verify the card end to end and replace every VRAM estimate with a measurement, before
any queued GPU job runs.
**Weeks:** as soon as the GPU is connected (target ≤ week 13) · **Depends on:** 0 · **Unlocks:** all LLM rows of 1–2, 8, 9 (neural), 10 (neural), 11 (quality path)

| # | Task | Detail | HW | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| 7.1 | Remote access | VS Code Remote-SSH (or equivalent); repo cloned; `.env` provisioned out-of-band | `GPU` | Contributor can run `pytest` remotely | ❌ |
| 7.2 | Driver + CUDA | Driver and CUDA runtime compatible with the pinned PyTorch build | `GPU` | `nvidia-smi` shows RTX 4500 Ada, 24 GB | ❌ |
| 7.3 | GPU environment | `pip install -e ".[dev,gpu]"`; `dvc pull`; MLflow pointed at shared server | `GPU` | Import test for torch/vllm/peft/trl/bitsandbytes | ❌ |
| 7.4 | Verification script | `python -m terser.hardware.verify`: device name, memory, capability (8, 9), bf16 matmul benchmark + parity with fp32 | `GPU` | Report in `docs/checkpoints/C7/hardware.md` | ❌ |
| 7.5 | vLLM smoke test | Qwen2.5-0.5B-Instruct generates 32 tokens | `GPU` | Logged | ❌ |
| 7.6 | Serving measurement | Llama-3.1-8B-Instruct bf16, `gpu_memory_utilization=0.90`, max len 4096: peak VRAM, tokens/s at batch 1 and 16 | `GPU` | VRAM table updated | ❌ |
| 7.7 | Training measurement | Qwen2.5-0.5B full FT and 1.5B LoRA, seq 1024, micro-batch 4, grad checkpointing: peak VRAM, step time | `GPU` | VRAM table updated | ❌ |
| 7.8 | Thermal check | 30-minute sustained training step | `GPU` | Throughput drop < 5% | ❌ |
| 7.9 | GPU test suite | All `@pytest.mark.gpu` tests | `GPU` | Green | ❌ |
| 7.10 | Queue drain plan | Dry-run of `gpu_queue` with measured VRAM | `GPU` | Every job's `est_vram_gb` ≤ 22 GB measured headroom | ❌ |

### Checkpoint C7 — GPU Ready 🚧

| # | Check | Target | Status |
| :---: | :--- | :--- | :---: |
| C7.1 🚧 | Environment | Verification script all green | ❌ |
| C7.2 | Memory plan | VRAM table in [Compute Allocation](#compute-allocation-cpu-local-open-models-and-the-rtx-4500-ada) replaced with measured values | ❌ |
| C7.3 | CPU/GPU parity | Phase 4 MLP trains to the same val AUC (± 0.005) on GPU as on CPU with the same seed | ❌ |
| C7.4 | Queue | Dry-run orders all jobs without dependency errors | ❌ |

---

# Phase 8 — Deep Learning: CNN Span Scorer and Generative Rewriter (Unit 4)

**Goal:** a convolutional span-importance scorer and a small fine-tuned generative rewriter that,
together with the quality gate and fallback, reach the main quality/savings gate.
**Weeks:** 14–17 · **Depends on:** 2, 5, 6, 7 · **Unlocks:** 9, 10, 11

### 8.1 Deep Learning Foundations

| # | Task | Detail | HW | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| 8.1.1 | Introduction write-up | Representation learning; why learned features beat Phase 3 hand features for span importance; decoder-only transformer overview | `CPU` | `docs/derivations/deep_learning_intro.md` | ❌ |
| 8.1.2 | Convolution operation | Manual 1-D convolution (kernel, stride, padding, output length) | `CPU` | Equals `nn.Conv1d` on fixture input (test) | ❌ |

### 8.2 TextCNN Span Scorer

| # | Task | Detail | HW | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| 8.2.1 | Architecture | Token embeddings (d = 384) → parallel Conv1d k = 3 and k = 5, 128 filters each, 'same' padding → per-token KEEP head + pooled prompt-compressibility head | `CPU→GPU` | `models/neural/textcnn.py` | ❌ |
| 8.2.2 | **Parameter calculation** | Conv k=3: 384×3×128+128 = 147,584 · Conv k=5: 384×5×128+128 = 245,888 · token head 256+1 = 257 · prompt head 256+1 = 257 · **total 393,986** | `CPU` | Equals `sum(p.numel())` in a test; derivation in docs | ❌ |
| 8.2.3 | **Max vs average pooling** | Prompt head on global max, global avg, and concatenated (head becomes 512+1 = 513 params); 3 seeds each | `CPU→GPU` | Table + interpretation ("any droppable region" vs "how much is droppable") | ❌ |
| 8.2.4 | Comparison | TextCNN vs HMM vs GBM span taggers on identical folds | `CPU→GPU` | McNemar + table | ❌ |

### 8.3 Generative Rewriter (Supervised Fine-Tuning)

| # | Task | Detail | HW | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| 8.3.1 | SFT data | (casual prompt, oracle preserved rewrite) from train split; `<target=…>` conditioning prefix | `CPU` | HF dataset in DVC | ❌ |
| 8.3.2 | Tiny-config smoke test | Full training loop on a tiny model on CPU | `CPU` | CI test passes; GPU jobs queued | ❌ |
| 8.3.3 | Rung 1: Qwen2.5-0.5B full FT | bf16, AdamW, grad checkpointing | `GPU` | Model registered; val report | ❌ |
| 8.3.4 | Rung 2: Qwen2.5-1.5B LoRA | r = 16, α = 32, bf16 | `GPU` | Same | ❌ |
| 8.3.5 | Rung 3: Qwen2.5-3B LoRA | r = 16, bf16, grad checkpointing | `GPU` | Same (only if rungs 1–2 fail C8.3) | ❌ |
| 8.3.6 | Rung 4: 7B QLoRA | NF4 base + LoRA | `GPU` | Same (only if rungs 1–3 fail C8.3) | ❌ |
| 8.3.7 | Decoding safeguards | Never longer than input on T; fall back to original if longer | `CPU` | Property test | ❌ |
| 8.3.8 | Inference-time quality gate | Champion gate rejects risky rewrites ⇒ lighter rewrite or original | `CPU` | Integration test | ❌ |
| 8.3.9 | Baselines | Original · best operator · HMM compressor · LLMLingua-2 · zero-shot Qwen2.5-32B-Instruct-AWQ "rewrite concisely" | `LLM` `GPU` | Same val table | ❌ |
| 8.3.10 | Failure analysis | 100 degraded cases hand-categorised: dropped constraint / tone / example / changed meaning / other | `CPU` | `reports/p8_failures.md` | ❌ |
| 8.3.11 | Tokenizer-awareness test | Rewrite conditioned on T vs on another target, both scored on T's tokenizer | `GPU` `LLM` | Table per target pair | ❌ |

### 8.4 Small Judge (optional throughput)

| # | Task | Detail | HW | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| 8.4.1 | Serve 8B judge | Llama-3.1-8B-Instruct AWQ on vLLM (not during training) for RL audits and shadow judging | `GPU` | Endpoint up | ❌ |
| 8.4.2 | Agreement with the 32B judge | 1,000 pairs | `GPU` `LLM` | κ reported | ❌ |

### Checkpoint C8 — Deep Learning

| # | Check | Target | Status |
| :---: | :--- | :--- | :---: |
| C8.1 | TextCNN parameters | Matches documented count exactly | ❌ |
| C8.2 | TextCNN tagging | Token macro-F1 **≥ C3.4 best + 0.05**, or documented reason | ❌ |
| C8.3 🚧 | **Rewriter main gate (val split)** | Median savings **≥ 25%** on every target; preservation **≥ 92%** overall and **≥ 85%** per category (gate + fallback on) | ❌ |
| C8.4 | vs zero-shot large open LLM | At equal preservation (±1 pp): savings within **5 pp** of zero-shot Qwen2.5-32B-AWQ while **≥ 10× lower** rewrite latency on the RTX 4500 Ada | ❌ |
| C8.5 | vs LLMLingua-2 | Higher preservation at equal or better savings | ❌ |
| C8.6 | Tokenizer-awareness | Target-conditioned rewrite saves **≥ 2 pp** more on its own tokenizer, or documented finding that it does not (ship one rewriter) | ❌ |
| C8.7 | Small judge (if 8.4 attempted) | κ **≥ 0.70** vs the 32B judge, else the 32B judge remains for all reported numbers | ❌ |

---

# Phase 9 — Reinforcement Learning with a Quality Gate (Unit 4)

**Goal:** treat rewriting as a sequential decision problem, learn it tabularly first, then
optimise the neural rewriter directly for "fewest tokens while quality is preserved" without
reward hacking.
**Weeks:** 17–19 · **Depends on:** 5, 8 · **Unlocks:** 10, 11

### 9.1 MDP Formulation

| # | Task | Detail | HW | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| 9.1.1 | Agent and environment | Agent = rewriter; environment = draft + target tokenizer + quality gate / judge | `CPU` | `docs/derivations/rl.md` §1 | ❌ |
| 9.1.2 | States | Tabular: (remaining filler count bucket, constraint count, length bucket, category, ops already applied) | `CPU` | `rl/env.py` | ❌ |
| 9.1.3 | Actions | Apply one of `F H D I L C S`, or `STOP` | `CPU` | Same | ❌ |
| 9.1.4 | Rewards | On `STOP`: Δtokens% − λ·1[degraded] (λ = 100); per step −0.01 | `CPU` | Reward unit tests | ❌ |
| 9.1.5 | Policy and value functions | π(a \| s), V^π(s), Q^π(s, a) defined for this MDP | `CPU` | `docs/derivations/rl.md` §2 | ❌ |
| 9.1.6 | Bellman equations | Expectation and optimality forms; Q-learning update derived from optimality | `CPU` | `docs/derivations/rl.md` §3 | ❌ |

### 9.2 Tabular Q-Learning

| # | Task | Detail | HW | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| 9.2.1 | Q-learning from scratch | ε-greedy; quality gate as cheap reward model (no LLM calls during learning) | `CPU` | Learning curve | ❌ |
| 9.2.2 | Hyperparameter sweep | α ∈ {0.05, 0.1, 0.3}, γ ∈ {0.9, 0.99}, ε schedule | `CPU` | Table | ❌ |
| 9.2.3 | Learned policy | Greedy operator order per category | `CPU` | `reports/p9_policy.md` | ❌ |
| 9.2.4 | Judge verification | Greedy policy on 300 val prompts with the real judge | `LLM` | Savings and preservation table | ❌ |

### 9.3 Policy Optimisation of the Neural Rewriter

| # | Task | Detail | HW | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| 9.3.1 | GRPO setup | Start from C8.3 model; LoRA; group size 4–8; KL penalty to SFT reference | `GPU` | Tiny CPU smoke test + queued job | ❌ |
| 9.3.2 | Reward | `tokens_saved_pct × gate_pass` | `GPU` | Reward logged per step | ❌ |
| 9.3.3 | Reward-hacking audits | Every N steps sample rewrites → 32B judge in a time-shared vLLM slot, never colocated with training | `LLM` | Audit log | ❌ |
| 9.3.4 | Degenerate-output monitor | Empty / truncated / non-natural-language rate | `GPU` | Metric logged | ❌ |
| 9.3.5 | RL vs SFT comparison | Same val table as C8.3 | `GPU` `LLM` | `reports/p9_rl_vs_sft.md` | ❌ |

### Checkpoint C9 — Reinforcement Learning

| # | Check | Target | Status |
| :---: | :--- | :--- | :---: |
| C9.1 | Tabular convergence | Q-table change **< 1e-3** over last 1,000 episodes | ❌ |
| C9.2 | Tabular policy value | Judge-verified savings **≥** best fixed operator order at equal preservation | ❌ |
| C9.3 | No reward hacking | Gate precision on RL rewrites within **5 pp** of its precision on Phase 2 data | ❌ |
| C9.4 | No degenerate outputs | **< 1%** on 1,000 samples | ❌ |
| C9.5 | RL improves on SFT | Savings **≥ SFT + 3 pp**; preservation not lower by > 1 pp overall or > 2 pp in any category. Else SFT stays champion | ❌ |

---

# Phase 10 — Hyperparameter Optimisation (Unit 4)

**Goal:** a formal comparison of grid, random and Bayesian optimisation, and tuned champions
for the gate, the CNN and the rewriter.
**Weeks:** 19–20 (grid and random already used in Phases 3–4) · **Depends on:** 3, 4, 5, 8 · **Unlocks:** 11, 12

| # | Task | Detail | HW | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| 10.1 | HPO module | `terser.hpo` wrapping grid / random / Optuna TPE / GP samplers; MLflow parent-child runs | `CPU` | Module + tests | ❌ |
| 10.2 | Grid search study | GBM quality gate, 60-point grid | `CPU` | Best-so-far curve | ❌ |
| 10.3 | Random search study | GBM quality gate, 60 trials × 5 repeats | `CPU` | Best-so-far curve | ❌ |
| 10.4 | Bayesian optimisation study | TPE and GP, 60 trials × 5 repeats | `CPU` | Best-so-far curve | ❌ |
| 10.5 | Method comparison | Grid vs Random vs Bayesian at equal budget | `CPU` | `reports/figures/p10_hpo_methods.png` + report | ❌ |
| 10.6 | TextCNN tuning | Kernel sizes, filters, dropout, lr (Bayesian) | `GPU` | Tuned model registered | ❌ |
| 10.7 | Rewriter LoRA tuning | Rank, α, lr, warmup, epochs; objective = savings s.t. preservation ≥ 0.92; median pruner | `GPU` | Tuned model registered | ❌ |
| 10.8 | Test-set isolation audit | Data-access logs show HPO touched only train/val | `CPU` | Automated check green | ❌ |

### Checkpoint C10 — Tuning

| # | Check | Target | Status |
| :---: | :--- | :--- | :---: |
| C10.1 | Method comparison | 3 methods × 5 repeats at equal budget, plot + table | ❌ |
| C10.2 | Gate improvement | Tuned AUC **≥ untuned + 0.01**, or documented plateau | ❌ |
| C10.3 | Rewriter improvement | Still passes C8.3 with **≥ 1 pp** more savings, or documented plateau | ❌ |
| C10.4 | No test leakage | 10.8 green | ❌ |

---

# Phase 11 — MLOps: Pipelines, Serving, CI/CD/CT, Monitoring (Unit 4)

**Goal:** Terser as a production system, with reproducible pipelines, versioned models, a
two-tier inference service with net-savings accounting, automated retraining, and drift and
quality monitoring.
**Weeks:** 20–23 · **Depends on:** 5, 6, 8 (9, 10 for champions) · **Unlocks:** 12, 13

### 11.1 Lifecycle, Data and Feature Pipelines

| # | Task | Detail | HW | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| 11.1.1 | MLOps overview ADR | Lifecycle diagram + Terser-specific production challenges: judge cost, tokenizer drift, silent over-compression, latency budget, training/serving skew | `CPU` | `docs/decisions/ADR-0010-mlops.md` | ❌ |
| 11.1.2 | Full DVC pipeline | `collect → casualise → candidates → respond → judge → label → features → split → stratify` | `CPU` | `dvc repro` end to end | ❌ |
| 11.1.3 | Shared feature code | One `terser.features` used offline and online | `CPU` | Parity test: 100/100 prompts identical | ❌ |
| 11.1.4 | Data validation | pandera schemas: nulls, token-ratio ranges, category balance | `CPU` | Fails on a corrupted fixture | ❌ |

### 11.2 Experiment Tracking and Model Versioning

| # | Task | Detail | HW | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| 11.2.1 | Tracking audit | Every reported number links to an MLflow run | `CPU` | Script finds 0 unlinked numbers in `reports/` | ❌ |
| 11.2.2 | Model registry | Stages `None → Staging → Production → Archived` for gate, tagger, rewriter | `CPU` | Registry populated | ❌ |
| 11.2.3 | Lineage | Each version records git SHA, DVC hash, config, checkpoint IDs passed | `CPU` | Any production model traceable | ❌ |

### 11.3 Deployment, Serving and Inference Systems

| # | Task | Detail | HW | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| 11.3.1 | API | FastAPI `POST /rewrite {prompt, target, mode: fast\|quality}` → `{rewrite, tokens_before, tokens_after, gate_score, fallback_used}` | `CPU` | OpenAPI docs | ❌ |
| 11.3.2 | Fast path | Span tagger + gate exported to ONNX Runtime | `CPU` | Runs with no GPU | ❌ |
| 11.3.3 | Quality path | Rewriter on vLLM with continuous batching + gate + fallback | `GPU` | Endpoint live on RTX 4500 Ada | ❌ |
| 11.3.4 | Caching | LRU keyed on normalised prompt hash + target | `CPU` | Warm-hit latency measured | ❌ |
| 11.3.5 | Timeout fallback | Quality path over budget ⇒ fast-path result | `CPU` | Fault-injection test | ❌ |
| 11.3.6 | Net-savings accounting | Per request: price_T × (tokens_before − tokens_after) − rewriter_cost | `CPU` | Logged per request | ❌ |
| 11.3.7 | Containers | `terser-cpu` and `terser-gpu` (CUDA base) images; docker-compose | `CPU` `GPU` | `docker compose up` serves both paths | ❌ |

### 11.4 CI / CD / CT

| # | Task | Detail | HW | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| 11.4.1 | CI | Every PR: lint, types, unit tests, CPU smoke training, `dvc repro --dry`, feature parity | `CPU` | Median ≤ 15 min | ❌ |
| 11.4.2 | CD | Tag `v*`: build + scan images → staging → smoke + latency tests → manual approval → production | `CPU` | Staging deploy automated | ❌ |
| 11.4.3 | CT | Weekly or on drift alert: new consented prompts → judge labels → retrain gate/tagger (CPU) → queue rewriter retrain (GPU) → evaluate → register as Staging if checkpoints pass | `CPU` `GPU` `LLM` | One full cycle completed | ❌ |
| 11.4.4 | Promotion guard | CT refuses promotion when a synthetic regression is injected | `CPU` | Test green | ❌ |

### 11.5 Monitoring, Drift and Observability

| # | Task | Detail | HW | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| 11.5.1 | Metrics | Prometheus: RPS, p50/p95/p99 latency per path, fallback rate, gate rejections, tokens saved, net savings, GPU util / VRAM | `CPU` | Exposed at `/metrics` | ❌ |
| 11.5.2 | Input drift | Evidently daily: embedding drift (PSI / MMD on PCA components), length and category distribution | `CPU` | Report generated | ❌ |
| 11.5.3 | Tokenizer drift | Re-count fixture set with current tokenizer versions; alert on any change | `CPU` | Alert fires on a pinned-version swap | ❌ |
| 11.5.4 | Shadow judging | 1% of consented rewrites judged asynchronously; rolling preservation | `LLM` | Metric on dashboard | ❌ |
| 11.5.5 | Dashboard and alerts | Grafana; alert at rolling preservation < 92% | `CPU` | Dashboard live | ❌ |
| 11.5.6 | Structured logging | Request IDs; hashes by default, raw text only if consented | `CPU` | Log schema doc | ❌ |

### 11.6 ML Orchestration

| # | Task | Detail | HW | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| 11.6.1 | Prefect flows | `data_flow`, `train_classical_flow`, `train_rewriter_flow` (submits to `gpu_queue`), `evaluate_flow`, `promote_flow`, `drift_flow` | `CPU` | Flows registered | ❌ |
| 11.6.2 | Reliability | Retries, cached LLM steps, failure notifications | `CPU` | Injected failure recovers | ❌ |

### Checkpoint C11 — Production Readiness

| # | Check | Target | Status |
| :---: | :--- | :--- | :---: |
| C11.1 | End-to-end orchestration | `data → train_classical → evaluate → promote` runs unattended on a fresh clone (CPU) | ❌ |
| C11.2 | Fast-path latency | p95 **≤ 150 ms** at 20 RPS | ❌ |
| C11.3 | Quality-path latency | p95 **≤ 400 ms** at 5 RPS, prompts ≤ 256 tokens, RTX 4500 Ada | ❌ |
| C11.4 | Throughput | Quality path ≥ 20 RPS sustained 30 min, no OOM | ❌ |
| C11.5 | Fallback correctness | Rewriter crash/timeout returns original prompt **100%** of the time | ❌ |
| C11.6 | Drift detection | Synthetic shift (only `code` prompts, or tokenizer version swap) alerts within one cycle | ❌ |
| C11.7 | CT guard | Synthetic regression correctly refused promotion | ❌ |

---

# Phase 12 — Final Evaluation, Ablations and Benchmark Release

**Goal:** a single, statistically rigorous answer to "does Terser save tokens without losing
quality?", on the untouched test split.
**Week:** 24 · **Depends on:** 8, 9, 10, 11 · **Unlocks:** 13

| # | Task | Detail | HW | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| 12.1 | Test-split unlock | Logged, one-time event | `CPU` | Access log entry | ❌ |
| 12.2 | Master results table | Every system (original, best operator, HMM, GBM tagger+gate, TextCNN+gate, SFT, RL, LLMLingua-2, zero-shot LLM) × savings × preservation × net savings × latency | `LLM` `GPU` | `reports/final_results.md` | ❌ |
| 12.3 | Stratified reporting | Per target × category × high-risk cluster | `CPU` | Tables | ❌ |
| 12.4 | Statistical rigour | 95% bootstrap CIs (1,000 resamples); ≥ 3 seeds; McNemar between top systems | `CPU` | Error bars on every headline number | ❌ |
| 12.5 | Cross-model generalisation matrix | Rewrite conditioned on A, evaluated on B, all pairs | `LLM` `GPU` | Matrix figure | ❌ |
| 12.6 | **Ablation matrix** | Remove gate / fallback / operator `S` / RL / HPO / cluster stratification one at a time | `LLM` `GPU` | 6-row ablation table | ❌ |
| 12.7 | Net savings by prompt length | Savings minus rewriter cost across length percentiles | `CPU` | Figure | ❌ |
| 12.8 | Error taxonomy | 200 degraded test cases categorised | `CPU` | `reports/final_errors.md` | ❌ |
| 12.9 | Qualitative gallery | Best and worst casual → rewrite → output pairs | `CPU` | `reports/gallery.md` | ❌ |
| 12.10 | Compute accounting | Generated tokens, GPU-hours, energy per phase | `CPU` | Table | ❌ |
| 12.11 | Benchmark release | Prompts, candidates, labels, rubrics, eval code (licence permitting) | `CPU` | Release tag + data card | ❌ |
| 12.12 | Reproduction | Second contributor reproduces headline numbers from a fresh clone | all | Within CI bounds | ❌ |

---

# Phase 13 — Documentation, Report and Viva Preparation

**Goal:** everything needed to understand, reproduce, present and defend the project.
**Weeks:** 23–24 · **Depends on:** 12 · **Unlocks:** delivery

| # | Task | Detail | HW | Definition of Done | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| 13.1 | Architecture diagram | Casual prompt → fast/quality path → gate → target LLM, with training loop | `CPU` | Figure in README | ❌ |
| 13.2 | Syllabus mapping doc | Unit 1–4 → phase / file / figure links | `CPU` | `docs/syllabus_map.md` | ❌ |
| 13.3 | Per-unit notebooks | One clean notebook per unit with results and plots | `CPU` | 4 notebooks | ❌ |
| 13.4 | Model cards | Gate, tagger, TextCNN, rewriter | `CPU` | Complete | ❌ |
| 13.5 | API docs | `/rewrite` reference + examples | `CPU` | Published | ❌ |
| 13.6 | Final report | Problem, gap in existing work, method, experiments, results, ablations, negative results, limitations, future work | `CPU` | `docs/report.md` | ❌ |
| 13.7 | Limitations | Creative prompts, multi-turn context, tokenizer drift, judge dependence | `CPU` | Section written | ❌ |
| 13.8 | Demo script | Live casual prompt → rewrite → side-by-side outputs + token counter | `CPU` `GPU` | Rehearsed; cached fallback if GPU unavailable | ❌ |
| 13.9 | Viva prep | ID3 gain, bias–variance, ROC, backprop, SVM dual, AdaBoost weights, EM, Viterbi, K-means ⇔ EM, Bellman / Q-update, CNN parameters, LoRA rank | `CPU` | `docs/viva.md` | ❌ |

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
| 7 | 0 (hardware available) | 1 (LLM rows), 2, 8, 9, 10, 11 |
| 8 | 2, 5, 6, 7 | 9, 10, 11 |
| 9 | 5, 8 | 10, 11 |
| 10 | 3, 4, 5, 8 | 11, 12 |
| 11 | 5, 6, 8 (9, 10 for champions) | 12, 13 |
| 12 | 8, 9, 10, 11 | 13 |
| 13 | 12 | delivery |

Hard gates on the critical path: **C7.1** (GPU, needed for all labelled data) → **C2.3–C2.5** (judge trustworthy) → **C2.7** (go/no-go) → **C8.3** (rewriter) → **S1–S5** (final).

---

## Syllabus Coverage Matrix

| Unit | Requirement | Where It Lives | Status |
| :--- | :--- | :--- | :---: |
| 1 | Introduction to ML, ML models | 1.1.1 (sub-problem taxonomy) | ❌ |
| 1 | Concept learning: hypotheses, version space | 1.3.2–1.3.7 | ❌ |
| 1 | Inductive bias | 1.3.8, 3.3.6 | ❌ |
| 1 | Accuracy, Precision, Recall, Sensitivity, Specificity | 3.2.1 | ❌ |
| 1 | ROC, AUC | 3.2.2, 3.2.3 | ❌ |
| 1 | Bias–variance decomposition and trade-off | 3.6.1, 3.6.2 | ❌ |
| 1 | Cross-validation strategies | 3.2.4–3.2.7 | ❌ |
| 1 | Decision trees: ID3 | 3.3.1 | ❌ |
| 1 | Decision boundary for decision trees (x–y axis) | 3.3.5 | ❌ |
| 1 | Hypothesis search and inductive bias in trees | 3.3.6 | ❌ |
| 1 | Overfitting and solutions | 3.3.3, 3.3.4 | ❌ |
| 1 | Continuous values | 3.1.4, 3.3.2 | ❌ |
| 1 | Logistic regression | 3.4 | ❌ |
| 1 | KNN classification and regression | 3.5.1, 3.5.2 | ❌ |
| 1 | Decision boundary for KNN | 3.5.3 | ❌ |
| 2 | Perceptron | 4.1.1 | ❌ |
| 2 | Multi-layer networks and backpropagation | 4.1.3–4.1.7 | ❌ |
| 2 | Activations: Step, Sigmoid, Tanh, ReLU | 4.2 | ❌ |
| 2 | Optimizers: GD, SGD, Momentum, Adagrad, Adam | 4.3 | ❌ |
| 2 | SVM margin maximisation, primal, Lagrangian dual, dual solution | 4.4.1, 4.4.2, 4.4.4 | ❌ |
| 2 | Hard margin and soft margin (classification) | 4.4.5, 4.4.6 | ❌ |
| 2 | Kernels: linear (derivation), polynomial | 4.4.3, 4.4.7 | ❌ |
| 3 | Combining weak learners | 5.1.1, 5.1.2 | ❌ |
| 3 | Gradient boosting | 5.1.5, 5.1.6 | ❌ |
| 3 | Random forest | 5.1.3, 5.1.4 | ❌ |
| 3 | Bayes theorem | 5.2.1 | ❌ |
| 3 | Maximum likelihood | 5.2.2, 5.3.2 | ❌ |
| 3 | Bayes optimal classifier | 5.2.4 | ❌ |
| 3 | Naive Bayes | 5.2.3 | ❌ |
| 3 | Expectation maximisation and GMM | 5.2.5, 5.2.6 | ❌ |
| 3 | Hidden Markov Models | 5.3 | ❌ |
| 4 | Hierarchical vs non-hierarchical clustering | 6.4 | ❌ |
| 4 | Agglomerative and divisive clustering | 6.5, 6.6 | ❌ |
| 4 | K-means, bisecting K-means | 6.2, 6.3 | ❌ |
| 4 | K-means as special case of EM | 6.7 | ❌ |
| 4 | RL: agent, environment, states, actions, rewards, policy, value functions | 9.1 | ❌ |
| 4 | Bellman equation and Q-learning | 9.1.6, 9.2 | ❌ |
| 4 | Introduction to deep learning | 8.1.1, 8.3 | ❌ |
| 4 | Convolution operation and CNNs | 8.1.2, 8.2.1 | ❌ |
| 4 | CNN parameter calculation, max and avg pooling | 8.2.2, 8.2.3 | ❌ |
| 4 | Hyperparameter tuning: Grid, Random, Bayesian optimisation | 3.6.4, 4.5, 10 | ❌ |
| 4 | MLOps overview, ML lifecycle and production challenges | 0.3.4, 11.1.1 | ❌ |
| 4 | Data and feature pipelines | 2.1.7, 11.1 | ❌ |
| 4 | Experiment tracking and model versioning | 0.2.4, 0.2.5, 11.2 | ❌ |
| 4 | Model deployment and serving | 11.3 | ❌ |
| 4 | CI/CD/CT for ML systems | 0.2.9, 11.4 | ❌ |
| 4 | Inference systems | 11.3.2–11.3.6 | ❌ |
| 4 | Monitoring, drift and observability | 11.5 | ❌ |
| 4 | ML orchestration | 11.6 | ❌ |

---

## Headline Success Criteria

Evaluated once, on the frozen **test** split, in Phase 12.

| # | Criterion | Target | Status |
| :---: | :--- | :--- | :---: |
| S1 | Median token savings on target tokenizers | ≥ 30% (95% CI lower bound ≥ 25%) | ❌ |
| S2 | Quality preservation, overall | ≥ 95% | ❌ |
| S3 | Quality preservation, worst task category | ≥ 90% | ❌ |
| S4 | Net savings after rewriter cost | Positive at the 25th-percentile prompt length for every target | ❌ |
| S5 | Cross-model generalisation | ≤ 3 pp preservation drop vs per-model rewriter, or per-model variants shipped and documented | ❌ |
| S6 | Judge validity | Judge–human κ ≥ 0.60 (C2.4) holds on a fresh 100-pair test sample | ❌ |
| S7 | Fast-path latency | p95 ≤ 150 ms | ❌ |
| S8 | Quality-path latency on RTX 4500 Ada | p95 ≤ 400 ms | ❌ |
| S9 | Independent reproduction | Headline numbers reproduced from a fresh clone by a second contributor | ❌ |
| S10 | All four syllabus units demonstrably covered | Every row of the Syllabus Coverage Matrix ✅ | ❌ |

Missing a target is reported, never hidden: the final report states which targets were met,
which were not, and why.

---

## Overall Progress

| Phase | Tasks | Checks | Done | Status |
| :--- | :---: | :---: | :---: | :---: |
| 0 — Foundations | 22 | 4 | 26 | ✅ |
| 1 — Framing and Concept Learning | 23 | 5 | 22 | ❌ |
| 2 — Data Pipeline and Judge | 21 | 9 | 7 | ❌ |
| 3 — Metrics, Trees, LR, KNN | 33 | 7 | 0 | ❌ |
| 4 — ANN and SVM | 29 | 5 | 0 | ❌ |
| 5 — Boosting, Bayes, GMM, HMM | 19 | 6 | 0 | ❌ |
| 6 — Clustering | 10 | 3 | 0 | ❌ |
| 7 — GPU Onboarding | 10 | 4 | 0 | ❌ |
| 8 — CNN and Rewriter | 19 | 7 | 0 | ❌ |
| 9 — Reinforcement Learning | 15 | 5 | 0 | ❌ |
| 10 — HPO | 8 | 4 | 0 | ❌ |
| 11 — MLOps | 26 | 7 | 0 | ❌ |
| 12 — Final Evaluation | 12 | — | 0 | ❌ |
| 13 — Documentation | 9 | — | 0 | ❌ |
| **Total** | **256** | **66** | **55** | ❌ |

---

# How to Contribute

### Getting started

```bash
git clone https://github.com/guru-bharadwaj20/Terser.git && cd Terser
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1    Linux/macOS: source .venv/bin/activate
pip install -e ".[dev]"            # CPU
pip install -e ".[dev,gpu]"        # on the RTX 4500 Ada machine
cp .env.example .env               # HF_TOKEN, TERSER_LOCAL_LLM_URL (never commit)
pre-commit install
pytest -m "not gpu and not api"
```

### Picking up work

1. Choose a ❌ row whose phase dependencies are ✅ (see the dependency map).
2. Open an issue titled with the row id, e.g. `3.3.1 ID3 from scratch`, and assign yourself.
3. Branch `phase<N>/<row-id>-<topic>`, e.g. `phase3/3.3.1-id3`.
4. In the same PR that satisfies the Definition of Done, flip the row to ✅ and update
   [Overall Progress](#overall-progress). If the row maps to a syllabus requirement, also update the
   [Syllabus Coverage Matrix](#syllabus-coverage-matrix).
5. For `GPU` rows before Phase 7 is ✅: merge the code with its CPU smoke test and a `gpu_queue/` job.
   The row stays ❌ until the job has run on the RTX 4500 Ada.

### Checkpoint protocol

1. When all task rows in a phase are ✅, open a `checkpoint` issue, e.g. `C3 — Classical baselines`.
2. Write `docs/checkpoints/C<N>/README.md`: each check, its **measured value**, pass/fail, MLflow
   run links, DVC data hash, git SHA, seeds, figures, deviations.
3. A reviewer **who did not do the work** re-runs at least one headline number.
4. Record the decision: `PASS`, `PASS WITH CONDITIONS` (conditions become issues due before the next
   checkpoint), or `FAIL`. 🚧 checks allow no conditional pass.
5. On `FAIL`, dependent phases do not start. Changing a target requires an ADR.

### Experiment standards

- Every run logs to MLflow with tags `phase`, `row`, `task`, `hardware` (`cpu` / `rtx4500ada`), `data_hash`, `git_sha`.
- Report mean ± std over ≥ 3 seeds (≥ 5 for Phase 4 studies).
- **Never tune on test.** Code touching `data/processed/test*` outside `terser.eval.final` fails CI.
- Every performance table has per-category rows. An aggregate alone is rejected in review.
- Every new model is compared to the current champion **and** the "original prompt" baseline
  (0% savings, 100% preservation), with McNemar or paired bootstrap for claimed wins.
- Log generated tokens and GPU-hours per experiment.
- Negative results are committed as reports. They are results.

### Coding standards

- Python ≥ 3.11, type hints throughout `src/`, `mypy --strict` clean; `black` + `ruff` via pre-commit.
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

Review: ≥ 1 approval. Changes to `judge/`, `eval/`, data splits or checkpoint targets need **2**. Squash-merge into `main`.

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
| Too little compression headroom | Low–Medium | Critical | Go/no-go with scope-narrowing plan | C2.7 |
| GPU arrives late | Medium | **Critical** | No hosted APIs, so all labelled data waits for the GPU: CPU-first ordering, CPU stand-in smoke runs (2.2.4), GPU queue, pull CPU halves of 9–11 forward | C7 |
| 24 GB VRAM insufficient | Low | Medium | Model ladder 0.5B → 7B QLoRA, grad checkpointing, no colocated serving + training | C7.2, C8.3 |
| Silent over-compression (dropped constraints) | High | High | Constraint features, gate at sensitivity 0.95, fallback, error taxonomy, shadow judging | C8.3, C11.6 |
| RL reward hacking | Medium | High | Judge audits, KL penalty, degenerate-output monitor | C9.3, C9.4 |
| Rewriter overhead wipes savings | Medium | High | Net-savings metric, CPU fast path, skip short prompts (3.5.4) | S4, C11.2 |
| Tokenizer drift | Medium | Medium | Tokenizer fixture alert, CT retraining | 11.5.3 |
| LLM compute overrun / GPU contention | Medium | Medium | Token caps, response caching, small judge (8.4), exclusive GPU slots | C2.9, C8.7 |
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
| GPU queue | Job specs in `gpu_queue/` waiting for the RTX 4500 Ada |
