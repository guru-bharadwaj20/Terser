# Contributing to Terser

Terser is a research-and-engineering project that builds a **casual-to-efficient prompt rewriter**. It takes a prompt written the way people naturally talk and rewrites it into a prompt that gets **the same quality and depth of output** from a target LLM while using **fewer tokens on that model's tokenizer**.

This document is the project's **contribution guide and execution plan**. It defines:

1. How to set up, write code, run experiments, and open pull requests.
2. The **phased roadmap** (about 24 weeks). Each phase has deliverables and **checkpoints** that must pass before the next phase starts.
3. How the project's ML components map onto the course syllabus (Units 1–4), so every concept is *used* for a real purpose and not bolted on.
4. How compute is split between **CPU / API** work (can start now) and **GPU** work on the **NVIDIA RTX 4500 Ada (24 GB)**, which is not connected yet and will be added later.

> **Golden rule:** no phase may claim success on token savings alone. Every result must report **token savings + quality preservation + (where relevant) cross-model behaviour**. A rewrite that saves 60% of tokens and silently drops a constraint is a failure.

---

## Table of Contents

- [0. Quick Start](#0-quick-start)
- [1. Project Goals and Success Criteria](#1-project-goals-and-success-criteria)
- [2. Repository Layout](#2-repository-layout)
- [3. Compute Plan: CPU, API, and the RTX 4500 Ada](#3-compute-plan-cpu-api-and-the-rtx-4500-ada)
- [4. Syllabus Coverage Matrix](#4-syllabus-coverage-matrix)
- [5. Roadmap Overview](#5-roadmap-overview)
- [6. Phase 0 — Foundations and Tooling (Week 1)](#phase-0--foundations-and-tooling-week-1)
- [7. Phase 1 — Problem Framing and Concept Learning (Weeks 2–3)](#phase-1--problem-framing-and-concept-learning-weeks-23)
- [8. Phase 2 — Data Generation Pipeline and Judge Validation (Weeks 3–6)](#phase-2--data-generation-pipeline-and-judge-validation-weeks-36)
- [9. Phase 3 — Metrics, Validation, Trees, Logistic Regression, KNN (Weeks 6–8)](#phase-3--metrics-validation-trees-logistic-regression-knn-weeks-68)
- [10. Phase 4 — Neural Networks and SVMs (Weeks 8–10)](#phase-4--neural-networks-and-svms-weeks-810)
- [11. Phase 5 — Ensembles, Bayesian Learning, HMMs (Weeks 10–12)](#phase-5--ensembles-bayesian-learning-hmms-weeks-1012)
- [12. Phase 6 — Unsupervised Structure of Prompts (Week 12–13)](#phase-6--unsupervised-structure-of-prompts-weeks-1213)
- [13. GPU Onboarding Gate (whenever the RTX 4500 Ada is connected)](#gpu-onboarding-gate-whenever-the-rtx-4500-ada-is-connected)
- [14. Phase 7 — Deep Learning: CNN Span Scorer and Generative Rewriter (Weeks 14–17)](#phase-7--deep-learning-cnn-span-scorer-and-generative-rewriter-weeks-1417)
- [15. Phase 8 — Reinforcement Learning with a Quality Gate (Weeks 17–19)](#phase-8--reinforcement-learning-with-a-quality-gate-weeks-1719)
- [16. Phase 9 — Hyperparameter Optimisation (Week 19–20, plus ongoing)](#phase-9--hyperparameter-optimisation-weeks-1920-plus-ongoing)
- [17. Phase 10 — MLOps: Deployment, Serving, CI/CD/CT, Monitoring (Weeks 20–23)](#phase-10--mlops-deployment-serving-cicdct-monitoring-weeks-2023)
- [18. Phase 11 — Final Evaluation, Benchmark Release, Write-up (Week 24)](#phase-11--final-evaluation-benchmark-release-write-up-week-24)
- [19. Checkpoint Protocol](#19-checkpoint-protocol)
- [20. Experiment Standards](#20-experiment-standards)
- [21. Coding Standards](#21-coding-standards)
- [22. Git Workflow and Pull Requests](#22-git-workflow-and-pull-requests)
- [23. Data, Privacy, and API Usage Policy](#23-data-privacy-and-api-usage-policy)
- [24. Risk Register](#24-risk-register)
- [25. Glossary](#25-glossary)

---

## 0. Quick Start

```bash
# 1. Clone
git clone <repo-url> terser && cd terser

# 2. Environment (Python 3.11)
python -m venv .venv
# Windows (PowerShell): .venv\Scripts\Activate.ps1
# Linux/macOS:          source .venv/bin/activate
pip install -e ".[dev]"          # CPU-only dependencies
# Later, on the GPU machine:
pip install -e ".[dev,gpu]"      # torch CUDA build, vllm, bitsandbytes, peft, trl

# 3. Secrets (never commit)
cp .env.example .env              # fill in ANTHROPIC_API_KEY, OPENAI_API_KEY, HF_TOKEN, ...

# 4. Pre-commit hooks
pre-commit install

# 5. Sanity check
pytest -m "not gpu and not api"   # must pass on any laptop
python -m terser.tokenizers.count "hey so can you like explain recursion to me"
```

Every script must run with `device: auto` and fall back to CPU cleanly. Tests that need a GPU are marked `@pytest.mark.gpu`, and tests that call paid APIs are marked `@pytest.mark.api`.

---

## 1. Project Goals and Success Criteria

### 1.1 Primary objective

Learn a function `R(prompt, target_model) -> prompt'` such that:

| Axis | Definition | Final target (Phase 11) |
|---|---|---|
| **Token savings** | `1 - tokens_T(prompt') / tokens_T(prompt)` on target model *T*'s real tokenizer | Median **≥ 30%** on the held-out casual-prompt test set |
| **Quality preservation** | Fraction of rewrites whose target-model output is judged *equivalent or better* than the original prompt's output | **≥ 95%** overall, and **≥ 90% in every task category** |
| **Net savings** | Token savings after subtracting the rewriter's own cost (tokens, $, latency) | **Positive** at the 25th-percentile prompt length for every supported target |
| **Latency overhead** | Added p95 latency of the rewrite step | **≤ 150 ms** (classifier path) / **≤ 400 ms** (generative path) on the RTX 4500 Ada |
| **Cross-model generalisation** | Savings and preservation when the rewrite is sent to a *different* target model | Degradation in preservation **≤ 3 percentage points** vs per-model rewriter, or a documented decision to ship per-model variants |

### 1.2 Non-goals

- Compressing long retrieved contexts or RAG chunks (LLMLingua already covers this; we compare against it as a baseline).
- Replacing prompt caching for repeated system prompts.
- Making prompts harder for humans to read. Rewrites stay human-readable natural language.

### 1.3 Supported task categories

Every dataset, metric, and checkpoint is **stratified** over these categories:

| ID | Category | Output quality measured by |
|---|---|---|
| `code` | Code generation / fixing | Unit-test pass rate plus pairwise judge |
| `fact` | Factual Q&A | Exact match / token F1 against reference, plus judge |
| `reason` | Math / multi-step reasoning | Final-answer accuracy |
| `explain` | Explanations / tutoring | Rubric judge (coverage, correctness, depth) |
| `creative` | Creative writing | Pairwise judge (constraint adherence, tone, depth) |
| `transform` | Summarise / rewrite / translate / format | Rubric judge plus constraint checklist |

---

## 2. Repository Layout

```
terser/
├── CONTRIBUTING.md
├── README.md
├── pyproject.toml
├── .env.example
├── dvc.yaml / dvc.lock               # data pipeline definitions
├── configs/                          # Hydra configs: data/, model/, experiment/, hardware/
│   └── hardware/{cpu.yaml, rtx4500ada.yaml}
├── data/                             # DVC-tracked, never committed raw
│   ├── raw/  interim/  processed/  gold/
├── docs/
│   ├── decisions/                    # ADRs (architecture decision records)
│   ├── checkpoints/                  # one report per checkpoint (see §19)
│   └── derivations/                  # SVM dual, Bellman, EM, backprop notes
├── notebooks/                        # exploration only; nothing imported from here
├── src/terser/
│   ├── tokenizers/                   # per-target token counting (ground truth)
│   ├── data/                         # collection, paraphrasing, candidate generation
│   ├── judge/                        # LLM-judge, task metrics, calibration
│   ├── features/                     # span/pair features, embeddings
│   ├── models/
│   │   ├── classical/                # ID3, logistic, KNN, SVM, NB, GMM, HMM, trees
│   │   ├── neural/                   # perceptron, MLP, TextCNN
│   │   ├── ensembles/                # RF, AdaBoost, gradient boosting
│   │   ├── clustering/               # k-means, bisecting, agglomerative, divisive
│   │   ├── rewriter/                 # seq2seq / decoder SFT
│   │   └── rl/                       # tabular Q-learning, GRPO/PPO policy
│   ├── eval/                         # three-axis evaluation harness
│   ├── hpo/                          # grid / random / Bayesian optimisation
│   ├── serving/                      # FastAPI app, inference clients
│   ├── monitoring/                   # drift, observability
│   └── pipelines/                    # Prefect flows
├── gpu_queue/                        # job specs waiting for the RTX 4500 Ada (see §3.3)
├── tests/
└── .github/workflows/                # CI, CT, release
```

---

## 3. Compute Plan: CPU, API, and the RTX 4500 Ada

### 3.1 Hardware profile

| Resource | Details | Status |
|---|---|---|
| Dev laptop (Windows 11) | CPU only | **Available now** |
| **NVIDIA RTX 4500 Ada Generation** | 24 GB GDDR6 ECC, Ada Lovelace (compute capability 8.9), supports **bf16** and FP8 kernels in vLLM, ~210 W | **Not connected yet**, to be attached later via remote access |
| Hosted LLM APIs | Claude, GPT family: candidate generation, judging, API-only targets, token counting | Available now (budget-capped, see §23) |

### 3.2 Task tags

Every task in the roadmap carries one tag:

- **[CPU]**: runs on the laptop. Start immediately.
- **[API]**: needs network and API keys. Start immediately, but watch the budget.
- **[GPU]**: needs the RTX 4500 Ada. **Do not block on it.** Write the code, test it on CPU with a tiny model (`configs/model/tiny.yaml`), and add a job spec to `gpu_queue/`.
- **[CPU→GPU]**: runs on CPU at reduced scale now and is re-run at full scale once the GPU is connected.

### 3.3 The GPU queue

Until the GPU is connected, GPU work is **written, unit-tested on CPU with tiny configs, and queued**:

```yaml
# gpu_queue/2026-10-14_sft-qwen0.5b-full.yaml
id: sft-qwen0.5b-full
phase: 7
depends_on: [gpu-onboarding-gate, ckpt-2.4]
command: python -m terser.models.rewriter.train experiment=sft_qwen05b_full hardware=rtx4500ada
est_vram_gb: 14
est_hours: 6
exclusive: true          # nothing else may use the GPU while this runs
priority: 2              # 1 = highest
cpu_smoke_test: tests/rewriter/test_train_tiny.py   # must pass before queuing
outputs: [models/rewriter/sft-qwen0.5b-full/, mlruns/]
```

Rules:
- A job may only be queued if its `cpu_smoke_test` passes in CI.
- `exclusive: true` jobs never share the GPU. vLLM serving and training **do not run at the same time** on 24 GB.
- Once the GPU is online, the queue is drained in `priority`, then `depends_on`, order by `python -m terser.gpu_queue run`.

### 3.4 Workload allocation and VRAM budget (24 GB)

Estimates are for planning. Real numbers get recorded at the GPU Onboarding Gate and replace these.

| Workload | Phase | Where | Precision / method | Est. VRAM | Notes |
|---|---|---|---|---|---|
| Token counting (tiktoken, HF tokenizers) | 1+ | CPU | — | 0 | Ground truth, never approximated |
| Claude token counts | 1+ | API | `count_tokens` endpoint | 0 | Cached to disk |
| Candidate rewrite generation | 2 | API | — | 0 | Strong LLM proposes 4–8 candidates per prompt |
| Target LLM outputs (API targets) | 2+ | API | — | 0 | Claude / GPT targets |
| Target LLM outputs (open targets: Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct) | 2 (small), 7+ (full) | **CPU→GPU** | vLLM bf16 | ~16 GB weights + KV cache at `gpu_memory_utilization=0.90` | Before GPU: use API-hosted versions of the same open models or a 0.5B–1.5B stand-in |
| Local judge (8B) | 7+ | GPU | vLLM AWQ int4 | ~6 GB + KV | Only after it agrees with the API judge (Checkpoint 7.1) |
| Sentence embeddings (bge-small / e5-base) | 3+ | CPU→GPU | fp32 / fp16 | < 2 GB | CPU is fine for ≤ 50k texts |
| Classical models (ID3, LR, KNN, SVM, NB, GMM, HMM, RF, GBM) | 3–5 | CPU | — | 0 | Never put these on GPU |
| MLP / perceptron / optimizer study | 4 | CPU | fp32 | 0 | Small enough for CPU |
| TextCNN span scorer | 7 | CPU→GPU | fp32 / bf16 | < 2 GB | CPU trains fine; GPU for sweeps |
| Rewriter SFT: Qwen2.5-0.5B **full fine-tune** | 7 | GPU | bf16 + AdamW, grad checkpointing | ~12–16 GB | About 16 bytes/param for weights+grads+Adam states (~8 GB) plus activations |
| Rewriter SFT: Qwen2.5-1.5B / 3B **LoRA** | 7 | GPU | bf16 base + LoRA r=16 | ~8–16 GB | Main candidate |
| Rewriter SFT: 7B **QLoRA** | 7 | GPU | 4-bit NF4 base + LoRA | ~10–18 GB at seq len ≤ 1024 | Upper bound. Only if smaller models miss Checkpoint 7.3 |
| RL (GRPO/PPO) rewriter policy | 8 | GPU | 0.5B–1.5B LoRA | ~12–20 GB | Reward calls target LLM via **API** or via time-shared vLLM, never colocated |
| Tabular Q-learning toy environment | 8 | CPU | — | 0 | |
| Bayesian HPO of rewriter | 9 | GPU | Optuna, 1 trial at a time | per trial | Trials pruned early (median pruner) |
| Production serving (rewriter) | 10 | GPU (CPU fallback) | vLLM / ONNX Runtime | 2–6 GB | Classifier path must also run on CPU |

### 3.5 GPU weekly schedule template (once connected)

| Slot | Use |
|---|---|
| Night (long, exclusive) | SFT / RL training runs from `gpu_queue/` |
| Day block A | vLLM serving target / judge models for data generation and evaluation |
| Day block B | Short sweeps (TextCNN, HPO trials), latency benchmarks |
| Weekend | Full evaluation suite re-runs, checkpoint reports |

---

## 4. Syllabus Coverage Matrix

Every syllabus topic has a **concrete job** in the project. Nothing is included just for coverage. Contributors working on a topic must link their PR to the matching row.

### Unit 1: Introduction, Performance Metrics, Decision Trees, KNN

| Concept | Where it is used in Terser | Phase |
|---|---|---|
| ML and its models | Taxonomy of our components: supervised (quality gate, span tagger), unsupervised (prompt clustering), RL (rewrite policy), generative (rewriter) | 1 |
| Concept learning: hypotheses | A hypothesis = a conjunction of *rewrite operators* (drop filler, drop hedge, merge duplicate constraints, convert to imperative, list-ify) that keeps quality | 1 |
| Version space | Candidate-Elimination over the operator lattice. Find the set of operator combinations consistent with all quality-labelled examples, per task category | 1 |
| Inductive bias | Written down explicitly for each model (ID3: shorter trees; KNN: local smoothness in embedding space; rewriter: "keep all constraint-bearing spans") | 1, 3, 7 |
| Accuracy, Precision, Recall, Sensitivity, Specificity | Quality-gate classifier: **positive = "degraded"**. Sensitivity = catching degradations, specificity = not rejecting good rewrites | 3+ |
| ROC, AUC | Choosing the quality-gate threshold; comparing every classifier on one plot | 3+ |
| Bias–variance decomposition and trade-off | Empirical decomposition with bootstrap for trees of varying depth and KNN of varying *k*; learning curves | 3 |
| Cross-validation strategies | **Grouped** k-fold (group = source prompt, so no paraphrase leakage), **stratified** by task category, nested CV for model selection, leave-one-category-out for generalisation | 3+ |
| ID3 decision trees | From-scratch ID3 for span keep/drop classification on discretised features | 3 |
| Decision boundary (x–y axis) | 2-D plots: `span_self_information` vs `span_token_cost` for tree, LR, KNN, SVM | 3, 4 |
| Hypothesis search and inductive bias of trees | Report on greedy information-gain search vs exhaustive search over small feature sets | 3 |
| Overfitting and solutions | Depth limits, min-samples, reduced-error pruning, rule post-pruning, validation curves | 3 |
| Continuous values | Threshold-based splits on token cost, perplexity, position (C4.5-style) | 3 |
| Logistic regression | Calibrated baseline quality gate; span tagger baseline | 3 |
| KNN classification and regression | Classification: "is this rewrite safe?" from nearest labelled pairs. **Regression: predict achievable compression ratio** of a new prompt from its neighbours | 3 |
| KNN decision boundary | 2-D embedding projection plots for k = 1, 5, 25 | 3 |

### Unit 2: ANNs and SVMs

| Concept | Where it is used | Phase |
|---|---|---|
| Perceptron | Linear keep/drop span classifier from scratch; convergence on linearly separable filler-word subset | 4 |
| Multi-layer networks, backpropagation | From-scratch NumPy MLP quality gate (verified against PyTorch gradients), then PyTorch MLP | 4 |
| Activations: Step, Sigmoid, Tanh, ReLU | Controlled ablation on the MLP quality gate | 4 |
| Optimizers: GD, SGD, Momentum, Adagrad, Adam | Controlled comparison study, same seed/architecture/budget | 4 |
| SVM margin maximisation, primal, Lagrangian dual, dual solution | Linear SVM span classifier solved via the **dual** with a QP solver (from scratch) and compared with scikit-learn; derivation in `docs/derivations/svm_dual.md` | 4 |
| Hard vs soft margin | Hard margin on separable filler subset; soft margin with C sweep on full data | 4 |
| Kernels: linear (derived), polynomial | Linear kernel derived in docs; polynomial kernel degree 2–3 compared empirically | 4 |

### Unit 3: Boosting, Bayesian Learning, HMMs

| Concept | Where it is used | Phase |
|---|---|---|
| Combining weak learners | AdaBoost with decision stumps on span features | 5 |
| Gradient boosting | Strongest tabular quality gate (LightGBM/XGBoost) with engineered pair features | 5 |
| Random forest | Quality gate and span tagger; feature importance to explain *what* makes a rewrite unsafe | 5 |
| Bayes theorem | Posterior probability that a phrase is filler given context tokens | 5 |
| Maximum likelihood | MLE of per-phrase token cost distributions across tokenizers; MLE of HMM parameters from labelled sequences | 5 |
| Bayes optimal classifier | Posterior-weighted vote over the hypothesis set from Phase 1 version space; compared against single MAP hypothesis | 5 |
| Naive Bayes | Filler/hedge phrase detector (multinomial NB on n-grams) | 5 |
| EM and Gaussian Mixture Models | 2-component GMM on judge score deltas to separate "preserved" vs "degraded" populations **without labels**; used to sanity-check judge thresholds | 5 |
| Hidden Markov Models | Token-sequence tagger with hidden states {KEEP, DROP, REPLACE}; Viterbi decoding produces a compressed prompt; Baum-Welch on unlabelled data | 5 |

### Unit 4: Unsupervised Learning, Deployment, Advanced Paradigms

| Concept | Where it is used | Phase |
|---|---|---|
| Hierarchical vs non-hierarchical clustering | Discover prompt *intent* families; compare with the 6 hand-defined categories | 6 |
| Agglomerative and divisive clustering | Dendrogram of intents (Ward, average linkage); divisive via recursive bisection | 6 |
| K-means, bisecting K-means | Intent clusters for stratified sampling and per-cluster compression analysis | 6 |
| K-means as special case of EM | Show GMM with shared spherical covariance σ²I and σ → 0 recovers K-means assignments on our embeddings | 6 |
| RL: agent, environment, states, actions, rewards, policy, value functions | Rewriting as an MDP: state = current prompt draft, actions = edit operators, reward = token savings gated by quality | 8 |
| Bellman equation, Q-learning | Tabular Q-learning over a discretised operator environment; derivation in `docs/derivations/bellman.md` | 8 |
| Introduction to deep learning | Transformer rewriter (SFT), sentence encoders | 7 |
| Convolution operation and CNNs; parameter calculation; max vs avg pooling | 1-D TextCNN span-importance scorer over token embeddings; parameter counts documented; max vs avg pooling ablation | 7 |
| Hyperparameter tuning: Grid, Random, Bayesian optimisation | Grid (trees/SVM), Random (MLP/TextCNN), Bayesian/TPE (rewriter LoRA, RL) | 3–9 |
| MLOps overview, ML lifecycle, production challenges | Lifecycle doc and ADRs | 0, 10 |
| Data and feature pipelines | DVC stages plus a feature store for span/pair features | 2, 10 |
| Experiment tracking and model versioning | MLflow tracking and Model Registry; DVC data versions | 0+ |
| Model deployment and serving | FastAPI + vLLM / ONNX Runtime | 10 |
| CI/CD/CT for ML | GitHub Actions: CI (tests), CD (container release), CT (scheduled retraining on new data) | 10 |
| Inference systems | Two-tier inference: fast classifier path, generative path, with batching and caching | 10 |
| Monitoring, drift, observability | Evidently drift reports on prompt embeddings and token ratios; Prometheus/Grafana; shadow-judge sampling | 10 |
| ML orchestration | Prefect flows for data → train → eval → register → deploy | 10 |

---

## 5. Roadmap Overview

```
Week:     1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24
Phase 0  ███
Phase 1      ███████
Phase 2          ███████████████
Phase 3                      ███████████
Phase 4                              ███████████
Phase 5                                      ███████████
Phase 6                                              ███████
GPU gate                                         (whenever connected, ideally ≤ week 13)
Phase 7                                                      ███████████████
Phase 8                                                                  ███████████
Phase 9                                                                          ███████  (+ ongoing)
Phase 10                                                                             ███████████████
Phase 11                                                                                         ████
```

| Phase | Weeks | Compute | Key output | Blocking checkpoint(s) |
|---|---|---|---|---|
| 0 | 1 | CPU | Repo, CI, tracking, tokenizers | 0.1 |
| 1 | 2–3 | CPU, API | Task definition, operator lattice, version space | 1.1, 1.2 |
| 2 | 3–6 | API, CPU→GPU | Labelled dataset, validated judge, headroom estimate | 2.1–2.4 (**go / no-go for the whole project**) |
| 3 | 6–8 | CPU | Classical baselines, CV protocol, metric suite | 3.1–3.3 |
| 4 | 8–10 | CPU | MLP, optimizer study, SVM | 4.1–4.3 |
| 5 | 10–12 | CPU | Ensembles, NB, GMM, HMM tagger | 5.1–5.3 |
| 6 | 12–13 | CPU | Intent clusters, stratification refresh | 6.1 |
| GPU gate | when connected | GPU | Verified environment and real VRAM numbers | G.1 |
| 7 | 14–17 | GPU | TextCNN scorer, SFT rewriter | 7.1–7.4 |
| 8 | 17–19 | CPU, GPU, API | Q-learning study, RL-tuned rewriter | 8.1–8.3 |
| 9 | 19–20 | CPU, GPU | Tuned models, HPO report | 9.1 |
| 10 | 20–23 | CPU, GPU | Deployed service, CI/CD/CT, monitoring | 10.1–10.4 |
| 11 | 24 | all | Final benchmark, report, release | 11.1 |

> **If the GPU is still not connected by week 13:** Phases 8 (tabular part), 9 (classical HPO), and 10 (classifier-path serving, CI/CD, monitoring) are pulled forward. Phases 7 and 8 (neural parts) shift right. The schedule slips; the checkpoints do not get lowered.

---

## Phase 0 — Foundations and Tooling (Week 1)

**Compute:** [CPU]

### Goals
Set up a reproducible project skeleton so every later experiment is tracked, versioned, and comparable.

### Tasks
- [ ] `pyproject.toml` with extras `dev`, `gpu`; Python 3.11; pinned lockfile.
- [ ] Tooling: `ruff`, `black`, `mypy --strict` on `src/`, `pytest`, `pre-commit`.
- [ ] **Hydra** configs, including `configs/hardware/cpu.yaml` and `configs/hardware/rtx4500ada.yaml` (bf16, `device: cuda:0`, `max_vram_gb: 22`).
- [ ] **MLflow** local tracking server (`mlruns/`) and Model Registry.
- [ ] **DVC** initialised with local remote (move to cloud/NAS remote later).
- [ ] `terser.tokenizers`: a unified `count_tokens(text, target)` for:
  - `gpt-4o` / `o200k_base` via `tiktoken`
  - `llama-3.1` and `qwen2.5` via Hugging Face `tokenizers`
  - `claude` via the Anthropic token-counting endpoint (disk-cached, rate-limited)
- [ ] `terser.utils.seed.set_seed()` covering `random`, `numpy`, `torch`, CUDA.
- [ ] GitHub Actions CI: lint, type-check, `pytest -m "not gpu and not api"`.
- [ ] `gpu_queue/` schema and validator (§3.3).
- [ ] ADR-0001: *ML lifecycle for Terser* (data → label → train → evaluate → register → deploy → monitor → retrain).

### Checkpoint 0.1: Foundations ready
| Check | Pass condition |
|---|---|
| CI | Green on `main` |
| Tokenizer correctness | `count_tokens` matches each official tokenizer on a 200-string fixture set **exactly (100%)** |
| Reproducibility | Same seed, two runs of a dummy experiment → identical metrics logged to MLflow |
| CPU fallback | `hardware=rtx4500ada` config on a CPU machine fails with a clear error, and `device: auto` falls back to CPU |

---

## Phase 1 — Problem Framing and Concept Learning (Weeks 2–3)

**Compute:** [CPU], [API]

### Goals
Frame compression as a learning problem, define the rewrite operators, and use **concept learning** to see what a "safe rewrite" looks like before building any statistical model.

### 1A. ML framing (Unit 1: "ML and its models")
Write `docs/decisions/ADR-0002-problem-framing.md` defining the sub-problems and model family for each:

| Sub-problem | Learning type | Input → Output |
|---|---|---|
| Quality gate | Supervised binary classification | (original, rewrite, task) → {preserved, degraded} |
| Span tagger | Supervised sequence labelling | token sequence → {KEEP, DROP, REPLACE} per token |
| Compression-ratio predictor | Supervised regression | prompt → achievable savings % |
| Intent discovery | Unsupervised clustering | prompt embedding → cluster |
| Rewriter | Conditional generation | (prompt, target) → prompt' |
| Rewrite policy | Reinforcement learning | draft state → edit action |

### 1B. Rewrite operator lattice (hypothesis space)
Define an initial set of **deterministic, interpretable rewrite operators**:

| Op | Name | Example |
|---|---|---|
| `F` | Drop filler | "so basically, like, " → "" |
| `H` | Drop hedging / politeness | "could you maybe please" → "" |
| `D` | Deduplicate restated requirements | "make it short… keep it brief" → "keep it brief" |
| `I` | Imperative conversion | "I was wondering if you could explain" → "Explain" |
| `L` | List-ify multiple constraints | run-on constraints → bullet list |
| `C` | Compress meta-talk | "I'm going to give you some text and I want you to" → "" |
| `S` | Tokenizer-aware synonym swap | choose a lower-token synonym on target *T* |

A **hypothesis** *h* is a conjunctive constraint vector over applicability conditions, for example `<task=explain, has_constraints=?, length=long, ops={F,H,I}>`. `?` means "any value" and `∅` means "no value".

### 1C. Concept learning experiments
- [ ] Hand-label **150 prompts** (25 per category) with which operator combinations preserve quality (labels come from a pilot judge run, reviewed by a human).
- [ ] Implement **Find-S** and **Candidate-Elimination** over the attribute-value representation above.
- [ ] Compute the **version space** (S and G boundaries) per task category.
- [ ] Write down the **inductive bias**: what the conjunctive representation cannot express (for example "drop hedging *unless* the hedge carries tone for creative tasks"). This motivates the statistical models in Phases 3–5.

### 1D. Pilot tokenizer economics study
- [ ] For 500 casual prompts, apply each single operator and record Δtokens on all 4 tokenizers.
- [ ] Report the per-operator token-savings distribution and **cross-tokenizer rank correlation** (Spearman ρ) of savings.

### Checkpoint 1.1: Framing and operators
| Check | Pass condition |
|---|---|
| ADR-0002 merged | Reviewed by at least 1 other contributor |
| Operator implementation | Every operator has unit tests including **no-op on already-efficient prompts** |
| Version space | Non-empty version space for **≥ 4 of 6** categories. If empty, the report explains which examples conflict (expected for `creative`) |

### Checkpoint 1.2: Tokenizer-aware signal exists
| Check | Pass condition |
|---|---|
| Savings from operators alone | Median single-best operator saves **≥ 10%** tokens on the pilot set |
| Tokenizer dependence | Either Spearman ρ < 0.9 for at least one tokenizer pair (per-model optimisation matters), **or** a documented finding that the tokenizers agree, which simplifies Phase 7 to a single rewriter |

---

## Phase 2 — Data Generation Pipeline and Judge Validation (Weeks 3–6)

**Compute:** [API], [CPU], [CPU→GPU] for open-model targets

This is the **most important phase**. If the judge cannot be trusted, nothing downstream can be measured.

### 2A. Prompt collection (DVC stage `collect`)
- [ ] Sources: public instruction datasets with permissive licences (for example Dolly, OpenAssistant, ShareGPT-style datasets whose licence allows it, HumanEval/MBPP prompts, GSM8K problems).
- [ ] **Casualisation**: an LLM rewrites clean instructions into several casual styles (rambling, hedging, texting-style, over-polite, repetitive). Keep the clean version as a reference.
- [ ] Target size: **≥ 6,000 casual prompts**, **≥ 800 per category**, deduplicated (MinHash, Jaccard ≥ 0.8 removed).
- [ ] Split **by source prompt** into train 70% / val 15% / test 15%. The test set is frozen at the end of Week 4 and hash-locked in DVC.

### 2B. Candidate generation (stage `candidates`)
- [ ] For each casual prompt, generate **6 candidates**: 3 from the strong LLM at different aggressiveness levels, 2 from operator combinations (Phase 1), 1 from LLMLingua-2 (external baseline).
- [ ] Count tokens for every candidate on every target tokenizer.

### 2C. Target outputs (stage `respond`)
- [ ] Targets: **1 Claude model, 1 GPT model** (API), **Llama-3.1-8B-Instruct and Qwen2.5-7B-Instruct** (open).
  - Before the GPU: open targets via a hosted inference API, or restricted to 500 prompts on CPU with a 4-bit 1.5B stand-in (clearly labelled as a stand-in, not used for final claims).
  - After the GPU: re-run open targets locally with vLLM bf16 (queued job `respond-open-targets`).
- [ ] Temperature 0 where supported. For creative tasks, sample **n = 3** and compare distributions.

### 2D. Quality judging (stage `judge`)
- [ ] **Task metrics first**: tests for `code`, EM/F1 for `fact`, answer match for `reason`.
- [ ] **Pairwise LLM judge** for everything, with **position swapping** (judge both A/B orders; a "tie or better" label counts only if both orders agree).
- [ ] Per-category rubric in `src/terser/judge/rubrics/*.yaml` (coverage, correctness, depth, constraint adherence, tone).
- [ ] Label definition: `preserved = 1` if task metric not worse **and** judge says tie-or-better in both orders; otherwise `degraded`.

### 2E. Human validation of the judge
- [ ] Two humans independently label **300 pairs** (50 per category), blind to which prompt is the rewrite.
- [ ] Measure inter-annotator agreement, then judge-vs-human agreement.

### 2F. Headroom study
- [ ] "Oracle" = the shortest candidate that is still `preserved`. Measure oracle savings per category and per tokenizer.

### Checkpoint 2.1: Dataset integrity
| Check | Pass condition |
|---|---|
| Size | ≥ 6,000 casual prompts; ≥ 800 per category |
| Leakage | 0 source prompts shared across splits (automated test) |
| Duplicates | < 1% near-duplicates after MinHash dedup |
| Licence audit | Every source listed with licence in `data/SOURCES.md` |

### Checkpoint 2.2: Judge is trustworthy (**hard gate**)
| Check | Pass condition |
|---|---|
| Human–human agreement | Cohen's κ **≥ 0.60** (otherwise fix the labelling guideline first) |
| Judge–human agreement | Cohen's κ **≥ 0.60** overall and **≥ 0.50 in every category** |
| Judge as a detector of degradation | Using human labels as truth: **sensitivity ≥ 0.85**, **specificity ≥ 0.80** |
| Position bias | Disagreement between A/B orders **≤ 10%** |
| **If failed** | Iterate rubrics, try a stronger judge, or add task-specific metrics. **Phases 3+ do not start.** |

### Checkpoint 2.3: There is something to learn (**go / no-go**)
| Check | Pass condition |
|---|---|
| Oracle savings | Median oracle savings **≥ 25%** on at least **2 targets**, while preserved |
| Category coverage | Oracle savings ≥ 15% in **≥ 5 of 6** categories |
| Class balance | Degraded share among candidates between **15% and 60%** (enough negatives to learn a gate) |
| **If failed** | Write a negative-result report and narrow the scope (for example drop `creative`, or focus only on long casual prompts) before continuing |

### Checkpoint 2.4: Pipeline reproducibility
| Check | Pass condition |
|---|---|
| `dvc repro` | Rebuilds the processed dataset from cached API responses with an identical hash |
| Cost report | Actual API spend within **±20%** of estimate; cost per labelled pair recorded |

---

## Phase 3 — Metrics, Validation, Trees, Logistic Regression, KNN (Weeks 6–8)

**Compute:** [CPU]

### Goals
Build the **evaluation backbone** (metrics and cross-validation) and the first interpretable models for the **quality gate**, **span tagger**, and **compression-ratio predictor**.

### 3A. Metric suite (`terser.eval.metrics`)
Implemented and unit-tested against scikit-learn:

- Accuracy, Precision, Recall, **Sensitivity** (TPR), **Specificity** (TNR), F1.
- ROC curve and **AUC**; PR curve and average precision (degraded class is the minority).
- Threshold selection rule for the quality gate: **maximise specificity subject to sensitivity ≥ 0.95**. Letting a degraded rewrite through costs more than rejecting a good one.
- Confusion matrices **per task category**, always.

### 3B. Cross-validation protocol (`terser.eval.cv`)
| Strategy | Used for |
|---|---|
| `StratifiedGroupKFold(k=5)`, group = source prompt, stratify = category × label | Default model selection |
| Nested CV (outer 5 / inner 3) | Unbiased performance estimate while tuning |
| Leave-one-category-out | Generalisation to unseen task types |
| Repeated hold-out on time-ordered data | Simulated drift (used again in Phase 10) |

A test comparing plain `KFold` with grouped CV **must show the leakage gap**. It demonstrates why grouping is mandatory.

### 3C. Features (`terser.features`)
**Span-level** (for the tagger): token cost on target T, self-information under a small LM (GPT-2 small on CPU), is-stopword ratio, relative position, POS tags, is-in-filler-lexicon, dependency depth, contains-number/negation/constraint-keyword ("must", "only", "don't", "at least").

**Pair-level** (for the quality gate): token ratio, removed-span features aggregated (max self-information removed, count of negations removed, count of constraint keywords removed, number removed), embedding cosine similarity, task category one-hot.

### 3D. Decision trees
- [ ] **From-scratch ID3** (`models/classical/id3.py`) using information gain on discretised features; tested against a hand-computed toy example.
- [ ] **Continuous values**: C4.5-style threshold search (sort, evaluate midpoints between class changes).
- [ ] **Overfitting study**: train/val accuracy vs tree depth 1–30; min-samples-leaf sweep.
- [ ] **Solutions**: pre-pruning (depth, min-samples), **reduced-error pruning**, **rule post-pruning**. Compare all three.
- [ ] **Decision boundary plot** on 2 features (`self_information` vs `token_cost`): axis-aligned rectangles.
- [ ] **Hypothesis search and inductive bias**: short report on ID3's greedy search (preference bias for short trees with high-gain attributes near the root) vs the restriction bias of Phase 1's conjunctive hypotheses.

### 3E. Logistic regression
- [ ] From-scratch batch-GD implementation, checked against scikit-learn (coefficients within 1e-3).
- [ ] L1 and L2 regularisation; calibration curve (reliability diagram) and Brier score.
- [ ] Linear decision boundary plotted on the same 2-D feature plane as the tree.

### 3F. K-Nearest Neighbours
- [ ] **KNN classification** quality gate on pair embeddings plus scalar features (standardised).
- [ ] **KNN regression**: predict a prompt's oracle savings % from its *k* nearest training prompts. Metrics: MAE, R².
- [ ] Distance metrics: Euclidean vs cosine; distance-weighted vs uniform voting.
- [ ] **Decision boundary plots** for k = 1, 5, 25 on a 2-D PCA projection.

### 3G. Bias–variance
- [ ] Empirical **bias–variance decomposition** (200 bootstrap resamples, squared-loss for KNN regression; 0-1 loss decomposition for classifiers) for tree depth ∈ {2, 4, 8, 16, ∞} and KNN k ∈ {1, 3, 5, 15, 51}.
- [ ] Learning curves (train size 10%→100%) to diagnose high bias vs high variance.

### 3H. Grid search (first HPO method)
- [ ] `GridSearchCV` over tree depth × min-samples and KNN k × metric × weighting, with nested CV.

### Checkpoint 3.1: Evaluation backbone
| Check | Pass condition |
|---|---|
| Metric implementations | Match scikit-learn to 1e-9 on 20 random fixtures |
| Leakage demonstration | Grouped CV AUC is **lower** than naive KFold AUC. The gap is recorded (expected > 0) |
| Per-category reporting | Every model report includes the 6-category table |

### Checkpoint 3.2: Classical baselines worth beating
| Model | Task | Pass condition (grouped 5-fold CV) |
|---|---|---|
| Best of Tree / LR / KNN | Quality gate | **ROC-AUC ≥ 0.75**; at sensitivity 0.95, specificity **≥ 0.40** |
| Best of Tree / LR | Span tagger | Token-level macro-F1 **≥ 0.65** vs oracle alignment labels |
| KNN regression | Savings predictor | **MAE ≤ 10 percentage points**, R² ≥ 0.3 |
| ID3 (from scratch) | Span tagger | Within 2 F1 points of scikit-learn `DecisionTreeClassifier(criterion="entropy")` at the same depth |

### Checkpoint 3.3: Understanding, not just numbers
| Check | Pass condition |
|---|---|
| Bias–variance report | Identifies the depth / k at the minimum validation error and explains it in terms of bias vs variance |
| Overfitting report | Shows at least one pruning method closing the train/val gap by **≥ 50%** |
| Decision boundary figures | Tree, LR, KNN (3 k values) plotted on identical axes in `docs/checkpoints/3.3/` |

---

## Phase 4 — Neural Networks and SVMs (Weeks 8–10)

**Compute:** [CPU] (all models here are small enough for CPU)

### 4A. Perceptron
- [ ] From-scratch perceptron with step activation for the **filler vs non-filler span** subset.
- [ ] Show convergence on a linearly separable subset (perceptron convergence theorem) and non-convergence on the full set, which motivates multi-layer networks.

### 4B. Multi-layer network and backpropagation
- [ ] **From-scratch NumPy MLP** (forward, backward, update), with a **gradient check** against finite differences (relative error < 1e-6) and against `torch.autograd`.
- [ ] PyTorch MLP quality gate on pair features `[e_o, e_r, |e_o − e_r|, e_o ⊙ e_r]` with 384-d embeddings → **1536-d input**.
- [ ] Reference architecture and **parameter count** (must be stated in every NN report):

  | Layer | Shape | Parameters |
  |---|---|---|
  | Linear 1536 → 256 | W: 1536×256, b: 256 | 393,472 |
  | Linear 256 → 64 | W: 256×64, b: 64 | 16,448 |
  | Linear 64 → 1 | W: 64×1, b: 1 | 65 |
  | **Total** | | **409,985** |

### 4C. Activation function ablation
Same architecture, seed set {0,1,2,3,4}, epochs, and optimizer (Adam, lr=1e-3):

| Activation | Record |
|---|---|
| Step (with straight-through estimator for the gradient) | val AUC, epochs to plateau, gradient norm per layer |
| Sigmoid | same, plus fraction of saturated units |
| Tanh | same |
| ReLU | same, plus fraction of dead units |

### 4D. Optimizer comparison
Same architecture (ReLU), same seeds, and same **compute budget (number of gradient evaluations)**:

| Optimizer | Settings to sweep |
|---|---|
| Full-batch GD | lr ∈ {0.1, 0.03, 0.01} |
| SGD (mini-batch 64) | lr ∈ {0.1, 0.03, 0.01} |
| SGD + Momentum | lr ∈ {0.03, 0.01}, β ∈ {0.9, 0.99} |
| Adagrad | lr ∈ {0.1, 0.01} |
| Adam | lr ∈ {1e-3, 3e-4}, β₁=0.9, β₂=0.999 |

Deliverables: loss curves (mean ± std over seeds), wall-clock, final val AUC, and a written explanation tying the observed behaviour to each update rule.

### 4E. Support Vector Machines
- [ ] `docs/derivations/svm_dual.md`: margin 2/‖w‖, **primal** (hard and soft margin), **Lagrangian**, KKT conditions, the **dual** problem, recovering *w* and *b* from support vectors, and the **linear kernel** derivation (K(x, z) = xᵀz).
- [ ] **From-scratch dual solver** (using `cvxopt` QP) for the linear SVM span classifier. Verify support vectors and *w* against `sklearn.svm.SVC(kernel="linear")`.
- [ ] **Hard margin** on the linearly separable filler subset; **soft margin** with C ∈ {0.01, 0.1, 1, 10, 100} on the full set; plot margin width and number of support vectors vs C.
- [ ] **Polynomial kernel** (degree 2, 3) comparison: accuracy, AUC, training time, support vector count.
- [ ] Decision boundary plots on the 2-D feature plane, same axes as Phase 3.

### 4F. Random search (second HPO method)
- [ ] Random search (60 trials) over MLP width, depth, dropout, lr, weight decay. Compare against a same-budget grid and report best-found AUC vs trials.

### Checkpoint 4.1: Correctness of from-scratch implementations
| Check | Pass condition |
|---|---|
| Backprop gradient check | Relative error **< 1e-6** on all layers |
| SVM dual solver | Same support-vector set as scikit-learn; ‖w_ours − w_sklearn‖ / ‖w_sklearn‖ **< 1e-3** |
| Perceptron | Converges in finite epochs on separable subset (logged) |

### Checkpoint 4.2: Studies complete and explained
| Check | Pass condition |
|---|---|
| Activation study | 5 seeds × 4 activations; includes saturation/dead-unit analysis |
| Optimizer study | 5 optimizers at equal gradient-evaluation budget; written interpretation |
| Random vs grid | Plot of best AUC vs number of trials for both |

### Checkpoint 4.3: Better quality gate
| Check | Pass condition |
|---|---|
| Best of MLP / SVM quality gate | ROC-AUC **≥ Checkpoint 3.2 best + 0.03**, or a documented explanation of why it is not (and the classical model stays the champion) |
| Worst category | AUC ≥ 0.70 in every category |

---

## Phase 5 — Ensembles, Bayesian Learning, HMMs (Weeks 10–12)

**Compute:** [CPU]

### 5A. Combining weak learners
- [ ] **AdaBoost** with depth-1 stumps (from scratch plus scikit-learn check). Plot training error vs rounds and the weight distribution over hard examples. Inspect *which* pairs get upweighted.
- [ ] **Random forest**: n_estimators, max_features, and bootstrap sweeps; out-of-bag error vs CV error; permutation feature importance ("what makes a rewrite unsafe?").
- [ ] **Gradient boosting** (LightGBM): learning rate × n_estimators × max_depth with early stopping on the grouped validation fold; SHAP summary per category.

### 5B. Bayesian learning
- [ ] **Bayes theorem** worked example in docs: P(filler | phrase, context) from counts.
- [ ] **Maximum likelihood**:
  - MLE of the token-cost distribution of each high-frequency casual phrase per tokenizer (for example Poisson / categorical), used as a prior by the tokenizer-aware synonym operator `S`.
  - MLE of HMM transition/emission parameters from labelled sequences (5C).
- [ ] **Naive Bayes** filler/hedge detector (multinomial, n-grams 1–3, Laplace smoothing). Compare against LR on the same features.
- [ ] **Bayes optimal classifier**: treat the Phase 1 hypotheses (operator combinations) as the hypothesis set H. Compute P(h | D) from validation performance and classify "is operator set safe for this prompt?" by the posterior-weighted vote. Compare with the single MAP hypothesis.
- [ ] **EM and GMM**: fit a 2-component 1-D GMM (from-scratch EM, checked against scikit-learn) to the **continuous judge score delta** (rewrite score − original score). Check whether the unsupervised component split agrees with the thresholded labels (Adjusted Rand Index). This is an independent sanity check of the judge threshold chosen in Phase 2.

### 5C. Hidden Markov Model span tagger
- [ ] Hidden states: `KEEP`, `DROP`, `REPLACE`. Observations: discretised token classes (filler lexicon id, POS, cost bucket).
- [ ] Supervised MLE training from alignment labels; **Viterbi** decoding; forward algorithm for sequence likelihood.
- [ ] **Baum-Welch** (EM) on unlabelled casual prompts initialised from supervised parameters; compare F1 before/after.
- [ ] Generate compressed prompts by applying Viterbi tags, then run them through the Phase 2 judge on **500 val prompts** (API).

### Checkpoint 5.1: Ensembles
| Check | Pass condition |
|---|---|
| Champion quality gate (likely GBM) | ROC-AUC **≥ 0.85**; at sensitivity 0.95, specificity **≥ 0.60** |
| RF OOB vs CV | OOB AUC within **0.02** of grouped CV AUC |
| Feature importance | Top-10 features documented; at least one finding turned into a new feature or operator rule |

### Checkpoint 5.2: Bayesian components
| Check | Pass condition |
|---|---|
| Naive Bayes filler detector | Precision **≥ 0.85** at recall ≥ 0.70 on filler spans |
| Bayes optimal vs MAP | Report the difference; Bayes optimal must be **no worse** than MAP on validation accuracy |
| GMM agreement | ARI **≥ 0.4** between GMM components and judge labels; if lower, open a judge-threshold review issue |

### Checkpoint 5.3: First end-to-end non-generative compressor
| Check | Pass condition |
|---|---|
| HMM (or span tagger + quality gate) compressor | On 500 val prompts: median savings **≥ 15%** with quality preservation **≥ 90%** on at least one API target |
| Comparison | Beats the Phase 1 best single operator on savings at equal preservation |

---

## Phase 6 — Unsupervised Structure of Prompts (Weeks 12–13)

**Compute:** [CPU]

### Tasks
- [ ] Embed all train prompts (bge-small, 384-d; CPU is fine).
- [ ] **Non-hierarchical**: **K-means** (k-means++, k ∈ 4..30, elbow and silhouette) and **Bisecting K-means**. Compare SSE and cluster size balance.
- [ ] **Hierarchical**: **Agglomerative** (Ward, average, complete linkage) with dendrograms on a 2,000-prompt sample. **Divisive** clustering via recursive bisection (top-down), then compare the resulting trees.
- [ ] **K-means as a special case of EM**: implement GMM-EM with shared spherical covariance σ²I. Anneal σ → 0 and show that responsibilities become hard 0/1 and assignments match K-means (report agreement %). Written derivation in `docs/derivations/kmeans_em.md`.
- [ ] Compare clusters with the 6 hand-defined categories (ARI, NMI, contingency table).
- [ ] **Per-cluster compression analysis**: oracle savings and degradation rate per cluster. Find clusters where compression is dangerous.
- [ ] Update the sampling strategy: stratify later training/eval by **category × cluster**.

### Checkpoint 6.1: Structure discovered and used
| Check | Pass condition |
|---|---|
| Clustering quality | Silhouette **≥ 0.10** for the chosen k (text embeddings are noisy, so the threshold is modest) |
| K-means ⇔ EM | Assignment agreement **≥ 98%** at the smallest σ |
| Actionable finding | At least one "high-risk cluster" (degradation rate ≥ 1.5× average) identified and added to the eval report as a tracked slice |
| Sampling update | DVC stage `stratify` updated and `dvc repro` passes |

---

## GPU Onboarding Gate (whenever the RTX 4500 Ada is connected)

**Compute:** [GPU]. **Run this before any queued GPU job.**

### Connection setup
- [ ] Remote access to the GPU machine (VS Code Remote-SSH or equivalent). Repo cloned there, `.env` provisioned out-of-band.
- [ ] NVIDIA driver and CUDA runtime compatible with the pinned PyTorch build. `nvidia-smi` shows the RTX 4500 Ada with 24 GB.
- [ ] `pip install -e ".[dev,gpu]"`; DVC pull of processed data; MLflow pointed at the shared tracking server.

### Verification script: `python -m terser.hardware.verify`
- [ ] `torch.cuda.is_available()`, device name, total memory, compute capability (8, 9).
- [ ] bf16 matmul benchmark and numeric parity with fp32 on a fixed seed.
- [ ] vLLM smoke test: load Qwen2.5-0.5B-Instruct, generate 32 tokens.
- [ ] **Real VRAM measurements** replacing the estimates in §3.4:
  - vLLM Llama-3.1-8B-Instruct bf16 at `gpu_memory_utilization=0.90`, max model len 4096: peak VRAM, tokens/s at batch 1 and 16.
  - SFT step on Qwen2.5-0.5B full FT, and Qwen2.5-1.5B LoRA, seq len 1024, micro-batch 4 with gradient checkpointing: peak VRAM, step time.
- [ ] Thermal / power check: 30-minute sustained training step, no throttling beyond 5% throughput drop.
- [ ] Run all `@pytest.mark.gpu` tests.

### Checkpoint G.1: GPU ready
| Check | Pass condition |
|---|---|
| Environment | Verification script all green; results committed to `docs/checkpoints/G.1/hardware.md` |
| Memory plan | §3.4 table updated with measured values; every queued job's `est_vram_gb` ≤ measured headroom (22 GB) |
| Parity | A Phase 4 MLP trains to the same val AUC (± 0.005) on GPU as on CPU with the same seed |
| Queue | `gpu_queue` dry-run orders jobs without dependency errors |

---

## Phase 7 — Deep Learning: CNN Span Scorer and Generative Rewriter (Weeks 14–17)

**Compute:** [GPU] primarily; [CPU] for smoke tests; [API] for judging

### 7A. Deep learning foundations (docs + small experiments)
- [ ] `docs/derivations/dl_intro.md`: representation learning, why deep models beat hand features for span importance, overview of the transformer decoder used by the rewriter.

### 7B. TextCNN span-importance scorer
Input: token embeddings (d = 384) for a prompt. Output: a KEEP probability per token, plus a prompt-level "compressibility" score from a pooled head.

- [ ] **Convolution operation** explained and unit-tested (manual 1-D convolution == `nn.Conv1d` on fixture input).
- [ ] Architecture: parallel Conv1d branches with kernel sizes 3 and 5, 128 filters each, 'same' padding → per-token head. Plus a pooled prompt-level head.
- [ ] **Parameter calculation** (must match `sum(p.numel())`):

  | Layer | Formula | Parameters |
  |---|---|---|
  | Conv1d(384 → 128, k=3) | 384 × 3 × 128 + 128 | 147,584 |
  | Conv1d(384 → 128, k=5) | 384 × 5 × 128 + 128 | 245,888 |
  | Per-token head Linear(256 → 1) | 256 × 1 + 1 | 257 |
  | Prompt head Linear(256 → 1) (after pooling) | 256 × 1 + 1 | 257 |
  | **Total** | | **393,986** |

- [ ] **Pooling ablation** for the prompt-level head: global **max pooling** vs global **average pooling** vs both concatenated (the concatenated variant's head becomes Linear(512 → 1) = 513 parameters). Report which works better for compressibility regression and explain why (max = "is there any droppable region", avg = "how much of the prompt is droppable").
- [ ] Compare against the HMM and GBM span taggers from Phase 5.

### 7C. Generative rewriter (supervised fine-tuning)
- [ ] Training pairs: (casual prompt, **oracle best preserved candidate**) from Phase 2, train split only. Conditioning prefix `<target=llama-3.1>` etc. for a single multi-target model.
- [ ] Model ladder (stop at the smallest one that passes Checkpoint 7.3):

  | Rung | Model | Method | Queue job |
  |---|---|---|---|
  | 1 | Qwen2.5-0.5B-Instruct | Full FT, bf16 | `sft-qwen0.5b-full` |
  | 2 | Qwen2.5-1.5B-Instruct | LoRA r=16, α=32, bf16 | `sft-qwen1.5b-lora` |
  | 3 | Qwen2.5-3B-Instruct | LoRA r=16, bf16, grad checkpointing | `sft-qwen3b-lora` |
  | 4 | 7B–8B instruct | QLoRA NF4 | `sft-7b-qlora` (only if 1–3 fail) |

- [ ] Constrained decoding safeguards: never output longer than the input (token count on target); fall back to the original prompt if the rewrite is longer.
- [ ] Inference-time **quality gate**: champion gate from Phase 5 (or MLP/TextCNN if better) rejects risky rewrites → fall back to original or a lighter rewrite.
- [ ] Baselines: original prompt, LLMLingua-2, HMM compressor, best operator, and zero-shot strong LLM ("rewrite concisely").

### 7D. Local judge (optional, cost reduction)
- [ ] vLLM-served 8B judge (AWQ int4). Measure agreement with the API judge on 1,000 pairs.

### Checkpoint 7.1: Local judge (only if 7D is attempted)
| Check | Pass condition |
|---|---|
| Local vs API judge | Cohen's κ **≥ 0.70**; otherwise the API judge stays in use for all reported results |

### Checkpoint 7.2: TextCNN
| Check | Pass condition |
|---|---|
| Parameter count | Matches the documented table exactly |
| Span tagging | Token macro-F1 **≥ Checkpoint 3.2 best + 0.05**, or documented reason |
| Pooling ablation | 3 variants × 3 seeds, results table and interpretation |

### Checkpoint 7.3: Rewriter quality (**main gate**)
Evaluated on the **validation** split. The test split stays untouched until Phase 11.

| Check | Pass condition |
|---|---|
| Token savings | Median **≥ 25%** on every target |
| Quality preservation | **≥ 92%** overall; **≥ 85%** per category (with gate + fallback enabled) |
| vs zero-shot strong LLM | At equal preservation (±1 pp), savings within **5 pp** of the zero-shot LLM while the rewriter is **≥ 10× cheaper** per call |
| vs LLMLingua-2 | Higher preservation at equal or better savings on casual prompts |
| Failure analysis | 100 degraded cases hand-inspected and categorised (dropped constraint / dropped tone / dropped example / changed meaning / other) |

### Checkpoint 7.4: Tokenizer-awareness
| Check | Pass condition |
|---|---|
| Per-target conditioning | Rewrites conditioned on target T save more tokens **on T's tokenizer** than rewrites conditioned on another target, by **≥ 2 pp** median, **or** a documented finding that target conditioning does not matter (in which case ship a single target-agnostic rewriter) |

---

## Phase 8 — Reinforcement Learning with a Quality Gate (Weeks 17–19)

**Compute:** [CPU] for tabular RL; [GPU] + [API] for policy optimisation

### 8A. RL fundamentals, applied to rewriting
`docs/derivations/bellman.md` defines the rewriting MDP:

| RL element | Terser definition |
|---|---|
| **Agent** | The rewriter |
| **Environment** | Prompt draft + target tokenizer + quality gate / judge |
| **State s** | Current draft (tabular version: discretised features, e.g. remaining filler count, constraint count, length bucket, category) |
| **Action a** | Apply one edit operator (F, H, D, I, L, C, S) or `STOP` |
| **Reward r** | On `STOP`: `r = Δtokens% − λ·1[degraded]` (λ large, e.g. 100). Per step: small −0.01 to discourage loops |
| **Policy π(a \| s)** | Which operator to apply next |
| **Value functions** | V^π(s), Q^π(s, a) = expected return of applying a in s |
| **Bellman optimality** | Q*(s, a) = E[r + γ · max_a′ Q*(s′, a′)] |

### 8B. Tabular Q-learning
- [ ] Discretised environment using the operator set and the **quality gate classifier as a cheap reward model** (no API calls during learning).
- [ ] ε-greedy Q-learning with α and γ sweeps. Plot learning curves and the greedy policy per category.
- [ ] Validate the learned operator orderings with the real judge on 300 val prompts (API).

### 8C. Policy optimisation of the neural rewriter
- [ ] Start from the Checkpoint 7.3 SFT model.
- [ ] **GRPO** (group-relative, no value network, lighter on 24 GB) with LoRA; group size 4–8 samples per prompt.
- [ ] Reward: `tokens_saved_pct × gate_pass` where `gate_pass` comes from the quality gate. Every N steps, a **judge audit** on a sample checks for reward hacking (the policy fooling the gate).
- [ ] Target LLM calls for reward audits go through the **API** (or a vLLM server in a separate time slot), **never colocated** with training on the GPU.
- [ ] KL penalty to the SFT reference to preserve fluency.

### Checkpoint 8.1: Tabular RL
| Check | Pass condition |
|---|---|
| Convergence | Q-table change **< 1e-3** over the last 1,000 episodes |
| Policy value | Greedy policy's judge-verified savings **≥** best fixed operator order at equal preservation |

### Checkpoint 8.2: No reward hacking
| Check | Pass condition |
|---|---|
| Gate vs judge on RL samples | Gate precision on RL-generated rewrites within **5 pp** of its precision on Phase 2 data |
| Degenerate outputs | < 1% empty, truncated, or non-natural-language rewrites on 1,000 samples |

### Checkpoint 8.3: RL improves on SFT
| Check | Pass condition |
|---|---|
| Savings | **≥ SFT + 3 pp** median savings |
| Preservation | **Not lower** than SFT by more than 1 pp overall, and no category drops by more than 2 pp |
| **If failed** | Keep the SFT model as champion; document the RL result |

---

## Phase 9 — Hyperparameter Optimisation (Weeks 19–20, plus ongoing)

**Compute:** [CPU] for classical models; [GPU] for rewriter / TextCNN

Grid search (Phase 3) and random search (Phase 4) are already in use. This phase adds **Bayesian optimisation** and a formal comparison.

### Tasks
- [ ] **Bayesian optimisation** with Optuna (TPE sampler) and a Gaussian-process sampler for low-dimensional spaces.
  - Quality gate GBM: learning rate, num_leaves, min_child_samples, feature_fraction, regularisation.
  - TextCNN: kernel sizes, filters, dropout, lr.
  - Rewriter LoRA: rank, α, lr, warmup, epochs. Objective: `savings_pct` subject to `preservation ≥ 0.92` (penalise infeasible trials). **Median pruner** kills bad GPU trials early.
- [ ] **Method comparison** on the GBM quality gate at a fixed budget of 60 trials: Grid vs Random vs Bayesian. Best-so-far curves over 5 repeats.
- [ ] All trials logged to MLflow with parent/child runs.

### Checkpoint 9.1: HPO
| Check | Pass condition |
|---|---|
| Method comparison | Plot and table for Grid vs Random vs Bayesian at equal budget, 5 repeats |
| Improvement | Tuned champion quality gate AUC **≥ untuned + 0.01**, or documented plateau |
| Rewriter | Tuned rewriter passes Checkpoint 7.3 thresholds with **≥ 1 pp** more savings, or documented plateau |
| No test leakage | HPO touched only train/val (automated check on data access logs) |

---

## Phase 10 — MLOps: Deployment, Serving, CI/CD/CT, Monitoring (Weeks 20–23)

**Compute:** [CPU] for pipelines/CI; [GPU] for generative serving and load tests

### 10A. MLOps overview and lifecycle
- [ ] `docs/decisions/ADR-0010-mlops-architecture.md` covering the lifecycle diagram and production challenges specific to Terser: judge cost, tokenizer drift, silent over-compression, latency budget, and training/serving skew in feature computation.

### 10B. Data and feature pipelines
- [ ] DVC pipeline stages finalised: `collect → casualise → candidates → respond → judge → label → features → split → stratify`.
- [ ] Shared feature code: **one** `terser.features` implementation used by both training and serving (a test asserts identical features for 100 prompts computed offline vs through the API).
- [ ] Data validation (pandera / Great Expectations): schema, null rates, token-ratio ranges, category balance.

### 10C. Experiment tracking and model versioning
- [ ] MLflow Model Registry stages: `None → Staging → Production → Archived`.
- [ ] Every registered model records: git SHA, DVC data hash, config, eval report, checkpoint IDs passed.
- [ ] Promotion to `Production` requires the Phase 11 evaluation harness to pass on the frozen **validation** set and a human approval.

### 10D. Model deployment and serving
- [ ] FastAPI service `POST /rewrite {prompt, target, mode: "fast"|"quality"}` → `{rewrite, tokens_before, tokens_after, gate_score, fallback_used}`.
- [ ] **Fast path** [CPU-capable]: span tagger (GBM/HMM/TextCNN exported to ONNX) + quality gate.
- [ ] **Quality path** [GPU]: generative rewriter served by vLLM (continuous batching) + quality gate + fallback.
- [ ] Docker images: `terser-cpu`, `terser-gpu` (CUDA base). Health and readiness endpoints.

### 10E. Inference systems
- [ ] Request batching; LRU cache keyed on normalised prompt hash + target.
- [ ] Timeouts: if the quality path exceeds its latency budget, return the fast-path result.
- [ ] **Net-savings accounting** on every request: `net = price_T × (tokens_before − tokens_after) − rewriter_cost`. Logged per request.

### 10F. CI / CD / CT
| Pipeline | Trigger | Steps |
|---|---|---|
| **CI** | Every PR | lint, types, unit tests, CPU smoke training, `dvc repro --dry`, feature parity test |
| **CD** | Tag `v*` on `main` | Build and scan images, deploy to staging, run smoke + latency tests, manual approval → production |
| **CT** (continuous training) | Weekly schedule or drift alert | Pull new logged prompts (consented) → label via judge → retrain gate / tagger (CPU) → queue rewriter retrain (GPU) → evaluate → register as `Staging` if checkpoints pass |

### 10G. Monitoring, drift, observability
- [ ] Metrics (Prometheus): request rate, p50/p95/p99 latency per path, fallback rate, gate rejection rate, tokens saved, net savings, GPU utilisation / VRAM.
- [ ] **Drift** (Evidently, daily): input drift on prompt embeddings (MMD or PSI on PCA components), length and category distribution, token-ratio distribution; **tokenizer drift** check (re-count a fixture set with the current tokenizer versions and alert on any change).
- [ ] **Shadow judging**: 1% of production rewrites (consented) sent to the judge asynchronously. Rolling preservation rate with alerting.
- [ ] Grafana dashboard and alert rules; structured logs with request IDs (no raw prompt text unless consented, see §23).

### 10H. ML orchestration
- [ ] Prefect flows: `data_flow`, `train_classical_flow`, `train_rewriter_flow` (submits to `gpu_queue`), `evaluate_flow`, `promote_flow`, `drift_flow`.
- [ ] Retries, caching of expensive API steps, and failure notifications.

### Checkpoint 10.1: Pipelines and versioning
| Check | Pass condition |
|---|---|
| End-to-end | Prefect `data_flow → train_classical_flow → evaluate_flow → promote_flow` runs unattended on a fresh clone (CPU) |
| Lineage | Any production model traceable to git SHA + data hash + checkpoint reports |
| Feature parity | Offline == online features for 100/100 prompts |

### Checkpoint 10.2: Serving performance (on RTX 4500 Ada)
| Check | Pass condition |
|---|---|
| Fast path latency | p95 **≤ 150 ms** at 20 RPS (CPU or GPU) |
| Quality path latency | p95 **≤ 400 ms** at 5 RPS for prompts ≤ 256 tokens |
| Throughput | Quality path ≥ 20 RPS sustained without OOM for 30 minutes |
| Fallback correctness | Fault-injection test: the rewriter timing out or crashing returns the original prompt 100% of the time |

### Checkpoint 10.3: CI/CD/CT
| Check | Pass condition |
|---|---|
| CI | Median PR pipeline **≤ 15 min** |
| CD | Staging deploy + smoke tests automated; production requires approval |
| CT | One full scheduled retrain cycle completed and correctly **refused** promotion when a synthetic regression was injected |

### Checkpoint 10.4: Monitoring
| Check | Pass condition |
|---|---|
| Drift detection | A synthetic distribution shift (e.g. only `code` prompts, or a changed tokenizer version) triggers an alert within one monitoring cycle |
| Shadow judge | Rolling preservation metric visible on dashboard with an alert at < 92% |

---

## Phase 11 — Final Evaluation, Benchmark Release, Write-up (Week 24)

**Compute:** all

### Tasks
- [ ] Unlock the **frozen test split** (logged event, once).
- [ ] Full three-axis evaluation for every system: original, best operator, HMM, GBM-tagger+gate, TextCNN+gate, SFT rewriter, RL rewriter, LLMLingua-2, zero-shot strong LLM.
- [ ] Report per target × per category × per high-risk cluster, with **95% bootstrap confidence intervals** (1,000 resamples).
- [ ] **Cross-model generalisation matrix**: rewriter conditioned on target A, evaluated on target B, for all pairs.
- [ ] Net savings analysis across prompt-length percentiles, including rewriter cost.
- [ ] Error taxonomy of 200 degraded test cases.
- [ ] Publish the **casual-prompt compression benchmark** (prompts, candidates, labels, judge rubrics, evaluation code) if licences allow.
- [ ] Final report `docs/report.md` covering method, results, negative results, limitations, and syllabus concept mapping with links to evidence.

### Checkpoint 11.1: Project success criteria (§1.1)
| Check | Pass condition |
|---|---|
| Savings | Median **≥ 30%** (95% CI lower bound ≥ 25%) |
| Preservation | **≥ 95%** overall; **≥ 90%** every category |
| Net savings | Positive at P25 prompt length for every target |
| Latency | Checkpoint 10.2 re-verified on release build |
| Generalisation | ≤ 3 pp cross-model preservation drop, or per-model variants shipped and documented |
| Reproducibility | A second contributor reproduces headline numbers within CI bounds from a fresh clone |

Missing a target is **not** hidden. The report states which targets were met, which were not, and why.

---

## 19. Checkpoint Protocol

1. **Open a checkpoint issue** using the `checkpoint` template when a phase's work is believed complete: `Checkpoint 3.2 — Classical baselines`.
2. **Write the report** at `docs/checkpoints/<id>/README.md` containing:
   - Each check from the table, its **measured value**, pass/fail, and a link to the MLflow run(s).
   - Data version (DVC hash), git SHA, config, seeds.
   - Plots referenced by the checkpoint.
   - Deviations and known issues.
3. **Review**: at least **one reviewer who did not do the work** re-runs at least one headline number from the report.
4. **Decision** recorded on the issue: `PASS`, `PASS WITH CONDITIONS` (conditions become tracked issues due before the next checkpoint), or `FAIL` (see below).
5. **On FAIL**: the phase continues. You may **not** start dependent work. Changing a threshold requires an ADR explaining why the original threshold was wrong. "We didn't reach it" is not a valid reason.
6. **Hard gates**: Checkpoints **2.2**, **2.3**, **G.1** (for GPU work), **7.3**, and **11.1**. Failing these blocks all dependent phases, with no conditional pass.

---

## 20. Experiment Standards

- **Tracking**: every run logs to MLflow with tags `phase`, `checkpoint`, `task` (`quality_gate` / `span_tagger` / `savings_regressor` / `rewriter` / `policy`), `hardware` (`cpu` / `rtx4500ada`), `data_hash`, `git_sha`.
- **Seeds**: report mean ± std over **≥ 3 seeds** (≥ 5 for studies in Phase 4).
- **Splits**: never tune on test. The test split is read-only and access is logged. Any code touching `data/processed/test*` outside `terser.eval.final` fails CI.
- **Stratified reporting**: every result table includes per-category rows. An aggregate alone is not accepted in PR review.
- **Baselines**: every new model is compared against the current champion **and** the "original prompt" baseline (0% savings, 100% preservation).
- **Statistical comparisons**: paired bootstrap or McNemar's test when claiming one model beats another. Report p-values or CIs.
- **Costs**: log API tokens and $ per experiment; log GPU hours per job.
- **Negative results** are committed as checkpoint reports too. They are results.

---

## 21. Coding Standards

- Python 3.11, type hints everywhere in `src/`, `mypy --strict` clean.
- Formatting: `black`, linting: `ruff`. Both enforced by pre-commit.
- **From-scratch implementations** (ID3, logistic regression, KNN, perceptron, MLP backprop, SVM dual, AdaBoost, NB, EM/GMM, HMM, K-means, bisecting K-means, Q-learning) live in `src/terser/models/**/scratch_*.py`. Each one has:
  - a docstring with the algorithm and a reference to its derivation in `docs/derivations/`,
  - a test comparing against a trusted library implementation, or against a hand-computed example.
- Library implementations (scikit-learn, LightGBM, PyTorch, HF) are used for production models.
- Device-agnostic code: no hard-coded `"cuda"`. Use `terser.hardware.get_device(cfg)`.
- No notebook code imported into `src/`. Notebooks are for exploration and must be cleared of outputs before commit (`nbstripout`).
- Configuration through Hydra. No magic numbers in training code.
- Tests: `pytest`; markers `slow`, `gpu`, `api`. New code needs tests; target **≥ 80%** line coverage for `src/terser/{tokenizers,eval,features,judge}`.

---

## 22. Git Workflow and Pull Requests

### Branches
- `main`: always green, protected.
- Feature branches: `phase<N>/<short-topic>`, e.g. `phase3/id3-scratch`, `phase7/textcnn-pooling`.
- Checkpoint branches: `checkpoint/<id>` for report-only PRs.

### Commits
Conventional Commits:
```
feat(judge): add position-swapped pairwise judging
fix(tokenizers): cache Claude token counts per model version
exp(phase4): optimizer comparison at equal gradient budget
docs(derivations): SVM Lagrangian dual
chore(gpu_queue): add sft-qwen1.5b-lora job
```

### Pull request checklist
Every PR description must include:

- [ ] Linked issue and **syllabus matrix row(s)** (§4) if applicable
- [ ] Phase and checkpoint this contributes to
- [ ] Compute tag(s): [CPU] / [API] / [GPU]. For GPU work: queue job file added and CPU smoke test passing
- [ ] Tests added/updated; `pytest -m "not gpu and not api"` passes locally
- [ ] MLflow run links for any reported number
- [ ] Per-category results (if the PR reports model performance)
- [ ] API spend for this PR's experiments (if any)
- [ ] No secrets, no raw personal data, no test-split access
- [ ] Docs / derivations updated

### Review
- At least **1 approval**. Changes to `judge/`, `eval/`, data splits, or checkpoint thresholds need **2 approvals**.
- Squash-merge into `main`.

---

## 23. Data, Privacy, and API Usage Policy

- **Licences**: only datasets whose licences permit research use and (for benchmark release) redistribution. Recorded in `data/SOURCES.md`.
- **Personal data**: never collect real user prompts without explicit consent. Production logging stores hashes and metrics by default; raw text only for opted-in requests, retained ≤ 30 days.
- **PII scrubbing**: run a PII detector over all collected text before it enters `data/interim/`.
- **Secrets**: `.env` only, never committed. CI uses repository secrets.
- **API budget**: each phase has a spend cap in `configs/budget.yaml`. The API client refuses calls once the cap is reached. All responses are cached on disk (keyed by model + params + prompt hash) so re-runs are free.
- **Provider terms**: using model outputs to train the rewriter must comply with each provider's terms of service. Document the check in ADR-0003 before Phase 7 begins.

---

## 24. Risk Register

| Risk | Likelihood | Impact | Mitigation | Owner checkpoint |
|---|---|---|---|---|
| Judge disagrees with humans | Medium | Critical | Hard gate 2.2; task metrics first; position swapping; GMM sanity check (5.2) | 2.2, 5.2 |
| Too little compression headroom | Low–Medium | Critical | Go/no-go 2.3; scope narrowing plan | 2.3 |
| GPU arrives late | Medium | High | CPU-first ordering; GPU queue; pull Phases 9/10 CPU parts forward | G.1 |
| 24 GB VRAM insufficient for chosen rewriter | Low | Medium | Model ladder (0.5B → 7B QLoRA); gradient checkpointing; no co-located serving and training | G.1, 7.3 |
| Silent over-compression (dropped constraints) | High | High | Constraint-keyword features, quality gate with sensitivity ≥ 0.95, fallback, error taxonomy, shadow judging | 7.3, 10.4 |
| RL reward hacking | Medium | High | Periodic judge audits, KL penalty, Checkpoint 8.2 | 8.2 |
| Rewriter overhead wipes savings | Medium | High | Net-savings metric; fast CPU path; skip rewriting short prompts (predicted savings from KNN regressor below threshold) | 10.2, 11.1 |
| Tokenizer drift | Medium | Medium | Tokenizer fixture test in monitoring; CT retraining | 10.4 |
| API costs overrun | Medium | Medium | Budget caps, disk caching, local judge (7.1) | 2.4 |
| Train/test leakage via paraphrases | Medium | High | Group-by-source splits; automated leakage tests | 2.1, 3.1 |

---

## 25. Glossary

| Term | Meaning in this project |
|---|---|
| **Target (T)** | The LLM that will receive the rewritten prompt; determines the tokenizer |
| **Casual prompt** | A naturally phrased, possibly redundant user instruction |
| **Candidate** | A proposed rewrite of a casual prompt |
| **Oracle rewrite** | The shortest candidate that is still judged `preserved` |
| **Preserved / Degraded** | Label from task metrics + position-swapped pairwise judge |
| **Quality gate** | Classifier predicting `degraded` for (original, rewrite); positive class = degraded |
| **Span tagger** | Model labelling tokens KEEP / DROP / REPLACE |
| **Fast path / Quality path** | Non-generative (tagger + gate) vs generative (rewriter + gate) serving modes |
| **Net savings** | Token cost saved on T minus the rewriter's own cost |
| **Hard gate** | A checkpoint that blocks dependent work with no conditional pass |
| **GPU queue** | Job specs in `gpu_queue/` waiting for the RTX 4500 Ada |

---

*Questions or proposals to change the plan: open an issue with the `plan-change` label and, for threshold or scope changes, an ADR in `docs/decisions/`.*
