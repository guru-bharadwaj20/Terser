# Terser: Final Report

Plan row 13.6. This report describes the problem, the method, the validation-split experiments and
every negative result of the Terser project. Test-split results and ablations come from the Phase 12
evaluation ([final_results.md](../reports/final_results.md)), reported in § 5.1.

Every number below comes from a file in [`reports/`](../reports/), linked where it is used. The
checkpoint ids (C*.*) and row ids refer to [CONTRIBUTING.md](../CONTRIBUTING.md); the current pass/fail
state of every checkpoint is in [checkpoints/status.md](checkpoints/status.md).

**Summary.** Casual prompts do contain removable text: a per-prompt oracle over judged candidate
rewrites reaches median savings of 46.5% (Qwen2.5) and 40.5% (Phi-3.5) at preserved quality
([p2_headroom.md](../reports/p2_headroom.md)). None of the systems built so far captures that
headroom at the required quality. Rule operators save 1.7% (C1.4). The supervised fine-tuned (SFT)
rewriters trade savings for preservation along a curve that never reaches the C8.3 bar of 25%
savings at 92% preservation: 35.0% savings at 57.0% preserved for the 1.5B LoRA, 2.0% at 75.3% for
the 7B QLoRA ([p8_rewriter_val.md](../reports/p8_rewriter_val.md)). The serving system is complete
and safe (fallbacks are 100% correct, the fast path meets its latency target), but its quality
path cannot answer inside the 0.4 s budget, and its quality gate rejects almost every rewrite, so
in practice the service returns the original prompt. No judgement in this report has been
checked against human labels yet.

---

## 1. Problem

### 1.1 Task

A user writes a prompt `x` for a target model `T`. Terser produces a rewrite `x'` with two goals:

- **Savings.** `(n_T(x) - n_T(x')) / n_T(x)`, where `n_T` counts tokens on the target's own
  tokenizer ([ADR-0002](decisions/ADR-0002-problem-framing.md),
  [p2_headroom.md](../reports/p2_headroom.md)).
- **Preservation.** The target's answer to `x'` is as good as its answer to `x`. A rewrite is
  `preserved` iff the category's task metric is not worse **and** a pairwise LLM judge says
  tie-or-rewrite-better in both presentation orders (row 2.3.4). Otherwise it is `degraded`.

Six task categories carry their own quality measures: `code` (unit tests), `fact` (EM / F1),
`reason` (final-answer accuracy), `explain`, `creative` and `transform` (rubric judge)
([categories.md](categories.md)).

### 1.2 Why it is hard

- Savings and preservation pull in opposite directions; the cost of a dropped constraint is a
  wrong answer, which is much larger than the value of a few saved tokens.
- The quality of an answer can only be measured by generating it and judging it, which is
  expensive and, with an LLM judge, noisy.
- The rewriter itself costs compute and latency, so a rewrite can cost more than it saves.

### 1.3 Decomposition

[ADR-0002](decisions/ADR-0002-problem-framing.md) frames six learning sub-problems, each covered
by a phase:

| Sub-problem | Type | Where |
| :--- | :--- | :--- |
| Quality gate: is a rewrite degraded? | binary classification (positive = degraded) | Phases 3, 4, 5, 10, 11 |
| Span tagger: KEEP / DROP / REPLACE per token | sequence labelling | Phases 3, 5 (HMM), 8 (TextCNN), 11 |
| Savings predictor | regression | Phase 3 (KNN) |
| Intent discovery | clustering | Phase 6 |
| Rewriter | conditional generation | Phase 8 (SFT) |
| Rewrite policy over operators | reinforcement learning | Phase 9 |

### 1.4 Success criteria

The headline criteria S1–S10 are evaluated once, on the frozen test split, in Phase 12. The main
ones are median savings ≥ 30% (S1), preservation ≥ 95% overall (S2) and ≥ 90% in the worst
category (S3), positive net savings after rewriter cost (S4), and per-target robustness with one
target-agnostic rewriter (S5) ([CONTRIBUTING.md § Headline Success Criteria](../CONTRIBUTING.md#headline-success-criteria)).
On the validation split the gate for the rewriter is C8.3: median savings ≥ 25% on every target,
preservation ≥ 92% overall and ≥ 85% per category.

---

## 2. Related work

**Selective Context** (Li et al., 2023) prunes a context by the self-information of its lexical
units, computed with a small causal language model: tokens, phrases or sentences that the small
model finds predictable carry little information and are removed until a target ratio is reached.
It is task-agnostic and needs no training. Terser's span features include the GPT-2
self-information of each token for the same reason (row 3.1.2).

**LLMLingua** (Jiang et al., 2023) compresses prompts coarse-to-fine with a small language model:
a budget controller allocates compression ratios across prompt parts (instruction, demonstrations,
question), demonstrations are dropped at a coarse level, and an iterative token-level step removes
tokens the small model rates as predictable, conditioned on what has already been kept. The small
model can be instruction-tuned to align its distribution with the target LLM.

**LLMLingua-2** (Pan et al., 2024) recasts compression as token classification. Compression
labels are distilled from a large LLM's extractive compressions (on MeetingBank transcripts), and a
bidirectional Transformer encoder is trained to predict keep or drop for each token. It is
task-agnostic, faster than the perplexity-based methods, and only deletes tokens; it never
rewrites. Terser uses the released `llmlingua-2-bert-base-multilingual-cased-meetingbank` model at
rate 0.5 as a candidate generator and as a baseline ([p2_candidates.md](../reports/p2_candidates.md)).

**How Terser differs.** All three methods delete tokens from a given text and are usually evaluated
on long contexts (documents, demonstrations, transcripts). Terser targets short, conversational
prompts whose redundancy is phrasing (filler, hedging, politeness, restated requirements), counts
savings on each target's own tokenizer, and allows generative rewriting as well as deletion. Its
quality measure is the target model's answer to the rewrite, judged against its answer to the
original, per task category. Terser's span tagger and HMM compressor are deletion methods in the
same family as LLMLingua-2; its SFT rewriter is not.

---

## 3. Method

### 3.1 Data pipeline (Phase 2)

The pipeline is a DVC graph, `collect (frozen) → casualise → dedupe → split → label_subset →
candidates → token_counts → respond (per target) → label → oracle → features`
([p11_dvc_repro.md](../reports/p11_dvc_repro.md), [data_pipeline.md](data_pipeline.md)).

| Step | What happens | Numbers |
| :--- | :--- | :--- |
| collect | Clean instructions from Dolly, OpenAssistant, HumanEval, MBPP and GSM8K, licence-audited ([license_audit.md](../reports/license_audit.md), [SOURCES.md](../data/SOURCES.md)) | |
| casualise | Qwen2.5-32B-Instruct-AWQ rewrites each clean instruction in all five styles (rambling, hedging, texting, over-polite, repetitive); outputs that add or drop requirements or change numbers are rejected | 7,108 accepted, 1,292 rejected ([p2_casualise_stats.json](../reports/p2_casualise_stats.json)) |
| dedupe | MinHash LSH, exact Jaccard ≥ 0.8 on word 5-grams | 7,108 → 7,080 records, 1,671 → 1,665 source groups ([p2_duplicates.md](../reports/p2_duplicates.md)) |
| split | 70 / 15 / 15 by source prompt; the test split is frozen and hash-locked | 0 source prompts shared (C2.2) |
| label_subset | 300 prompts per category, so judging fits the token budget | 1,800 prompts: train 1,262 / val 285 / test 253 ([p2_candidates.md](../reports/p2_candidates.md)) |
| candidates | 6 per prompt: Qwen2.5-32B rewrites at low / medium / high aggressiveness, operator stacks `ops_FHC` and `ops_CHFIDSL`, LLMLingua-2 | 10,800 candidates ([p2_candidates.md](../reports/p2_candidates.md)) |
| token_counts | Every text on the gpt-4o (tiktoken `o200k_base`), Qwen2.5 and Llama-3.1 tokenizers | 37,800 rows, 0 missing ([p2_token_counts.md](../reports/p2_token_counts.md)) |
| respond | Qwen2.5-7B-Instruct and Phi-3.5-mini-instruct answer the original and every candidate (vLLM bf16, greedy; 3 samples for `creative`) | 13,100 responses per target ([p11_dvc_repro.md](../reports/p11_dvc_repro.md)) |
| label | Judge + task metrics → `preserved` / `degraded` | 20,986 labelled candidates, 43.1% degraded ([p2_headroom.md](../reports/p2_headroom.md)) |

Llama-3.1-8B was the planned second target and second judge, but its repository is gated for this
project's token; Phi-3.5-mini-instruct replaced it in both roles
([ADR-0011](decisions/ADR-0011-second-family-phi35.md), [p2_blockers.md](../reports/p2_blockers.md)).

A separate **real-prompt slice** of 1,000 first-turn prompts from WildChat-1M is used for
evaluation only, never casualised and never trained on ([real_slice.md](data_cards/real_slice.md)).
It guards against a rewriter that only learns to undo the verbosity that casualisation injected.

### 3.2 Judge and labels

The judge is Qwen2.5-32B-Instruct-AWQ on vLLM, comparing the target's answer to the original with
its answer to the rewrite, with a per-category rubric. On 836 pilot pairs the plain pairwise
protocol flipped its verdict with the A/B order on **40.9%** of pairs, 77% of those towards the
first-shown answer, concentrated in `creative` (69.1%) and `explain` (62.7%)
([p2_judge_bias.md](../reports/p2_judge_bias.md)). The project therefore uses `pairwise_avg`: both
orders are judged with `top_logprobs`, the verdict probabilities are averaged over the two orders,
and the label rule is applied to the averaged verdict
([ADR-0005](decisions/ADR-0005-judge-protocol.md)). This makes the label independent of order by
construction, but it does not remove the underlying bias; C2.6 (raw disagreement ≤ 10%) stays
failed.

The full labelling run made 41,986 judge calls (24,132 from cache) and took 13,042 s of GPU wall
time ([p2_label_stats.json](../reports/p2_label_stats.json)).

**Self-preference control (row 2.2.5).** Phi-3.5 judged 1,000 pairs (500 per target family) with the
same protocol. Its label agreement with the Qwen judge is 61.4% on Qwen-target outputs and 66.2% on
Phi-target outputs, a gap of 4.8 pp (threshold 5 pp); the side-agreement gap is 3.6 pp
([p2_self_preference.json](../reports/p2_self_preference.json)). No self-preference was detected, but
the two judges agree on only about two labels in three.

**Oracle.** For each prompt and target, the oracle rewrite is the shortest non-rejected candidate
labelled `preserved` and strictly shorter on the target's tokenizer, else the original
([p2_headroom.md](../reports/p2_headroom.md)).

### 3.3 Rewrite operators (Phase 1)

Seven deterministic operators, applied in the canonical order `C H F I D S L`: drop filler (`F`),
drop hedging and politeness (`H`), deduplicate restated requirements (`D`), imperative conversion
(`I`), list-ify constraints (`L`), compress meta-talk (`C`) and a tokenizer-aware phrase
substitution (`S`) that picks the cheapest alternative on the target's tokenizer. Every operator is
a no-op on 50 already-efficient prompts (C1.2). The operators supply discrete hypotheses for concept
learning (Find-S, Candidate-Elimination; rows 1.3.1–1.3.5) and the action space for Phase 9.

One shared helper, `terser.operators.base.tidy`, strips each line and upper-cases its first letter.
This corrupts code (see § 6.4).

### 3.4 Classical models (Phases 3–6)

All classical models run on CPU under one protocol: grouped stratified 5-fold cross-validation with
the source prompt as the group, so paraphrases of one source never sit on both sides of a fold
(row 3.2.4). Gate thresholds follow one rule: maximise specificity subject to sensitivity ≥ 0.95
(row 3.2.3), since a missed degradation is worse than a rejected good rewrite.

- **Features.** Pair features for the gate (token ratio, maximum and summed self-information
  removed, negations / numbers / constraint keywords removed, embedding cosine, category);
  span features for the tagger (token cost, GPT-2 self-information, stopword ratio, position, POS,
  filler-lexicon hit, keyword flags). Token-level KEEP / DROP / REPLACE labels come from diff-aligning
  each prompt with its oracle rewrite (rows 3.1.1–3.1.3).
- **Phase 3.** ID3 from scratch and C4.5-style trees with four pruning methods, logistic regression
  from scratch with L1 / L2 / elastic-net and calibration, KNN classification and regression,
  bias–variance decomposition, learning curves, nested grid search.
- **Phase 4.** Perceptron, a NumPy MLP with hand-written backprop (gradient check), a PyTorch MLP
  gate on 1536-d bge-small pair embeddings (409,985 parameters), activation and optimizer studies,
  and an SVM dual solver from scratch (cvxopt QP) with linear and polynomial kernels.
- **Phase 5.** AdaBoost from scratch, random forest, LightGBM, permutation importance and SHAP;
  multinomial Naive Bayes filler detector; a Bayes optimal classifier over the Phase 1 hypothesis
  space; EM for GMMs from scratch, applied to judge score deltas; and an HMM span tagger (supervised
  MLE, forward algorithm, Viterbi, Baum-Welch) turned into the first end-to-end compressor.
- **Phase 6.** K-means, bisecting K-means, agglomerative and divisive clustering of bge-small prompt
  embeddings; K-means derived as the σ → 0 limit of EM; per-cluster risk.

### 3.5 Deep learning: TextCNN and the SFT rewriter (Phase 8)

**TextCNN span scorer.** Word-averaged bge-small token states feed parallel 1-D convolutions
(kernel 3 and 5, 128 filters each) with a per-token KEEP head and a pooled prompt-compressibility
head; max, average and concatenated pooling are compared over 3 seeds
([p8_textcnn.md](../reports/p8_textcnn.md), [textcnn.md](model_cards/textcnn.md)).

**SFT rewriter.** One target-agnostic rewriter (justified by C1.5, § 6.1) is fine-tuned on
(casual prompt, oracle rewrite) pairs from the train split. The escalation ladder is Qwen2.5-0.5B
full fine-tune → 1.5B LoRA (r = 16, α = 32) → 3B LoRA → 7B QLoRA (NF4), each rung trained only if
the previous one fails C8.3. Decoding is greedy, with a safeguard that falls back to the original
whenever the rewrite is not shorter on the target tokenizer (row 8.3.7)
([rewriter.md](model_cards/rewriter.md)).

The first SFT set required a candidate preserved on *both* targets, which capped the median target
savings near 10%, so every rung learned to save about 7.6%. `sft_v2` added two aggressive rewrite
strengths (`llm_max`, `llm_min`) for the train and val prompts only. This raised the median target
savings to 21.4% on train and 17.4% on val, but 32.6% of train prompts and 35.4% of val prompts
still have no preserved rewrite and keep the original as their target
([p8_sft_v2.md](../reports/p8_sft_v2.md)). Even a perfect imitation of the val targets would
stay below the 25% bar.

### 3.6 Reinforcement learning: tabular Q-learning (Phase 9)

Rewriting is cast as an MDP. The state holds the remaining-filler bucket, constraint count, length
bucket, category and the operators already applied; the actions are the seven operators or `STOP`.
The reward at `STOP` is `Δtokens% − 100 · 1[degraded]`, with −0.01 per step, and the LightGBM gate
serves as a cheap reward model, so no LLM is called during learning
([p9_qlearning.md](../reports/p9_qlearning.md), [rl.md](derivations/rl.md)). Tabular Q-learning with
ε-greedy exploration is swept over α (including a visit-count step 1 / N(s, a)), γ and the ε
schedule on 300 train prompts, and the greedy policy is then verified with the real judge on 300
val prompts.

### 3.7 Hyperparameter optimisation (Phase 10)

`terser.hpo` wraps grid, random, Optuna TPE and GP samplers with MLflow parent-child runs. The
method study tunes the LightGBM gate (60 trials × 5 seeds per method) and the MLP gate (random and
TPE) on grouped out-of-fold AUC, train split only ([p10_hpo.md](../reports/p10_hpo.md)). Rewriter
LoRA tuning (row 10.7) searched rank, α, learning rate, warmup and epochs for the 1.5B rewriter with
Optuna TPE and a median pruner (12 trials, 5 pruned). A cheap proxy for judged preservation, fitted
on the val rewrites of five rungs, scored the trials; the top three were then judged
([p10_rewriter_hpo.md](../reports/p10_rewriter_hpo.md)). The chosen trial (`hpo08`, r 32, α 64,
1 epoch) saves 54.3% at 49.1% judged preservation against 35.0% at 57.0% for the untuned rung. It is
registered as `terser-rewriter` v10, and the plateau is documented (C10.3).
A data-access audit confirms that HPO touched only train and val (C10.4).

### 3.8 Serving and MLOps (Phase 11)

- **API.** FastAPI `POST /rewrite {prompt, target, mode}` returns the rewrite, token counts, gate
  score, the path that served it, fallback information and per-request net savings
  `price_T × (tokens_before − tokens_after) − rewriter_cost` ([api.md](serving/api.md)). An LRU
  cache keyed on the normalised prompt hash and target answers warm hits 39× faster than misses
  (median, [p11_cache_latency.md](../reports/p11_cache_latency.md)).
- **Fast path (CPU).** A depth-10 decision-tree span tagger on 10 cheap word features
  (droppable macro-F1 0.631) and a GradientBoosting gate (200 trees of depth 3) on 30 shared surface
  features (grouped-CV AUC 0.760), both exported to ONNX Runtime
  ([p11_fast_path.md](../reports/p11_fast_path.md)). The gate threshold 0.264564 follows the row
  3.2.3 rule.
- **Quality path (GPU).** The registered Staging rewriter (Qwen2.5-3B + `sft-qwen3b-lora`) on vLLM
  with continuous batching, then the same gate. A request over the 0.4 s budget is answered by the
  fast path; a crash in both paths returns the original prompt
  ([p11_quality_path.md](../reports/p11_quality_path.md)).
- **Containers.** `terser-cpu` (fast path baked in) and `terser-gpu` (vLLM base image) with
  Prometheus, Pushgateway and Grafana in one `compose.yaml`
  ([p11_containers_dashboard.md](../reports/p11_containers_dashboard.md)).
- **Registry and lineage.** MLflow model registry for gate, tagger and every rewriter rung, each
  version tagged with git SHA, data hash, config hash and the checkpoints it passed
  ([p11_registry.md](../reports/p11_registry.md)).
- **Orchestration and CT.** Seven Prefect deployments (`data_flow`, `train_classical_flow`,
  `train_rewriter_flow`, `evaluate_flow`, `promote_flow`, `drift_flow`, `ct_flow`). `ct_flow` runs
  on a drift alert or a new data batch, retrains gate and tagger, queues the rewriter retrain on the
  GPU queue, evaluates, and promotes to Staging only if a promotion guard passes
  ([p11_flows.md](../reports/p11_flows.md), [p11_ct_and_fresh_clone.md](../reports/p11_ct_and_fresh_clone.md)).
- **Monitoring.** Request metrics at `/metrics`; daily input drift (PSI and MMD on PCA components of
  prompt embeddings, length, category; [p11_input_drift.md](../reports/p11_input_drift.md));
  tokenizer drift on a fixed fixture set; a nightly 200-prompt quality canary judged on both targets,
  with a Grafana alert below 92% rolling preservation ([p11_canary.md](../reports/p11_canary.md)).

---

## 4. Experiments

### 4.1 Setup

| Item | Value |
| :--- | :--- |
| Labelled data | 1,800 prompts × 6 candidates × 2 targets = 20,986 labels (with identical-text pairs) ([p2_label_stats.json](../reports/p2_label_stats.json)) |
| Classical training data | 14,732 labelled train pairs over 1,262 source prompts; 129,380 train tokens for span tagging ([p3_gate_checkpoint.md](../reports/p3_gate_checkpoint.md), [p3_span_tagger.md](../reports/p3_span_tagger.md)) |
| Rewriter evaluation | 285 val prompts, both targets, judged with `pairwise_avg` ([p8_rewriter_val.md](../reports/p8_rewriter_val.md)) |
| Savings tokenizer for rewriter tables | Qwen2.5 (tokenizers agree, § 6.1) |
| Seeds | 3 for classical tables, 5 for Phase 4 studies and HPO; greedy decoding, one run per rewriter rung |
| GPU | RTX 4500 Ada (24 GB) for generation, judging and training; RTX 4090 for the Phase 2 smoke run |
| Compute | 57,986,571 generated tokens, 63.22 active / 100.49 reserved GPU-hours, est. ≤ 13.30 kWh active (row 12.10) ([compute_accounting.md](../reports/compute_accounting.md)) |

### 4.2 Conventions for the rewriter tables

Median savings are over all prompts, with fallbacks at 0%. Preservation is pooled over (prompt,
target) pairs, and a fallback counts as preserved by construction. This favours systems that fall
back often: the 3B rung falls back on 130 of 285 prompts, which is part of why it has high
preservation and low savings ([rewriter.md](model_cards/rewriter.md)). The quality gate is **not**
applied in the C8.3 table; § 5.4 shows what it does.

---

## 5. Results (validation split)

All results in this section are on the **validation** split or on train-split cross-validation,
except § 5.1, which reports the one-time test-split evaluation.

### 5.1 Test-split results

<!-- test-results:begin -->
The test split was unlocked once (row 12.1, [access log](checkpoints/C12/test_access_log.jsonl)) and
evaluated with the validated protocol: both targets, the Qwen2.5-32B judge with position swapping,
and 95% cluster-bootstrap CIs over 234 source groups (1,000 resamples). It covers 992 main-split
prompts and the 1,000-prompt real slice. Every number here is from
[final_results.md](../reports/final_results.md).

| System | Median savings (Qwen2.5 / Phi-3.5) | Preserved, pooled | Worst category | Mean net savings (Qwen2.5) |
| :--- | :--- | :--- | :--- | :--- |
| Terser full (`sft3bv2` + gate) | 0.0% / 0.0% | 96.7% [95.4, 97.8] | 82.8% | −54.6% |
| Fast path (deployed ONNX) | 0.0% / 0.0% | 99.1% [98.3, 99.7] | 96.9% | 0.8% |
| `sft3bv2`, no gate | 12.6% / 11.8% | 69.8% [67.0, 72.6] | 61.5% | −35.7% |
| `sfthpo` (row 10.7), no gate | 61.3% / 59.8% | 48.9% [46.0, 51.8] | 37.4% | 1.2% |
| Zero-shot Qwen2.5-32B, medium | 54.3% / 52.9% | 55.3% [51.8, 58.8] | 39.5% | −202.1% |
| LLMLingua-2 | 51.4% / 50.0% | 26.5% [23.7, 29.6] | 12.3% | 49.7% |
| Best operator stack (`ops_FHC`) | 0.5% / 0.4% | 82.6% [80.5, 84.6] | 72.6% | 2.3% |
| HMM compressor | 9.1% / 8.7% | 59.9% [55.6, 64.3] | 31.3% | 20.3% |

**Headline criteria.** S2 (96.7% ≥ 95%) and S7 (fast-path p95 20.9 ms) pass. S1 (0% median
savings), S3 (82.8% in `reason` on Qwen2.5), S4 (net savings −58.6% / −57.5% at the 25th-percentile
length), S5 and S8 fail. S2 passes for the same reason S1 fails: the gate rejects 466 (Qwen2.5) and
589 (Phi-3.5) of 992 rewrites, the safeguard sends back the original for another 366, and the median
request is served unchanged. The test numbers match validation closely: `sft3bv2` preserved 71.4% on
val and 69.8% on test, and `sft15v2` 57.0% and 60.4%. Validation did not overfit the selection.

**Trade-off.** No system reaches S1 and S2 together. Every system that saves more than 30% of tokens
preserves at most 56%: the tuned 1.5B rewriter, the zero-shot 32B rewrites and LLMLingua-2. In net
terms only the non-generative compressors come out positive. The zero-shot 32B rewriter costs more
than it saves at every prompt-length decile (§ 5.7 prices), and the 1.5B rewriter is positive only
because it is small ([final_results.md § 12.7](../reports/final_results.md)).

**Stratified (row 12.3).** `reason` is the weak category for every gated system on Qwen2.5 (68% for
the full system), because the gate passes word-problem rewrites whose numbers changed. No Phase 6
cluster met the 1.5× high-risk bar. In the highest-risk cluster (c6, risk 1.19) the full system
preserves 98–100%. On the real-prompt slice the full system preserves 97.3% at 0% savings. Zero-shot
32B saves far less there (17.6% medium against 54.3% on the main split), because real prompts are
short and already terse.

**Statistics (row 12.4).** Exact McNemar tests on paired preserved flags rank the zero-shot 32B
`medium` rewrites above `high` (+6.0 pp preserved, p = 7e-7) and the full system far above both on
preservation (+41 to +47 pp, p < 1e-190), at 54–59 pp less savings. The seeded systems vary little
across 3 seeds: Q-learning 87.5% ± 0.8% and TextCNN 92.7% ± 0.3% pooled preservation. The full
system over 3 rewriter training seeds holds 97.0% ± 0.3% preserved at 0% median savings; the bare
seed-1 and seed-2 rewriters save 4.8% / 4.5% (Qwen2.5) at 73.1% / 74.4% preserved
([final_results.md](../reports/final_results.md)).

**Error taxonomy (row 12.8).** See [final_errors.md](../reports/final_errors.md): 200 degraded
`sft3bv2` test cases, categorised with LLM assistance and not human-reviewed.
<!-- test-results:end -->

### 5.2 Headroom (C2.7, C2.8)

| Target | Oracle median savings | 95% CI | Any preserved candidate |
| :--- | ---: | :--- | ---: |
| Qwen2.5-7B | 46.5% | [43.5%, 48.6%] | 95.9% |
| Phi-3.5-mini | 40.5% | [36.0%, 43.9%] | 93.0% |

Pooled over targets, every category clears 15% (code 21.8%, transform 19.0%, the others 47–59%),
so C2.7 passes. The degraded share among candidates is 43.1% (C2.8 passes). By method, LLMLingua-2
candidates degrade most often (72.7%) and the light operator stack `ops_FHC` least often (16.8%).
The `texting` style has the least headroom (median 16.7% on Qwen2.5, 14.2% on Phi-3.5), since texting
prompts are already short ([p2_headroom.md](../reports/p2_headroom.md)).

The oracle is a per-prompt, per-target selection with hindsight over judged candidates. A single
rewriter that has to serve both targets sees a much lower ceiling: requiring preservation on both
targets leaves about 10% median target savings in the first SFT set
([rewriter.md](model_cards/rewriter.md)).

### 5.3 Rewriter ladder and baselines (C8.3, C8.4, C8.5)

From [p8_baselines.md](../reports/p8_baselines.md) (same 285 val prompts and metric definitions as
[p8_rewriter_val.md](../reports/p8_rewriter_val.md)):

| System | Median savings | Fell back | Preserved | Qwen2.5-7B | Phi-3.5 | Latency (s / prompt) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original prompt | 0.0% | 285 | 100.0% | 100.0% | 100.0% | 0.00 |
| Best operator stack `ops_FHC` | 0.3% | 138 | 81.4% | 81.8% | 81.1% | n/r |
| HMM compressor (124 shared prompts) | 10.9% | 51 | 58.1% | 60.5% | 55.6% | n/r |
| LLMLingua-2 | 50.7% | 1 | 28.9% | 30.2% | 27.7% | n/r |
| Zero-shot Qwen2.5-32B, `llm_low` | 40.1% | 24 | 58.8% | 64.9% | 52.6% | 0.25 |
| Zero-shot Qwen2.5-32B, `llm_medium` | 47.9% | 26 | 57.5% | 62.8% | 52.3% | 0.15 |
| Zero-shot Qwen2.5-32B, `llm_max` | 66.4% | 25 | 50.7% | 55.1% | 46.3% | 0.20 |
| SFT 0.5B full FT (`sft05`) | 7.6% | 98 | 67.4% | 70.9% | 63.9% | 0.33 |
| SFT 1.5B LoRA (`sft15`) | 7.6% | 101 | 69.3% | 69.8% | 68.8% | 0.50 |
| SFT 3B LoRA (`sft3b`) | 2.1% | 130 | 74.7% | 78.6% | 70.9% | 0.76 |
| SFT 1.5B LoRA, `sft_v2` (`sft15v2`) | **35.0%** | 72 | 57.0% | 61.1% | 53.0% | 0.44 |
| SFT 3B LoRA, `sft_v2` (`sft3bv2`) | 12.5% | 104 | 71.4% | 75.1% | 67.7% | 0.98 |
| SFT 7B QLoRA, `sft_v2` (`sft7bv2`) | 2.0% | 130 | **75.3%** | 77.9% | 72.6% | 1.71 |

C8.3 bar: 25% savings at 92% preservation. n/r = not recorded. The HMM row is scored on the 124 of
its 500 judged val prompts that fall in this table's 285, so it is not directly comparable with the
other rows. The zero-shot `llm_high` and `llm_min` levels are omitted here for space (55.2% / 52.3%
and 52.2% / 50.0%); see the source table.

Preservation by category for the main systems:

| System | code | creative | explain | fact | reason | transform |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ops_FHC` | 78.4% | 70.2% | 84.3% | 80.0% | 97.9% | 78.6% |
| LLMLingua-2 | 7.8% | 27.7% | 47.1% | 24.5% | 49.0% | 24.5% |
| Zero-shot 32B `llm_medium` | 52.9% | 34.0% | 60.0% | 65.5% | 77.1% | 55.1% |
| `sft15v2` | 48.0% | 54.3% | 65.7% | 68.2% | 44.8% | 62.2% |
| `sft3b` | 57.8% | 75.5% | 85.7% | 80.0% | 82.3% | 70.4% |
| `sft7bv2` | 56.9% | 85.1% | 65.7% | 80.0% | 81.2% | 80.6% |

**Reading.**

1. **No rung meets C8.3.** Every system sits on one trade-off curve: more savings, less
   preservation. The augmented targets moved the 1.5B rung from 7.6% to 35.0% savings but cut its
   preservation from 69.3% to 57.0%. Larger models on the same `sft_v2` targets became more
   conservative, falling back more often (104 and 130 times) and saving less (12.5%, 2.0%). The
   escalation ladder is exhausted ([checkpoints/status.md](checkpoints/status.md)).
2. **C8.4 fails.** Only `sft15v2` matches a zero-shot 32B level within ±1 pp preservation
   (`llm_medium`), and that level saves 12.9 pp more. The SFT rungs are also not 10× faster than the
   32B under the recorded batching: the best latency ratio is below 1× for every rung. The two
   latencies come from different batching set-ups, so a same-GPU batch-1 benchmark is still needed
   for a final latency comparison ([p8_baselines.md](../reports/p8_baselines.md)).
3. **C8.5 fails.** Every rung preserves 28–46 pp more than LLMLingua-2 but saves 16–49 pp less; the
   check needs both at once.
4. **Code is the weakest category for the SFT rungs.** Part of the cause is the `tidy` defect
   (§ 6.4).

### 5.4 The quality gate on rewriter outputs

The deployed gate was trained on Phase 2 candidates, not on rewriter outputs. Applied to the val
rewrites of five rungs at the deployed threshold 0.264564
([p10_rewriter_hpo/proxy.json](../reports/p10_rewriter_hpo/proxy.json)):

| Rung | Gate pass share | Preserved, gate on | Median savings, gate on |
| :--- | ---: | ---: | ---: |
| `sft05` | 7.7% | 98.6% | 0.0% |
| `sft15` | 6.7% | 98.1% | 0.0% |
| `sft3b` | 6.3% | 97.5% | 0.0% |
| `sft15v2` | 4.2% | 97.9% | 0.0% |
| `sft3bv2` | 6.0% | 98.1% | 0.0% |

The gate rejects 92–96% of SFT rewrites, which lifts preservation near 98% only by returning the
original prompt; median savings drop to 0. Its ROC-AUC for "degraded" on these rewrites is 0.598,
against 0.760 on the Phase 2 candidates it was trained on, so it barely ranks SFT rewrites. The same
pattern appears everywhere the gate meets new outputs:

| Where | Rejected | Source |
| :--- | ---: | :--- |
| Fast-path tagger rewrites, 300 train prompts | 298 of 300 | [p11_fast_path.md](../reports/p11_fast_path.md) |
| Fast-path traffic in the CPU container | 595 of 600 | [p11_containers_dashboard.md](../reports/p11_containers_dashboard.md) |
| Quality-path load test, 5 RPS | 596 of 600 | [p11_quality_path.md](../reports/p11_quality_path.md) |
| Quality-path load test, 20 RPS for 30 min | 34,707 of 36,000 | [p11_quality_path.md](../reports/p11_quality_path.md) |
| Nightly canary, first run | 372 of 400 | [p11_canary.md](../reports/p11_canary.md) |

On the gate's own out-of-fold candidates the same threshold passes 23.7% of rewrites
([p11_fast_path.md](../reports/p11_fast_path.md)). The gap between 23.7% and the rates above is a
distribution shift between Phase 2 candidates and the outputs the gate actually sees. Whether the
gate is mis-calibrated on these outputs or the outputs really degrade cannot be told apart without
judge labels for them.

### 5.5 Classical models

| Check | Measured | Target | Result | Source |
| :--- | :--- | :--- | :---: | :--- |
| C3.2 leakage (naive − grouped AUC) | 0.001 ± 0.001 | gap recorded | pass | [p3_cv_leakage.md](../reports/p3_cv_leakage.md) |
| C3.3 gate (Tree depth 4) | AUC 0.758, specificity 0.405 at sensitivity 0.95 | ≥ 0.75, ≥ 0.40 | pass | [p3_gate_checkpoint.md](../reports/p3_gate_checkpoint.md) |
| C3.4 span tagger (LR) | 3-tag macro-F1 0.440 | ≥ 0.65 | fail | [p3_span_tagger.md](../reports/p3_span_tagger.md) |
| C3.5 savings predictor (KNN k = 25) | MAE 16.7 pp, R² 0.290 | ≤ 10 pp, ≥ 0.3 | fail | [p3_knn_regression.md](../reports/p3_knn_regression.md) |
| C3.6 overfitting control | cost-complexity pruning: train–val gap 0.409 → 0.001 | close ≥ 50% | pass | [p3_tree_pruning.md](../reports/p3_tree_pruning.md) |
| C4.4 better gate | NumPy MLP 0.774, tuned MLP 0.775 < C3.3 + 0.03; LightGBM stays champion | or documented reason | pass (documented) | [p4_quality_gate.md](../reports/p4_quality_gate.md), [p10_hpo.md](../reports/p10_hpo.md) |
| C4.5 worst-category AUC | embedding MLP `explain` 0.673 | ≥ 0.70 | fail | [p4_quality_gate.md](../reports/p4_quality_gate.md) |
| C5.1 champion gate (LightGBM) | AUC 0.778; tuned 0.783 | ≥ 0.85 | fail | [p5_ensembles.md](../reports/p5_ensembles.md), [p10_hpo.md](../reports/p10_hpo.md) |
| C5.2 RF OOB vs grouped CV | OOB 0.079–0.082 above CV | ≤ 0.02 | fail | [p5_ensembles.md](../reports/p5_ensembles.md) |
| C5.3 Naive Bayes filler detector | precision 0.594 at recall 0.70 | ≥ 0.85 | fail | [p5_naive_bayes.md](../reports/p5_naive_bayes.md) |
| C5.4 Bayes optimal vs MAP | −0.008 accuracy | ≥ 0 | fail | [p5_bayes_optimal.md](../reports/p5_bayes_optimal.md) |
| C5.5 GMM on judge deltas | ARI 0.090; judge-threshold review opened | ≥ 0.4 or review | pass (via review) | [p5_gmm_deltas.md](../reports/p5_gmm_deltas.md) |
| C5.6 HMM compressor | see § 5.6 | | fail | [p5_hmm_compressor.md](../reports/p5_hmm_compressor.md) |
| C6.1 clustering | silhouette 0.066 at k = 9 | ≥ 0.10 | fail | [p6_results.md](../reports/p6_results.md) |
| C6.2 K-means ⇔ EM | agreement 98.9% at the smallest σ | ≥ 98% | pass | [p6_results.md](../reports/p6_results.md) |
| C6.3 high-risk cluster | riskiest 1.195× average degradation | ≥ 1.5× | fail | [p6_cluster_risk.md](../reports/p6_cluster_risk.md) |
| C8.2 TextCNN tagging (binary) | 0.679 vs 0.658 best classical (+0.021, McNemar p < 1e-18) | + 0.05 | fail | [p8_textcnn.md](../reports/p8_textcnn.md) |
| C10.2 gate HPO | 0.7733 → 0.7829 (+0.0095), all methods within 0.001 | + 0.01 or plateau | pass (plateau) | [p10_hpo.md](../reports/p10_hpo.md) |

**Reading.** The quality gate plateaus near AUC 0.78 regardless of model family (trees, LR, KNN,
MLPs, SVMs, boosting) and of search method. [p10_hpo.md](../reports/p10_hpo.md) puts the remaining
error in the features rather than the hyperparameters, and [p4_quality_gate.md](../reports/p4_quality_gate.md)
shows that sentence embeddings do worse than hand-built removal features, because the label depends
on *which* tokens were removed. A noisy label caps every gate as well (§ 7.4). The random forest's
OOB estimate is optimistic by about 0.08 because bootstrap rows share source prompts, which is the
same leakage the grouped CV removes. Clusters recover the code and reason categories well (ARI 0.490
against categories overall) but carry no cluster with markedly higher risk.

### 5.6 Non-generative compressors (C5.6, C9.1, C9.2)

| System | Prompts | Median savings | Preserved | Source |
| :--- | ---: | ---: | ---: | :--- |
| HMM (Viterbi DROP tags deleted) | 500 val | 16.5% | 55.4% (Qwen 58.0%, Phi 52.8%) | [p5_hmm_compressor.md](../reports/p5_hmm_compressor.md) |
| Best single operator `op_F` | 500 val | 0.0% | 89.5% | [p5_hmm_compressor.md](../reports/p5_hmm_compressor.md) |
| Q-learning greedy policy | 300 val | 0.0% | 89.3% | [p9_judge_verification.md](../reports/p9_judge_verification.md) |
| Best fixed operator order `FHIS` | 300 val | 1.4% | 77.8% | [p9_judge_verification.md](../reports/p9_judge_verification.md) |

The HMM clears the savings clause of C5.6 but reaches 90% preservation on neither target, and when it
is mixed with the identity rewrite to match `op_F`'s preservation, both save 0%. Its preservation is
lowest on `transform` (24.7%) and `code` (39.7%), where deletion removes payload.

The Q-learning policy learns to do almost nothing. `STOP` is its most frequent sequence in 4 of 6
categories, and median gate-estimated savings are 0–2% ([p9_policy.md](../reports/p9_policy.md)). That
is the correct response to a reward in which the operators save little and a degradation costs 100
points. Judged on val, it preserves 11.5 pp more than the fixed order but saves 1.4 pp less, so C9.2
fails. The Q-table does not converge to the 1e-3 criterion (C9.1): the state lumps different prompts
together, so the STOP reward for one state-action pair ranges from about +40 to about −100. Even with a
1 / N step and 200,000 episodes the largest update in the last window is 0.058
([p9_convergence.md](../reports/p9_convergence.md)).

### 5.7 Serving

| Check | Measured | Target | Result | Source |
| :--- | :--- | :--- | :---: | :--- |
| C11.1 orchestration on a fresh clone (CPU) | four flows completed unattended in 585 s, after two fixes | runs unattended | pass | [p11_ct_and_fresh_clone.md](../reports/p11_ct_and_fresh_clone.md) |
| C11.2 fast-path latency, 20 RPS, 1,200 distinct prompts | p50 10.8 ms, p95 20.9 ms, 0 errors | p95 ≤ 150 ms | pass | [p11_serving_checks.md](../reports/p11_serving_checks.md) |
| C11.3 quality-path latency, 5 RPS (3B LoRA) | p95 438 ms, 0.0% quality-served | p95 ≤ 400 ms | fail | [p11_quality_path.md](../reports/p11_quality_path.md) |
| C11.3, 1.5B LoRA v2 | p95 437 ms, 0.2% quality-served (1 of 600) | | fail | [p11_quality_path_15v2.md](../reports/p11_quality_path_15v2.md) |
| C11.4 throughput, 20 RPS for 30 min | 36,000 requests, 0 errors, no OOM, 0.0% quality-served | ≥ 20 RPS quality-served | fail | [p11_quality_path.md](../reports/p11_quality_path.md) |
| C11.5 fallback correctness | 100% in 5 fault scenarios × 1,000 requests | 100% | pass | [p11_serving_checks.md](../reports/p11_serving_checks.md) |
| C11.6 drift detection | `code`-only window and real slice flagged; val window not flagged | alert within one cycle | pass | [p11_input_drift.md](../reports/p11_input_drift.md) |
| C11.7 CT guard | synthetic regression refused; real CT cycle refused Staging because no rung passes C8.3 | refuse | pass | [p11_ct_and_fresh_clone.md](../reports/p11_ct_and_fresh_clone.md) |

**Why the quality path misses its budget.** Without load, API or timeout, the 3B rewriter needs
p50 2,374 ms and p95 4,298 ms per request, and the 1.5B v2 rewriter p50 1,145 ms and p95 1,768 ms.
A rewrite is roughly as long as its prompt (mean prompt 167.5 tokens, mean output 130.9 / 102.1
tokens), and decoding costs 19.3 / 11.0 ms per output token, so a full regeneration cannot fit in
400 ms on this GPU before any queueing ([p11_quality_path.md](../reports/p11_quality_path.md)).
Inside the GPU container an unloaded quality request takes 0.95–1.3 s end to end (954 ms and
1,296 ms for the two checked prompts) ([p11_containers_dashboard.md](../reports/p11_containers_dashboard.md)).
Meeting C11.3 needs a shorter output (edit-style output or a span-deletion head) or a much smaller
rewriter, not a serving change.

**Canary.** The first nightly run (2026-10-18, 1.5B LoRA v2 in quality mode) measured 98.0%
preservation over 400 (prompt, target) pairs, and the metric reached the dashboard. But 380 of the
400 pairs were unchanged prompts, which are preserved by construction: 263 requests hit the quality
timeout and the gate rejected 372 rewrites. Of the 20 pairs with an edited prompt, 12 (60%) were
preserved ([p11_canary.md](../reports/p11_canary.md)). The canary therefore shows that the service
rarely harms users. It does not show that the rewriter is good.

**Net savings.** The aggregate net-savings measurement (row 12.7) is in
[final_results.md](../reports/final_results.md): −58.6% (Qwen2.5) and −57.5% (Phi-3.5) at the
25th-percentile length for the full system. Two earlier measurements bound it. First, when the gate rejects a quality-path rewrite the original prompt is
served, so the request's net savings are the rewriter cost alone, −8.5e-6 USD in the container
check ([p11_containers_dashboard.md](../reports/p11_containers_dashboard.md)). Second, at the
configured prices the rewriter's cost is 6% of a `gpt-4o` prompt but 75% of a prompt at the assumed
self-hosted `qwen2.5` price ([p3_skip_rule.md](../reports/p3_skip_rule.md)). With the gate on, median
realised savings are 0% (§ 5.4), so the generative path loses money on the median request at these
prices. Without the gate, no rung's median savings (at most 35.0%) reach the 75% overhead of the
self-hosted targets. This is derived from the two reports, not measured end to end.

### 5.8 Ablations

<!-- ablations:begin -->
Row 12.6, test split, one change at a time from the full system
([final_results.md § 12.6](../reports/final_results.md)):

| Variant | Median savings (Qwen2.5 / Phi-3.5) | Preserved, pooled | Δ preserved |
| :--- | :--- | :--- | :--- |
| Terser full (`sft3bv2` + gate + fallback) | 0.0% / 0.0% | 96.7% | — |
| − gate | 12.6% / 11.8% | 69.8% | −26.9 pp |
| − fallback (8.3.7 safeguard off) | 0.0% / 0.0% | 95.9% | −0.8 pp |
| − HPO (untuned gate) | 0.0% / 0.0% | 97.5% | +0.8 pp |
| Operator-only (`ops_FHC` + gate) | 0.0% / 0.0% | 97.3% | +0.7 pp |

The gate does all the work: removing it buys 12 pp of median savings for 27 pp of preservation. The
fallback and the gate's HPO move preservation by less than 1 pp, inside the CIs. With the gate on,
the generative rewriter is no better than the operator stack, and its rewriter cost makes its net
savings worse (−54.6% against +0.5% mean on Qwen2.5).
<!-- ablations:end -->

---

## 6. Negative results

Negative results are results. This section lists every failed or open checkpoint
([checkpoints/status.md](checkpoints/status.md)) and the findings behind them.

### 6.1 C1.4 and C1.5: operators do not save tokens, and tokenizers agree

**C1.4 (fail, non-blocking).** On 500 pilot prompts the median best-single-operator savings are 1.7%
(gpt-4o), 1.6% (Qwen2.5) and 1.6% (Llama-3.1) against a 10% target. The full operator stack saves
about 2% at the median, and every single operator leaves the median prompt unchanged
([p1_tokenizer_study.md](../reports/p1_tokenizer_study.md)). The deterministic operators were
written to be conservative, and on casual prompts conservative deletion removes little. Because the
target assumed operators would be the main source of savings,
[ADR-0004](decisions/ADR-0004-plan-revision.md) made C1.4 non-blocking and moved the savings question
to the generative candidates and C2.7. The same pattern recurs at full scale: the operator stacks
save 0.4% and 1.6% at the median, against 45–62% for the LLM rewrites and 51–52% for LLMLingua-2
([p2_token_counts.md](../reports/p2_token_counts.md)).

**C1.5 (pass by documented agreement).** The plan expected tokenizer-specific rewriting to matter.
Spearman ρ of per-(prompt, operator) savings is at least 0.991 for every tokenizer pair, and
full-scale candidate savings agree across tokenizers to within about 1.5 pp per method
([p1_tokenizer_study.md](../reports/p1_tokenizer_study.md), [p2_token_counts.md](../reports/p2_token_counts.md)).
The tokenizer-aware operator `S` changes almost nothing (mean savings 0.02%). The negative finding is
that target-specific rewriting has no effect left to measure on these tokenizers, so Phase 8 ships
one target-agnostic rewriter and the target-conditioned rewriter was removed from the plan
(ADR-0004).

### 6.2 Concept learning collapses (C1.3)

Candidate-Elimination over the conjunctive attribute language collapses in all 6 categories (target:
non-empty in at least 4). Each collapse names a minimal conflicting set of 3–5 examples that no
conjunction separates ([p1_version_spaces.md](../reports/p1_version_spaces.md)). The labels come
from the judge (row 1.3.6). The finding is consistent
with the inductive-bias write-up: whether a rewrite is safe depends on interactions (for example,
"drop hedging unless tone matters") that a conjunctive language cannot express. The Bayes optimal
classifier over the same hypothesis space is 0.8 pp below MAP, and both are below the majority
class (C5.4, [p5_bayes_optimal.md](../reports/p5_bayes_optimal.md)).

### 6.3 The generative rewriter does not reach the main gate (C8.3, C8.4, C8.5, C10.3)

See § 5.3. The binding limit is the training targets, not model size: the `sft_v2` val targets have a
median of 17.4%, below the 25% bar even under perfect imitation ([p8_sft_v2.md](../reports/p8_sft_v2.md)).
Larger rungs on the same targets fall back more and save less. C10.3 (rewriter improvement through
HPO) passes as a documented plateau: LoRA rank, α, learning rate, warmup and epochs trade savings for
preservation (best judged trial 54.3% savings at 49.1% preserved), but none moves preservation toward
0.92 ([p10_rewriter_hpo.md](../reports/p10_rewriter_hpo.md)). The test split confirms this
(§ 5.1): the tuned rewriter saves 61.3% at 48.9% preserved.

The failure analysis of 100 degraded `sft3b` cases ([p8_failures.md](../reports/p8_failures.md)) found
a visible loss in only 28: dropped examples or context passages (12), changed meaning (10, half of
them in `reason` word problems), dropped constraints (5) and lost tone (1). The other 72 show no
task-relevant difference in 63 cases. If every visible loss were fixed, preservation would rise to
about 82% by the report's extrapolation, still below 92%. **Caveat:** these categories were assigned
by an LLM assistant (Claude), not by a human annotator, and have not been human-reviewed.

### 6.4 `tidy` corrupts code

`terser.operators.base.tidy` strips each line and upper-cases its first letter. In code this turns
`assert` into `Assert` and `def` into `Def`, renames identifiers that start a line and strips
docstring indentation. 33 of 220 `code` SFT training targets contain such a line-initial keyword that
their prompt does not, so the rewriter learned the pattern: 16 of 41 non-fallback `code` rewrites on
val carry it, and 12 of the 100 analysed failures are of this kind, 2 of which change meaning
([p8_failures.md](../reports/p8_failures.md)). The fix, leaving code lines untouched in `tidy` and
rebuilding the SFT targets, is not implemented.

### 6.5 The quality gate does not transfer to new outputs

See § 5.4. At the serving operating point the gate rejects 92–96% of SFT rewrites and 99% of fast-path
rewrites, so it acts as an almost unconditional fallback. The fast path's 28.6% median savings before
the gate become 0.0% after it ([p11_fast_path.md](../reports/p11_fast_path.md)). The gate itself
plateaus at AUC 0.78 on its training distribution (C5.1 fail) and falls below the sensitivity floor
in `reason` (0.794 at the pooled threshold).

### 6.6 Classical and RL checkpoints

C3.4, C3.5, C4.5, C5.1–C5.4, C5.6, C6.1, C6.3, C8.2, C9.1 and C9.2 fail with the measured values in
§§ 5.5 and 5.6. The recurring causes are a noisy label (§ 7.4), small per-category samples, and an
operator action space that cannot produce large savings.

### 6.7 Serving checkpoints (C11.3, C11.4)

See § 5.7. Both fail because a full-regeneration rewriter is too slow for a 400 ms budget on one
24 GB GPU; the safeguards worked as designed (20 RPS for 30 min, 0 errors, every request answered by
the fallback).

### 6.8 Judge and infrastructure checkpoints

| Checkpoint | State | Cause |
| :--- | :--- | :--- |
| C2.6 position bias | fail: 40.9% raw A/B disagreement on the pilot | § 3.2; labels use order averaging |
| C7.2 memory plan, C7.4 queue | partial | Several VRAM rows are still estimates; the revised 32B queue estimate is unverified on the RTX 4090 workers ([C7 README](checkpoints/C7/README.md)) |
| C9.3–C9.5 (stretch GRPO) | ❌ measured, all fail | See § 6.10 |

Infrastructure findings that do not map to one checkpoint:

- **Registry alias.** MLflow `register_one` moves the `staging` alias to every version it promotes.
  Registering the three `sft_v2` rungs as Archived moved the alias off v6, and it had to be set back by
  hand ([p11_registry.md](../reports/p11_registry.md)).
- **DVC remote on one machine.** The DVC remote is now in sync with every committed output
  (`dvc status -c` clean, including `data/interim/rewrites/` and the tuned adapter), but it is a
  local directory on the workstation. Another machine needs a copy or a shared remote, so C11.1 holds
  on the workstation only ([p11_ct_and_fresh_clone.md](../reports/p11_ct_and_fresh_clone.md)). The
  LLM response cache is not tracked, so `respond` and `judge` cannot rerun on a fresh clone without a
  GPU ([p11_dvc_repro.md](../reports/p11_dvc_repro.md)).
- **Budget cap during the test judge.** The Phase 12 respond and judge runs charged the Phase 2
  ledger bucket by default and hit its 250M-token cap partway through the judge. The run resumed from
  the call cache under a resized Phase 12 cap (300M tokens, local compute only). No label was lost
  or recomputed; the compute accounting attributes those calls to Phase 12 by time window.
- **LLMLingua-2 needs the network.** Its real test downloads the model and is behind the `network`
  marker ([tests/test_data_candidates.py](../tests/test_data_candidates.py)); offline reproduction
  works only from the copied `.cache/llmlingua2` ([p11_dvc_repro.md](../reports/p11_dvc_repro.md)).
- **CI was red for most of the project.** Unpinned `black` disagreed with the local version and a
  billing block (from 2026-10-17) stopped every job ([p11_ci.md](../reports/p11_ci.md)). Both were
  fixed: CI run 39890354266 passed on all 4 jobs (median 8.8 min), and the v1.0.2 tag's CD run
  published the images, closing rows 11.4.1 and 11.4.2.

### 6.9 Open because they need people

These rows and checkpoints cannot be produced by a script, and no script output stands in for them:

| Item | What is needed |
| :--- | :--- |
| C1.1 | Review of ADR-0002 by a second contributor |
| 2.1.3 | Manual check of a 500-row sample for residual PII |
| 12.12 (S9) | A second contributor reproduces the headline numbers from a fresh clone |

Every preservation number in this report is an estimate from the order-averaged LLM judge.

### 6.10 Stretch work (GRPO)

GRPO policy optimisation of the rewriter (rows 9.3.1–9.3.5) was attempted although C8.3 had not
passed, against the stretch rule in [CONTRIBUTING.md § Stretch Goals](../CONTRIBUTING.md#stretch-goals-never-block-a-checkpoint).
All three checks fail ([p9_grpo_v2_val.md](../reports/p9_grpo_v2_val.md)): C9.3 gate precision on
RL rewrites 71.4% vs 91.0% on Phase 2 data (v1 reward-hacked at 42.4%), C9.4 1.1% degenerate on val,
C9.5 savings 25.2 pp below SFT. SFT stays champion.

---

## 7. Limitations

### 7.1 Creative prompts

`creative` is where the judge is least stable (69.1% raw A/B disagreement on the pilot,
[p2_judge_bias.md](../reports/p2_judge_bias.md)) and where zero-shot rewrites preserve worst
(26.6–39.4% across the five 32B levels, [p8_baselines.md](../reports/p8_baselines.md)). Creative
answers are sampled 3 times at temperature 0.7, so preservation there also measures sampling variance.
The failure analysis shows a related effect: removing "non-required" imagery from a creative request
can count as degraded because the judge rewards answers that used it ([p8_failures.md](../reports/p8_failures.md)).

### 7.2 Multi-turn context

Every prompt is a single turn: casualised instructions, and first-turn prompts in the real slice
([real_slice.md](data_cards/real_slice.md)). Rewriting a turn that refers to earlier turns, or
compressing a conversation history, was not studied.

### 7.3 Tokenizer drift

Savings are defined on pinned tokenizer versions, and a tokenizer-drift check alerts on any count
change in a fixed fixture set (C11.6). A new tokenizer version invalidates the stored counts and the
`S` operator's phrase choices, and requires retraining through CT. Claude's tokenizer was never
counted, because no Anthropic key is configured ([p2_token_counts.md](../reports/p2_token_counts.md)),
and only two target families were judged.

### 7.4 Judge dependence

Every label comes from one open 32B judge that has not been validated against humans (§ 6.9). The
evidence that its labels are noisy is consistent:

- 40.9% of verdicts flip with the presentation order (C2.6).
- A second judge (Phi-3.5) agrees with it on 61.4–66.2% of labels ([p2_self_preference.json](../reports/p2_self_preference.json)).
- A two-component GMM on the judge's score deltas agrees with its labels at ARI 0.090
  ([p5_gmm_deltas.md](../reports/p5_gmm_deltas.md)).
- 63 of 100 analysed degraded rewrites show no visible loss ([p8_failures.md](../reports/p8_failures.md)).
- AdaBoost's hardest examples are light rewrites labelled degraded that remove little information
  ([p5_ensembles.md](../reports/p5_ensembles.md)).

Label noise lowers every preservation number and caps every gate's AUC, and the rewriter is trained on
targets chosen by the same judge. The annotator in the failure analysis was also an LLM.

### 7.5 Synthetic prompts

The training and validation prompts are clean instructions made casual by an LLM. The real-prompt
slice differs from them clearly: median length 29 vs 62 words (length PSI 0.70) and embedding drift on
7 PCA components, so live traffic like it would raise the drift alert every day
([p11_input_drift.md](../reports/p11_input_drift.md)). A rewriter trained on these prompts may mostly
learn to undo the injected verbosity. On the real slice the Phase 12 evaluation (row 12.3) scores the full
system at 97.3% preserved and 0% median savings ([final_results.md](../reports/final_results.md)). The `texting` style, the closest to short real prompts, also has the least
oracle headroom (§ 5.2).

### 7.6 Further limitations

- **Small labelled set.** 1,800 prompts were judged (285 val), limited by the judge's token budget
  ([p2_candidates.md](../reports/p2_candidates.md)). Per-category val samples are small, and there are
  no confidence intervals on the rewriter tables yet.
- **Fallbacks count as preserved.** The rewriter tables reward falling back (§ 4.2).
- **Latency comparisons mix set-ups.** SFT latencies are HF `generate` at batch 16; zero-shot 32B
  latencies are vLLM run wall time (§ 5.3).
- **Assumed prices.** The self-hosted target prices and the rewriter cost are assumptions in the serving
  config, not measured GPU-hour costs ([p3_skip_rule.md](../reports/p3_skip_rule.md)).

---

## 8. Future work

In rough order of expected value:

1. **Re-judge with more samples** for the order-inconsistent and low-margin pairs, to reduce label
   noise before any further rewriter training.
2. **Fix `tidy` for code** and rebuild the SFT targets (§ 6.4).
3. **Cheap safeguards from the failure analysis:** fall back when generation reaches the token cap,
   and a "no number lost" check, which would catch most visible losses in the sample
   ([p8_failures.md](../reports/p8_failures.md)).
5. **Better targets, not bigger models.** Per-target oracle targets (46.5% / 40.5% median) are far
   above the both-targets SFT targets (17.4% on val); a target-conditioned rewriter or more
   candidate strengths could close that gap.
6. **Edit-style output.** Predict deletions or short edit scripts instead of regenerating the whole
   prompt, so the quality path can fit the 400 ms budget ([p11_quality_path.md](../reports/p11_quality_path.md)).
7. **Retrain the gate on the outputs it gates.** Label tagger and SFT rewrites with the judge, refit
   the gate on them, and re-choose the threshold (§ 5.4).
8. **Rewriter HPO with judged preservation** in the objective, instead of the proxy used in
   row 10.7 (done; the plateau is documented in C10.3).
9. **Infrastructure:** push the missing DVC outputs, track the LLM response cache, pin the formatter
   versions in CI, and fix `register_one`'s alias move.
10. **Multi-turn prompts and more target families**, including a validated small judge for cheaper
    audits.
