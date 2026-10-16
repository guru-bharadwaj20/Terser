# Model Card: `span_tagger` (KEEP / DROP / REPLACE span tagger)

| Field | Value |
| :--- | :--- |
| Card version | `1.0` |
| Owner | `guru-bharadwaj20` |
| Last updated | `2026-10-17` |
| Registry name / version | `terser-tagger / 2` ([p11_registry.md](../../reports/p11_registry.md)) |
| Registry stage | `Production` (history `None -> Staging -> Production`) |

The deployed tagger is the CPU fast-path decision tree exported to `models/fast/tagger.onnx` by
`python -m terser.phase11.train_fast_path` ([train_fast_path.py](../../src/terser/phase11/train_fast_path.py),
[fast_models.py](../../src/terser/serving/fast_models.py)). It solves the **binary** task
"droppable" (DROP or REPLACE = 1, KEEP = 0), not the 3-tag task of C3.4. The research taggers
(Phase 3 tree / LR, Phase 5 HMM, Phase 8 LightGBM / HMM) are listed in §5.5 for comparison. The
TextCNN tagger has its own card ([textcnn.md](textcnn.md)).

---

## 1. Model details

| Field | Value |
| :--- | :--- |
| Component | Span tagger: per-word P(droppable); drives the fast-path rewrite |
| Base model / revision | None (classical model) |
| Architecture | `sklearn.tree.DecisionTreeClassifier(max_depth=10, min_samples_leaf=50, random_state=0)` on 10 per-word features (`SPAN_FAST_FEATURES`): `token_cost`, `stopword_ratio` (5-word window), `rel_position`, `is_filler`, `is_negation`, `is_number`, `is_constraint`, `is_punct`, `word_len`, `prev_filler` |
| Parameters (total / trainable) | One tree, depth ≤ 10, ≥ 50 words per leaf; no weight count applies |
| Precision | float32 features; ONNX graph |
| Training method | 5-fold `GroupKFold` by prompt to score, then a refit on all words |
| Config | No Hydra experiment config; CLI defaults in `train_fast_path.py`. Registry `config_sha256` `65c4501a…09b2` |
| Hardware / GPU-hours | CPU only; 0 GPU-hours |
| Serving path | fast (ONNX Runtime, `CPUExecutionProvider`), `tagger_threshold: 0.5` in [configs/serving/fast_onnx.yaml](../../configs/serving/fast_onnx.yaml) |

ONNX vs scikit-learn parity: max |dP| 2.9e-08 on 5,000 words
([p11_fast_path.json](../../reports/p11_fast_path.json)).

`span_matrix` is the single feature implementation used for training and serving, and the word
tokenisation is the row 3.1.1 `TOKEN_RE`, so training and serving features cannot diverge.

## 2. Intended use

- Primary use: the fast-path rewriter (`OnnxSpanRewriter`). It deletes every word with
  P(droppable) ≥ 0.5 **except** negations, numbers, constraint keywords and anything inside fenced
  code, inline backticks or double quotes. The rest of the text is kept byte for byte. The result
  is then scored by the quality gate ([quality_gate.md](quality_gate.md)).
- Users: `POST /rewrite` in fast mode; fast-path fallback for the quality path (C11.5).
- Out of scope:
  - producing replacement text. REPLACE spans are merged into "droppable", so a word the oracle
    *replaced* is *deleted* on the fast path;
  - a 3-tag KEEP / DROP / REPLACE output (C3.4 task);
  - multi-turn conversations, non-English prompts, code that must be reproduced verbatim outside
    a fenced block, safety-critical instructions;
  - real user prompts (not evaluated on the WildChat slice).

## 3. Supported target models and tokenizers

| Target | Tokenizer | Tokenizer version / revision | Evaluated | Notes |
| :--- | :--- | :--- | :---: | :--- |
| gpt-4o | `tiktoken` `o200k_base` | `o200k_base` | no | `token_cost` uses the request's target tokenizer at serving time unless fixed |
| Qwen2.5-7B-Instruct | HF `tokenizers` | `Qwen/Qwen2.5-7B-Instruct`, no revision pinned ([count.py](../../src/terser/tokenizers/count.py)) | yes | Training `token_cost` and oracle labels |
| Phi-3.5-mini-instruct | HF `tokenizers` | `microsoft/Phi-3.5-mini-instruct` | no | Labels are from the Qwen2.5 oracle only |
| Llama-3.1-8B-Instruct | HF `tokenizers` (ungated mirror) | `unsloth/Meta-Llama-3.1-8B-Instruct` | no | Not a label target ([ADR-0011](../decisions/ADR-0011-second-family-phi35.md)) |
| Claude (optional) | Anthropic token-counting endpoint | — | no | Omitted |

Target conditioning: `no` (no target input). `token_cost` was computed with the Qwen2.5 tokenizer
in training (`--target qwen2.5`). At serving time `OnnxSpanRewriter` uses the request's target
tokenizer when no fixed `cost` is given, which is a train/serve difference for non-Qwen targets.
Its size has not been measured.

## 4. Training data

| Dataset | Data card | Split(s) used | DVC hash |
| :--- | :--- | :--- | :--- |
| Span table `data/processed/span_features.parquet` (row 3.1.2) | none for the labelled set; source cards in [docs/data_cards/sources/](../data_cards/sources/dolly.md) | train | md5 `0fbf27fc45f6677c770005d283a08b9f` |
| Oracle table `data/processed/oracle.csv` | — | train | md5 `1fef5d14404ca0e37b3e332776854254` |

- Size: 129,380 train words from 1,262 train prompts
  ([p11_fast_path.md](../../reports/p11_fast_path.md), [p3_span_tagger.md](../../reports/p3_span_tagger.md)).
- Labels: each word of the prompt is aligned with the prompt's Qwen2.5 oracle rewrite (row 3.1.1)
  and tagged KEEP, DROP or REPLACE. The oracle is the shortest non-rejected candidate the judge
  labelled `preserved` on the target. Prompts with no such candidate keep the original, so every
  word is KEEP ([p2_headroom.md](../../reports/p2_headroom.md)). On the ≤ 256-word version of the
  table, 42.5% of words are droppable ([p8_textcnn.md](../../reports/p8_textcnn.md)).
- Label source: `Qwen/Qwen2.5-32B-Instruct-AWQ` judge, `pairwise_avg`
  ([ADR-0005](../decisions/ADR-0005-judge-protocol.md)), on Qwen2.5-7B-Instruct outputs. Phi-3.5
  labels are not used.
- Split: `data/processed/splits/` (4,955 train / 1,133 val / 992 test records; no shared groups,
  [splits.json](../../data/processed/splits/splits.json)). The labelled subset has 1,262 train
  prompts.

Test-split isolation audit: `pass` (C10.4, [status.md](../checkpoints/status.md)).

## 5. Metrics

Split: `train`, `GroupKFold(5)` by prompt, out-of-fold · Words: `129,380` · Prompts: `1,262` ·
Seeds: deterministic folds, single run · MLflow run: `9f315f7c7673448a827a809f6e861d8f`

### 5.1 Overall, per target

| Target | Droppable token macro-F1 (2 classes) | Median savings (fast-path rewrite, before gate) | Preservation | Net savings / 1k requests | Fallback rate | MLflow |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| Qwen2.5 (training tokenizer) | 0.631 | 28.6% (300 train prompts) | not measured | not measured | not measured | `9f315f7c…` |

Sources: [p11_fast_path.json](../../reports/p11_fast_path.json) (`tagger_macro_f1` 0.6313,
`median_savings` 0.2857). Savings are measured before the gate on train prompts. They are not a
held-out estimate and do not include gate rejections. The judged end-to-end run of tagger + gate
on 500 val prompts is checkpoint C5.6, still pending ([status.md](../checkpoints/status.md)).

### 5.2 Per category

| Category | Median savings | Preservation | Macro-F1 | n |
| :--- | ---: | ---: | ---: | ---: |
| code | not reported | not measured | not reported | — |
| fact | not reported | not measured | not reported | — |
| reason | not reported | not measured | not reported | — |
| explain | not reported | not measured | not reported | — |
| creative | not reported | not measured | not reported | — |
| transform | not reported | not measured | not reported | — |
| High-risk cluster(s) (C6.3) | none (riskiest cluster 1.195×, below 1.5×, [p6_cluster_risk.md](../../reports/p6_cluster_risk.md)) | | | |

No per-category tagger metric exists in any report. For context, oracle savings (the labels'
upper bound) differ strongly by category on Qwen2.5: code 25.8%, transform 24.2%, reason 60.5%
median ([p2_headroom.md](../../reports/p2_headroom.md)).

### 5.3 Latency

| Path | Hardware | Load (RPS) | Prompt length | p50 | p95 | p99 |
| :--- | :--- | ---: | :--- | ---: | ---: | ---: |
| fast, in-process (tagger rewrite + gate), warm | CPU | 1 request at a time | 300 train prompts | 1.1 ms | 7.3 ms | — |
| fast, HTTP API, 60 s open loop | CPU | 20.0 | 1,200 distinct train prompts | 10.8 ms | 20.9 ms | 27.9 ms |
| quality | RTX 4500 Ada | — | — | n/a | n/a | n/a |

Sources: [p11_fast_path.md](../../reports/p11_fast_path.md),
[p11_serving_checks.md](../../reports/p11_serving_checks.md). C11.2: pass.

### 5.4 Cross-model generalisation

Not measured. Labels exist only for the Qwen2.5 oracle.

### 5.5 Comparison with baselines

Two tasks are in use and their numbers are not interchangeable: **3-tag** macro-F1 (KEEP / DROP /
REPLACE, C3.4) and **binary** droppable macro-F1 (C8.2 and the deployed tagger).

| System | Task | Macro-F1 | Per tag (KEEP / DROP / REPLACE) | Data / folds | Source |
| :--- | :--- | ---: | :--- | :--- | :--- |
| Majority (all KEEP) | 3-tag | 0.245 | 0.735 / 0.000 / 0.000 | 129,380 tokens, grouped 5-fold | [p3_span_tagger.md](../../reports/p3_span_tagger.md) |
| Stratified | 3-tag | 0.337 | 0.587 / 0.277 / 0.145 | same | p3_span_tagger.md |
| Tree (depth 8, leaf 50) | 3-tag | 0.425 | 0.610 / 0.393 / 0.271 | same | p3_span_tagger.md |
| LR (multinomial, balanced), C3.4 best | 3-tag | 0.440 | 0.646 / 0.422 / 0.251 | same | p3_span_tagger.md |
| HMM, supervised MLE | 3-tag | 0.438 | — | 1,009 train / 253 held-out labelled prompts (both from the train split) | [p5_baum_welch.md](../../reports/p5_baum_welch.md) |
| HMM + Baum-Welch (3,000 unlabelled prompts) | 3-tag | 0.346 | — | same | p5_baum_welch.md |
| **Deployed: DecisionTree depth 10, 10 cheap features** | binary | **0.631** | — | 129,380 words, `GroupKFold(5)` | [p11_fast_path.md](../../reports/p11_fast_path.md) |
| HMM (Viterbi, KEEP vs not) | binary | 0.6571 | — | 122,815 words (≤ 256 / prompt), `StratifiedGroupKFold(5)` | [p8_textcnn.md](../../reports/p8_textcnn.md) |
| LightGBM on row 3.1.2 span features | binary | 0.6579 | — | same | p8_textcnn.md |
| TextCNN (concat) | binary | 0.6791 | — | same | p8_textcnn.md, [textcnn.md](textcnn.md) |

The deployed tree scores below the Phase 8 HMM and LightGBM taggers. It uses only microsecond
features (no surprisal, no embeddings), and its folds and word cap differ, so the gap is indicative
only. No McNemar test was run against it.

Checkpoint status ([status.md](../checkpoints/status.md)):

| Checkpoint | Target | Result |
| :--- | :--- | :--- |
| C3.4 | 3-tag token macro-F1 ≥ 0.65 | fail (best 0.440, LR) |
| C5.6 | first non-generative compressor: HMM / fast tagger + gate on 500 val prompts with the judge | open (not run) |
| C11.2 | fast-path p95 ≤ 150 ms at 20 RPS | pass (20.9 ms) |

The registry lineage of `terser-tagger / 2` lists `checkpoints_passed: []`
([p11_registry.md](../../reports/p11_registry.md)).

## 6. Quality-gate threshold

The tagger has its own decision threshold, `tagger_threshold: 0.5` (P(droppable) ≥ 0.5 deletes the
word). It was not tuned. Precision and recall at that threshold are not reported separately; the
0.631 macro-F1 uses `predict`, which is the same 0.5 cut.

The gate used with this tagger, its threshold (0.5, not selected by the row 3.2.3 rule) and its
metrics are documented in [quality_gate.md §6](quality_gate.md#6-quality-gate-threshold).

| Scope | Threshold | Sensitivity | Specificity | ROC-AUC | AP | Selected on |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| Overall (tagger, droppable) | 0.5 | not reported | not reported | not reported | not reported | default, not selected |
| per category | 0.5 | not reported | not reported | not reported | not reported | — |

## 7. Limitations and failure modes

| Failure mode | Share of degraded cases | Example (hashed id or link) | Mitigation in place |
| :--- | ---: | :--- | :--- |
| Dropped constraint | not measured | — | Hard-keep for negations, numbers and constraint keywords; gate `constraint_ratio` feature |
| Lost tone | not measured | — | none specific |
| Dropped example | not measured | — | Quoted and fenced spans are protected; unquoted examples are not |
| Changed meaning | not measured | — | Gate |
| Other | not measured | — | — |

The hand-categorised failure analysis (rows 8.3.10 / 12.8) has not been done for any compressor.

- **Deletion only.** The fast path cannot rephrase. Where the oracle replaced a phrase with a
  shorter one, the tagger can only delete it. Grammatical breakage is possible and is not checked.
- **Low agreement with the oracle.** 0.631 binary macro-F1 means many words are tagged against the
  oracle. The oracle itself is a single candidate chosen from a fixed candidate set, so the labels
  carry the bias of those candidate generators (operator stacks, LLMLingua-2, Qwen2.5-32B rewrites).
- **Judge dependence.** The oracle depends on the Qwen2.5-32B-AWQ `preserved` labels. Judge-human
  agreement (C2.4, C2.5) is unmeasured; raw A/B-order disagreement is 40.9% on the pilot (C2.6
  fail, [p2_judge_bias.md](../../reports/p2_judge_bias.md)).
- **Tokenizer drift.** `token_cost` is a target-tokenizer count. A tokenizer change shifts the
  feature; the drift job (`terser.monitoring.tokenizer_drift`, C11.6 pass) alerts on any changed
  fixture count. HF tokenizers are loaded by repo id without a pinned revision.
- **Savings are not net savings.** The 28.6% figure is before the gate and on train prompts.

## 8. Ethical considerations

- **Silent over-compression.** The tagger deletes words from the user's prompt. Only the gate
  stands between a harmful deletion and the target model. A gate rejection returns the original
  prompt. `fallback_used` ([docs/serving/api.md](../serving/api.md)) marks path fallbacks, not gate
  misses.
- **Privacy.** Training text passed the PII scrubber ([pii_policy.md](../pii_policy.md)); the regex
  backend does not detect names. At serving time logs carry SHA-256 hashes unless the request set
  `consent_raw_text: true`; raw text is kept at most 30 days
  ([docs/serving/logging.md](../serving/logging.md)).
- **Licences.** Labels derive from Qwen2.5-32B-AWQ verdicts on Qwen2.5-7B outputs (both
  Apache-2.0, [license_audit.md](../../reports/license_audit.md)). Dolly (CC-BY-SA-3.0) is the
  largest source (1,109 of 1,665 source groups in `splits.json`). ADR-0003 (training on model
  outputs) is referenced by CONTRIBUTING but has not been written.
- **Language.** English only; synthetic casual styles (hedging, over_polite, rambling, repetitive,
  texting) produced by an LLM. The lexicons (filler, stopword, constraint) are English word lists.
- **Compute.** CPU-only training. Labels inherit the judge stage cost (13,042 s wall time,
  [p2_label_stats.json](../../reports/p2_label_stats.json)).

## 9. Lineage

| Field | Value |
| :--- | :--- |
| Git SHA | `ae079d6d397107d8d3468d01ae511ac96a7382d6` |
| MLflow experiment / run id | `9f315f7c7673448a827a809f6e861d8f` |
| Hydra config (resolved) | none (CLI); `config_sha256` `65c4501aa58dc4b1e2bb08de61ffa2e38a200e9521c03d0f48b70e458de009b2` |
| Data DVC hash(es) | registry `data_hash` `0c451d108599d886`; `span_features.parquet` md5 `0fbf27fc45f6677c770005d283a08b9f`; `models/fast` md5 `9a0cd36b04ad789e706282c2aa304d05.dir` |
| Parent model | none |
| Judge model / revision used for labels | `Qwen/Qwen2.5-32B-Instruct-AWQ`, `pairwise_avg` (via the Qwen2.5 oracle); revision not recorded in the reports read |
| `gpu_queue` job id | none (CPU) |
| Checkpoints passed | registry: none. C11.2 pass; C3.4 fail; C5.6 open ([status.md](../checkpoints/status.md)) |
| Related ADRs | [ADR-0005](../decisions/ADR-0005-judge-protocol.md), [ADR-0011](../decisions/ADR-0011-second-family-phi35.md) |

## 10. Version history

| Card version | Registry version | Date | Change |
| :--- | :--- | :--- | :--- |
| 1.0 | `terser-tagger / 2` | 2026-10-17 | Initial |
