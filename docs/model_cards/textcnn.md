# Model Card: `textcnn` (TextCNN span scorer)

| Field | Value |
| :--- | :--- |
| Card version | `1.0` |
| Owner | `guru-bharadwaj20` |
| Last updated | `2026-10-17` |
| Registry name / version | not registered |
| Registry stage | `None` (research model; no exported weights) |

The TextCNN is a Phase 8 research model (rows 8.2.1-8.2.4, checkpoints C8.1 and C8.2). It is
trained inside each cross-validation fold by `python -m terser.phase8.textcnn_study`
([textcnn_study.py](../../src/terser/phase8/textcnn_study.py)). No final model is refit, saved or
served: `models/` holds only `fast/` and `rewriter/`, and the registry has no TextCNN entry
([p11_registry.md](../../reports/p11_registry.md)). The deployed tagger is the decision tree in
[span_tagger.md](span_tagger.md).

---

## 1. Model details

| Field | Value |
| :--- | :--- |
| Component | Span scorer: per-word droppable logit plus a pooled prompt-compressibility logit |
| Base model / revision | Inputs are frozen `BAAI/bge-small-en-v1.5` token states (384-d), sub-word states averaged per word; the encoder is not fine-tuned and not part of the parameter count. Revision not pinned |
| Architecture | Two parallel `Conv1d(384 -> 128)`, kernels 3 and 5, 'same' padding, ReLU → 256-d per-word features → token head `Linear(256 -> 1)`; prompt head on global max, global average, or both concatenated (`Linear(256 or 512 -> 1)`); masked pooling ([textcnn.py](../../src/terser/models/neural/textcnn.py)) |
| Parameters (total / trainable) | 393,986 (max or avg pooling) / 394,242 (concat); all trainable. Conv k=3 147,584 · conv k=5 245,888 · token head 257 · prompt head 257 (513 for concat) |
| Precision | float32 |
| Training method | Adam, lr 1e-3, 12 epochs, batch 16, dropout 0.0; token loss = class-weighted BCE (positive weight (1 - p)/p, p = droppable share of the training fold) masked to real words; plus unweighted BCE on the prompt head |
| Config | No Hydra experiment config; constants in `textcnn_study.py` (`MAX_WORDS = 256`, `epochs = 12`) |
| Hardware / GPU-hours | CPU (`torch.set_num_threads(8)`; bge-small states computed with `device="cpu"`). Wall time not recorded |
| Serving path | none |

C8.1 (parameter count matches the documented 393,986 exactly): pass ([status.md](../checkpoints/status.md);
the closed form `textcnn_param_count` is tested against `sum(p.numel())`).

## 2. Intended use

- Primary use: research comparison of a learned span scorer against the classical taggers (row
  8.2.4, C8.2), and the max vs average pooling study (row 8.2.3).
- Users: Phase 8 experiments and this project's reports. No API or pipeline calls it.
- Out of scope:
  - serving: there are no saved weights and no ONNX export, and it needs a bge-small forward pass
    per request, which the CPU fast path avoids;
  - the 3-tag KEEP / DROP / REPLACE task (it predicts droppable vs KEEP only);
  - prompts longer than 256 words (truncated) or longer than 512 bge-small sub-word tokens (words
    past the truncation get a zero vector);
  - non-English prompts (bge-small-en), multi-turn conversations, code that must be reproduced
    verbatim, real user prompts.

## 3. Supported target models and tokenizers

| Target | Tokenizer | Tokenizer version / revision | Evaluated | Notes |
| :--- | :--- | :--- | :---: | :--- |
| gpt-4o | `tiktoken` `o200k_base` | — | no | — |
| Qwen2.5-7B-Instruct | HF `tokenizers` | `Qwen/Qwen2.5-7B-Instruct` (labels only) | yes | Token labels and prompt-head target come from the Qwen2.5 oracle |
| Phi-3.5-mini-instruct | HF `tokenizers` | — | no | Second target family ([ADR-0011](../decisions/ADR-0011-second-family-phi35.md)); not used |
| Llama-3.1-8B-Instruct | HF `tokenizers` (ungated mirror) | — | no | Not available (ADR-0011) |
| Claude (optional) | Anthropic token-counting endpoint | — | no | Omitted |

Target conditioning: `no`. The model reads only the bge-small tokenizer's view of the text, so a
target-tokenizer change does not change its inputs. It changes the labels, through oracle savings.

## 4. Training data

| Dataset | Data card | Split(s) used | DVC hash |
| :--- | :--- | :--- | :--- |
| Span table `data/processed/span_features.parquet` (row 3.1.2) | none for the labelled set; [source cards](../data_cards/sources/dolly.md) | train | md5 `0fbf27fc45f6677c770005d283a08b9f` |
| Oracle table `data/processed/oracle.csv` (prompt-head target) | — | train | md5 `1fef5d14404ca0e37b3e332776854254` |

- Size: 1,262 train prompts, 122,815 words (≤ 256 per prompt), 42.5% droppable (DROP or REPLACE
  against the Qwen2.5 oracle rewrite). 14 words with no sub-word token keep a zero vector
  ([p8_textcnn.md](../../reports/p8_textcnn.md)).
- Token labels: word alignment of each prompt with its Qwen2.5 oracle rewrite (row 3.1.1). The
  oracle is the shortest candidate labelled `preserved` by the `Qwen/Qwen2.5-32B-Instruct-AWQ`
  judge (`pairwise_avg`, [ADR-0005](../decisions/ADR-0005-judge-protocol.md)) on Qwen2.5-7B
  outputs ([p2_headroom.md](../../reports/p2_headroom.md)).
- Prompt-head target: Qwen2.5 oracle savings above the median of the 1,262 prompts (binary;
  non-stand-in rows of `oracle.csv`).
- Folds: `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)`, stratified by category,
  grouped by prompt. The same folds are used for TextCNN, HMM and LightGBM.

Test-split isolation audit: `pass` (C10.4, [status.md](../checkpoints/status.md)). Only the train
split's span table is read.

## 5. Metrics

Split: `train`, grouped 5-fold out-of-fold · Prompts: `1,262` · Words: `122,815` · Seeds: `0, 1, 2`
(pooling study); seed 0 for the tagger comparison · MLflow run: not recorded in the report

### 5.1 Overall

| Pooling | Token macro-F1 (droppable, 2 classes) | Prompt-head ROC-AUC | Seeds |
| :--- | ---: | ---: | :--- |
| avg | 0.6759 ± 0.0019 | 0.7477 ± 0.0016 | 0, 1, 2 |
| max | 0.6763 ± 0.0031 | 0.7526 ± 0.0028 | 0, 1, 2 |
| concat | 0.6765 ± 0.0028 | 0.7549 ± 0.0007 | 0, 1, 2 |
| concat, seed 0 (used in the comparison) | 0.6791 (accuracy 0.6831) | — | 0 |

Source: [p8_textcnn.md](../../reports/p8_textcnn.md) (± is the sd over 3 seeds). Savings,
preservation, net savings and fallback rate do not apply: the TextCNN was never wired into a
rewriter, and nothing it tags has been judged.

### 5.2 Per category

| Category | Token macro-F1 | Prompt-head AUC | n |
| :--- | ---: | ---: | ---: |
| code | not reported | not reported | — |
| fact | not reported | not reported | — |
| reason | not reported | not reported | — |
| explain | not reported | not reported | — |
| creative | not reported | not reported | — |
| transform | not reported | not reported | — |
| High-risk cluster(s) (C6.3) | none (C6.3 fail: riskiest cluster 1.195×, [p6_cluster_risk.md](../../reports/p6_cluster_risk.md)) | | |

The study reports pooled metrics only. A per-category breakdown is needed before this card can
support any category-level claim.

### 5.3 Latency

Not measured. The model is not served.

### 5.4 Cross-model generalisation

Not applicable (labels exist for the Qwen2.5 oracle only).

### 5.5 Comparison with baselines

Same 122,815 words and identical folds ([p8_textcnn.md](../../reports/p8_textcnn.md)):

| Tagger | Token macro-F1 | Accuracy |
| :--- | ---: | ---: |
| TextCNN (concat) | 0.6791 | 0.6831 |
| LightGBM on row 3.1.2 span features (200 trees, lr 0.1) | 0.6579 | 0.6688 |
| HMM (supervised, Viterbi, KEEP vs not) | 0.6571 | 0.6619 |

McNemar on discordant tokens:

| A | B | only A right | only B right | statistic | p |
| :--- | :--- | ---: | ---: | ---: | ---: |
| TextCNN (concat) | HMM | 20,063 | 17,450 | 181.9 | 1.757e-41 |
| TextCNN (concat) | LightGBM | 20,021 | 18,261 | 80.82 | 2.433e-19 |
| HMM | LightGBM | 16,083 | 16,936 | 21.98 | 2.745e-06 |

The report calls the test "exact binomial" but tabulates a `chi2` column; the column is the
`statistic` field returned by `terser.eval.significance.mcnemar`.

For reference, the deployed fast-path decision tree reaches 0.631 binary macro-F1 on its own folds
(129,380 words, no 256-word cap, [p11_fast_path.md](../../reports/p11_fast_path.md)). The best 3-tag
classical result (C3.4) is 0.440 ([p3_span_tagger.md](../../reports/p3_span_tagger.md)), which is a
different task.

Checkpoint status ([status.md](../checkpoints/status.md)):

| Checkpoint | Target | Result |
| :--- | :--- | :--- |
| C8.1 | parameter count matches the documented count exactly | pass (393,986) |
| C8.2 | token macro-F1 ≥ C3.4 best + 0.05, or documented reason | fail: 0.679 vs best classical binary tagger 0.658 (+0.021; target 0.708), McNemar p = 2.4e-19 vs LightGBM |

C8.2 was scored against the best classical tagger **on the same binary task** (+0.05 → 0.708),
not against the 3-tag C3.4 value (0.440 + 0.05). The report gives the reason: the two tasks are
not comparable ([p8_textcnn.md](../../reports/p8_textcnn.md)). The gain over the classical taggers
is statistically clear but small (2.1 pp).

## 6. Quality-gate threshold

Not applicable. The TextCNN is not a gate. Its token decision uses logit > 0 (P > 0.5), which was
not tuned. Prompt-head AUC is in §5.1.

| Scope | Threshold | Sensitivity | Specificity | ROC-AUC | AP | Selected on |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| Overall | n/a | n/a | n/a | n/a | n/a | n/a |

## 7. Limitations and failure modes

| Failure mode | Share of degraded cases | Example (hashed id or link) | Mitigation in place |
| :--- | ---: | :--- | :--- |
| Dropped constraint | not measured | — | none (no rewrite is produced) |
| Lost tone | not measured | — | none |
| Dropped example | not measured | — | none |
| Changed meaning | not measured | — | none |
| Other | not measured | — | none |

No rewrite from TextCNN tags was ever generated or judged, so there is no failure analysis.

- **Small gain.** +0.021 macro-F1 over LightGBM / HMM at the cost of a bge-small forward pass per
  prompt. The C8.2 margin (+0.05) was not reached.
- **Pooling matters little.** Token F1 is within 0.0006 across max / avg / concat. Prompt-head AUC
  ranges 0.748-0.755, with concat best ([p8_textcnn.md](../../reports/p8_textcnn.md)).
- **Label ceiling.** The labels align to one oracle rewrite per prompt. A word that several
  equally good rewrites treat differently is labelled by whichever candidate was shortest.
- **Truncation.** Words past 256 are dropped from training and evaluation. Words past bge-small's
  512-token limit get zero vectors. Long prompts are under-represented.
- **Judge dependence.** Oracle labels rest on the Qwen2.5-32B-AWQ judge. C2.4 / C2.5 are
  unmeasured (no human labels); C2.6 raw order disagreement is 40.9% on the pilot
  ([p2_judge_bias.md](../../reports/p2_judge_bias.md)).
- **Tokenizer drift.** A change of the bge-small tokenizer changes the word averaging. A change of
  the Qwen2.5 tokenizer changes oracle savings and the prompt-head target. Neither revision is
  pinned in the study code. The C11.6 drift job covers target tokenizers, not bge-small.

## 8. Ethical considerations

- **Over-compression.** Not applicable in deployment: the model never edits user text.
- **Privacy.** Inputs are scrubbed synthetic prompts ([pii_policy.md](../pii_policy.md)). The model
  is not served and logs nothing.
- **Licences.** Labels derive from Apache-2.0 Qwen2.5 models ([license_audit.md](../../reports/license_audit.md)).
  Dolly (CC-BY-SA-3.0) is the largest source (1,109 of 1,665 source groups in `splits.json`).
  bge-small-en-v1.5 is used as a frozen feature extractor; its licence was not part of the audit
  read for this card. [ADR-0003](../decisions/ADR-0003-training-on-model-outputs.md) (training on model outputs) records the licence check.
- **Language.** English-only encoder and English synthetic casual prompts.
- **Compute.** CPU-only; 3 poolings × 3 seeds × 5 folds = 45 trainings of 12 epochs, plus one
  bge-small pass over 1,262 prompts.

## 9. Lineage

| Field | Value |
| :--- | :--- |
| Git SHA | not recorded in the report |
| MLflow experiment / run id | not recorded in the report |
| Hydra config (resolved) | none; constants in [textcnn_study.py](../../src/terser/phase8/textcnn_study.py) |
| Data DVC hash(es) | `span_features.parquet` md5 `0fbf27fc45f6677c770005d283a08b9f`; `oracle.csv` md5 `1fef5d14404ca0e37b3e332776854254` |
| Parent model | `BAAI/bge-small-en-v1.5` (frozen inputs) |
| Judge model / revision used for labels | `Qwen/Qwen2.5-32B-Instruct-AWQ`, `pairwise_avg` (via the Qwen2.5 oracle) |
| `gpu_queue` job id | none (CPU) |
| Checkpoints passed | C8.1 pass; C8.2 fail ([status.md](../checkpoints/status.md)) |
| Related ADRs | [ADR-0005](../decisions/ADR-0005-judge-protocol.md) |

## 10. Version history

| Card version | Registry version | Date | Change |
| :--- | :--- | :--- | :--- |
| 1.0 | none | 2026-10-17 | Initial |
