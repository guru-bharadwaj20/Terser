# Model Card: `<model-name>`

<!--
Copy this file to docs/model_cards/<model-name>.md and fill every section (row 13.4 requires cards for
the gate, tagger, TextCNN and rewriter). Delete the guidance comments when done. Every number must link
to an MLflow run. Every metric table has per-category rows; an aggregate alone is not accepted.
-->

| Field | Value |
| :--- | :--- |
| Card version | `<e.g. 1.0>` |
| Owner | `<name>` |
| Last updated | `<YYYY-MM-DD>` |
| Registry name / version | `<e.g. terser-rewriter / 3>` |
| Registry stage | `None` / `Staging` / `Production` / `Archived` |

---

## 1. Model details

<!--
Component type (quality gate, span tagger, TextCNN span scorer, generative rewriter, RL policy), base
model and revision if fine-tuned, architecture, parameter count (total and trainable), precision,
training method (e.g. LoRA r=16, α=32), key hyperparameters with the Hydra config path, and the
hardware used (cpu / rtx4500ada) with GPU-hours.
-->

| Field | Value |
| :--- | :--- |
| Component | |
| Base model / revision | |
| Architecture | |
| Parameters (total / trainable) | |
| Precision | |
| Training method | |
| Config | `configs/experiment/<name>.yaml` |
| Hardware / GPU-hours | |
| Serving path | fast (ONNX Runtime, CPU) / quality (vLLM, GPU) |

## 2. Intended use

<!--
What the model does inside Terser, who calls it (POST /rewrite, CT pipeline, evaluation), and what it is
not for. Out-of-scope examples: multi-turn conversations, non-English prompts, prompts containing code
the target must reproduce verbatim, safety-critical instructions, targets not listed in §3.
-->

- Primary use:
- Users:
- Out of scope:

## 3. Supported target models and tokenizers

<!--
Targets the model was trained or evaluated for. Pin tokenizer versions: a tokenizer change is a drift
event (row 11.5.3). State whether the model is target-conditioned (`<target=…>` prefix) or target-agnostic
(C8.6 finding). Only Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct are generation targets (served locally
on vLLM), so only they have preservation numbers. gpt-4o is tokenizer-only (savings only). A Claude row
is optional and only filled if an Anthropic key is ever added for token counting; it is never required
for a reported number.
-->

| Target | Tokenizer | Tokenizer version / revision | Evaluated | Notes |
| :--- | :--- | :--- | :---: | :--- |
| gpt-4o | `tiktoken` `o200k_base` | | | Tokenizer only: savings, no preservation |
| Qwen2.5-7B-Instruct | HF `tokenizers` | | | |
| Llama-3.1-8B-Instruct | HF `tokenizers` (ungated mirror) | | | |
| Claude `<model id>` (optional) | Anthropic token-counting endpoint | | | Tokenizer only; omit unless a key was added |

Target conditioning: `<yes / no>`

## 4. Training data

<!--
Link the data card(s). Record the DVC hash of every dataset version used and the split used for
training, validation and threshold selection. Confirm the test split was not touched before Phase 12
(row 10.8 audit result).
-->

| Dataset | Data card | Split(s) used | DVC hash |
| :--- | :--- | :--- | :--- |
| | `docs/data_cards/<name>.md` | train / val | |

Test-split isolation audit: `<pass / fail, link>`

## 5. Metrics

<!--
State the split (val or test), number of prompts, seeds (≥ 3), and give mean ± std or 95% bootstrap CI.
Definitions:
- Token savings: (tokens_before − tokens_after) / tokens_before on the target tokenizer; report the median.
- Quality preservation: share of rewrites labelled `preserved` (task metric not worse and judge
  tie-or-better in both orders), with gate and fallback on. Name the judge model and revision
  (Qwen2.5-32B-Instruct-AWQ, or the 14B-AWQ fallback, on vLLM; the 8B small judge only if C8.7 passed).
  Outputs from the CPU smoke-test stand-in (`stand_in: true`) never feed these numbers.
- Net savings: price_T × (tokens_before − tokens_after) − rewriter_cost, per request.
- Latency: p95 end-to-end for the stated path, load and prompt length.
Always include the "original prompt" baseline (0% savings, 100% preservation) and the current champion.
-->

Split: `<val / test>` · Prompts: `<n>` · Seeds: `<list>` · MLflow run(s): `<links>`

### 5.1 Overall, per target

| Target | Median savings | Preservation | Net savings / 1k requests | Fallback rate | MLflow |
| :--- | ---: | ---: | ---: | ---: | :--- |
| | | | | | |

### 5.2 Per category

| Category | Median savings | Preservation | n |
| :--- | ---: | ---: | ---: |
| code | | | |
| fact | | | |
| reason | | | |
| explain | | | |
| creative | | | |
| transform | | | |
| High-risk cluster(s) (C6.3) | | | |

### 5.3 Latency

| Path | Hardware | Load (RPS) | Prompt length | p50 | p95 | p99 |
| :--- | :--- | ---: | :--- | ---: | ---: | ---: |
| fast | CPU | | | | | |
| quality | RTX 4500 Ada | | ≤ 256 tokens | | | |

### 5.4 Cross-model generalisation

<!-- Rewrite conditioned on row target, evaluated on column target (row 12.5). Cells: preservation / median savings.
Preservation needs target outputs, so columns are the locally served generation targets only; savings for
tokenizer-only targets (gpt-4o, optional Claude) go in §5.1. -->

| Conditioned on ↓ / evaluated on → | Qwen2.5 | Llama-3.1 |
| :--- | :---: | :---: |
| Qwen2.5 | | |
| Llama-3.1 | | |

### 5.5 Comparison with baselines

| System | Median savings | Preservation | McNemar / bootstrap p vs this model |
| :--- | ---: | ---: | :--- |
| Original prompt | 0% | 100% | — |
| Current champion | | | |
| | | | |

## 6. Quality-gate threshold

<!--
For the quality gate itself, or for the gate used with this model. Positive class = degraded.
Threshold rule (row 3.2.3): maximise specificity subject to sensitivity ≥ 0.95, chosen on validation
out-of-fold predictions, never on test. Report the threshold, sensitivity, specificity, ROC-AUC and AP,
overall and per category.
-->

| Scope | Threshold | Sensitivity | Specificity | ROC-AUC | AP | Selected on |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| Overall | | | | | | val OOF |
| code | | | | | | |
| fact | | | | | | |
| reason | | | | | | |
| explain | | | | | | |
| creative | | | | | | |
| transform | | | | | | |

## 7. Limitations and failure modes

<!--
Use the failure analysis (8.3.10, 12.8): dropped constraint, lost tone, dropped example, changed meaning,
other, with counts. Include known weak categories, prompt-length ranges where net savings are negative,
judge dependence (numbers are only as valid as the open-model judge, C2.6; self-preference toward the
judge's own model family, 2.2.5), and behaviour under tokenizer drift.
-->

| Failure mode | Share of degraded cases | Example (hashed id or link) | Mitigation in place |
| :--- | ---: | :--- | :--- |
| Dropped constraint | | | |
| Lost tone | | | |
| Dropped example | | | |
| Changed meaning | | | |
| Other | | | |

## 8. Ethical considerations

<!--
Silent over-compression can change what a user asked for without them noticing; describe how fallback
and `fallback_used` expose this. Privacy: production stores hashes by default; raw text only with consent,
retained ≤ 30 days. Training on open-model outputs: state the licences that apply
(Llama 3.1 Community Licence, Qwen2.5 licence) and link ADR-0003. Language and cultural bias from the
data card. Energy and compute used.
-->

## 9. Lineage

<!--
Everything needed to trace this version back to code, data and decisions (row 11.2.3).
-->

| Field | Value |
| :--- | :--- |
| Git SHA | |
| MLflow experiment / run id | |
| Hydra config (resolved) | MLflow artifact path |
| Data DVC hash(es) | |
| Parent model (e.g. SFT for RL) | |
| Judge model / revision used for labels | |
| `gpu_queue` job id | |
| Checkpoints passed | e.g. C2.7, C7.1, C8.3 (links to `docs/checkpoints/C<N>/README.md`) |
| Related ADRs | |

## 10. Version history

| Card version | Registry version | Date | Change |
| :--- | :--- | :--- | :--- |
| 1.0 | | | Initial |
