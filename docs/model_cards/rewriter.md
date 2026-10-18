# Model Card: `rewriter` (SFT generative rewriter ladder)

| Field | Value |
| :--- | :--- |
| Card version | `1.3` |
| Owner | `guru-bharadwaj20` |
| Last updated | `2026-10-18` |
| Registry name / version | `terser-rewriter / 16` (3B LoRA); earlier rungs `/ 14` (0.5B full), `/ 15` (1.5B LoRA) ([p11_registry.md](../../reports/p11_registry.md)) |
| Registry stage | `Staging` (v16); v14 and v15 `Archived`. No rewriter is in `Production` |

Every SFT rung fails the main gate C8.3 ([p8_rewriter_val.md](../../reports/p8_rewriter_val.md)).
v16 (3B LoRA) is held at Staging. The quality path is not served with any of
them. The retrain on augmented targets (`sft_v2`) also fails C8.3 (sft15 v2: 22.6% savings, 58.1%
preserved; sft3b v2: 24.2%, 65.3%; sft7b v2: 35.8%, 61.9%); see [Retrain on sft_v2](#retrain-on-sft_v2).

---

## 1. Model details

| Field | sft05 (v14) | sft15 (v15) | sft3b (v16, Staging) |
| :--- | :--- | :--- | :--- |
| Component | generative rewriter | generative rewriter | generative rewriter |
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` | `Qwen/Qwen2.5-1.5B-Instruct` | `Qwen/Qwen2.5-3B-Instruct` |
| Base revision | not pinned in config | not pinned | not pinned |
| Training method | full fine-tune, AdamW, grad checkpointing | LoRA r = 16, α = 32, dropout 0.05, grad checkpointing | LoRA r = 16, α = 32, dropout 0.05, grad checkpointing |
| Steps / batch / lr / max_len | 3,000 / 4 / 1e-5 / 1,024 | 3,000 / 4 / 2e-4 / 1,024 | 1,000 / 4 / 2e-4 / 1,024 |
| Precision | bf16 | bf16 | bf16 |
| Parameters (total / trainable) | not reported (all trainable) | not reported (adapter file 73.9 MB) | not reported (adapter file 119.8 MB) |
| Training pairs | 1,262 | 1,262 | 1,262 |
| Last logged step losses | 1.4e-4, 1.1e-3, 7.2e-4 | 3.3e-4, 5.8e-3, 2.2e-3 | 2.6e-2, 1.0e-2, 1.5e-2 |
| Config | [sft_qwen05b_full.yaml](../../configs/experiment/sft_qwen05b_full.yaml) | [sft_qwen15b_lora.yaml](../../configs/experiment/sft_qwen15b_lora.yaml) | [sft_qwen3b_lora.yaml](../../configs/experiment/sft_qwen3b_lora.yaml) |
| Weights | `models/rewriter/sft-qwen0.5b-full` (DVC md5 `db6457ff56acd778cba8b46edf69bd89.dir`) | `models/rewriter/sft-qwen1.5b-lora` (`12c0ba167f9ac82af6383d4a3daaf464.dir`) | `models/rewriter/sft-qwen3b-lora` (`4b7256a1f6f23decc58e2cbf5fffdb53.dir`) |

Training losses are from each output directory's `loss.json` (`steps`, `n_pairs`, `loss`). The 3B
config notes that "3000 steps drove rung 1/2 loss to ~0", which is why rung 3 uses 1,000 steps
(about 3 epochs). Losses near zero on 1,262 pairs suggest the targets were memorised.

Hardware / GPU-hours: the `gpu_queue` jobs request `hardware=rtx4090`, with estimates of 8 h
(1.5B, measured 12.19 GiB reserved at seq 1,024, micro-batch 4) and 3 h (3B, est. 20 GB VRAM)
(`gpu_queue/sft-qwen1.5b-lora.yaml`, `gpu_queue/sft-qwen3b-lora.yaml`). Measured GPU-hours were
not recorded in the files read for this card.

Input format: prompt tokens, then the separator `"\n### Rewrite:\n"`, then the rewrite up to EOS.
Inference is greedy with at most 512 new tokens
([generate.py](../../src/terser/models/rewriter/generate.py)). Every output passes the row 8.3.7
safeguard: a rewrite that is not shorter than the prompt on the base tokenizer is replaced by the
original prompt (`fell_back`).

Serving path: quality (vLLM, GPU), **not enabled** (`quality.base_url: null` in
[configs/serving/fast_onnx.yaml](../../configs/serving/fast_onnx.yaml)).

## 2. Intended use

- Primary use (planned): the quality path of `POST /rewrite`. The rewriter generates a shorter
  prompt, the quality gate checks it, and the service falls back to the fast path or the original
  prompt on rejection, crash or timeout.
- Users: none in production. The Phase 8 val report and the registry are the only consumers.
- Out of scope:
  - production use of any current rung (C8.3 fails);
  - commercial use of the 3B rung. Qwen2.5-3B-Instruct is under the Qwen Research Licence
    (non-commercial), so v16 may be released for research only
    ([license_audit.md](../../reports/license_audit.md));
  - multi-turn conversations, non-English prompts, prompts with code the target must reproduce
    verbatim, safety-critical instructions;
  - real user prompts (no evaluation on the WildChat slice);
  - targets other than Qwen2.5-7B-Instruct and Phi-3.5-mini-instruct for preservation claims.

## 3. Supported target models and tokenizers

| Target | Tokenizer | Tokenizer version / revision | Evaluated | Notes |
| :--- | :--- | :--- | :---: | :--- |
| gpt-4o | `tiktoken` `o200k_base` | `o200k_base` | no | Savings not reported for this target |
| Qwen2.5-7B-Instruct | HF `tokenizers` | `Qwen/Qwen2.5-7B-Instruct`, no revision pinned ([count.py](../../src/terser/tokenizers/count.py)) | yes | Savings and preservation |
| Phi-3.5-mini-instruct | HF `tokenizers` | `microsoft/Phi-3.5-mini-instruct`, no revision pinned | preservation only | Replaces Llama-3.1 as second family ([ADR-0011](../decisions/ADR-0011-second-family-phi35.md)); savings not computed on its tokenizer |
| Llama-3.1-8B-Instruct | HF `tokenizers` (ungated mirror) | `unsloth/Meta-Llama-3.1-8B-Instruct` | no | Gated repo not accessible (ADR-0011) |
| Claude (optional) | Anthropic token-counting endpoint | — | no | Omitted |

Target conditioning: `no`. One target-agnostic rewriter (row 8.3.1), justified by C1.5: savings
rankings agree across tokenizers, lowest Spearman ρ 0.991 (gpt-4o vs qwen2.5)
([p1_tokenizer_study.md](../../reports/p1_tokenizer_study.md)). The training prompt has no
`<target=…>` prefix.

## 4. Training data

| Dataset | Data card | Split(s) used | DVC hash |
| :--- | :--- | :--- | :--- |
| `data/processed/sft` (row 8.3.1) | none; [source cards](../data_cards/sources/dolly.md) | train (training), val (report) | md5 `c9843e9ba4fcd39ac9019a40843e52e5.dir` (`sft.dvc`) |
| Judge labels `data/processed/labels.jsonl` (target selection) | — | train, val | md5 `c5a38491cbd84f4094d4a46ada4ca47b` |

- Construction ([sft_data.py](../../src/terser/models/rewriter/sft_data.py)): for each labelled
  prompt, the target is the **shortest candidate the judge labelled `preserved` for every target
  model** (Qwen2.5-7B-Instruct and Phi-3.5-mini-instruct), measured on the Qwen2.5 tokenizer and
  strictly shorter than the original. Prompts with no such candidate keep the original prompt
  ("identity" example), so the model also learns when not to compress. The test split is never
  read.
- Sizes and target savings ([data/processed/sft/stats.json](../../data/processed/sft/stats.json)):

| Split | Prompts | Identity targets | Median target savings |
| :--- | ---: | ---: | ---: |
| train | 1,262 | 470 | 10.1% |
| val | 285 | 109 | 10.0% |

- Label source: `Qwen/Qwen2.5-32B-Instruct-AWQ` judge, `pairwise_avg`
  ([ADR-0005](../decisions/ADR-0005-judge-protocol.md), [p2_judge_bias.md](../../reports/p2_judge_bias.md)),
  on Qwen2.5-7B and Phi-3.5 outputs. Candidates come from operator stacks (`ops_FHC`,
  `ops_CHFIDSL`), LLMLingua-2 and Qwen2.5-32B rewrites (`llm_low`, `llm_medium`, `llm_high`)
  ([p2_headroom.md](../../reports/p2_headroom.md)).
- Why the targets are short: requiring `preserved` on *both* targets and picking from a fixed
  candidate pool leaves a median of about 10% savings. The per-target oracle reaches 46.5%
  (Qwen2.5) and 40.5% (Phi-3.5) median ([p2_headroom.md](../../reports/p2_headroom.md)). That figure
  allows a different candidate per target and pools all 1,800 labelled prompts.

Test-split isolation audit: `pass` (C10.4, [status.md](../checkpoints/status.md)).

## 5. Metrics

Split: `val` · Prompts: `285` · Seeds: greedy decoding, single run per rung · Judge:
`Qwen/Qwen2.5-32B-Instruct-AWQ`, `pairwise_avg`, on both targets · MLflow run:
`2ec9104893714b53ad71a2b639a31b00` (registry source run of v16); runs for v14 / v15 and the val
report are not listed in the reports read.

How the numbers are computed ([val_report.py](../../src/terser/models/rewriter/val_report.py)):

- Median savings is one median over all 285 rewrites on the Qwen2.5 tokenizer, including rewrites
  that fell back (0% savings).
- Preservation is pooled over (rewrite, target) pairs for both targets. A rewrite that fell back
  counts as preserved by construction.
- The 8.3.7 safeguard (fallback to the original) is on. The quality gate is **not** applied in this
  report, although C8.3 is defined "gate + fallback on".
- Latency is the mean generation time per prompt in batches of 16.

### 5.1 Overall, per target

| Rung | n | Median savings (Qwen2.5 tok.) | Oracle (SFT target) median | Fell back | Preserved (both targets pooled) | Latency (s / prompt) | C8.3 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Original prompt | 285 | 0% | — | — | 100% | 0 | — |
| sft05 (v14) | 285 | 4.1% | 10.0% | 108 | 68.4% | 0.39 | fail |
| sft15 (v15) | 285 | 10.2% | 10.0% | 99 | 68.2% | 0.50 | fail |
| sft3b (v16) | 285 | 26.1% | 10.0% | 66 | 65.6% | 0.70 | fail |

Source: [p8_rewriter_val.md](../../reports/p8_rewriter_val.md),
[p8_rewriter_val.json](../../reports/p8_rewriter_val.json). Per-target preservation, savings on
the Phi-3.5 tokenizer, net savings and the post-gate fallback rate are not reported. C8.3 requires
median savings ≥ 25% on every target, preservation ≥ 92% overall and ≥ 85% per category. sft3b clears
25% on the Qwen2.5 tokenizer (26.1%; other targets not reported); all three rungs miss both
preservation bars.

### 5.2 Per category

Preservation (both targets pooled, fallbacks counted as preserved):

| Category | sft05 | sft15 | sft3b |
| :--- | ---: | ---: | ---: |
| code | 70.6% | 63.7% | 57.8% |
| fact | 66.4% | 74.5% | 64.5% |
| reason | 62.5% | 67.7% | 82.3% |
| explain | 71.4% | 78.6% | 72.9% |
| creative | 64.9% | 60.6% | 52.1% |
| transform | 75.5% | 66.3% | 66.3% |
| High-risk cluster(s) (C6.3) | none (riskiest cluster 1.195×, [p6_cluster_risk.md](../../reports/p6_cluster_risk.md)) | | |

Per-category median savings and per-category n are not reported. No rung reaches the 85%
per-category bar in any category (closest: sft3b `reason`, 82.3%).

### 5.3 Latency

| Path | Hardware | Load (RPS) | Prompt length | p50 | p95 | p99 |
| :--- | :--- | ---: | :--- | ---: | ---: | ---: |
| quality (offline batch 16, HF `generate`, not vLLM) | GPU (type not recorded in the report) | offline | val prompts | mean 0.39 / 0.50 / 0.70 s (sft05 / sft15 / sft3b) | not reported | not reported |
| quality (served) | RTX 4500 Ada | — | ≤ 256 tokens | not measured | not measured | not measured |

C11.3 (quality-path latency) and C11.4 (throughput) were measured and fail: p95 438 ms at 5 RPS with 0% quality-served, and 20 RPS sustained but every request answered by the fallback ([p11_quality_path.md](../../reports/p11_quality_path.md))
([status.md](../checkpoints/status.md)).

### 5.4 Cross-model generalisation

Not applicable (target-agnostic; no conditioning). Per-target preservation is not broken out.

### 5.5 Comparison with baselines

| System | Median savings | Preservation | McNemar / bootstrap p vs this model |
| :--- | ---: | ---: | :--- |
| Original prompt | 0% | 100% | — |
| Current champion | none (no rung passes C8.3) | — | — |
| SFT target (val oracle) | 10.0% | not judged as a system | — |
| Best operator, HMM compressor, LLMLingua-2, zero-shot Qwen2.5-32B-AWQ (row 8.3.9) | not run | not run | — |

The row 8.3.9 baselines have not been run. C8.4 (vs zero-shot Qwen2.5-32B-AWQ) and C8.5 (vs
LLMLingua-2) are open and depend on C8.3.

## 6. Quality-gate threshold

The quality gate was not applied to rewriter outputs in the val report. The deployed gate
(`terser-gate / 2`, threshold 0.264564 by the row 3.2.3 rule) was trained on Phase 2 candidates only and has not been
validated on SFT rewrites; see [quality_gate.md §6](quality_gate.md#6-quality-gate-threshold).

| Scope | Threshold | Sensitivity | Specificity | ROC-AUC | AP | Selected on |
| :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| Overall | not applied | — | — | — | — | — |

## 7. Limitations and failure modes

**Code fixes (2026-10-18), applied before the 2026-10-26 retrain.** All SFT numbers in this card
come from checkpoints retrained and re-judged after these changes:

- training, `generate` and GRPO use the whole `### Rewrite:` separator, not its last token;
- a rewrite too long for `max_len` no longer drops the prompt or ends in a mid-rewrite EOS;
- validation decoding passes `repetition_penalty=1.1`, the base model's generation config value;
- GRPO weights micro-batches by size, and its retention proxy matches whole items;
- the val report leaves unjudged rewrites out of the preservation rate (`missing_labels`)
  instead of counting them as degraded.

The failure-mode shares below come from the row 8.3.10 analysis of the first sft3b run
(2026-10-17) and were not redone on the retrained checkpoint.

| Failure mode | Share of degraded cases | Example (hashed id or link) | Mitigation in place |
| :--- | ---: | :--- | :--- |
| Dropped constraint | 5% | `dolly:97:hedging` (second request dropped) | 8.3.7 safeguard (length only) |
| Lost tone | 1% | `dolly:1050:repetitive` | none |
| Dropped example | 12% | `dolly:5264:rambling`; 4 of 12 are truncations at the 512-token cap | none |
| Changed meaning | 10% | `gsm8k:train:6101:repetitive` | none |
| Other | 72% | `dolly:5382:repetitive` (only "Um," removed); 63 of 72 show no visible loss | — |

Shares are from the row 8.3.10 failure analysis ([p8_failures.md](../../reports/p8_failures.md)):
100 of the 144 non-fallback degraded (prompt, target) pairs of the first sft3b run (v6), sampled by prompt
category with seed 0. The categorisation was done by an LLM assistant (Claude), not by a human
annotator, and has not been human-reviewed. Treat the shares as a first-pass triage.

- **Code line-case corruption.** 12 of the 100 analysed cases (all `code`) have line-initial
  `Assert` / `Def` and stripped indentation. The operators' `tidy` step upper-cases line starts,
  and 33 of 220 `code` SFT training targets carry the pattern.

- **Target ceiling.** The model imitates targets with a 10.0% val median. Perfect imitation would
  still miss the 25% savings bar. The sft_v2 retrain addresses this (17.4% val median), but that
  target is also below 25%.
- **Fallbacks inflate preservation.** Rungs fall back on 66-108 of 285 prompts. Those count as
  preserved and as 0% savings; sft05 falls back most (108) and has both the best preservation
  (68.4%) and the lowest savings (4.1%).
- **Memorisation.** Training loss reached about 1e-3 or below on rungs 1 and 2 (1,262 pairs, 3,000
  steps, batch 4).
- **Weak categories.** `code` preservation falls from 70.6% (sft05) to 57.8% (sft3b). `creative` is
  lowest for sft3b (52.1%).
- **Judge dependence.** All preservation numbers rest on the Qwen2.5-32B-AWQ judge. Raw A/B disagreement is 40.9% on the pilot (C2.6 fail).
  The judge is from the same family as the Qwen2.5 target and as the rewriter base models. The
  2.2.5 self-preference check found a 4.8 pp label-agreement gap between target families (≤ 5 pp,
  [p2_self_preference.json](../../reports/p2_self_preference.json)). The judge's preference for
  Qwen-written rewrites was not measured.
- **Tokenizer drift.** Savings use the Qwen2.5 tokenizer loaded by repo id without a pinned
  revision. The safeguard uses the base model's tokenizer. The C11.6 drift job alerts on changed
  fixture counts for target tokenizers.

## 8. Ethical considerations

- **Silent over-compression.** At 65.6-68.4% preservation, the judge labels 32-34% of
  (rewrite, target) pairs `degraded`, and the gate was not applied. That is why no rung is in
  Production. If deployed, gate
  rejections and path fallbacks are exposed through `fallback_used` / `fallback_reason`
  ([docs/serving/api.md](../serving/api.md)).
- **Privacy.** Training prompts are scrubbed synthetic text ([pii_policy.md](../pii_policy.md)). A
  generative model can reproduce training text. With near-zero training loss, verbatim output of
  training targets is plausible; this has not been tested. Serving logs store hashes unless the request sets
  `consent_raw_text: true`; raw text is kept at most 30 days ([docs/serving/logging.md](../serving/logging.md)).
- **Licences** ([license_audit.md](../../reports/license_audit.md)): Qwen2.5-0.5B / 1.5B are
  Apache-2.0. **Qwen2.5-3B is under the Qwen Research Licence (non-commercial)**, so v16 is
  research-only. Targets come from Qwen2.5-32B-AWQ (Apache-2.0) rewrites and operator / LLMLingua-2
  outputs, judged on Qwen2.5-7B (Apache-2.0) and Phi-3.5-mini (MIT) outputs. No Llama output is
  used. Dolly (CC-BY-SA-3.0) is the largest source (1,109 of 1,665 source groups in
  `splits.json`); a rewriter that reproduces Dolly text cannot claim a non-ShareAlike licence.
  [ADR-0003](../decisions/ADR-0003-training-on-model-outputs.md) (training on model outputs, written retroactively on 2026-10-16) finds the
  Apache-2.0 and MIT sources compliant and limits the Qwen2.5-3B rungs to research use.
- **Language.** English only; synthetic casual styles generated by an LLM; not real user prompts.
- **Compute.** Queue estimates are 8 GPU-h (1.5B) and 3 GPU-h (3B) on RTX 4090-class GPUs.
  Measured GPU-hours and energy are not recorded.

## 9. Lineage

| Field | Value |
| :--- | :--- |
| Git SHA | `cbee594ceec840e930bbe08226aa9733ad2901f0` (registry lineage of v16) |
| MLflow experiment / run id | v16 `2ec9104893714b53ad71a2b639a31b00`; v14 / v15 not listed |
| Hydra config (resolved) | v16 `config_sha256` `b5a60452ba4734242198bffd56cc56be43924e3a7192c05438ced263f364d444` |
| Data DVC hash(es) | registry `data_hash` `0c451d108599d886`; `sft` md5 `c9843e9ba4fcd39ac9019a40843e52e5.dir` |
| Parent model | `Qwen/Qwen2.5-{0.5B,1.5B,3B}-Instruct` |
| Judge model / revision used for labels | `Qwen/Qwen2.5-32B-Instruct-AWQ`, `pairwise_avg`; revision not recorded in the reports read |
| `gpu_queue` job id | `sft-qwen0.5b-full`, `sft-qwen1.5b-lora`, `sft-qwen3b-lora` |
| Checkpoints passed | registry: none. C8.3 fail; C8.4, C8.5, C9.3-C9.5, C10.3, C11.3, C11.4 blocked on C8.3 ([status.md](../checkpoints/status.md)) |
| Related ADRs | [ADR-0005](../decisions/ADR-0005-judge-protocol.md), [ADR-0011](../decisions/ADR-0011-second-family-phi35.md), [ADR-0002](../decisions/ADR-0002-problem-framing.md) |

## Retrain on sft_v2

Status as of 2026-10-18 (retrained 2026-10-26): sft15 v2, sft3b v2 and sft7b v2 (QLoRA) trained and
judged; all three fail C8.3. The row 8.3.4 escalation ladder (1.5B LoRA, 3B LoRA, 7B QLoRA) is exhausted.

**What changes.** Only the training data. Two extra rewrite levels, `llm_max` and `llm_min`, were
generated by `Qwen/Qwen2.5-32B-Instruct-AWQ` (prompt `rewrite_v1`, temperature 0) for the labelled
prompts. They were judged like the Phase 2 candidates and added to the pool the shortest
preserved-on-every-target target is chosen from ([p8_target_aug.yaml](../../configs/experiment/p8_target_aug.yaml),
`sft_data.py --candidates/--labels/--tokens`). The config notes that the 2026-10-16 augmentation
run predates `exclude_splits` and also rewrote test prompts; `sft_data` reads only train / val, so
those rows are dropped.

| Split | Prompts | Identity targets | Median target savings |
| :--- | ---: | ---: | ---: |
| train | 1,262 | 412 | 21.4% |
| val | 285 | 101 | 17.4% |

Source: [data/processed/sft_v2/stats.json](../../data/processed/sft_v2/stats.json). DVC: `sft_v2`
md5 `56c7be7283112eb9a5f82b3e5216cd8a.dir`; `labels_aug.jsonl` md5
`637952960b2b03280370c784c1cc1c34`.

| Rung | Config | Base | Method | Steps / batch / lr | Output dir |
| :--- | :--- | :--- | :--- | :--- | :--- |
| sft15 v2 | [sft_qwen15b_lora_v2.yaml](../../configs/experiment/sft_qwen15b_lora_v2.yaml) | Qwen2.5-1.5B-Instruct | LoRA r 16, α 32 | 3,000 / 4 / 2e-4 | `models/rewriter/sft-qwen1.5b-lora-v2` |
| sft3b v2 | [sft_qwen3b_lora_v2.yaml](../../configs/experiment/sft_qwen3b_lora_v2.yaml) | Qwen2.5-3B-Instruct | LoRA r 16, α 32 | 1,000 / 4 / 2e-4 | `models/rewriter/sft-qwen3b-lora-v2` |
| sft7b v2 | [sft_qwen7b_qlora_v2.yaml](../../configs/experiment/sft_qwen7b_qlora_v2.yaml) | Qwen2.5-7B-Instruct | QLoRA (NF4, double quant, bf16 compute) r 16, α 32 | 1,000 / 4 / 2e-4 | `models/rewriter/sft-qwen7b-qlora-v2` |

Expectation stated before results: the val target median (17.4%) is below the 25% C8.3 savings
bar, so imitation alone cannot pass C8.3 on savings.

Results (val split, same protocol as §5; [p8_rewriter_val.md](../../reports/p8_rewriter_val.md)).
Generation reads `data/processed/sft/val.jsonl`, so the oracle column is the v1 val target
(10.0%), not the v2 one (17.4%).

| Rung | n | Median savings | Oracle median | Fell back | Preserved | Latency (s) | C8.3 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| sft15 v2 | 285 | 22.6% | 10.0% | 73 | 58.1% | 0.49 | fail |
| sft3b v2 | 285 | 24.2% | 10.0% | 70 | 65.3% | 0.72 | fail |
| sft7b v2 | 285 | 35.8% | 10.0% | 63 | 61.9% | 1.24 | fail |

Per-category preservation:

| Rung | code | creative | explain | fact | reason | transform |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| sft15 v2 | 52.9% | 56.4% | 72.9% | 70.0% | 41.7% | 57.1% |
| sft3b v2 | 48.0% | 66.0% | 67.1% | 72.7% | 71.9% | 66.3% |
| sft7b v2 | 54.9% | 53.2% | 62.9% | 69.1% | 71.9% | 59.2% |

Reading:

- **sft15 v2 cuts the wrong things.** It saves 22.6% against a 17.4% target median at 58.1%
  preserved, the lowest of any rung; `reason` falls to 41.7%. It cuts worked steps or context the
  answer needs.
- **sft3b v2 matches sft3b v1.** 24.2% savings at 65.3% preserved (sft3b: 26.1%, 65.6%). The
  augmented targets did not move the 3B rung.
- **sft7b v2 saves the most.** 35.8% median savings, the only v2 rung above 25%, at 61.9%
  preserved and 1.24 s per prompt (1.8x sft3b).
- No rung is near the 92% overall / 85% per-category preservation bar. More target savings alone
  does not close C8.3, and neither does a larger base model.

Registry ([p11_registry.md](../../reports/p11_registry.md)): the post-audit rebuild registers only
the v1 ladder (v14-v16); v16 (sft3b v1) keeps Staging. The v2 weights are in DVC
(`models/rewriter/sft-qwen7b-qlora-v2.dvc`).

Data note: the first report run showed 0.0% preservation for both v2 rungs. The merge step
`cat ... > val_labels_all.jsonl` had failed silently because that file is a read-only DVC
hardlink, so the v2 labels never reached the report. The merged labels are now written to
`val_labels_all_v2.jsonl`, and the numbers above come from that file.

## 10. Version history

| Card version | Registry version | Date | Change |
| :--- | :--- | :--- | :--- |
| 1.0 | `terser-rewriter / 4, 5, 6` | 2026-10-17 | Initial; sft_v2 retrain in progress |
| 1.1 | `terser-rewriter / 4, 5, 6` | 2026-10-17 | sft_v2 results (sft15 v2, sft3b v2): both fail C8.3 |
| 1.2 | `terser-rewriter / 4-9` | 2026-10-17 | sft7b v2 (QLoRA) result, fails C8.3; v2 rungs registered as v7-v9 (Archived) |
| 1.3 | `terser-rewriter / 14-16` | 2026-10-18 | Post-audit rebuild: every rung retrained and re-judged with the code fixes in §7; all tables refreshed |
