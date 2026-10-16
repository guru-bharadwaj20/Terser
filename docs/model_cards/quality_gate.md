# Model Card: `quality_gate` (CPU fast-path gate)

| Field | Value |
| :--- | :--- |
| Card version | `1.1` |
| Owner | `guru-bharadwaj20` |
| Last updated | `2026-10-17` |
| Registry name / version | `terser-gate / 2` ([p11_registry.md](../../reports/p11_registry.md)) |
| Registry stage | `Production` (history `None -> Staging -> Production`) |

This card describes the gate that is **deployed**: the scikit-learn gradient-boosting model exported
to `models/fast/gate.onnx` by `python -m terser.phase11.train_fast_path`
([train_fast_path.py](../../src/terser/phase11/train_fast_path.py),
[p11_fast_path.md](../../reports/p11_fast_path.md)). It is **not** the Phase 5 / Phase 10 LightGBM
that the checkpoints call the champion (AUC 0.778 / 0.783). That model uses GPT-2 surprisal and
embedding features that the fast path does not compute. The research gates are listed in §5.5 for
comparison only.

---

## 1. Model details

| Field | Value |
| :--- | :--- |
| Component | Quality gate: binary classifier, output `P(degraded)` for an (original, rewrite) pair |
| Base model / revision | None (classical model, trained from scratch) |
| Architecture | `sklearn.ensemble.GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=0)` on 30 features: 14 surface features of the original, the same 14 of the rewrite, `token_ratio`, `constraint_ratio` (`GATE_FEATURE_NAMES` in [features/shared.py](../../src/terser/features/shared.py)) |
| Parameters (total / trainable) | 200 trees of depth ≤ 3; no weight count applies |
| Precision | float32 inputs; ONNX graph (export via `terser.serving.onnx_export.export_sklearn`) |
| Training method | 5-fold grouped CV (groups = source prompt) to score, then a refit on all pairs |
| Config | No Hydra experiment config. CLI defaults in [train_fast_path.py](../../src/terser/phase11/train_fast_path.py). Registry `config_sha256` `2f65400f…64c6` |
| Hardware / GPU-hours | CPU only; 0 GPU-hours for training (labels came from GPU judge runs, see §8) |
| Serving path | fast (ONNX Runtime, `CPUExecutionProvider`), [configs/serving/fast_onnx.yaml](../../configs/serving/fast_onnx.yaml) |

ONNX vs scikit-learn parity: max |dP| 1.0e-07 on 2,000 training rows
([p11_fast_path.json](../../reports/p11_fast_path.json): `gate_parity` 9.99e-08).

## 2. Intended use

- Primary use: in the CPU fast path, score each rewrite made by the span tagger
  ([span_tagger.md](span_tagger.md)). If `P(degraded) > gate_threshold` (0.264564 in
  `configs/serving/fast_onnx.yaml`, the row 3.2.3 operating point, §6), the service returns the
  original prompt (`terser.serving.service`, line `rejected = score > self.gate_threshold`).
- At that threshold the gate passes 23.7% of the labelled train candidates and rejects 298 of 300
  tagger rewrites of train prompts, so the fast path currently saves almost nothing after the
  gate (median 0.0%, mean 0.1%; §5.1).
- Users: `POST /rewrite` in fast mode, and the fast-path fallback when the quality path crashes or
  times out (C11.5, [p11_serving_checks.md](../../reports/p11_serving_checks.md)).
- Out of scope:
  - gating outputs of the generative rewriter. The gate was trained only on Phase 2 candidates
    (operator stacks, LLMLingua-2, `llm_low/medium/high`) and was never evaluated on SFT rewrites;
  - acting as the C5.1 champion gate (it fails C5.1, like every other gate);
  - a quality guarantee: the sensitivity ≥ 0.95 operating point (§6) holds on grouped
    out-of-fold scores of Phase 2 candidates, pooled over categories. Sensitivity on tagger
    rewrites is unmeasured, and in the `reason` category it is 0.794 at the pooled threshold;
  - multi-turn conversations, non-English prompts, prompts with code that the target must
    reproduce verbatim, safety-critical instructions;
  - real user prompts: it was trained and scored only on synthetic casual prompts; the WildChat
    evaluation slice ([real_slice.md](../data_cards/real_slice.md)) has not been run through it;
  - feature computation with a tokenizer other than Qwen2.5 (`gate_feature_target: qwen2.5`).

## 3. Supported target models and tokenizers

| Target | Tokenizer | Tokenizer version / revision | Evaluated | Notes |
| :--- | :--- | :--- | :---: | :--- |
| gpt-4o | `tiktoken` `o200k_base` | `o200k_base` | no | Not used by the gate; features always use Qwen2.5 counts |
| Qwen2.5-7B-Instruct | HF `tokenizers` | `Qwen/Qwen2.5-7B-Instruct`, no revision pinned in [count.py](../../src/terser/tokenizers/count.py) | yes (labels) | Feature tokenizer and label target |
| Phi-3.5-mini-instruct | HF `tokenizers` | `microsoft/Phi-3.5-mini-instruct`, no revision pinned | yes (labels) | Second target family, replaces Llama-3.1 ([ADR-0011](../decisions/ADR-0011-second-family-phi35.md)) |
| Llama-3.1-8B-Instruct | HF `tokenizers` (ungated mirror) | `unsloth/Meta-Llama-3.1-8B-Instruct` | no | Gated repo not accessible; no labels (ADR-0011) |
| Claude (optional) | Anthropic token-counting endpoint | — | no | Omitted; no key |

Target conditioning: `no`. The 30 features contain no target indicator. Each (prompt, candidate)
pair appears once per target with that target's label, so identical feature rows can carry
different labels. The gate scores the average risk over the two target families, not the risk for
the request's target.

## 4. Training data

| Dataset | Data card | Split(s) used | DVC hash |
| :--- | :--- | :--- | :--- |
| Judge labels `data/processed/labels.jsonl` | none for the labelled set; source cards [dolly](../data_cards/sources/dolly.md), [gsm8k](../data_cards/sources/gsm8k.md), [mbpp](../data_cards/sources/mbpp.md), [humaneval](../data_cards/sources/humaneval.md) | train | md5 `c5a38491cbd84f4094d4a46ada4ca47b` (`labels.jsonl.dvc`) |
| Prompts `data/interim/label_prompts.jsonl`, candidates `data/interim/candidates.jsonl` | as above | train | registry `data_hash` `0c451d108599d886` |
| Splits `data/processed/splits/` | — | train | `splits.json` (seed 0) |

- Size: 14,732 labelled train pairs from 1,262 train prompts
  ([p11_fast_path.md](../../reports/p11_fast_path.md), [p3_gate_checkpoint.md](../../reports/p3_gate_checkpoint.md)).
  The pairs cover both targets ([p6_cluster_risk.md](../../reports/p6_cluster_risk.md): "14,732 train
  label records (both targets)"). Degraded share 0.431 ([p3_loco.md](../../reports/p3_loco.md)).
- Labelled subset: 1,800 prompts, 1,262 train / 285 val / 253 test (counted from the `split` field
  of `data/interim/label_prompts.jsonl`). Only train rows are read (`split == "train"` filter in
  `train_fast_path.py`). Stand-in and rejected candidates are skipped.
- Full split (`splits.json`): 4,955 train / 1,133 val / 992 test records, 1,165 / 266 / 234 source
  groups; no shared groups and no near-duplicate pairs at threshold 0.8.
- Label source: `Qwen/Qwen2.5-32B-Instruct-AWQ` judge, `pairwise_avg` protocol (order-averaged
  verdict probabilities, [ADR-0005](../decisions/ADR-0005-judge-protocol.md),
  [p2_judge_bias.md](../../reports/p2_judge_bias.md)), on outputs of Qwen2.5-7B-Instruct and
  Phi-3.5-mini-instruct. Label = `degraded` unless tie-or-better in both orders. Full-scale judge
  stage: 20,986 records, 17,644 judged pairs, 3,342 identical-output pairs, 4,874 low-margin
  verdicts ([p2_label_stats.json](../../reports/p2_label_stats.json)).

Test-split isolation audit: `pass` (C10.4, `python -m terser.hpo.audit --scan src/terser`,
[status.md](../checkpoints/status.md)).

## 5. Metrics

Split: `train`, grouped 5-fold out-of-fold · Pairs: `14,732` (1,262 prompts) · Seeds: `0` only ·
MLflow run: `930498fc3b7343abb9b510a7e4a8e6a2` (registry `source_run_id`)

Only one seed was run for the deployed gate, so no spread is available. The research gates in
§5.5 report mean ± sd over 3 or 5 seeds.

### 5.1 Overall, per target

| Target | Grouped-CV ROC-AUC | Median savings (fast path, before gate) | Median savings after gate (0.264564) | Preservation | Net savings / 1k requests | Fallback rate | MLflow |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| Pooled (Qwen2.5 + Phi-3.5 labels) | 0.760 | 28.6% (300 train prompts, Qwen2.5 tokenizer) | 0.0% (mean 0.1%; 298 of 300 rewrites rejected) | not measured | not measured | not measured | `930498fc…` |
| Qwen2.5-7B-Instruct | not reported separately | — | — | not measured | — | — | — |
| Phi-3.5-mini-instruct | not reported separately | — | — | not measured | — | — | — |

Source: [p11_fast_path.json](../../reports/p11_fast_path.json) (`gate_auc` 0.7605,
`median_savings` 0.2857, `operating_point.savings_oof`). The after-gate savings use, for each
prompt, the grouped-CV fold model that did not see that prompt's labelled pairs. At the old 0.5
threshold the same run rejects 199 of 300 rewrites (median 0.0%, mean 8.5%). With the deployed
`gate.onnx`, which was refit on these prompts' pairs, the counts are 298 of 300 at 0.264564 and
179 of 300 at 0.5 (mean 0.1% and 9.9%). The tagger was trained on these prompts too
([p11_fast_path.md](../../reports/p11_fast_path.md)). The fast path (tagger + gate) has never been judged end to end. That
run is checkpoint C5.6 (500 val prompts with the judge), still pending
([status.md](../checkpoints/status.md)).

### 5.2 Per category

The deployed gate column is per-category ROC-AUC of its pooled grouped 5-fold OOF scores (seed 0,
[p11_fast_path.md](../../reports/p11_fast_path.md)). For reference, per-category ROC-AUC of
research gates trained on the row 3.1.3 pair table (`pair_features.parquet`), on the same 14,732
pairs (grouped OOF for LR and KNN, leave-one-category-out for the tree):

| Category | Deployed GB gate, grouped OOF | LR, grouped OOF ([p3_cv_leakage.md](../../reports/p3_cv_leakage.md)) | KNN k = 25, manhattan, distance ([p3_knn_grid.md](../../reports/p3_knn_grid.md)) | Tree depth 4, leave-category-out ([p3_loco.md](../../reports/p3_loco.md)) | n pairs | Degraded share |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| code | 0.751 | 0.753 ± 0.001 | 0.731 ± 0.003 | 0.735 | 2,620 | 0.458 |
| fact | 0.756 | 0.732 ± 0.001 | 0.751 ± 0.003 | 0.730 | 2,242 | 0.430 |
| reason | 0.710 | 0.669 ± 0.001 | 0.731 ± 0.008 | 0.689 | 2,556 | 0.237 |
| explain | 0.699 | 0.692 ± 0.000 | 0.698 ± 0.003 | 0.697 | 2,534 | 0.472 |
| creative | 0.700 | 0.705 ± 0.000 | 0.700 ± 0.005 | 0.702 | 2,546 | 0.502 |
| transform | 0.792 | 0.782 ± 0.001 | 0.775 ± 0.003 | 0.763 | 2,234 | 0.493 |
| High-risk cluster(s) (C6.3) | none: riskiest cluster 1.195× the mean degradation rate (< 1.5×) | | | | | |

n and degraded share are the leave-one-category-out held-out counts from p3_loco.md. C4.5 (every
category ≥ 0.70) fails for every Phase 3/4 gate ([status.md](../checkpoints/status.md)). The
deployed gate is also below 0.70 on `explain` (0.699).

### 5.3 Latency

| Path | Hardware | Load (RPS) | Prompt length | p50 | p95 | p99 |
| :--- | :--- | ---: | :--- | ---: | ---: | ---: |
| fast, in-process (tagger rewrite + gate), warm | CPU | 1 request at a time | 300 train prompts | 1.1 ms | 7.3 ms | — |
| fast, HTTP API (`uvicorn`, 1 worker), 60 s open loop | CPU | 20.0 | 1,200 distinct train prompts | 10.8 ms | 20.9 ms | 27.9 ms |
| quality | RTX 4500 Ada | — | — | n/a | n/a | n/a |

Sources: [p11_fast_path.md](../../reports/p11_fast_path.md),
[p11_serving_checks.md](../../reports/p11_serving_checks.md). C11.2 (p95 ≤ 150 ms at 20 RPS): pass.
The quality path has no rewriter that passes C8.3, so it is not measured.

### 5.4 Cross-model generalisation

Not applicable. The gate is not target-conditioned and per-target AUC was not reported.

### 5.5 Comparison with baselines

ROC-AUC under grouped 5-fold CV on the same 14,732 train pairs. Specificity is at sensitivity 0.95
where reported.

| System | ROC-AUC | Specificity at sens. 0.95 | Source |
| :--- | ---: | ---: | :--- |
| Majority / always-preserved baseline | 0.500 ± 0.000 | 0.000 | [p3_gate_checkpoint.md](../../reports/p3_gate_checkpoint.md) |
| **Deployed: GradientBoosting, 30 surface features** | **0.760** (seed 0) | **0.380** (seed 0) | [p11_fast_path.md](../../reports/p11_fast_path.md) |
| LR (l2, C = 1) | 0.748 ± 0.000 | 0.303 ± 0.005 | p3_gate_checkpoint.md |
| Tree C4.5 depth 4, min leaf 10 (C3.3 choice) | 0.758 ± 0.002 | 0.405 ± 0.004 | p3_gate_checkpoint.md |
| KNN k = 51, manhattan, distance | 0.763 ± 0.001 | 0.377 ± 0.003 | p3_gate_checkpoint.md |
| PyTorch MLP on bge-small pair embeddings | 0.721 ± 0.003 | 0.283 ± 0.013 | [p4_quality_gate.md](../../reports/p4_quality_gate.md) |
| Random forest (300 trees, max_features 0.3) | 0.774 | — | [p5_ensembles.md](../../reports/p5_ensembles.md) |
| LightGBM (lr 0.1, depth 3, early stopping) | 0.778 | — | p5_ensembles.md |
| LightGBM tuned (GP search, 60 trials × 5 seeds) | 0.7829 ± 0.0007 | — | [p10_hpo.md](../../reports/p10_hpo.md) |
| MLP tuned (TPE) | 0.7749 ± 0.0006 | — | p10_hpo.md |

AdaBoost (0.797 from scratch, 0.791 scikit-learn, [p5_ensembles.md](../../reports/p5_ensembles.md))
is left out of the ranking. It was scored on a single grouped fold (`next(grouped_folds(...,
seed=0))` in `terser/phase5/ensembles.py`), not 5-fold OOF, so it is not comparable. No McNemar or
bootstrap test compares the deployed gate with the others.

Checkpoint status ([status.md](../checkpoints/status.md)):

| Checkpoint | Target | Result |
| :--- | :--- | :--- |
| C3.3 | AUC ≥ 0.75, specificity ≥ 0.40 at sens. 0.95 | pass (tree depth 4: 0.758 / 0.405) |
| C4.4 | MLP/SVM ≥ C3.3 + 0.03, or documented reason | pass (documented: MLPs 0.775 / 0.774 < 0.788) |
| C4.5 | every category ≥ 0.70 | fail (`explain` 0.673 embedding MLP, 0.698 best KNN) |
| C5.1 | AUC ≥ 0.85, specificity ≥ 0.60 | fail (LightGBM 0.778, tuned 0.783) |
| C5.2 | RF OOB within 0.02 of grouped CV | fail (+0.08) |
| C10.2 | tuning gain ≥ +0.01 or plateau | pass (+0.0095, documented plateau) |
| C11.2 | fast-path p95 ≤ 150 ms at 20 RPS | pass (20.9 ms) |

The registry lineage of `terser-gate / 2` lists `checkpoints_passed: ['C3.1']` only
([p11_registry.md](../../reports/p11_registry.md)).

## 6. Quality-gate threshold

All rates are on the grouped 5-fold out-of-fold scores of the 14,732 train pairs (seed 0), with
the service's rule: reject when `P(degraded) > gate_threshold`. Sensitivity = share of degraded
rewrites rejected, specificity = share of preserved rewrites passed. Source:
[p11_fast_path.md](../../reports/p11_fast_path.md), `operating_point` in
[p11_fast_path.json](../../reports/p11_fast_path.json).

| Scope | Threshold | Sensitivity | Specificity | Rewrites passed | ROC-AUC | AP | Selected on |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| **Overall (deployed)** | **0.264564** | **0.950** | **0.380** | 23.7% | 0.760 | 0.680 | grouped OOF, train split, row 3.2.3 rule |
| Overall, old default | 0.5 | 0.646 | 0.708 | 55.6% | 0.760 | 0.680 | fixed in config, not selected |
| code | 0.264564 | 0.966 | 0.298 | 17.7% | 0.751 | 0.730 | pooled threshold |
| fact | 0.264564 | 0.974 | 0.337 | 20.3% | 0.756 | 0.693 | pooled threshold |
| reason | 0.264564 | 0.794 | 0.517 | 44.3% | 0.710 | 0.406 | pooled threshold |
| explain | 0.264564 | 0.974 | 0.333 | 18.8% | 0.699 | 0.629 | pooled threshold |
| creative | 0.264564 | 0.969 | 0.341 | 18.5% | 0.700 | 0.651 | pooled threshold |
| transform | 0.264564 | 0.952 | 0.393 | 22.3% | 0.792 | 0.749 | pooled threshold |

The serving threshold `gate_threshold: 0.264564` in `configs/serving/fast_onnx.yaml` is the row
3.2.3 rule (maximise specificity subject to sensitivity ≥ 0.95,
`terser.eval.threshold.choose_threshold`) applied to the out-of-fold scores, written as the
shortest decimal that flags the same pairs under the service's strict `>`
(`terser.phase11.train_fast_path.service_threshold`). Rerun with
`python -m terser.phase11.train_fast_path --operating-point-only`. The `gate_threshold: 0.5` in
`configs/serving/default.yaml` and `docker/serving.compose.yaml` belongs to the rule-based
`ConstraintGate` (`gate_onnx: null`), not to this model; `terser.serving.api.build_service` now
refuses an ONNX gate without an explicit `gate_threshold`.

At the old 0.5 the gate passed 35.4% of degraded rewrites (sensitivity 0.646), and 27.5% of the
rewrites it passed were degraded. At 0.264564, 9.0% of the rewrites it passes are degraded, at
the cost of rejecting 62.0% of preserved ones. Specificity 0.380 is below the C3.3 floor of 0.40
that the Phase 3 tree met (0.405). The threshold is pooled: `reason` (degraded share 0.237) stays
below the floor at 0.794, the other five categories are at or above 0.95. The risk-table
mitigation "gate at sensitivity 0.95" therefore holds for pooled Phase 2 candidates only; it is
unmeasured on tagger rewrites (C5.6), and the threshold was fitted on fold-model scores and is
applied to the refit `gate.onnx`.

## 7. Limitations and failure modes

| Failure mode | Share of degraded cases | Example (hashed id or link) | Mitigation in place |
| :--- | ---: | :--- | :--- |
| Dropped constraint | not measured | — | Tagger never deletes negations, numbers, constraint keywords, code or quoted spans; `constraint_ratio` feature |
| Lost tone | not measured | AdaBoost hard examples `dolly:12133`, `dolly:9088` (over_polite under `ops_FHC` / `ops_CHFIDSL`) | none specific |
| Dropped example | not measured | — | none specific |
| Changed meaning | not measured | — | none specific |
| Other | not measured | short `texting` prompts labelled degraded after light rewrites (`dolly:12203`, `dolly:1140`, `dolly:12208`) | none specific |

The hand-categorised failure analysis (rows 8.3.10 / 12.8) has not been done; the examples are the
AdaBoost highest-weight pairs from [p5_ensembles.md](../../reports/p5_ensembles.md).

- **Surface features only.** The strongest predictors in Phase 5 are `sum_si_removed` (AUC drop
  0.134), `drop_ratio` (0.070) and `cat_reason` (0.068)
  ([p5_ensembles.md](../../reports/p5_ensembles.md)). The fast gate has no GPT-2 surprisal and no
  category input, so it cannot see how much information a deletion removed.
- **Weak categories.** `explain` and `creative` stay near 0.70 AUC for every KNN
  ([p3_knn_grid.md](../../reports/p3_knn_grid.md)). `reason` has the lowest leave-category-out AUC
  (0.689 tree, 0.659 LR, [p3_loco.md](../../reports/p3_loco.md)) and a different base rate (0.237).
- **Target-agnostic.** See §3. Degraded share differs by target: 41.3% Qwen2.5, 44.9% Phi-3.5
  ([p2_headroom.md](../../reports/p2_headroom.md), all 1,800 labelled prompts).
- **Judge dependence.** Labels are only as valid as the Qwen2.5-32B-AWQ judge. Judge-human
  agreement (C2.4, C2.5) is unmeasured because no human labels exist. Raw A/B-order disagreement
  is 40.9% on the pilot (C2.6 fail); `pairwise_avg` removes the order dependence of the label but
  not the judge's uncertainty (209 of 342 order-inconsistent pilot pairs have margin < 0.2,
  [p2_judge_bias.md](../../reports/p2_judge_bias.md)). Self-preference check (row 2.2.5, Phi-3.5 as
  second judge): label-agreement gap 4.8 pp between target families, within the 5 pp limit
  ([p2_self_preference.json](../../reports/p2_self_preference.json)).
- **Tokenizer drift.** `n_tokens`, `chars_per_token` and `token_ratio` are Qwen2.5 token counts.
  A tokenizer change shifts these features without any retraining. `terser.monitoring.tokenizer_drift`
  re-counts 200 fixture strings against a pinned snapshot and alerts on any change (C11.6 pass).
  The HF tokenizers themselves are loaded by repo id without a pinned revision.
- **Prompt length.** Net savings by prompt length were not measured for this path.

## 8. Ethical considerations

- **Silent over-compression.** A false negative (degraded rewrite scored safe) changes the user's
  prompt without notice. Responses carry `fallback_used` / `fallback_reason`
  ([docs/serving/api.md](../serving/api.md)) when the requested path did not produce the rewrite.
  A gate rejection returns the original prompt. A gate miss is not flagged to the user.
- **Privacy.** Training text was scrubbed before it entered `data/interim/`
  ([pii_policy.md](../pii_policy.md)). The regex backend does not detect personal names, and the
  500-row manual review required by row 2.1.3 is a human step. At serving time prompts and
  rewrites are logged as SHA-256 hashes; raw text only with `consent_raw_text: true`, retained at
  most 30 days ([docs/serving/logging.md](../serving/logging.md)).
- **Licences.** Labels come from Qwen2.5-32B-Instruct-AWQ (Apache-2.0) judging Qwen2.5-7B-Instruct
  (Apache-2.0) and Phi-3.5-mini-instruct (MIT, ADR-0011) outputs. No Llama-3.1 output is used, so
  the Llama naming and notice conditions in [license_audit.md](../../reports/license_audit.md) do not
  apply. Dolly (CC-BY-SA-3.0) is the largest source: 1,109 of 1,665 source groups in `splits.json`.
  The licence audit records model weights trained on Dolly-derived data as an unsettled
  ShareAlike question. [ADR-0003](../decisions/ADR-0003-training-on-model-outputs.md) (training
  on model outputs) records the licence check.
- **Language and population.** Prompts are English, from Dolly, GSM8K, MBPP and HumanEval,
  rewritten by an LLM into five casual styles (hedging, over_polite, rambling, repetitive,
  texting; [p2_headroom.md](../../reports/p2_headroom.md)). They are not real user prompts.
- **Compute.** Gate training is CPU-only. The judge stage behind `labels.jsonl` logged 13,042 s
  wall time and 805,371 generated tokens ([p2_label_stats.json](../../reports/p2_label_stats.json)).

## 9. Lineage

| Field | Value |
| :--- | :--- |
| Git SHA | `ae079d6d397107d8d3468d01ae511ac96a7382d6` |
| MLflow experiment / run id | `930498fc3b7343abb9b510a7e4a8e6a2` |
| Hydra config (resolved) | none (CLI); `config_sha256` `2f65400f92ec96e7896fd3c6971410c5edc47a62f329e7cde52e7aca8d4f64c6` |
| Data DVC hash(es) | registry `data_hash` `0c451d108599d886`; `labels.jsonl` md5 `c5a38491cbd84f4094d4a46ada4ca47b`; `models/fast` md5 `9a0cd36b04ad789e706282c2aa304d05.dir` |
| Parent model | none |
| Judge model / revision used for labels | `Qwen/Qwen2.5-32B-Instruct-AWQ`, `pairwise_avg`; revision not recorded in the reports read |
| `gpu_queue` job id | none (CPU) |
| Checkpoints passed | registry: C3.1. Gate checkpoints, scored on the research gates rather than this model: C3.3, C4.4, C10.2 pass; C4.5, C5.1, C5.2 fail. C11.2 (fast-path latency, this model) pass ([status.md](../checkpoints/status.md)) |
| Related ADRs | [ADR-0005](../decisions/ADR-0005-judge-protocol.md), [ADR-0011](../decisions/ADR-0011-second-family-phi35.md), [ADR-0010](../decisions/ADR-0010-mlops.md) |

## 10. Version history

| Card version | Registry version | Date | Change |
| :--- | :--- | :--- | :--- |
| 1.0 | `terser-gate / 2` | 2026-10-17 | Initial |
| 1.1 | `terser-gate / 2` | 2026-10-17 | Row 3.2.3 operating point measured and applied (`gate_threshold` 0.5 -> 0.264564 in `fast_onnx.yaml`); per-category OOF AUC/AP; savings after gate. Model file unchanged |
