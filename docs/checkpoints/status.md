# Checkpoint status

Status of all 63 checkpoints as of 2026-10-16, read from the ✅ / ❌ marks in [CONTRIBUTING.md](../../CONTRIBUTING.md): **30 pass, 33 open**. Open checkpoints are either measured failures (negative results, kept as reports), blocked on human annotation, or waiting on an earlier checkpoint.

| checkpoint | name | status | evidence |
|---|---|---|---|
| C0.1 | CI | ✅ | see the phase checkpoint section |
| C0.2 | Tokenizer correctness | ✅ | see the phase checkpoint section |
| C0.3 | Reproducibility | ✅ | see the phase checkpoint section |
| C0.4 | CPU fallback | ✅ | see the phase checkpoint section |
| C1.1 | ADR-0002 merged | ❌ | ADR-0002 review by a second contributor: needs human annotation / review; cannot be produced by a script |
| C1.2 | Operator safety | ✅ | see the phase checkpoint section |
| C1.3 | Version space | ❌ | 0 of 6 categories non-empty on the GPU judge labels; every collapse named ([p1_version_spaces.md](../../reports/p1_version_spaces.md)) |
| C1.4 | Savings exist (non-blocking, [ADR-0004](docs/decisions/ADR-0004-plan-revision.md)) | ❌ | non-blocking (ADR-0004): 1.7% median best-single-operator savings, recorded negative result |
| C1.5 | Tokenizer dependence | ✅ | see the phase checkpoint section |
| C2.1 | Dataset size | ✅ | see the phase checkpoint section |
| C2.2 | Leakage | ✅ | see the phase checkpoint section |
| C2.3 | Human–human agreement | ❌ | human-human κ: needs human annotation / review; cannot be produced by a script |
| C2.4 | Judge–human agreement | ❌ | judge-human κ: needs human annotation / review; cannot be produced by a script |
| C2.5 | Judge as degradation detector | ❌ | judge vs human labels: needs human annotation / review; cannot be produced by a script |
| C2.6 | Position bias | ❌ | raw A/B disagreement 40.9% (≤ 10%); labels use order-averaged `pairwise_avg` ([p2_judge_bias.md](../../reports/p2_judge_bias.md)) |
| C2.7 | **Go / no-go: oracle savings** | ✅ | see the phase checkpoint section |
| C2.8 | Class balance | ✅ | see the phase checkpoint section |
| C2.9 | Reproducibility | ❌ | respond / judge / label are not DVC stages; `collect` deliberately stale since 2.1.3 ([p2_dvc_repro.md](../../reports/p2_dvc_repro.md)) |
| C3.1 | Metric implementations | ✅ | see the phase checkpoint section |
| C3.2 | Leakage demonstration | ✅ | see the phase checkpoint section |
| C3.3 | Quality gate (best of Tree / LR / KNN) | ✅ | tree (depth 4, min leaf 10) AUC 0.758, specificity 0.405 at sensitivity 0.95 (≥ 0.75, ≥ 0.40); the top-AUC KNN k = 51 (0.763) misses specificity (0.377) ([p3_gate_checkpoint.md](../../reports/p3_gate_checkpoint.md)) |
| C3.4 | Span tagger (best of Tree / LR) | ❌ | best 3-tag token macro-F1 0.440 (≥ 0.65) ([p3_span_tagger.md](../../reports/p3_span_tagger.md)) |
| C3.5 | Savings predictor (KNN regression) | ❌ | KNN regression MAE 16.7 pp, R² 0.290 (≤ 10 pp, ≥ 0.3) ([p3_knn_regression.md](../../reports/p3_knn_regression.md)) |
| C3.6 | Overfitting control | ✅ | see the phase checkpoint section |
| C3.7 | Figures | ✅ | tree, LR and KNN (k = 1, 5, 25) boundaries, learning curves and the bias-variance decomposition ([p3_bias_variance.md](../../reports/p3_bias_variance.md)) |
| C4.1 | Backprop correctness | ✅ | see the phase checkpoint section |
| C4.2 | SVM dual correctness | ✅ | see the phase checkpoint section |
| C4.3 | Studies complete | ✅ | see the phase checkpoint section |
| C4.4 | Better quality gate | ✅ | documented reason: tuned MLP 0.775 and NumPy MLP 0.774 are below C3.3 best + 0.03 (0.788); the classical tuned LightGBM (0.783) stays champion ([p10_hpo.md](../../reports/p10_hpo.md), [p4_quality_gate.md](../../reports/p4_quality_gate.md)) |
| C4.5 | Worst category | ❌ | every Phase 3/4 gate has a category below 0.70: embedding MLP `explain` 0.673, best KNN `explain` 0.698 ([p4_quality_gate.md](../../reports/p4_quality_gate.md), [p3_knn_grid.md](../../reports/p3_knn_grid.md)) |
| C5.1 | Champion quality gate | ❌ | best gate LightGBM 0.778 grouped AUC, tuned 0.783 (≥ 0.85) ([p5_ensembles.md](../../reports/p5_ensembles.md), [p10_hpo.md](../../reports/p10_hpo.md)) |
| C5.2 | RF OOB vs CV | ❌ | OOB 0.08 above grouped CV (≤ 0.02): bootstrap rows share source prompts ([p5_ensembles.md](../../reports/p5_ensembles.md)) |
| C5.3 | Naive Bayes filler detector | ❌ | NB precision 0.594 at recall 0.70 (≥ 0.85) ([p5_naive_bayes.md](../../reports/p5_naive_bayes.md)) |
| C5.4 | Bayes optimal vs MAP | ❌ | Bayes optimal 0.8 pp below MAP on validation over the 708,589 Phase 1 conjunctions; both below majority ([p5_bayes_optimal.md](../../reports/p5_bayes_optimal.md)) |
| C5.5 | GMM agreement | ✅ | see the phase checkpoint section |
| C5.6 | First non-generative compressor | ❌ | pending: HMM / fast tagger + gate on 500 val prompts with the judge |
| C6.1 | Clustering quality | ❌ | best K-means silhouette 0.066 at k = 9 (≥ 0.10) ([p6_results.md](../../reports/p6_results.md)) |
| C6.2 | K-means ⇔ EM | ✅ | see the phase checkpoint section |
| C6.3 | Actionable finding | ❌ | riskiest cluster 1.195x the average degradation rate (≥ 1.5x) ([p6_cluster_risk.md](../../reports/p6_cluster_risk.md)) |
| C7.1 | Environment | ✅ | see the phase checkpoint section |
| C7.2 | Memory plan | ❌ | partial: several VRAM rows still estimates; needs RTX 4090 worker measurements ([C7 details](C7/README.md)) |
| C7.4 | Queue | ❌ | partial: queue validates and dry-runs; revised 32B estimate unverified on the RTX 4090 workers ([follow-up](C7/README.md#follow-up)) |
| C8.1 | TextCNN parameters | ✅ | see the phase checkpoint section |
| C8.2 | TextCNN tagging | ❌ | TextCNN 0.679 vs best classical 0.658 on the binary task (+0.021, McNemar p < 1e-18; target +0.05) ([p8_textcnn.md](../../reports/p8_textcnn.md)) |
| C8.3 | **Rewriter main gate (val split)** | ❌ | SFT rungs at 7.6% (0.5B, 1.5B) and 2.1% (3B) median savings, targets capped near 10%; on `sft_v2` targets 1.5B LoRA saves 35.0% at 57.0% preserved and 3B LoRA 12.5% at 71.4% (bar 25% / 92%); 7B QLoRA rung in training ([p8_rewriter_val.md](../../reports/p8_rewriter_val.md)) |
| C8.4 | vs zero-shot large open LLM | ❌ | depends on a rewriter that passes C8.3 |
| C8.5 | vs LLMLingua-2 | ❌ | depends on a rewriter that passes C8.3 |
| C9.1 | Tabular convergence | ❌ | most settled learner max |dQ| 0.164 (< 1e-3); aliased states make the reward stochastic ([p9_qlearning.md](../../reports/p9_qlearning.md)) |
| C9.2 | Tabular policy value | ❌ | pending: row 9.2.4 judge verification of the greedy policy |
| C10.1 | Method comparison | ✅ | 4 GBM methods + 2 MLP methods, 60 trials x 5 seeds ([p10_hpo.md](../../reports/p10_hpo.md)) |
| C10.2 | Gate improvement | ✅ | +0.0095 AUC, documented plateau ([p10_hpo.md](../../reports/p10_hpo.md)) |
| C10.3 | Rewriter improvement | ❌ | depends on a rewriter that passes C8.3 |
| C10.4 | No test leakage | ✅ | test-set isolation audit green (`python -m terser.hpo.audit --scan src/terser`) |
| C11.1 | End-to-end orchestration | ❌ | not run end to end on a fresh clone |
| C11.2 | Fast-path latency | ✅ | see the phase checkpoint section |
| C11.3 | Quality-path latency | ❌ | pending: quality path needs a rewriter that passes C8.3 |
| C11.4 | Throughput | ❌ | pending: quality path needs a rewriter that passes C8.3 |
| C11.5 | Fallback correctness | ✅ | see the phase checkpoint section |
| C11.6 | Drift detection | ✅ | see the phase checkpoint section |
| C11.7 | CT guard | ✅ | see the phase checkpoint section |
| C9.3 | No reward hacking | ❌ | RL stage not started (depends on C8.3) |
| C9.4 | No degenerate outputs | ❌ | RL stage not started (depends on C8.3) |
| C9.5 | RL improves on SFT | ❌ | RL stage not started (depends on C8.3) |

## Blocked on people

C1.1, C2.3, C2.4 and C2.5 need a second reviewer or human labels. No script output stands in for them.

## Critical path

C8.3 (rewriter main gate) blocks C8.4, C8.5, C9.3-C9.5, C10.3, C11.3 and C11.4. The SFT targets capped savings near 10%. The augmented `llm_max` / `llm_min` targets (`configs/experiment/p8_target_aug.yaml`, train/val prompts only) are judged and built into `data/processed/sft_v2` (median target savings 21.4% train, 17.4% val); the rewriter is retrained on it next.
