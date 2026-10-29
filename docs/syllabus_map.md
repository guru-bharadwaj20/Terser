# Syllabus Map (row 13.2)

This page maps every syllabus requirement (Units 1–4) to the plan rows, the code, the derivation,
the tests and the figure or report that show it. Paths are relative to the repository root.
Every requirement below is covered.
A figure or report marked *(planned)* is produced when its row runs on labelled data. The code
and tests for that row already exist.

## Unit 1: Introduction, concept learning, evaluation, trees, LR, KNN

| Requirement | Phase / rows | Code | Derivation | Tests | Figure / report |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Introduction to ML, ML models | 1 / 1.1.1 | — | [ADR-0002](decisions/ADR-0002-problem-framing.md) | — | — |
| Concept learning: hypotheses, version space | 1 / 1.3.2–1.3.7 | [`concept/hypothesis.py`](../src/terser/concept/hypothesis.py), [`find_s.py`](../src/terser/concept/find_s.py), [`candidate_elimination.py`](../src/terser/concept/candidate_elimination.py), [`version_space.py`](../src/terser/concept/version_space.py), [`phase1/version_spaces.py`](../src/terser/phase1/version_spaces.py) | [concept_learning.md](derivations/concept_learning.md) | `tests/concept/` | [p1_version_spaces.md](../reports/p1_version_spaces.md) |
| Inductive bias | 1, 3 / 1.3.8, 3.3.6 | as above + [`scratch_id3.py`](../src/terser/models/classical/scratch_id3.py) | [inductive_bias.md](derivations/inductive_bias.md) | `tests/phase1/test_inductive_bias_numbers.py` | `reports/p3_tree_bias.md` |
| Accuracy, precision, recall, sensitivity, specificity | 3 / 3.2.1 | [`eval/metrics.py`](../src/terser/eval/metrics.py) | — | `tests/test_eval_metrics.py` | — |
| ROC, AUC | 3 / 3.2.2–3.2.3 | [`eval/curves.py`](../src/terser/eval/curves.py), [`eval/threshold.py`](../src/terser/eval/threshold.py) | — | `tests/test_eval_curves.py`, `tests/test_eval_threshold.py` | — |
| Bias–variance decomposition and trade-off | 3 / 3.6.1 | [`eval/bias_variance.py`](../src/terser/eval/bias_variance.py) | — | `tests/test_eval_studies.py` | `reports/figures/p3_bias_variance.png` |
| Cross-validation strategies | 3 / 3.2.4–3.2.7 | [`eval/cv.py`](../src/terser/eval/cv.py), [`data/splits.py`](../src/terser/data/splits.py) | — | `tests/test_eval_cv.py` | `reports/p3_cv_leakage.md` |
| Decision trees: ID3 | 3 / 3.3.1 | [`models/classical/scratch_id3.py`](../src/terser/models/classical/scratch_id3.py) | [decision_trees.md](derivations/decision_trees.md) | `tests/test_scratch_id3.py` | — |
| Decision boundary for trees (x–y axis) | 3 / 3.3.5 | [`eval/boundaries.py`](../src/terser/eval/boundaries.py) | — | `tests/phase3/test_studies_runner.py` | `reports/figures/p3_tree_boundary.png` |
| Hypothesis search and inductive bias in trees | 3 / 3.3.6 | `scratch_id3.py` | [inductive_bias.md](derivations/inductive_bias.md) | — | `reports/p3_tree_bias.md` |
| Overfitting and solutions (pre-pruning, reduced-error, rule post-pruning, CCP) | 3 / 3.3.3–3.3.4 | `scratch_id3.py` | [decision_trees.md](derivations/decision_trees.md) | `tests/test_scratch_id3.py` | `reports/figures/p3_tree_overfit.png` |
| Continuous values | 3 / 3.1.4, 3.3.2 | [`features/continuous.py`](../src/terser/features/continuous.py), C4.5 splitter in `scratch_id3.py` | [decision_trees.md](derivations/decision_trees.md) | `tests/test_features_continuous.py`, `tests/test_scratch_id3.py` | — |
| Logistic regression | 3 / 3.4, 3.5.3 | [`logistic.py`](../src/terser/models/classical/logistic.py), [`scratch_penalized_lr.py`](../src/terser/models/classical/scratch_penalized_lr.py), [`eval/calibration.py`](../src/terser/eval/calibration.py) | [logistic_regression.md](derivations/logistic_regression.md) | `tests/test_logistic_gd.py`, `tests/test_penalized_lr_calibration.py` | `reports/figures/p3_lr_calibration.png` |
| KNN classification and regression | 3 / 3.5.1–3.5.2 | [`scratch_knn.py`](../src/terser/models/classical/scratch_knn.py), [`skip_rule.py`](../src/terser/models/classical/skip_rule.py) | — | `tests/test_scratch_knn.py` | — |
| Decision boundary for KNN | 3 / 3.5.3 | `eval/boundaries.py` | — | `tests/phase3/test_studies_runner.py` | `reports/figures/p3_knn_boundaries.png` |

## Unit 2: Neural networks and SVMs

| Requirement | Phase / rows | Code | Derivation | Tests | Figure / report |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Perceptron | 4 / 4.1.1 | [`scratch_perceptron.py`](../src/terser/models/neural/scratch_perceptron.py) | [backprop.md](derivations/backprop.md) | `tests/test_scratch_classifiers.py` | — |
| Multi-layer networks and backpropagation | 4 / 4.1.2–4.1.5, 4.1.7 | [`numpy_mlp.py`](../src/terser/models/neural/numpy_mlp.py), [`mlp.py`](../src/terser/models/neural/mlp.py), [`pair_embedding.py`](../src/terser/features/pair_embedding.py) | [backprop.md](derivations/backprop.md) | `tests/test_numpy_mlp.py`, `tests/test_mlp_params.py`, `tests/test_backprop_doc.py` | — |
| Activations: step, sigmoid, tanh, ReLU | 4 / 4.2.1 | [`phase4/nn_studies.py`](../src/terser/phase4/nn_studies.py) | [backprop.md](derivations/backprop.md) | `tests/phase4/test_phase4_studies.py` | `reports/p4_activations.md` |
| Optimizers: GD, SGD, Momentum, Adagrad, Adam | 4 / 4.3.1 | [`scratch_optim.py`](../src/terser/models/neural/scratch_optim.py), `nn_studies.py` | [backprop.md](derivations/backprop.md) | `tests/phase4/test_phase4_studies.py` | `reports/figures/p4_optimizers.png` |
| SVM margin, primal, Lagrangian dual, dual solution | 4 / 4.4.1, 4.4.2, 4.4.4 | [`scratch_svm.py`](../src/terser/models/classical/scratch_svm.py) | [svm.md](derivations/svm.md) | `tests/test_scratch_svm.py`, `tests/test_svm_doc.py` | — |
| Hard and soft margin | 4 / 4.4.5–4.4.6 | `scratch_svm.py`, [`phase4/svm_studies.py`](../src/terser/phase4/svm_studies.py) | [svm.md](derivations/svm.md) | `tests/phase4/test_phase4_studies.py` | — |
| Kernels: linear (derivation), polynomial | 4 / 4.4.3, 4.4.7 | `scratch_svm.py`, `svm_studies.py` | [svm.md](derivations/svm.md) | `tests/test_scratch_svm.py` | `reports/figures/p4_svm_surfaces.png` |

## Unit 3: Ensembles, Bayesian learning, EM, HMM

| Requirement | Phase / rows | Code | Derivation | Tests | Figure / report |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Combining weak learners (AdaBoost) | 5 / 5.1.1 | [`scratch_adaboost.py`](../src/terser/models/ensembles/scratch_adaboost.py) | — | `tests/test_scratch_classifiers.py` | — |
| Gradient boosting | 5 / 5.1.5–5.1.6 | [`phase5/ensembles.py`](../src/terser/phase5/ensembles.py) | — | `tests/phase5/test_phase5_studies.py` | `reports/figures/p5_shap.png` |
| Random forest | 5 / 5.1.3 | `phase5/ensembles.py` | — | `tests/phase5/test_phase5_studies.py` | — |
| Bayes theorem | 5 / 5.2.1 | [`phase5/bayes_counts.py`](../src/terser/phase5/bayes_counts.py) | [bayes.md](derivations/bayes.md) | `tests/test_bayes_doc.py` | — |
| Maximum likelihood | 5 / 5.2.3, 5.3.2 | [`scratch_naive_bayes.py`](../src/terser/models/classical/scratch_naive_bayes.py), [`hmm.py`](../src/terser/models/classical/hmm.py) | [bayes.md](derivations/bayes.md) | `tests/test_scratch_classifiers.py`, `tests/test_hmm.py` | — |
| Bayes optimal classifier | 5 / 5.2.4 | [`bayes_optimal.py`](../src/terser/models/classical/bayes_optimal.py) | [bayes.md](derivations/bayes.md) | `tests/phase5/test_phase5_studies.py` | `reports/p5_bayes_optimal.md` |
| Naive Bayes | 5 / 5.2.3 | `scratch_naive_bayes.py` | [bayes.md](derivations/bayes.md) | `tests/test_scratch_classifiers.py` | — |
| EM and GMM | 5 / 5.2.5–5.2.6 | [`gmm.py`](../src/terser/models/classical/gmm.py) | [bayes.md](derivations/bayes.md) | `tests/test_gmm_em.py` | — |
| Hidden Markov models (forward, Viterbi, Baum-Welch) | 5 / 5.3 | `hmm.py`, [`phase5/hmm_tagger.py`](../src/terser/phase5/hmm_tagger.py) | [bayes.md](derivations/bayes.md) | `tests/test_hmm.py`, `tests/test_hmm_tagger.py`, `tests/phase5/test_hmm_compressor.py` | `reports/p5_hmm_compressor.md` |

## Unit 4: Clustering, RL, deep learning, HPO, MLOps

| Requirement | Phase / rows | Code | Derivation | Tests | Figure / report |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Hierarchical vs non-hierarchical clustering | 6 / 6.4 | [`clustering/hierarchy.py`](../src/terser/models/clustering/hierarchy.py) | [kmeans_em.md](derivations/kmeans_em.md) | `tests/test_clustering.py` | [p6_clustering.md](../reports/p6_clustering.md) |
| Agglomerative and divisive clustering | 6 / 6.5–6.6 | `hierarchy.py` | — | `tests/test_clustering.py` | dendrograms *(planned)* |
| K-means, bisecting K-means | 6 / 6.2–6.3 | [`scratch_kmeans.py`](../src/terser/models/clustering/scratch_kmeans.py) | [kmeans_em.md](derivations/kmeans_em.md) | `tests/test_clustering.py` | `reports/figures/p6_kmeans_k.png` |
| K-means as a special case of EM | 6 / 6.7 | [`kmeans_em.py`](../src/terser/models/clustering/kmeans_em.py) | [kmeans_em.md](derivations/kmeans_em.md) | `tests/test_clustering.py` | — |
| RL: agent, environment, states, actions, rewards, policy, value functions | 9 / 9.1.1–9.1.4 | [`rl/env.py`](../src/terser/models/rl/env.py) | [rl.md](derivations/rl.md) | `tests/test_rl.py` | — |
| Bellman equation and Q-learning | 9 / 9.1.1, 9.2 | [`scratch_qlearning.py`](../src/terser/models/rl/scratch_qlearning.py) | [rl.md](derivations/rl.md) | `tests/test_rl.py` | `reports/p9_policy.md` |
| Introduction to deep learning | 8 / 8.1.1, 8.3 | [`rewriter/train.py`](../src/terser/models/rewriter/train.py) | [deep_learning_intro.md](derivations/deep_learning_intro.md) | `tests/rewriter/test_train_tiny.py` | — |
| Convolution operation and CNNs | 8 / 8.1.2, 8.2.1 | [`conv.py`](../src/terser/models/neural/conv.py), [`textcnn.py`](../src/terser/models/neural/textcnn.py) | [deep_learning_intro.md](derivations/deep_learning_intro.md) | `tests/test_conv_textcnn.py` | — |
| CNN parameter calculation, max and avg pooling | 8 / 8.2.2–8.2.3 | `textcnn.py` | [deep_learning_intro.md](derivations/deep_learning_intro.md) | `tests/test_conv_textcnn.py` | — |
| Hyperparameter tuning: grid, random, Bayesian | 3, 10 / 3.6.4, 10.2, 10.5 | [`hpo/search.py`](../src/terser/hpo/search.py), [`hpo/audit.py`](../src/terser/hpo/audit.py) | — | `tests/hpo/` | `reports/figures/p10_hpo_methods.png` |
| MLOps overview, lifecycle, production challenges | 0, 11 / 0.3.4, 11.1.1 | — | [ADR-0001](decisions/ADR-0001-ml-lifecycle.md), [ADR-0010](decisions/ADR-0010-mlops.md) | — | — |
| Data and feature pipelines | 2, 11 / 2.1.7, 11.1 | [`dvc.yaml`](../dvc.yaml), [`features/shared.py`](../src/terser/features/shared.py), [`data/validation.py`](../src/terser/data/validation.py) | — | `tests/features/test_parity.py`, `tests/test_data_validation.py` | [p2_dvc_repro.md](../reports/p2_dvc_repro.md), [p11_dvc_repro.md](../reports/p11_dvc_repro.md), [data_pipeline.md](data_pipeline.md) |
| Experiment tracking and model versioning | 0, 11 / 0.2.4, 0.2.5, 11.2 | [`tracking/runs.py`](../src/terser/tracking/runs.py), [`tracking/lineage.py`](../src/terser/tracking/lineage.py) | — | `tests/test_tracking.py`, `tests/pipelines/test_promotion_lineage.py` | — |
| Model deployment and serving | 11 / 11.3 | [`serving/api.py`](../src/terser/serving/api.py), [`serving/service.py`](../src/terser/serving/service.py), [`compose.yaml`](../compose.yaml), [`docker/`](../docker) | [serving/api.md](serving/api.md) | `tests/serving/` | [p11_cache_latency.md](../reports/p11_cache_latency.md) |
| CI/CD/CT for ML systems | 0, 11 / 0.2.9, 11.4 | [`.github/workflows/ci.yml`](../.github/workflows/ci.yml), [`pipelines/promotion.py`](../src/terser/pipelines/promotion.py) | — | `tests/pipelines/` | — |
| Inference systems | 11 / 11.3.2–11.3.6 | [`onnx_export.py`](../src/terser/serving/onnx_export.py), [`cache.py`](../src/terser/serving/cache.py), [`accounting.py`](../src/terser/serving/accounting.py), [`paths.py`](../src/terser/serving/paths.py) | — | `tests/serving/` | — |
| Monitoring, drift and observability | 11 / 11.5 | [`serving/metrics.py`](../src/terser/serving/metrics.py), [`monitoring/tokenizer_drift.py`](../src/terser/monitoring/tokenizer_drift.py), [`request_log.py`](../src/terser/serving/request_log.py) | [serving/logging.md](serving/logging.md) | `tests/monitoring/` | — |
| ML orchestration | 11 / 11.6 | [`flows/`](../src/terser/flows), [`gpu_queue/`](../src/terser/gpu_queue) | [orchestration.md](orchestration.md) | `tests/test_flows.py`, `tests/test_gpu_queue.py` | [p11_flows.md](../reports/p11_flows.md) |
