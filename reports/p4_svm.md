# Phase 4 SVM studies (rows 4.4.5-4.4.8)

From-scratch `DualSVM` (cvxopt QP, row 4.4.4) on the row 3.1.3 pair table, first grouped fold as
validation, 2,000 standardised training pairs (QP memory is O(n²)). Tables from
`python -m terser.phase4.svm_studies --table data/processed/p4_pairs.parquet` (`p4_svm_soft_margin.csv`,
`p4_svm_kernels.csv`), figures from `python -m terser.phase4.svm_figures`.

> **Note (post-audit):** re-run on 2026-10-26 on a seeded random sample of 2,000 training pairs
> (the first run used the first 2,000 indices, i.e. the alphabetically first prompts).

## 4.4.5 Hard margin

![hard margin](figures/p4_svm_hard_margin.png)

DROP vs KEEP span tokens in the (`self_information`, `stopword_ratio`) plane. The full token set
is not linearly separable, so the subset keeps the 60 tokens per class that a C = 1 soft SVM
scores most confidently. The hard-margin fit separates all 120 with **3 support vectors**,
margin 2/‖w‖ = 1.737.

## 4.4.6 Soft margin

![soft margin](figures/p4_svm_soft_margin.png)

| C | margin width | # SVs | val AUC | val F1 |
|---|---|---|---|---|
| 0.01 | 2.465 | 1478 | 0.758 | 0.658 |
| 0.1 | 1.590 | 1417 | 0.757 | 0.666 |
| 1 | 1.227 | 1403 | 0.756 | 0.669 |
| 10 | 1.117 | 1398 | 0.756 | 0.668 |
| 100 | 1.117 | 1398 | 0.756 | 0.668 |

Raising C shrinks the margin (2.47 → 1.12) and slightly reduces the support-vector count, then
both flatten above C = 10, once few slack variables remain to trade off. About 70-74% of the 2,000
points stay support vectors at every C, which shows how much the classes overlap. Val AUC is
flat (0.756-0.758), so the gate is limited by the features, not by C.

## 4.4.7 Polynomial kernel

| kernel | degree | coef0 | C | val AUC | fit (s) | # SVs |
|---|---|---|---|---|---|---|
| linear | - | - | 0.1 | 0.757 | 16.8 | 1417 |
| linear | - | - | 1 | 0.756 | 18.3 | 1403 |
| linear | - | - | 10 | 0.756 | 19.8 | 1398 |
| poly | 2 | 0 | 0.1 | 0.730 | 14.2 | 1686 |
| poly | 2 | 0 | 1 | 0.734 | 13.9 | 1462 |
| poly | 2 | 0 | 10 | 0.710 | 16.3 | 1317 |
| poly | 2 | 1 | 0.1 | **0.775** | 14.4 | 1441 |
| poly | 2 | 1 | 1 | 0.756 | 14.4 | 1315 |
| poly | 2 | 1 | 10 | 0.716 | 15.6 | 1245 |
| poly | 3 | 0 | 0.1 | 0.749 | 15.1 | 1642 |
| poly | 3 | 0 | 1 | 0.736 | 14.2 | 1428 |
| poly | 3 | 0 | 10 | 0.705 | 14.9 | 1283 |
| poly | 3 | 1 | 0.1 | 0.761 | 15.2 | 1379 |
| poly | 3 | 1 | 1 | 0.720 | 16.0 | 1296 |
| poly | 3 | 1 | 10 | 0.691 | 15.9 | 1193 |

The best kernel is degree 2 with coef0 = 1 and C = 0.1 (AUC 0.775, +0.018 over linear).
coef0 = 1 keeps the linear terms alongside the pairwise interactions; coef0 = 0 has only pure
degree-d terms and is worse at small C (0.730 vs 0.775 at degree 2, 0.749 vs 0.761 at degree 3). Larger C overfits the polynomial feature space quickly
(degree 3, coef0 = 1, C = 10: 0.691). Fit time is about the same for every kernel because the QP size is
fixed by n.

## 4.4.8 Decision surfaces

![surfaces](figures/p4_svm_surfaces.png)

Linear vs degree-3 polynomial (coef0 = 1, C = 1) on the Phase 3 plane (`max_si_removed` x
`tokens_saved`, as in rows 3.3.5 / 3.5.3), 600 standardised pairs; red is "degraded", blue
"preserved". The linear boundary is almost flat: it predicts "degraded" once a pair saves more
than about 0.6-1.5 standard deviations of tokens, and `max_si_removed` barely moves it. The
polynomial boundary sits near tokens_saved = 0 while `max_si_removed` is below about 1.5, then
bends sharply upward, so pairs that drop a very high-information token are called "preserved"
unless they also save many tokens. It also encloses a "preserved" island at high tokens_saved and
low `max_si_removed`. Both shapes are fitted to a handful of sparse outliers (the blue pairs at
`max_si_removed` = 10.7 and at tokens_saved = 6-7), not to the dense cloud, where the two
boundaries agree. This is the overfitting the 4.4.7 table shows for degree 3 at C = 1 (AUC 0.720
vs 0.756 linear). The bend starts near `max_si_removed` = 1.5-2, close to the first split of the
Phase 3 tree (`max_si_removed > 1.90`), but it goes the way the outliers push it, so it is not
evidence about that split.
