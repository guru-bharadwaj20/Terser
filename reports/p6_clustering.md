# Phase 6: unsupervised structure of prompts

Rows 6.2–6.8. Code: `src/terser/models/clustering/`. This file holds the method comparison
(section 1) only; the measured results are in [p6_results.md](p6_results.md) (rows 6.2–6.8, C6.1,
C6.2) and [p6_cluster_risk.md](p6_cluster_risk.md) (row 6.9, C6.3).

## 1. Hierarchical vs non-hierarchical clustering (row 6.4)

| | Non-hierarchical (K-means, bisecting K-means as a flat result) | Hierarchical (agglomerative, divisive) |
| :--- | :--- | :--- |
| Output | One flat partition into exactly `k` clusters | A tree (dendrogram) of nested partitions; any `k` is a cut of the tree |
| Must choose `k` first? | Yes, then compare k by elbow / silhouette (row 6.2) | No; `k` (or a height) is chosen after building the tree |
| Objective | Explicit: minimise within-cluster SSE (K-means is EM on an equal-σ spherical GMM as σ → 0, see `docs/derivations/kmeans_em.md`) | Agglomerative: none global; a greedy merge rule (linkage). Divisive: greedy top-down splits (here 2-means bisection) |
| Assumptions | Clusters are compact, roughly spherical, similar in radius; Euclidean geometry; the mean is meaningful | Only a distance and a linkage. Ward keeps the SSE (spherical) bias; average and complete linkage allow other shapes; single linkage chains |
| Determinism | Depends on initialisation (k-means++ and restarts reduce this); local optimum | Agglomerative is deterministic given ties; divisive inherits the 2-means initialisation |
| Reversibility | Points move between clusters every iteration | Merges (or splits) are never undone, so an early mistake stays in the tree |
| Time | O(n · k · d · iterations); scales to all train prompts | Agglomerative O(n²) memory and O(n² log n) time with scipy, so row 6.5 uses a 2,000-prompt sample. Divisive to singletons: O(n · depth) 2-means runs |
| Bisecting K-means | Top-down, splits the highest-SSE cluster; flat result but its split history is a tree. Typically more balanced sizes than K-means | — |
| What Terser uses it for | Intent families for stratified training / evaluation and per-cluster risk (rows 6.8, 6.9) | Checking whether intent families nest (e.g. code requests → debug vs write) and whether the flat clusters are stable cuts of a tree (cophenetic correlation, row 6.6) |

**Choice for Terser.** The downstream use is a flat label per prompt (eval slices in 12.3,
stratification in 8 and 11), so K-means / bisecting K-means on all train prompts is primary. The
hierarchical trees are a diagnostic on a sample: if Ward's tree cut at the chosen `k` agrees with
K-means and the cophenetic correlation is high, the flat clusters are not an artefact of
initialisation.
