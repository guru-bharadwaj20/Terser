# K-means as a special case of EM (row 6.7)

Code: [`src/terser/models/clustering/kmeans_em.py`](../../src/terser/models/clustering/kmeans_em.py),
[`scratch_kmeans.py`](../../src/terser/models/clustering/scratch_kmeans.py) ·
Test: `tests/test_clustering.py::test_annealed_em_agrees_with_kmeans`

## 1. The two algorithms

**K-means** minimises the distortion

$$J(\mu, c) = \sum_{i=1}^{n} \lVert x_i - \mu_{c(i)} \rVert^2$$

by alternating (a) $c(i) = \arg\min_k \lVert x_i - \mu_k \rVert^2$ and
(b) $\mu_k = \frac{1}{|C_k|}\sum_{i \in C_k} x_i$.

**EM for a Gaussian mixture** $p(x) = \sum_k \pi_k \, \mathcal N(x \mid \mu_k, \Sigma_k)$
alternates an E-step (responsibilities $r_{ik} = p(z_i = k \mid x_i)$) and an M-step
(weighted maximum likelihood), and never decreases the log-likelihood (row 5.2.5).

## 2. Restrict the mixture

Fix every component to the same spherical covariance $\Sigma_k = \sigma^2 I$ with $\sigma$ known
(not learned), and equal weights $\pi_k = 1/K$. Then

$$\mathcal N(x \mid \mu_k, \sigma^2 I) = (2\pi\sigma^2)^{-d/2}
\exp\!\Big(-\frac{\lVert x - \mu_k\rVert^2}{2\sigma^2}\Big).$$

## 3. E-step

The normalising constant $(2\pi\sigma^2)^{-d/2}$ and $\pi_k$ are the same for every $k$ and cancel:

$$r_{ik} = \frac{\exp(-\lVert x_i - \mu_k\rVert^2 / 2\sigma^2)}
{\sum_j \exp(-\lVert x_i - \mu_j\rVert^2 / 2\sigma^2)}.$$

This is a softmax over $-\lVert x_i - \mu_k\rVert^2$ with temperature $2\sigma^2$.

## 4. M-step

Maximising $\sum_i \sum_k r_{ik} \log \mathcal N(x_i \mid \mu_k, \sigma^2 I)$ over $\mu_k$ only
($\sigma$ and $\pi$ are fixed): the objective is
$-\frac{1}{2\sigma^2}\sum_i r_{ik}\lVert x_i - \mu_k\rVert^2 + \text{const}$; setting the gradient
$\frac{1}{\sigma^2}\sum_i r_{ik}(x_i - \mu_k)$ to zero gives

$$\mu_k = \frac{\sum_i r_{ik}\, x_i}{\sum_i r_{ik}}.$$

## 5. The limit σ → 0

Let $k^* = \arg\min_k \lVert x_i - \mu_k\rVert^2$ (assume no tie) and
$\Delta_k = \lVert x_i - \mu_k\rVert^2 - \lVert x_i - \mu_{k^*}\rVert^2 > 0$ for $k \ne k^*$. Dividing
numerator and denominator by $\exp(-\lVert x_i - \mu_{k^*}\rVert^2/2\sigma^2)$:

$$r_{ik} = \frac{e^{-\Delta_k/2\sigma^2}}{1 + \sum_{j \ne k^*} e^{-\Delta_j/2\sigma^2}}
\;\xrightarrow{\sigma \to 0}\; \begin{cases}1 & k = k^* \\ 0 & k \ne k^*\end{cases}$$

So the E-step becomes the hard assignment (a), and with $r_{ik} \in \{0,1\}$ the M-step becomes the
cluster mean (b). EM in this limit **is** K-means. The log-likelihood also tends to
$-\frac{1}{2\sigma^2} J(\mu, c) + \text{const}$, so maximising it means minimising the K-means
distortion.

What is lost in the limit: no uncertainty (a point halfway between centroids is assigned fully to
one), no learned shape or size per cluster (K-means prefers equal-radius spherical clusters), and no
mixture weights.

## 6. Annealing, and the measured check

`AnnealedSphericalEM` runs EM to convergence at
σ ∈ {4, 2, 1, 0.5, 0.25, 0.1, 0.05, 0.01}, warm-starting each stage from the previous means. At large
σ every responsibility is near $1/K$ (soft); at the smallest σ the mean of $\max_k r_{ik}$ exceeds
0.999 (hard). `assignment_agreement` compares the hard EM labels with a K-means run started from the
EM means, after the best one-to-one relabelling (Hungarian algorithm).

On the synthetic 4-blob fixture in the test (3 seeds, 240 points, 3-d) agreement is **≥ 98%** at
σ = 0.01, which the test asserts. Checkpoint **C6.2** asks for the same ≥ 98% on the bge-small
prompt embeddings of row 6.1; that number is not yet measured because the Phase 2 train split does
not exist yet.
