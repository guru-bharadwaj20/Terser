# Bayesian learning: Bayes' theorem, naive Bayes, MAP and the Bayes optimal classifier

Row **5.2.1** of `CONTRIBUTING.md`, with the theory used by rows 5.2.3 (naive Bayes filler
detector) and 5.2.4 (Bayes optimal classifier). The worked numbers in §2 are recomputed from the
committed pilot corpus by `tests/test_bayes_doc.py`.

## 1. Bayes' theorem

For events $F$ ("this word occurrence is filler") and $E$ (the evidence: the phrase $w$ in context
$c$), the product rule gives $P(F, E) = P(F \mid E) P(E) = P(E \mid F) P(F)$, so

$$
P(F \mid w, c) = \frac{P(w, c \mid F)\, P(F)}{P(w, c)},
\qquad
P(w, c) = P(w, c \mid F) P(F) + P(w, c \mid \neg F) P(\neg F).
$$

* $P(F)$ is the **prior**: how often any word is filler before looking at it.
* $P(w, c \mid F)$ is the **likelihood**: among filler words, how often they are $w$ in context $c$.
* $P(w, c)$ is the **evidence**, a normaliser.
* $P(F \mid w, c)$ is the **posterior** the filler detector wants.

When all three factors are estimated from the same counts, the theorem reduces to the direct
relative frequency: with $N$ word tokens, $N_F$ filler tokens, $N_{wc}$ occurrences of $w$ in
context $c$ and $N_{wcF}$ of those that are filler,

$$
\frac{(N_{wcF}/N_F)\,(N_F/N)}{N_{wc}/N} = \frac{N_{wcF}}{N_{wc}}.
$$

Bayes' theorem earns its keep when the factors come from **different** sources, e.g. a likelihood
estimated on a labelled corpus combined with a prior for a new population where filler is rarer
(the posterior then shrinks in proportion), or when the evidence $E$ factorises into many features
(§3).

## 2. Worked example from corpus counts

**Corpus.** The 500 pilot prompts of row 1.4.1 (`data/pilot/prompts.jsonl`), 18,519 word tokens.

**Labels.** Silver labels: a word occurrence counts as filler when the deterministic `F` operator
(`terser.operators.filler`) deletes it, found by aligning the word sequences before and after
`F`. These are **not** the Phase 3 alignment labels (row 3.1.1), which do not exist yet; when they
do, `python -m terser.phase5.bayes_counts` recomputes the table on them.

**Context.** $c = $ `initial` (prompt- or sentence-initial, or right after a comma) or `medial`.

Counts (from `python -m terser.phase5.bayes_counts --prompts data/pilot/prompts.jsonl`):
$N = 18{,}519$, $N_F = 220$, so the prior is $P(F) = 220 / 18{,}519 = 0.01188$.

| phrase $w$ | context $c$ | $N_{wc}$ | $N_{wcF}$ | $P(w,c \mid F)$ | $P(w,c)$ | $P(F \mid w,c)$ |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| so | initial | 78 | 43 | 0.19545 | 0.00421 | **0.551** |
| so | medial | 135 | 25 | 0.11364 | 0.00729 | **0.185** |
| um | initial | 10 | 10 | 0.04545 | 0.00054 | 1.000 |
| basically | medial | 7 | 7 | 0.03182 | 0.00038 | 1.000 |
| honestly | medial | 9 | 8 | 0.03636 | 0.00049 | 0.889 |
| like | initial | 26 | 0 | 0 | 0.00140 | 0.000 |
| like | medial | 59 | 0 | 0 | 0.00319 | 0.000 |
| well | medial | 10 | 0 | 0 | 0.00054 | 0.000 |

**Worked case: "so" at the start of a sentence.**

$$
P(F \mid \text{so}, \text{initial})
= \frac{P(\text{so}, \text{initial} \mid F)\, P(F)}{P(\text{so}, \text{initial})}
= \frac{(43/220)\,(220/18{,}519)}{78/18{,}519}
= \frac{0.19545 \times 0.01188}{0.00421}
= \frac{43}{78} = 0.551.
$$

Context matters: the same word in medial position has posterior $25/135 = 0.185$, three times
lower, because medial "so" is mostly the intensifier or conjunction ("so many rows", "so that").
The prior alone (0.012) would badly under-call sentence-initial "so", and the phrase alone
($68/213 = 0.319$ pooled over contexts) would over-call medial "so".

**What the counts say about the silver labels.** "like" is never deleted (0 / 85): `F` only
removes comma-bracketed ", like ," and the pilot prompts contain none, so every "like" here is the
verb or the comparison. "um" and "basically" are always deleted. Posteriors of exactly 0 or 1 come
from small counts and a deterministic labeller; the naive Bayes detector of §3 smooths them.

## 3. Naive Bayes, MLE, and why Laplace smoothing is MAP

A prompt span is a bag of n-grams $x = (x_1, \dots, x_{|V|})$ (counts over vocabulary $V$). The
**naive** assumption is that n-grams are conditionally independent given the class $y$, which turns
the intractable joint likelihood into a product (multinomial event model):

$$
P(y \mid x) \propto P(y) \prod_{j} \theta_{y j}^{x_j},
\qquad \sum_j \theta_{y j} = 1.
$$

**MLE.** Maximising $\sum_i \log P(x_i, y_i)$ with a Lagrange multiplier for each simplex
constraint gives $\hat P(y) = N_y / N$ and $\hat\theta_{yj} = N_{yj} / \sum_{j'} N_{yj'}$, where
$N_{yj}$ is the total count of n-gram $j$ in class $y$.

**The zero-count problem.** If n-gram $j$ never occurs in class $y$, $\hat\theta_{yj} = 0$ and any
document containing $j$ gets $P(y \mid x) = 0$ regardless of all other evidence. With 1–3-grams
most n-grams are rare, so this happens constantly.

**Laplace smoothing is MAP.** Put a symmetric Dirichlet prior on each class's parameter vector,
$\theta_y \sim \operatorname{Dir}(\beta, \dots, \beta)$. The posterior is
$\operatorname{Dir}(N_{y1} + \beta, \dots)$, whose **mode** (the MAP estimate) is

$$
\hat\theta^{\text{MAP}}_{yj} = \frac{N_{yj} + \beta - 1}{\sum_{j'} N_{yj'} + |V|(\beta - 1)}.
$$

With $\beta = 1 + \alpha$ this is exactly additive smoothing,
$(N_{yj} + \alpha) / (\sum_{j'} N_{yj'} + \alpha |V|)$, and $\alpha = 1$ is Laplace's rule
($\beta = 2$). $\alpha = 0$ ($\beta = 1$, a uniform prior) recovers the MLE. So smoothing is not an
ad-hoc fix: it is the most probable parameter under a prior that says every n-gram is possible in
every class. `terser.models.classical.scratch_naive_bayes` implements this and matches
scikit-learn's `MultinomialNB` (`tests/test_scratch_classifiers.py`).

## 4. MAP hypothesis vs Bayes optimal classifier

Let $H$ be a finite hypothesis space (row 5.2.4 uses the Phase 1 version-space members). With
a prior $P(h)$ and a likelihood $P(D \mid h)$, the posterior is
$P(h \mid D) = P(D \mid h) P(h) / \sum_{h'} P(D \mid h') P(h')$. If each label in $D$ is flipped
with probability $\epsilon$, a hypothesis making $m_h$ mistakes on $n$ examples has
$P(D \mid h) = (1 - \epsilon)^{n - m_h} \epsilon^{m_h}$.

* **MAP hypothesis:** $h_{\text{MAP}} = \arg\max_h P(h \mid D)$; predict $h_{\text{MAP}}(x)$.
* **Bayes optimal classifier:**
  $\hat y(x) = \arg\max_{v} \sum_{h \in H} P(v \mid h, x)\, P(h \mid D)$, a posterior-weighted vote.

No classifier using the same $H$, prior and data has lower expected error than the Bayes optimal
one, because it predicts the value with the highest posterior probability given everything known.
The two disagree when the MAP hypothesis is outvoted: with posteriors $0.4, 0.3, 0.3$ and
predictions $+, -, -$, MAP says $+$ but the vote gives $P(+) = 0.4 < P(-) = 0.6$. This is the
textbook case reproduced in `tests/test_scratch_classifiers.py::test_bayes_optimal_textbook_example`;
`terser.models.classical.bayes_optimal` implements both rules for checkpoint C5.4.
