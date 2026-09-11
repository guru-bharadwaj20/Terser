# Inductive bias of the Phase 1 concept learner

Row **1.3.8** of `CONTRIBUTING.md`. Companion to
[`concept_learning.md`](concept_learning.md), which defines the instance language, conjunctive
hypotheses, Find-S and Candidate-Elimination (CE). This note explains what that learner
*assumes*, what Terser concepts it therefore cannot represent, and why Phases 3–5 move to
statistical learners with different biases. The per-category results it interprets are in
`reports/p1_version_spaces.md` (row 1.3.7).

Every number in section 2 and section 4 is recomputed from `terser.concept.ATTRIBUTES` by
`tests/phase1/test_inductive_bias_numbers.py`, which parses the tables below; the
counter-examples of section 1 are replayed through `candidate_elimination` by the same test.

Notation. An instance is the 12-vector of `ATTRIBUTES`:

```
<length, has_constraints, has_payload, tone_cue, filler_density, op_F, op_H, op_D, op_I, op_L, op_C, op_S>
```

A positive example means the rewrite **preserved** output quality (see `concept_learning.md`
section 1; the quality gate of ADR-0002 uses the opposite convention). In the instances below
the prompt attributes not mentioned are `has_payload=no`, `filler_density=low`, and operator
flags not mentioned are `no`.

---

## 1. Restriction bias: what a conjunction cannot say

CE searches only the conjunctive language `H`: each attribute is fixed to one value or left
free (`?`). This is a **restriction bias** (also called language bias): concepts outside `H`
are never considered, whatever the data say. If the target concept `c` is not in `H`, some
labelled sample makes the version space empty, because a set of examples with at least one
positive is consistent with `H` iff `find_s` of its positives covers none of its negatives
(`concept_learning.md` section 6). Each subsection gives a realistic Terser concept outside `H`
and a two- or three-example set that collapses CE.

A useful pattern: `find_s` of two positives replaces every disagreeing attribute by `?`. Any
concept whose positives disagree on an attribute that also matters for a negative cannot be
conjunctive.

### 1.1 Exceptions: "drop hedging unless tone matters in creative prompts"

Concept: `op_H=yes` is safe, except when `tone_cue=yes` (hedges such as "maybe a little
playful" carry the requested register). Equivalently, preserved iff
`not (op_H=yes and tone_cue=yes)`, a negated conjunction.

| # | instance | label |
|---|---|---|
| e1 | `<short, no, no, no, low, no, yes, no, no, no, no, no>` (no tone cue, `H`) | preserved |
| e2 | `<short, no, no, yes, low, yes, no, no, no, no, no, no>` (tone cue, `F`) | preserved |
| e3 | `<short, no, no, yes, low, no, yes, no, no, no, no, no>` (tone cue, `H`) | degraded |

`find_s({e1, e2}) = <short, no, no, ?, low, ?, ?, no, no, no, no, no>`: `tone_cue` must be
free (e1, e2 disagree) and so must `op_H`. That hypothesis covers e3, so the version space is
empty. This is exactly the collapse the synthetic fixture of row 1.3.7 reproduces in the
`creative` category.

The complement ("degraded iff `op_H=yes` and `tone_cue=yes`") *is* conjunctive:
`<?, ?, ?, yes, ?, ?, yes, ?, ?, ?, ?, ?>` fits e1–e3 with flipped labels. The language is not
closed under complement, so the choice of positive class is itself part of the bias.

### 1.2 Disjunctions: "`I` is safe if the prompt is short OR states no constraints"

Concept: the imperative rewrite only hurts when a long prompt carries explicit requirements
(the rewrite reorders and drops the softening around them). Preserved iff
`length=short or has_constraints=no`.

| # | instance | label |
|---|---|---|
| d1 | `<short, yes, no, no, low, no, no, no, yes, no, no, no>` | preserved |
| d2 | `<long, no, no, no, low, no, no, no, yes, no, no, no>` | preserved |
| d3 | `<long, yes, no, no, low, no, no, no, yes, no, no, no>` | degraded |

`find_s({d1, d2}) = <?, ?, no, no, low, no, no, no, yes, no, no, no>` covers d3. A conjunction
can fix `length` to one value or leave it free; it cannot say "short, or anything with
`has_constraints=no`". The complement `(length=medium or length=long) and has_constraints=yes`
is also outside `H`, because `length` has three values.

### 1.3 Thresholds finer than the buckets: "`L` is safe below about 40 tokens"

Concept: list-ifying breaks prompts whose instructions span several clauses, which starts
around 40 gpt-4o tokens. Both 30 and 70 tokens fall in the `medium` bucket (25–80).

| # | prompt | instance | label |
|---|---|---|---|
| t1 | 30 tokens | `<medium, no, no, no, low, no, no, no, no, yes, no, no>` | preserved |
| t2 | 70 tokens | `<medium, no, no, no, low, no, no, no, no, yes, no, no>` | degraded |

The two instances are identical, so no hypothesis of any kind over this instance space can
separate them: CE sets `S = <t1>` and removes it at t2. The fault is in the attribute
discretisation, not in the conjunction itself. The same happens at `filler_density`'s single
cut at 0.15. Phase 3 keeps token counts continuous (C4.5 threshold search, row 3.3.2;
standardised features for LR and KNN, row 3.1.4).

### 1.4 Operator interactions: "`H` alone is safe, `I` alone is safe, `H` + `I` is not"

Concept: removing hedges keeps the imperative's politeness, and the imperative keeps the
hedges' softness, but both together turn an "explain this gently for a beginner" request into a
curt command and the target answers tersely. Preserved iff `not (op_H=yes and op_I=yes)`.

| # | instance | label |
|---|---|---|
| i1 | `<medium, no, no, no, low, no, yes, no, no, no, no, no>` (`H`) | preserved |
| i2 | `<medium, no, no, no, low, no, no, no, yes, no, no, no>` (`I`) | preserved |
| i3 | `<medium, no, no, no, low, no, yes, no, yes, no, no, no>` (`HI`) | degraded |

`find_s({i1, i2}) = <medium, no, no, no, low, no, ?, no, ?, no, no, no>` covers i3. The
operator stacks of row 1.3.6 (`FHC`, `FHCI`, ...) are the instances where such interactions
appear, which is why per-category collapses are expected on real labels. Count-based concepts
("at most two operators") fail the same way: the positives `F` and `H` generalise to `?` on
both flags and cover the negative `FH` when the concept is "at most one operator".

---

## 2. The unbiased learner has nothing to generalise with

### 2.1 Sizes of the spaces

Domain sizes, from `ATTRIBUTES`: `length` 3, the four other prompt attributes 2 each, and 7
operator flags of 2 each.

Instances:

- prompt values: `3 · 2 · 2 · 2 · 2 = 48`
- operator flag values: `2^7 = 128`
- `|X| = 48 · 128 = 6,144`

Syntactic conjunctive hypotheses (each attribute is a value, `?` or `∅`):

- `∏(|D_i| + 2) = 5 · 4^4 · 4^7 = 5 · 256 · 16,384 = 20,971,520`

Semantically distinct conjunctive hypotheses (every hypothesis containing `∅` denotes the same
empty concept, so count the `∅`-free ones plus one):

- `1 + ∏(|D_i| + 1) = 1 + 4 · 3^4 · 3^7 = 1 + 4 · 81 · 2,187 = 1 + 708,588 = 708,589`

Every concept over `X` (the power set of `X`, the hypothesis space of an unbiased learner):

- `|P(X)| = 2^|X| = 2^6144 ≈ 3.375 × 10^1849`, a number with 1,850 decimal digits.

So the conjunctive language can express `708,589 / 2^6144 ≈ 10^-1843.7` of all concepts over
`X`.

| Quantity | Formula | Value |
|---|---|---|
| `prompt_values` | `3 · 2^4` | 48 |
| `op_flag_values` | `2^7` | 128 |
| `instance_space` | `∏ \|D_i\|` | 6,144 |
| `syntactic_hypotheses` | `∏ (\|D_i\| + 2)` | 20,971,520 |
| `semantic_hypotheses` | `1 + ∏ (\|D_i\| + 1)` | 708,589 |
| `power_set_log2` | `\|X\|` | 6,144 |
| `power_set_digits` | `⌊6144 · log10 2⌋ + 1` | 1,850 |
| `unbiased_vs_log2_m300` | `\|X\| − 300` | 5,844 |

### 2.2 Why an unbiased learner cannot classify an unseen instance

Take `H' = P(X)` and run CE on `m` distinct labelled instances `D`. A concept is consistent iff
it agrees with `D` on those `m` instances, and is free on the other `6,144 − m`, so

$$|VS(H', D)| = 2^{6144 - m}.$$

For any unseen `x`, exactly half of these concepts contain `x` (pair each concept `h` with
`h △ {x}`, which agrees with `D` too). A unanimous vote never happens and a majority vote is a
tie, so the learner can only classify the instances it has already seen: it memorises. `S` is
the disjunction of the positives and `G` the negation of the disjunction of the negatives.

Row 1.3.6 gives at most 25 prompts × 12 operator subsets = 300 labelled pairs per category,
hence at most 300 distinct instances and `|VS| ≥ 2^5844` per category. Even all 6,144
instances would require labelling every prompt-attribute combination under every one of the
128 operator sets, and prompts that share an instance can still disagree (section 1.3).

The conjunctive restriction is what lets CE predict at all: once a single positive is seen,
`S` is one concrete-or-`?` vector and each member of `G` bounds at most `2^12 = 4,096`
hypotheses, so unseen instances covered by `S` (or outside every `G`) get a unanimous vote. The price is section 1. A learner without bias does
not generalise; a learner with the wrong bias generalises to the wrong concept or collapses.
Phases 3–5 therefore swap the restriction bias for **preference biases** over far richer
spaces.

---

## 3. Preference biases of the Phase 3–5 learners

A preference (search) bias does not forbid hypotheses; it orders them and the learner returns
the first good-enough one. Each model below can represent the concepts of section 1, but its
preference can still point it at the wrong concept on Terser data.

| Learner (rows) | Preference bias | Terser failure mode it risks |
|---|---|---|
| ID3 / C4.5 (3.3) | Shorter trees; attributes with high information gain near the root (greedy, no backtracking) | An interaction (section 1.4) or exception (1.1) whose attributes have little gain alone is split late or pruned away: `op_H` and `op_I` each look harmless marginally, so the tree never tests them together. XOR-like operator pairs are the classic greedy blind spot. |
| KNN (3.5) | Instances close in feature space have the same label; the metric and scaling define "close"; smoothness grows with `k` | Token-ratio and cost features dominate Euclidean distance unless standardised (3.1.4), so a rewrite that removed one negation ("do **not** use recursion") sits next to harmless filler removals with the same ratio. Cross-category neighbours leak labels unless CV groups by source prompt (3.2.4). |
| Logistic regression (3.4) | A linear decision boundary in the features (log-odds additive), smaller weights under L1/L2 | The interaction "safe alone, unsafe together" is not linear in `op_H`, `op_I`; without a product feature LR averages it into a mild penalty on both and over-rejects or under-rejects each. |
| SVM (4.4) | Maximum margin; with a linear kernel, a hyperplane; kernels choose the similarity | Max margin trusts the boundary cases, and the boundary cases are the ambiguous, judge-noisy pairs (4.4.9); a few mislabelled support vectors move the whole hyperplane. Polynomial kernels can express interactions but prefer the widest margin, not the most plausible rule. |
| Naive Bayes (5.2.3) | Features conditionally independent given the class | n-grams of a hedge ("I think", "maybe") are strongly correlated, so their evidence is counted several times and the detector is over-confident on hedge-heavy but meaningful phrases ("I think the bug is in line 4"). |
| HMM tagger (5.3) | First-order Markov: each KEEP/DROP/REPLACE tag depends only on the previous tag; emissions depend only on the current tag | Dropping a span is safe or unsafe depending on distant context (a constraint two sentences earlier, "keep it formal" at the end of the prompt); a first-order chain cannot see it and happily drops the softener the constraint depends on. |
| GMM + EM (5.2.5–5.2.6) | Data are a mixture of a few Gaussian components; EM finds a local optimum of the likelihood | Judge score deltas are bounded, discrete and skewed, so two Gaussians may split "large improvement vs everything else" rather than preserved vs degraded; C5.5 (ARI ≥ 0.4) is the check. |

Ensembles (5.1) mainly reduce variance; their bias is that of the base learner (depth-1
stumps for AdaBoost are additive in single features, so interactions again need depth ≥ 2).

---

## 4. Label noise breaks Candidate-Elimination

CE assumes the labels are **noise-free**: it deletes every hypothesis that misclassifies a
single example. The pilot labels come from a position-swapped LLM judge (row 1.3.6) and, even
after human review, some will be wrong. Row 2.3.6 measures judge–human agreement for this
reason.

One wrong label is enough to eliminate the target:

- If the target concept `c` is in `H` and one example `(x, ¬c(x))` is mislabelled, `c` is
  inconsistent with `D` and is removed from the version space permanently. CE never revisits a
  deletion.
- The outcome is then either a **collapse**, indistinguishable from the restriction-bias
  collapses of section 1, or, worse, a **non-empty version space that excludes `c`** and is
  reported as a success. The collapse diagnosis of row 1.3.7 (the minimal conflicting set) is
  the tool for telling the two apart: a conflicting set that a human reviewer judges mislabelled
  is noise, one that is correctly labelled is bias.

If each label is wrong independently with probability `ε`, the target survives `m` examples
only if all `m` labels are right, with probability `(1 − ε)^m`:

| Quantity | `ε` | `m` | `(1 − ε)^m` |
|---|---|---|---|
| `survival_eps01_m100` | 0.01 | 100 | 0.366 |
| `survival_eps01_m300` | 0.01 | 300 | 0.0490 |
| `survival_eps02_m300` | 0.02 | 300 | 0.00233 |
| `survival_eps05_m100` | 0.05 | 100 | 0.00592 |
| `survival_eps05_m300` | 0.05 | 300 | 2.08e-7 |

At a 300-example category, a judge that is 99% accurate still gives the true concept under a
5% chance of surviving. Collecting more data makes this *worse*, whereas statistical learners
improve with more data.

This is why the project moves on after Phase 1:

1. **Minimise error, not consistency.** ID3 with pruning, LR, SVM with soft margin and the
   ensembles minimise an empirical loss that tolerates misclassified training examples;
   reduced-error pruning (3.3.4) and the soft-margin `C` (4.4.6) set how much.
2. **Probabilistic outputs.** LR, naive Bayes and GMM output probabilities; a label that
   contradicts many similar examples lowers confidence instead of deleting a hypothesis.
   The Bayes optimal classifier of row 5.2.4 reuses the Phase 1 conjunctive hypotheses with a
   likelihood that allows a per-example error rate, turning CE's hard deletion into a soft
   reweighting.
3. **Measured noise.** Grouped cross-validation (3.2.4), the human-annotated 300 pairs (2.3.5)
   and bootstrap CIs (3.2.8) estimate generalisation error directly instead of assuming zero
   training error.

Phase 1 keeps CE because its failures are informative: each collapse names a concrete set of
examples that is either a labelling mistake to fix or a concept (exception, disjunction,
threshold, interaction) that the later learners must be able to represent.
