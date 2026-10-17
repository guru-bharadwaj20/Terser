# Decision trees: ID3, C4.5 thresholds and pruning

Code: [`src/terser/models/classical/scratch_id3.py`](../../src/terser/models/classical/scratch_id3.py).
Tests: [`tests/test_scratch_id3.py`](../../tests/test_scratch_id3.py). Plan rows 3.3.1, 3.3.2, 3.3.4, 3.3.7.

## 1. Entropy and information gain

For a set `S` with class proportions `p_c`, the entropy in bits is

    H(S) = - sum_c p_c log2 p_c

which is 0 for a pure set and `log2 K` for a uniform one over `K` classes. A multiway split of `S` on
attribute `A` with values `v` gives subsets `S_v`; the expected entropy after the split is
`sum_v |S_v|/|S| H(S_v)` and the information gain is

    IG(S, A) = H(S) - sum_v |S_v|/|S| H(S_v)  >= 0.

ID3 grows the tree greedily: at each node pick the attribute with the largest gain, create one child
per value, recurse on each child with that attribute removed, and stop when the node is pure, no
attribute is left, or a pre-pruning limit is hit. The leaf predicts the majority class.

**Hand check (Mitchell 1997, Table 3.2, PlayTennis, 9 yes / 5 no).** `H(S) = -(9/14) log2(9/14) -
(5/14) log2(5/14) = 0.940`. Outlook splits into Sunny (2+, 3-), Overcast (4+, 0-), Rain (3+, 2-):

    IG = 0.940 - (5/14)(0.971) - (4/14)(0) - (5/14)(0.971) = 0.247

and likewise Humidity 0.152, Wind 0.048, Temperature 0.029, so Outlook is the root. Under Sunny,
Humidity separates the classes perfectly; under Rain, Wind does. The test asserts these gains and
this exact five-leaf tree.

## 2. Continuous attributes (C4.5 thresholds)

A continuous attribute `x` is turned into a binary test `x <= t`. Sort the examples by `x`; only
midpoints `t = (a + b)/2` between adjacent distinct values `a < b` need be tried. Fayyad and Irani
(1992) prove that the cut minimising expected entropy lies on a **boundary point**: a midpoint where
the class changes, i.e. not between two values that are both pure in the same single class. Moving
`t` across a run of same-class examples changes the split entropy monotonically (it is concave in
the run position), so the optimum is at the ends of runs. The implementation therefore evaluates
only boundary midpoints, and finds the same best cut as scikit-learn, which tries all midpoints.
That equivalence holds without a leaf-size limit; with `min_samples_leaf > 1` the implementation
tries every midpoint, because a boundary cut can leave a leaf too small.
Unlike ID3 attributes, a continuous attribute can be tested again deeper in the tree.

C4.5 normally ranks splits by gain ratio; this implementation uses information gain so it is
directly comparable to `DecisionTreeClassifier(criterion="entropy")`.

## 3. Pruning

* **Pre-pruning**: stop at `max_depth`, below `min_samples_split`, when a child would hold fewer than
  `min_samples_leaf` examples, or when the best gain is not above `min_gain`.
* **Reduced-error pruning**: with a held-out validation set, visit nodes bottom-up; replace the
  subtree at `t` by a leaf (its training majority) if the leaf makes no more validation errors than
  the subtree. Validation accuracy never decreases. Nodes reached by no validation example are
  pruned (0 errors either way).
* **Rule post-pruning**: write one rule per leaf (conjunction of the path's tests). For each rule,
  repeatedly delete the precondition whose deletion gives the highest Laplace accuracy estimate
  `(correct + 1)/(covered + 2)` on the validation set, as long as it does not fall. Sort rules by that
  estimate; the first matching rule classifies, else the training majority. Because each rule is
  pruned on its own, the result need not be a tree.
* **Minimal cost-complexity pruning** (Breiman et al. 1984): for a subtree `T` let
  `R_alpha(T) = R(T) + alpha |leaves(T)|`. For each internal node `t`, collapsing `T_t` into a leaf
  is worthwhile once `alpha >= g(t) = (R(t) - R(T_t)) / (|leaves(T_t)| - 1)`. Repeatedly collapsing
  the weakest link (smallest `g`) gives a nested sequence of trees with increasing effective
  `alpha`. With `R(t) = (n_t / N) H(t)` this is exactly scikit-learn's `ccp_alphas`; the test checks
  the whole alpha path against scikit-learn to 1e-9. `R(t)` = training misclassification rate is
  Breiman's original choice and is also available.
