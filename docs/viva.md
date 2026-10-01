# Viva Preparation (row 13.9)

Each topic has: the result to state, a worked example (from Terser where possible), a pointer to
the code that implements it, and the likely follow-up questions. Full derivations are in
[`docs/derivations/`](derivations). The syllabus-to-code index is [syllabus_map.md](syllabus_map.md).

---

## 1. ID3 information gain

**State.** For a node with sample set S and classes c,
`H(S) = −Σ_c p_c log₂ p_c`, and for attribute A with values v,
`Gain(S, A) = H(S) − Σ_v (|S_v| / |S|) · H(S_v)`. ID3 splits on the attribute with the highest
gain, one branch per value, and recurses until a node is pure or no attributes are left.

**Worked example (PlayTennis, 9+ / 5−).** `H(S) = −(9/14)log₂(9/14) − (5/14)log₂(5/14) = 0.940`.
Outlook: Sunny 2+/3− (H = 0.971), Overcast 4+/0− (H = 0), Rain 3+/2− (H = 0.971), so
`Gain = 0.940 − (5/14)(0.971) − 0 − (5/14)(0.971) = 0.246`. Humidity 0.151, Wind 0.048,
Temperature 0.029, so the root is Outlook. `tests/test_scratch_id3.py` rebuilds this exact
tree (5 leaves, depth 2).

**Code.** [`scratch_id3.py`](../src/terser/models/classical/scratch_id3.py).

**Follow-ups.**
- *Why does gain favour many-valued attributes?* More branches give purer subsets by chance.
  Gain ratio (C4.5) divides by the split information `−Σ (|S_v|/|S|) log₂(|S_v|/|S|)`.
- *Why compare against scikit-learn only at full depth?* A multiway ID3 node with 4 bins has
  up to 4^d leaves at depth d, but a binary CART tree has 2^d. Equal depth is not equal capacity.
  Fully grown, both fit the training set, so the comparison is fair (row 3.3.1).
- *Inductive bias?* A preference bias for short trees with high-gain attributes near the root
  (Occam). Compare the restriction bias of Phase 1's conjunctive hypothesis space.

## 2. Bias–variance decomposition

**State (squared loss).** For `y = f(x) + ε` with `Var(ε) = σ²`, and a model trained on a random
dataset D:
`E_D[(y − ĥ_D(x))²] = (f(x) − E_D[ĥ_D(x)])² + E_D[(ĥ_D(x) − E_D[ĥ_D(x)])²] + σ²`
= bias² + variance + irreducible noise.

**How Terser estimates it (row 3.6.1).** Draw 200 bootstrap training sets and score each model
on a fixed test set. For each point, the mean prediction across bootstraps gives bias² and the
spread around that mean gives variance. For 0-1 loss we use the main-prediction (mode)
decomposition. Tree depth ∈ {2, 4, 8, 16, ∞}: bias falls and variance rises with depth. For
KNN, k = 1 has low bias and high variance, and k = 51 is the reverse. The minimum of the total
error is the trade-off point.

**Code.** [`eval/bias_variance.py`](../src/terser/eval/bias_variance.py).

**Follow-ups.** Bagging lowers variance and leaves bias about the same. Boosting lowers bias.
Regularisation (L2, pruning, larger k) trades a little bias for less variance.

## 3. ROC and AUC

**State.** As the threshold t sweeps from +∞ to −∞, plot TPR = TP/(TP+FN) (sensitivity)
against FPR = FP/(FP+TN) (1 − specificity). AUC equals the probability that a random positive
scores above a random negative (the Mann–Whitney U statistic divided by n₊n₋). Ties count ½.

**Why it matters here.** The quality gate's positive class is *degraded*. The product needs
high sensitivity, because a missed degradation reaches the user. So the checkpoints fix the
operating point: specificity is measured **at sensitivity 0.95** (C3.3, C5.1), not at the
default 0.5 threshold.

**Code.** [`eval/curves.py`](../src/terser/eval/curves.py),
[`eval/threshold.py`](../src/terser/eval/threshold.py). Both are tested against scikit-learn
to 1e-9.

**Follow-ups.** ROC is insensitive to class balance, and PR curves are not. That is why the
Naive Bayes filler detector (C5.3) is judged on precision/recall. AUC can hide a bad region of
the curve, which is why the operating point is fixed.

## 4. Backpropagation

**State.** For layer l with `z⁽ˡ⁾ = W⁽ˡ⁾a⁽ˡ⁻¹⁾ + b⁽ˡ⁾` and `a⁽ˡ⁾ = g(z⁽ˡ⁾)`, define
`δ⁽ˡ⁾ = ∂L/∂z⁽ˡ⁾`. Then
`δ⁽ᴸ⁾ = ∂L/∂a⁽ᴸ⁾ ⊙ g′(z⁽ᴸ⁾)`, `δ⁽ˡ⁾ = (W⁽ˡ⁺¹⁾)ᵀ δ⁽ˡ⁺¹⁾ ⊙ g′(z⁽ˡ⁾)`,
`∂L/∂W⁽ˡ⁾ = δ⁽ˡ⁾ (a⁽ˡ⁻¹⁾)ᵀ`, `∂L/∂b⁽ˡ⁾ = δ⁽ˡ⁾`.
With a sigmoid output and binary cross-entropy, `δ⁽ᴸ⁾ = ŷ − y`, because the g′ term cancels.

**Verification (C4.1).** A central finite-difference gradient check gives relative error
< 1e-6 on every layer, and the result matches `torch.autograd`.

**Code.** [`numpy_mlp.py`](../src/terser/models/neural/numpy_mlp.py), derivation in
[backprop.md](derivations/backprop.md).

**Follow-ups.** Vanishing gradients: sigmoid has g′ ≤ ¼ and saturates (row 4.2.1 reports the
fraction of saturated units). With ReLU, dead units get zero gradient. The step function has
zero gradient almost everywhere, so it is trained with a straight-through estimator.

## 5. SVM dual

**Primal (soft margin).** `min ½‖w‖² + C Σ ξ_i` subject to `y_i(wᵀx_i + b) ≥ 1 − ξ_i` and
`ξ_i ≥ 0`.
**Lagrangian → dual.** `max_α Σα_i − ½ ΣΣ α_i α_j y_i y_j x_iᵀx_j` subject to
`0 ≤ α_i ≤ C` and `Σ α_i y_i = 0`.
**Solution.** `w = Σ α_i y_i x_i`. Support vectors are the points with α_i > 0. The points with
0 < α_i < C lie exactly on the margin and give b. Prediction is
`sign(Σ α_i y_i K(x_i, x) + b)`. The data enter only through dot products, which is the kernel
trick. The polynomial kernel is `K = (γ xᵀx′ + coef0)^d`.

**Margin width.** 2/‖w‖. A larger C gives a narrower margin, fewer margin violations and fewer
support vectors (row 4.4.6).

**Code.** [`scratch_svm.py`](../src/terser/models/classical/scratch_svm.py) solves the dual QP
with cvxopt. C4.2 requires the support-vector set to equal scikit-learn's and the weight error
to be < 1e-3. Derivation in [svm.md](derivations/svm.md).

## 6. AdaBoost weights

**State.** Start with `w_i = 1/n`. In round t, fit a stump h_t, take the weighted error
`ε_t = Σ w_i [h_t(x_i) ≠ y_i]`, set `α_t = ½ ln((1 − ε_t)/ε_t)`, then update
`w_i ← w_i · exp(−α_t y_i h_t(x_i)) / Z_t`. The final classifier is `sign(Σ α_t h_t(x))`.

**Worked numbers.** ε = 0.3 gives α = ½ ln(7/3) ≈ 0.424. Misclassified points are multiplied
by e^{0.424} ≈ 1.53 and correct ones by 0.65. After normalisation the misclassified points hold
exactly half the weight.

**Terser angle (5.1.1).** The pairs that end up with the largest weights are the hard examples. The
hard-example note for 5.1.1 lists them and says what they have in common.

**Code.** [`scratch_adaboost.py`](../src/terser/models/ensembles/scratch_adaboost.py).

**Follow-ups.** The training error is bounded by `Π_t 2√(ε_t(1−ε_t))`. AdaBoost minimises
exponential loss stagewise. It is sensitive to label noise, and Terser's labels come from a
judge.

## 7. EM (Gaussian mixture)

**E-step.** `γ_ik = π_k N(x_i | μ_k, Σ_k) / Σ_j π_j N(x_i | μ_j, Σ_j)`.
**M-step.** `N_k = Σ_i γ_ik`, `π_k = N_k/n`, `μ_k = Σ γ_ik x_i / N_k`,
`Σ_k = Σ γ_ik (x_i − μ_k)(x_i − μ_k)ᵀ / N_k`.
Each iteration cannot decrease the log-likelihood, because EM maximises a lower bound (the ELBO)
that touches the log-likelihood after the E-step. It converges to a local optimum, so it needs
several restarts.

**Terser use (5.2.6).** A 2-component GMM on (rewrite judge score − original judge score)
recovers the preserved/degraded split without labels. ARI against the threshold labels tests
the judge's threshold (C5.5).

**Code.** [`gmm.py`](../src/terser/models/classical/gmm.py).

## 8. Viterbi

**State.** For an HMM (π, A, B), `δ_1(j) = π_j b_j(o_1)` and
`δ_t(j) = max_i δ_{t−1}(i) a_ij · b_j(o_t)`, with `ψ_t(j) = argmax_i …`. Backtrack from
`argmax_j δ_T(j)`. Work in log space so long prompts do not underflow. Cost is O(T·N²). The
forward algorithm has the same structure with Σ in place of max, and gives P(O | λ).

**Terser use (5.3).** States are KEEP/DROP/REPLACE and observations are token features.
Viterbi tags a casual prompt, and the DROP tags become the deterministic compressor (C5.6).
Baum-Welch (EM with forward-backward) refines the parameters on unlabelled casual prompts.

**Code.** [`hmm.py`](../src/terser/models/classical/hmm.py).

## 9. K-means ⇔ EM

**State.** Take a GMM with shared isotropic covariance σ²I and equal priors. Then
`γ_ik ∝ exp(−‖x_i − μ_k‖² / 2σ²)`. As σ → 0, γ_ik becomes 1 for the nearest centroid and 0
otherwise (a hard assignment), and the M-step mean becomes the cluster mean. That is exactly
Lloyd's K-means.

**Verification (C6.2).** Assignment agreement ≥ 98% between EM at the smallest σ and K-means.

**Code.** [`kmeans_em.py`](../src/terser/models/clustering/kmeans_em.py), derivation in
[kmeans_em.md](derivations/kmeans_em.md).

**Follow-ups.** K-means minimises SSE and assumes spherical clusters of similar size.
k-means++ seeding gives an O(log k)-competitive start. Bisecting K-means is the divisive
(top-down) alternative.

## 10. Bellman equation and the Q-learning update

**Bellman optimality.** `Q*(s, a) = E[r + γ max_{a′} Q*(s′, a′) | s, a]`.
**Q-learning (off-policy TD).** `Q(s,a) ← Q(s,a) + α [r + γ max_{a′} Q(s′,a′) − Q(s,a)]`, with
ε-greedy exploration. The tabular version converges if every (s, a) is visited infinitely
often and the step sizes satisfy Robbins–Monro conditions.

**Terser MDP (Phase 9).** The state is (filler bucket, constraint bucket, length bucket,
category, operators already applied). An action applies one of the operators F H D I L C S
(each at most once) or STOPs. Each operator step costs −0.01. At STOP the reward is
`savings_pct − 100 · 1[degraded]`, where *degraded* comes from the quality gate, so no LLM is
called during learning. C9.1 requires
the Q-table change to be < 1e-3 over the last 1,000 episodes.

**Code.** [`scratch_qlearning.py`](../src/terser/models/rl/scratch_qlearning.py),
[`rl/env.py`](../src/terser/models/rl/env.py), derivation in [rl.md](derivations/rl.md).

**Follow-ups.** SARSA is on-policy and uses the action actually taken. γ trades off immediate
savings against later operators. Overestimation bias comes from the max.

## 11. CNN parameter count

**State.** A Conv1d layer has `C_in · k · C_out` weights plus `C_out` biases. Pooling has no
parameters. The output length is `⌊(L + 2p − k)/s⌋ + 1`, and 'same' padding uses
p = (k − 1)/2.

**Terser TextCNN (8.2.2).**
| Layer | Params |
| :--- | ---: |
| Conv k = 3, 384 → 128 | 384·3·128 + 128 = 147,584 |
| Conv k = 5, 384 → 128 | 384·5·128 + 128 = 245,888 |
| Token head (256 → 1) | 257 |
| Prompt head (256 → 1) | 257 |
| **Total** | **393,986** |

With max and avg pooling concatenated (8.2.3), the prompt head takes 512 inputs and has 513
parameters. `tests/test_conv_textcnn.py` checks the closed form against `sum(p.numel())`.

**Max vs avg pooling.** Max pooling answers "is there any droppable region?". Average pooling
answers "how much of the prompt is droppable?".

## 12. LoRA rank

**State.** Freeze W₀ ∈ ℝ^{d×k} and learn `ΔW = (α/r) · B A` with B ∈ ℝ^{d×r} and
A ∈ ℝ^{r×k}, where r ≪ min(d, k). B starts at zero, so training starts from the base model.
Trainable parameters per adapted matrix: `r(d + k)` instead of `dk`.

**Worked number (rung 2, Qwen2.5-1.5B, r = 16, α = 32).** For one 1536 × 1536 `q_proj`, LoRA
trains 16 · (1536 + 1536) = 49,152 parameters instead of 2,359,296 (about 2%). The scale is
α/r = 2. After training, BA can be merged into W₀, so inference cost does not change.

**Choosing r.** The rank caps how much the update can express. Too small underfits the
rewrite task. Too large costs memory and can overfit a small SFT set. Rank is tuned with α, the
learning rate and the epochs in row 10.7. QLoRA (rung 4) keeps the same adapters on an NF4
4-bit base model, which fits 7B in 24 GB.

**Code.** [`rewriter/train.py`](../src/terser/models/rewriter/train.py),
[`configs/experiment/sft_qwen15b_lora.yaml`](../configs/experiment/sft_qwen15b_lora.yaml).
