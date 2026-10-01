# ADR-0005: Judge protocol: order-averaged pairwise verdicts (`pairwise_avg`)

| Field | Value |
| :--- | :--- |
| Status | Proposed (becomes Accepted after review by ≥ 1 other contributor) |
| Date | 2026-10-01 |
| Plan row(s) | 2.3.3, 2.3.4 (label rule unchanged), 1.3.6 (pilot relabel); checkpoint C2.6 |
| Deciders | guru-bharadwaj20 |
| Reviewers | pending |
| Related | [reports/p2_judge_bias.md](../../reports/p2_judge_bias.md), [reports/p1_pilot_labels.md](../../reports/p1_pilot_labels.md), ADR-0002 (problem framing) |

## Context

Row 2.3.3 specifies a pairwise judge (Qwen2.5-32B-Instruct-AWQ) that compares the target model's
output for the original prompt with its output for the rewrite, in both A/B orders. Row 2.3.4
labels a pair `preserved` iff the judge says tie-or-rewrite-better in **both** orders. C2.6
requires an A/B-order disagreement of at most 10%.

The 1.3.6 GPU pilot showed that this judge has a strong first-position bias: on 836 pilot pairs
the two orders disagree on 40.7%, and 77% of the disagreements are "whichever response is shown
first wins". Under the both-orders rule every such disagreement becomes `degraded`, so 314 of the
428 `degraded` pilot labels came from position-inconsistent verdicts. Labels built this way
measure presentation order as much as quality.

We measured four protocols on the same 836 pilot pairs (296 distinct judge requests; target
outputs from the pilot cache, Qwen2.5-7B-Instruct, temperature 0). The **reference set** is the
494–496 pairs on which the original pilot judge was position-consistent. On those pairs the
both-orders verdict is not in doubt, so a protocol that disagrees with them is changing labels
that were never affected by position.

| Protocol | Judge | A/B disagreement | Share of disagreements won by the first-shown answer | Degraded share | Agreement with reference (κ) | Judge calls per pair | Generated tokens (836 pairs) | Wall time |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| (1) pairwise (pilot run) | 32B-AWQ | 40.7% | 76.8% | 51.2% | 1.000 (1.000) | 2 | — | — |
| (1) pairwise (re-run with logprobs) | 32B-AWQ | 40.9% | 77.2% | 51.4% | 0.996 (0.989) | 2 | 23,740 | 351 s |
| (2) pointwise rubric scores | 32B-AWQ | 0% by construction | — | 13.9% | 0.841 (0.478) | 2 (1 per output) | 38,885 | 268 s |
| (3) pairwise | 14B-AWQ | 28.7% | 10.8% (second position wins both orders in 64 pairs vs first in 26) | 26.0% | 0.847 (0.556) | 2 | 22,158 | 141 s |
| (3b) pointwise rubric scores | 14B-AWQ | 0% by construction | — | 11.0% | 0.804 (0.293) | 2 | 39,668 | 103 s |
| (4) **pairwise_avg**: order-averaged logprobs | 32B-AWQ | **0% by construction** (raw orders: 40.9%) | — | 31.6% | **1.000 (1.000)** | 2 | 23,740 (same calls as 1) | 351 s |
| (4b) pairwise_avg | 14B-AWQ | 0% by construction (raw: 28.7%) | — | 17.3% | 0.895 (0.650) | 2 | 22,158 | 141 s |

MLflow (experiment `phase2`): 32B pairwise `09d66708740b`, 32B pointwise `3494f098f0ba`, 14B
pairwise `d8b9796c2626`, 14B pointwise `c96954c422bc`. Raw data: `data/pilot/judge_bias/`.
Table: [reports/p2_judge_bias.csv](../../reports/p2_judge_bias.csv).

Findings:

- **No protocol brings the per-order disagreement of a pairwise judge to ≤ 10%.** Switching to
  14B lowers it to 28.7% but its bias leans to the second position (64 vs 26 pairs where one
  position wins both orders), so it is a different bias, not the absence of one.
- **Pointwise scoring is order-free but loses discrimination.** It labels only 11–14% of pairs
  `degraded` and disagrees with 16–20% of the reference pairs (κ 0.29–0.48), mostly by calling
  pairs the pairwise judge consistently found worse a tie. Absolute 1–5 rubric scores are too
  coarse to separate two good answers.
- **Averaging the two orders' verdict probabilities removes the order effect from the label and
  keeps every reference label** (32B: 100%, κ = 1.000). The 342 position-inconsistent pairs
  are resolved by the averaged probabilities: 191 → `preserved`, 151 → `degraded` (under the
  both-orders rule 315 of them were `degraded` in the pilot run). On 209 of them the averaged margin between the top two sides is < 0.2, so these are
  close calls and are flagged `low_margin` for human review.

## Decision

Use **`pairwise_avg` with Qwen2.5-32B-Instruct-AWQ** as the judge protocol for row 2.3.3 and all
labels from now on:

1. Judge every pair in both A/B orders with the unchanged `pairwise_v1` prompt and rubric,
   temperature 0, requesting `top_logprobs=20` (`terser.judge.protocols.averaged_judge`).
2. Read P(A), P(B), P(tie) at the verdict token of each reply
   (`terser.judge.pairwise.verdict_probs`), map them to sides (original / rewrite / tie) and
   average the two orders (`terser.judge.protocols.order_averaged_side`). Exact ties in the
   averaged probabilities are broken towards `tie`, then `original` (conservative).
3. Apply the **row 2.3.4 rule unchanged** to the averaged side, used for both orders
   (`label_pair(task_not_worse, side, side)`): `preserved` iff the task metric is not worse and
   the averaged verdict is tie-or-rewrite-better. The meaning of the label ("the judge, over both
   presentation orders, does not prefer the original's output") is kept; only the aggregation of
   the two orders changes from "both must agree" to "their average decides".
4. Keep both raw per-order verdicts in every record (`verdicts`), plus `aggregate`
   (side, probabilities, margin). Flag `position_inconsistent` and `low_margin` (< 0.2) pairs for
   human review; they are the first candidates for the 2.3.5 annotation sample.
5. Report the raw per-order disagreement alongside every labelled dataset; it remains the
   measurement C2.6 refers to.

**C2.6 is not changed by this ADR and stays ❌.** The raw A/B disagreement of the chosen judge on
the pilot is 40.9% (target ≤ 10%). The labels no longer depend on it, but a checkpoint target may
only be changed by an accepted ADR that shows it was wrong. This ADR proposes, for the reviewer
to accept or reject, that C2.6 be re-stated as "the **label** does not depend on presentation
order (A/B swap changes ≤ 10% of labels), with the raw per-order disagreement reported". Under
that wording `pairwise_avg` passes by construction, which is exactly why the reviewer, not the
author, should decide whether the re-statement is legitimate.

## Alternatives considered

| Alternative | Pros | Cons | Why not chosen |
| :--- | :--- | :--- | :--- |
| Keep `pairwise` with the both-orders rule | As planned; no new code | 41% of verdict pairs disagree; ~73% of `degraded` labels are position artefacts; C2.6 fails by 4× | Labels would measure position, not quality |
| Pointwise rubric scoring | Order-free; scores each output once (cache-friendly: one call per distinct output) | Coarse 1–5 anchors: κ 0.48 (32B) vs the reference set; degraded share 14% (lenient) | Loses the discrimination the gate classifier needs |
| Pairwise with 14B-AWQ | 2.5× faster | 28.7% disagreement with an opposite (second-position) bias; κ 0.56 vs reference | Swaps one bias for another and is less accurate |
| `pairwise_avg` with 14B-AWQ | Fast; order-free | κ 0.65 vs reference | 32B is clearly better at the same protocol; 32B fits the card (row 7.6) |
| Contextual calibration per order (subtract a fitted position prior) | Keeps per-order verdicts | Needs a held-out fit of the prior; extra parameters to validate against humans | Averaging is parameter-free and already removes the order effect from the label |

## Consequences

- Positive: labels are invariant to A/B order by construction; the 32B judge's consistent
  verdicts are kept exactly; no extra GPU cost over the planned pairwise judge (same two calls,
  one more response field); the averaged margin gives a confidence score for review sampling.
- Negative: on order-inconsistent pairs the label rests on small probability differences
  (median margin 0.13); `degraded` labels drop from 51% to 32% on the pilot, so the C2.8 class
  balance and the C2.7 oracle savings will be measured on a more permissive labelling than the
  plan assumed. Requires servers that return `top_logprobs` (vLLM does; verdicts recovered from a
  retry fall back to a one-hot vector).
- Risk: if human review (2.3.5 / 2.3.6) shows the averaged verdict is wrong on the low-margin
  pairs, the fallback is to treat `low_margin` pairs as unlabelled. This is the measurement that
  verifies the decision: judge–human κ (C2.4) and sensitivity/specificity (C2.5) on the 300-pair
  gold set, reported separately for low-margin pairs.
- Follow-up: the pilot labels were regenerated with this protocol
  (`data/pilot/labels.jsonl`; the old labels are kept as `data/pilot/labels_pairwise_v1.jsonl`);
  the Phase 2 judge stage (2.3.3) uses it; the reviewer decides the C2.6 wording.
