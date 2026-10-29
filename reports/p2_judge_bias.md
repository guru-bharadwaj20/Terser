# Phase 2 judge position bias: protocol comparison on the pilot pairs

> Supports row **2.3.3** and checkpoint **C2.6** (an A/B swap changes ≤ 10% of labels, [ADR-0012](../docs/decisions/ADR-0012-feasible-checkpoint-targets.md);
> originally raw A/B verdict disagreement ≤ 10%). Decision:
> [ADR-0005](../docs/decisions/ADR-0005-judge-protocol.md) (Accepted). All numbers are judge
> outputs on unreviewed pilot pairs.

## Setup

| Field | Value |
| :--- | :--- |
| Date | 2026-10-01 |
| Machine | RTX 4500 Ada, WSL2 (`hardware=rtx4500ada`), vLLM 0.30.0, one model on the card at a time |
| Pairs | The 836 labelled pilot pairs of row 1.3.6 (`data/pilot/labels_pairwise_v1.jsonl`): 127 prompts × operator subsets, Qwen2.5-7B-Instruct target outputs from the pilot cache (temperature 0, max 512 tokens). 296 distinct judge requests; 59 pairs have byte-identical outputs |
| Judges | `Qwen/Qwen2.5-32B-Instruct-AWQ` (`awq_marlin`, max len 4096, `max_num_batched_tokens=2048`, `max_num_seqs=64`) and `Qwen/Qwen2.5-14B-Instruct-AWQ` (`awq_marlin`, max len 4096); temperature 0; `--max-logprobs 20` |
| Code | `python -m terser.phase2.judge_bias run|report` ([judge_bias.py](../src/terser/phase2/judge_bias.py)); protocols in [`terser.judge.protocols`](../src/terser/judge/protocols.py), [`terser.judge.pointwise`](../src/terser/judge/pointwise.py) |
| Outputs | `data/pilot/judge_bias/<run>.jsonl` (+ `.stats.json`), table [p2_judge_bias.csv](p2_judge_bias.csv) |

Protocols:

1. **pairwise**: `pairwise_v1` prompt with the category rubric, both A/B orders, greedy verdict
   per order. Row 2.3.4 rule: `preserved` iff tie-or-rewrite-better in both orders.
2. **pointwise**: `pointwise_v1`: each output scored alone on the category rubric's dimensions
   (1–5 anchors) plus a hard-fail flag; rubric-weighted score; a hard fail loses; equal scores
   tie. Same label rule on the score comparison.
3. **pairwise, 14B**: protocol 1 with the 14B fallback judge.
4. **pairwise_avg**: protocol 1's two calls with `top_logprobs=20`; P(A/B/tie) at the verdict
   token is mapped to sides and averaged over the two orders; argmax side (ties → tie, then
   original); same label rule.

"Reference agreement" is the label agreement with the 494–496 pairs on which the original pilot
judge was position-consistent (its both-orders verdict is unambiguous there).

## Results

| Run | Protocol | Judge | A/B disagreement | First-position share of disagreements | Degraded share | Reference agreement | Reference κ | Identical pairs preserved | MLflow |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| pilot (1.3.6) | pairwise | 32B | 40.7% | 76.8% | 51.2% | 100% | 1.000 | 59/59 | `6b1831d5eb77` |
| pairwise_32b | pairwise | 32B | **40.9%** | 77.2% | 51.4% | 99.6% | 0.989 | 59/59 | `09d66708740b` |
| pairwise_32b | **pairwise_avg** | 32B | **0%** (by construction) | — | 31.6% | **100%** | **1.000** | 59/59 | `09d66708740b` |
| pointwise_32b | pointwise | 32B | 0% (by construction) | — | 13.9% | 84.1% | 0.478 | 59/59 | `3494f098f0ba` |
| pairwise_14b | pairwise | 14B | 28.7% | 10.8% | 26.0% | 84.7% | 0.556 | 59/59 | `d8b9796c2626` |
| pairwise_14b | pairwise_avg | 14B | 0% (by construction) | — | 17.3% | 89.5% | 0.650 | 59/59 | `d8b9796c2626` |
| pointwise_14b | pointwise | 14B | 0% (by construction) | — | 11.0% | 80.4% | 0.293 | 59/59 | `c96954c422bc` |

Raw A/B disagreement of the pairwise judges per category:

| Judge | code | creative | explain | fact | reason | transform |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32B (re-run) | 17.0% | 69.1% | 62.7% | 29.3% | 22.8% | 38.2% |
| 14B | 16.4% | 34.2% | 30.1% | 46.6% | 11.9% | 32.9% |

Compute (distinct requests only; repeats are cache hits):

| Run | Judge calls | Generated tokens | Wall time |
| :--- | ---: | ---: | ---: |
| pairwise_32b | 592 | 23,740 | 351 s |
| pointwise_32b | 592 | 38,885 | 268 s |
| pairwise_14b | 592 | 22,158 | 141 s |
| pointwise_14b | 592 | 39,668 | 103 s |
| **Total** | 2,368 | 124,451 | 863 s ≈ **0.24 GPU-h** |

## Findings

1. **The 32B judge's position bias is reproducible.** With logprobs requested the verdict pairs match
   the pilot run on 832 of 836 pairs (99.5%; label agreement on the 496 reference pairs is 99.6%); 40.9% of pairs still flip with the order, 77% of them towards
   the first-shown answer. The bias is concentrated in the open-ended categories (creative 69%,
   explain 63%), where outputs for slightly different prompts differ a lot at temperature 0.
2. **The 14B judge is less biased but not unbiased.** 28.7% disagreement. Where both orders name
   a winner, it picks the *second*-shown answer both times in 64 pairs and the first in 26 (the
   32B judge: first 264, second 20); the other 150 disagreements involve a tie. It also disagrees with 15% of the 32B judge's
   consistent verdicts.
3. **Pointwise scoring is order-free but too coarse.** It calls 86–89% of pairs a tie or better,
   including many pairs the 32B judge consistently judged worse (κ 0.48 against the reference).
4. **Order-averaging fixes the label without losing the consistent verdicts.** `pairwise_avg`
   (32B) keeps all reference labels and decides the 342 order-inconsistent pairs by the averaged
   probabilities (191 → preserved, 151 → degraded). The decision on those pairs is often close:
   median margin 0.13, and 209 pairs have a margin < 0.2 (flagged `low_margin`).
5. Every protocol labels all 59 identical-output pairs `preserved`, so none of them invents
   differences between identical answers.

## Decision and checkpoint status

`pairwise_avg` with Qwen2.5-32B-Instruct-AWQ is the judge protocol for 2.3.3 onwards (ADR-0005,
Accepted). It keeps the 2.3.4 label rule, applied to the order-averaged verdict.

**C2.6 ✅ under the revised target** ([ADR-0012](../docs/decisions/ADR-0012-feasible-checkpoint-targets.md), adopting the ADR-0005 re-statement): the label
produced by `pairwise_avg` averages both orders, so an A/B swap changes **0%** of labels (target
≤ 10%). The raw per-order verdict disagreement is still reported and still misses the original
target: **40.9%** on the pilot (original target ≤ 10%).
At full scale the 2.3.3 judge stage measured **44.0%** raw A/B disagreement (10,432 of 23,718
parsed per-order verdict pairs, [p2_label_stats.json](p2_label_stats.json)).

## Pilot relabel (row 1.3.6)

The pilot was relabelled with `pairwise_avg` (all 1,672 judge requests served from the cache, 0
new tokens; MLflow `508e0d17bce8`). The cached verdicts are those of the `pairwise_32b` re-run
above, not the original pilot verdicts. Old labels are kept as
`data/pilot/labels_pairwise_v1.jsonl` / `labels_review_pairwise_v1.csv`. See
[p1_pilot_labels.md](p1_pilot_labels.md).
