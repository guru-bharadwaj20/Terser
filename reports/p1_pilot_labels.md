# Phase 1 pilot labels (row 1.3.6): GPU run

## Run

| Field | Value |
| :--- | :--- |
| Date | 2026-10-01 |
| Machine | RTX 4500 Ada workstation, WSL2 (`hardware=rtx4500ada`), one model on the card at a time |
| Prompts | `data/pilot/prompts.jsonl`, first **25 per category** (`limit=25`) = **150 prompts** |
| Operator subsets | 12: `F H D I L C S FHC FHCI FHCID FHCIDS FHCIDSL` |
| Target | `Qwen/Qwen2.5-7B-Instruct`, vLLM 0.30.0, bf16, temperature 0, max 512 tokens |
| Judge | `Qwen/Qwen2.5-32B-Instruct-AWQ` (`awq_marlin`), max len 4096, `max_num_batched_tokens=2048`, temperature 0, prompt `pairwise_v1`, both A/B orders |
| Label rule | `preserved` iff tie-or-rewrite-better in **both** orders, else `degraded` |
| Stage 1 (respond) | MLflow `e71803a65709` (`1-pilotlabels-Qwen2.5_7B_Instruct-respond-0`): 443 unique target generations, 120,832 output tokens, 179 s, `workers=32` |
| Stage 2 (judge) | MLflow `6b1831d5eb77` (`1-pilotlabels-Qwen2.5_7B_Instruct-judge-0`): 1,672 judge calls (1,062 served from cache because rewrites repeat across subsets), 25,973 output tokens, 372 s, `workers=16` |
| Outputs | `data/pilot/labels.jsonl` (836 records), `data/pilot/labels_review.csv` (836 rows) |

Substitution: the plan also asks for `meta-llama/Llama-3.1-8B-Instruct` as a target. That repo is
gated and this machine has no Hugging Face token, so only the Qwen2.5-7B target was run.

## Counts

150 prompts × 12 subsets = 1,800 pairs. **964 are skipped** because the rewrite equals the prompt,
leaving **836 labelled pairs** from 127 prompts. 23 prompts are unchanged by every subset.

| Category | Pairs | Preserved | Degraded | Position-inconsistent (needs review) |
| :--- | ---: | ---: | ---: | ---: |
| code | 165 | 102 | 63 | 28 |
| fact | 116 | 79 | 37 | 34 |
| reason | 101 | 78 | 23 | 23 |
| explain | 153 | 45 | 108 | 94 |
| creative | 149 | 16 | 133 | 103 |
| transform | 152 | 88 | 64 | 58 |
| **Total** | **836** | **408** | **428** | **340** |

| Ops | F | H | D | I | L | C | S | FHC | FHCI | FHCID | FHCIDS | FHCIDSL |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Pairs | 89 | 46 | 2 | 71 | 1 | 8 | 2 | 110 | 126 | 127 | 127 | 127 |
| Preserved | 48 | 25 | 1 | 29 | 0 | 5 | 1 | 57 | 59 | 61 | 61 | 61 |

## Findings

1. **The judge has a strong first-position bias.** Verdict pairs, as (original-first winner,
   rewrite-first winner):

   | Pattern | Count | Meaning |
   | :--- | ---: | :--- |
   | (original, rewrite) | **261** | picks whichever response is shown first, both times → `degraded` |
   | tie, tie | 238 | consistent tie → `preserved` |
   | rewrite, rewrite | 144 | consistent rewrite better → `preserved` |
   | original, original | 114 | consistent original better → `degraded` |
   | original, tie | 33 | → `degraded` |
   | rewrite, original | 20 | picks the second-shown response both times → `degraded` |
   | tie, rewrite | 19 | → `preserved` |
   | rewrite, tie | 7 | → `preserved` |

   261 "always A" against 20 "always B" is far from symmetric. Under the both-orders rule, every
   one of those 261 becomes `degraded`. So **most `degraded` labels (314 of 428) come from
   position-inconsistent verdicts**, and only 114 are a consistent "original better". These 340
   rows have `needs_human_review = True` and should be reviewed first.
2. **Creative and explain dominate the degraded labels.** At temperature 0, small prompt edits
   change long open-ended outputs a lot, and the judge then prefers whichever it saw first. This
   is the main thing for a human to adjudicate. Is a different but equally good poem "degraded"?
3. **87 of 443 target outputs (20%) hit the 512-token cap**, so some comparisons are between two
   truncated responses.
4. Operators `D`, `L`, `S` and `C` almost never change these prompts (≤ 8 pairs each). The
   cumulative stacks repeat each other's rewrites (`FHCID`, `FHCIDS` and `FHCIDSL` give the same rewrite for 124 of 127 prompts), which is
   why so many judge calls were cache hits.

## Relabel with the ADR-0005 judge protocol (2026-10-01)

Because of finding 1, the judge protocol was compared on these 836 pairs
([p2_judge_bias.md](p2_judge_bias.md)) and changed to **`pairwise_avg`**
([ADR-0005](../docs/decisions/ADR-0005-judge-protocol.md), Proposed): the same two
`pairwise_v1` calls, with the A/B/tie probabilities of both orders averaged before the 2.3.4 rule
is applied. The pilot was relabelled with it
(`experiment.params.protocol=pairwise_avg`, stage `judge`; MLflow `508e0d17bce8`; all 1,672
requests from the cache, 0 new tokens). The cached verdicts are those of the 32B logprob re-run
(`data/pilot/judge_bias/pairwise_32b`, MLflow `09d66708740b`), not the original pilot verdicts:
they differ from `labels_pairwise_v1.jsonl` on 43 rows, and 4 pairs change position consistency
(`code-020:I`, `code-023:I`, `explain-009:F`, `explain-009:FHC`), which is why the flagged count
moves from 340 to 342.

| Labels | File | Preserved | Degraded | Flagged for review |
| :--- | :--- | ---: | ---: | ---: |
| `pairwise_v1` both-orders rule (original) | `data/pilot/labels_pairwise_v1.jsonl`, `labels_review_pairwise_v1.csv` | 408 | 428 | 340 |
| **`pairwise_avg` (current)** | `data/pilot/labels.jsonl`, `labels_review.csv` | **572** | **264** | **342** |

| Category | Preserved | Degraded |
| :--- | ---: | ---: |
| code | 118 | 47 |
| fact | 96 | 20 |
| reason | 86 | 15 |
| explain | 92 | 61 |
| creative | 58 | 91 |
| transform | 122 | 30 |

165 labels changed from `degraded` to `preserved` and 1 the other way (`transform-001:F`, an
order-inconsistent pair whose averaged probabilities favour the original). All 494 pairs
that are position-consistent under the re-run verdicts keep their label (496 pairs were
consistent under `pairwise_v1`; the two sets differ by the 4 pairs named above). The 342 flagged rows are the
position-inconsistent pairs; 209 of them also carry `low_margin` (averaged margin < 0.2), and
these are the closest calls. The review CSV has three new columns: `protocol`,
`averaged_side`, `averaged_margin`.

The provisional 1.3.7 analysis was re-run on these labels
([p1_version_spaces.md](p1_version_spaces.md), MLflow `1bd3ceeae1cb`): the version space still
collapses in all 6 categories.

The position-bias result is also direct input for the Phase 2 judge protocol (C2.6).
