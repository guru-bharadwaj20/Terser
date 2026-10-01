# Phase 1 pilot labels (row 1.3.6): GPU run, awaiting human review

> **Status: labels generated, human review PENDING.** Row 1.3.6's Definition of Done requires a
> human to review every label in `data/pilot/labels_review.csv`. That has not happened, so the row
> stays ❌. All numbers below are **judge** labels, not reviewed labels.

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
| Outputs | `data/pilot/labels.jsonl` (836 records), `data/pilot/labels_review.csv` (836 rows, `human_label` / `reviewer` / `notes` empty) |

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

## Findings for the reviewer

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

## Owner action required (to finish 1.3.6)

1. Open `data/pilot/labels_review.csv` and fill `human_label` (`preserved` / `degraded`),
   `reviewer` and `notes` for **every** row, starting with `needs_human_review = True`.
2. Merge the reviewed labels back into `data/pilot/labels.jsonl` as a `human_label` field. Note
   that `terser.phase1.version_spaces` lets a non-empty `human_label` override `label`.
3. Re-run 1.3.7:
   `python -m terser.phase1.version_spaces --labels data/pilot/labels.jsonl --out reports/p1_version_spaces.md`.
4. Then flip 1.3.6 (and 1.3.7, if the re-run stands) in CONTRIBUTING.md.

The position-bias result is also direct input for Phase 2 judge validation (C2.3–C2.5).
