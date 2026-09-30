# Phase 2 pipeline smoke run (pre-GPU)

Plan row **2.2.4**. Run on 2026-09-30 in a 4-core CPU container with no GPU.

This run checks that the pipeline stages (respond → judge → label) connect end to end through the
`terser.llm` client using small CPU stand-in models. It also checks that nothing they produce can
become a label.

> **SMOKE RUN.** Every output is tagged `stand_in: true` and lives only under the gitignored
> `data/pilot/smoke/`. None of the numbers below say anything about rewrite quality.

## Setup

| Item | Value |
| :--- | :--- |
| Pipeline | `python -m terser.phase1.pilot_labels experiment=p1_pilot_labels` (shared `respond → judge → label` path) |
| Overrides | `limit=9` per category, `operator_subsets=[FHCIDSL]`, `max_target_tokens=256` |
| Prompts | 54 (9 per category) from `data/pilot/prompts.jsonl` |
| Serving | Ollama 0.35.0 on CPU, OpenAI-compatible endpoint `http://localhost:11434/v1` (`local` provider) |
| Target model (stand-in) | `qwen2.5:1.5b`, temperature 0 |
| Judge model (stand-in) | `qwen2.5:0.5b`, prompt `pairwise_v1`, both A/B orders |
| Budget phase | `phase1` token cap (enforced by `LLMClient`) |

## Result: stages run end to end

| Stage | Outcome |
| :--- | :--- |
| Rewrite (operator stack `FHCIDSL`) | 54 pairs; 3 skipped because the stack left the prompt unchanged |
| Respond (target) | 102 target outputs (original + rewrite for each of the 51 pairs), disk-cached |
| Judge (position-swapped) | 102 judge calls; **0 unparseable verdicts** (`n_unlabelled = 0`) |
| Label (row 2.3.4 rule) | 51 labels: 3 `preserved`, 48 `degraded` |
| Review CSV | `labels_review.csv`, all 51 rows marked `needs_human_review` |

| Metric | Value |
| :--- | ---: |
| LLM calls (5 served from cache) | 204 |
| Generated tokens | 21,728 |
| Wall time | 1,262 s (21 min) |
| Throughput | about 17 generated tok/s on CPU |

## Result: stand-in output is excluded from labels

Every one of the 51 records has `stand_in: true`. Feeding them to the label consumers:

| Consumer | Default behaviour | With the explicit smoke flag |
| :--- | :--- | :--- |
| `terser.phase2.oracle.compute_oracle` (row 2.4.1) | **Refuses**: `StandInDataError: stand-in data found (51 candidate(s), 51 label(s)); stand-in outputs are never labels (row 2.2.4)` | `allow_stand_in=True`: 51 rows, every row flagged `stand_in` |
| `terser.phase1.version_spaces.check_stand_in` (row 1.3.7) | **Refuses**: `StandInLabelsError: 51 of 51 records have stand_in: true` | `--allow-stand-in` only |
| `terser.phase2.headroom`, `terser.phase2.judge_validation` | Raise on stand-in input unless `allow_stand_in` is set (same guard, in source) | — |

`PilotSettings` also refuses `stand_in=false` when a model name looks like an Ollama tag. Stand-in
output therefore cannot be relabelled as real by mistake.

## What the run shows about the stand-in judge (expected, not a finding)

| Verdict pair (original-first, rewrite-first) | Count |
| :--- | ---: |
| (A, A): the judge always picks the first response | 41 |
| (B, B) | 4 |
| (A, B): original wins in both orders | 3 |
| (B, A): rewrite wins in both orders | 3 |

The 0.5B judge has an extreme position bias: 45 of 51 pairs flip with the order. The position-swap
rule handles this as designed. A verdict that flips counts as "not tie-or-rewrite-better in both
orders", so the pair is labelled `degraded`.

This is exactly the failure checkpoint C2.6 (position disagreement ≤ 10%) exists to catch. It is
why real labels need the Qwen2.5-32B judge on the GPU (row 2.3.3).

## Projection for the GPU run

On CPU a pair costs about 25 s. The real Phase 2 volume is roughly:

- 6,000 prompts × (1 original + 6 candidates) × 2 targets is about 8.4×10⁴ target calls; 72,000 pairs × 2 orders is about
  1.4×10⁵ judge calls.

That is infeasible on CPU. On the RTX 4090 workers, vLLM batching is expected to deliver
10³–10⁴ tok/s. The tokens logged here (about 213 generated tokens per pair) are the per-pair
budget to size the `phase2` token cap and the GPU queue jobs.

## Reproduce

```bash
ollama pull qwen2.5:1.5b && ollama pull qwen2.5:0.5b && ollama serve &
python -m terser.phase1.pilot_labels experiment=p1_pilot_labels experiment.params.limit=9 \
  "experiment.params.operator_subsets=[FHCIDSL]" experiment.params.max_target_tokens=256
```
