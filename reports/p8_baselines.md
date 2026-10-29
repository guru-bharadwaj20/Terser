# Phase 8 baselines on the rewriter val table (row 8.3.9, checkpoints C8.4 / C8.5)

Same 285 val prompts and metric definitions as [p8_rewriter_val.md](p8_rewriter_val.md) (`val_report.rung_metrics`): median savings on the Qwen2.5 tokenizer over all prompts (fallbacks count 0%); preservation pooled over (prompt, target) for both targets, judged by Qwen2.5-32B-Instruct-AWQ (`pairwise_avg`), fallbacks counted as preserved. Baselines reuse the judged Phase 2 candidates; nothing was regenerated. A candidate that was rejected in row 2.2.1 or is not strictly shorter on Qwen2.5 falls back to the original, like the 8.3.7 safeguard of the SFT rungs.

| System | Source | n | Median savings | Fell back | Preserved | Preserved Qwen7B | Preserved Phi3.5 | Latency (s/prompt) | C8.3 |
|---|---|---|---|---|---|---|---|---|---|
| Original | identity | 285 | 0.0% | 285 | 100.0% | 100.0% | 100.0% | 0.00 | fail |
| Best operator | `ops_FHC` | 285 | 0.3% | 138 | 81.4% | 82.1% | 80.7% | n/r | fail |
| Other operator stack | `ops_CHFIDSL` | 285 | 1.2% | 101 | 75.1% | 75.4% | 74.7% | n/r | fail |
| HMM compressor | `hmm` (row 5.3.6, 124 shared prompts) | 124 | 10.9% | 51 | 58.1% | 60.5% | 55.6% | n/r | fail |
| LLMLingua-2 | `llmlingua2` | 285 | 50.7% | 1 | 28.9% | 30.2% | 27.7% | n/r | fail |
| Zero-shot 32B low | `llm_low` | 285 | 40.1% | 24 | 58.8% | 64.9% | 52.6% | 0.25 | fail |
| Zero-shot 32B medium | `llm_medium` | 285 | 47.9% | 27 | 57.5% | 62.8% | 52.3% | 0.15 | fail |
| Zero-shot 32B high | `llm_high` | 285 | 55.2% | 25 | 52.5% | 54.7% | 50.2% | 0.21 | fail |
| Zero-shot 32B max | `llm_max` | 285 | 66.4% | 25 | 50.7% | 55.1% | 46.3% | 0.20 | fail |
| Zero-shot 32B min | `llm_min` | 285 | 52.2% | 19 | 50.0% | 53.0% | 47.0% | 0.29 | fail |
| SFT sft05 | `sft05` | 285 | 4.1% | 108 | 68.4% | 73.3% | 63.5% | 0.39 | fail |
| SFT sft15 | `sft15` | 285 | 10.2% | 99 | 68.2% | 70.5% | 66.0% | 0.50 | fail |
| SFT sft3b | `sft3b` | 285 | 26.1% | 66 | 65.6% | 69.8% | 61.4% | 0.70 | pass |
| SFT sft15v2 | `sft15v2` | 285 | 22.6% | 73 | 58.1% | 62.1% | 54.0% | 0.49 | fail |
| SFT sft3bv2 | `sft3bv2` | 285 | 24.2% | 70 | 65.3% | 68.1% | 62.5% | 0.72 | fail |
| SFT sft7bv2 | `sft7bv2` | 285 | 35.8% | 63 | 61.9% | 69.1% | 54.7% | 1.24 | fail |

HMM compressor: row 5.3.6 judged its own 500 val prompts (seed 0, stratified), of which 124 are in this table's 285; its row is scored on those only, so it is not directly comparable with the other rows. Here an unchanged rewrite falls back and counts as preserved; the full 5.3.6 result is in [p5_hmm_compressor.md](p5_hmm_compressor.md).

Latency: SFT rungs are the mean generation time per prompt in batches of 16 (HF `generate`, RTX 4500 Ada). Zero-shot 32B levels are the measured wall time of the vLLM generation run divided by the prompts it wrote (concurrent requests, continuous batching, same RTX 4500 Ada); `llm_low` has no recorded GPU time (its run was client-bound, [p2_candidates.md](p2_candidates.md)), so it shows the batch-16 decode estimate below. Operators and LLMLingua-2 timings were not recorded (n/r).

Preservation by category:

| System | code | creative | explain | fact | reason | transform |
|---|---|---|---|---|---|---|
| Original | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| Best operator | 78.4% | 70.2% | 84.3% | 80.0% | 97.9% | 78.6% |
| Other operator stack | 64.7% | 64.9% | 77.1% | 73.6% | 96.9% | 74.5% |
| HMM compressor | 43.8% | 78.3% | 92.9% | 59.5% | 57.1% | 28.6% |
| LLMLingua-2 | 7.8% | 27.7% | 47.1% | 24.5% | 49.0% | 24.5% |
| Zero-shot 32B low | 57.8% | 39.4% | 48.6% | 63.6% | 83.3% | 56.1% |
| Zero-shot 32B medium | 52.9% | 34.0% | 60.0% | 65.5% | 77.1% | 55.1% |
| Zero-shot 32B high | 52.0% | 27.7% | 54.3% | 52.7% | 75.0% | 53.1% |
| Zero-shot 32B max | 55.9% | 30.9% | 44.3% | 52.7% | 75.0% | 42.9% |
| Zero-shot 32B min | 56.9% | 26.6% | 47.1% | 54.5% | 70.8% | 41.8% |
| SFT sft05 | 70.6% | 64.9% | 71.4% | 66.4% | 62.5% | 75.5% |
| SFT sft15 | 63.7% | 60.6% | 78.6% | 74.5% | 67.7% | 66.3% |
| SFT sft3b | 57.8% | 52.1% | 72.9% | 64.5% | 82.3% | 66.3% |
| SFT sft15v2 | 52.9% | 56.4% | 72.9% | 70.0% | 41.7% | 57.1% |
| SFT sft3bv2 | 48.0% | 66.0% | 67.1% | 72.7% | 71.9% | 66.3% |
| SFT sft7bv2 | 54.9% | 53.2% | 62.9% | 69.1% | 71.9% | 59.2% |

## Best operator (chosen on train)

Phase 2 judged two fixed operator stacks, not single operators, so the plan's "best single operator" is the better stack on the train split (1262 prompts) by the Phase 9 reward `100 * savings - 100 * 1[degraded]` per (prompt, target), fallbacks safe.

| Stack | Train median savings | Train preserved | Train reward | Chosen |
|---|---|---|---|---|
| `ops_FHC` | 0.4% | 83.8% | -13.80 | yes |
| `ops_CHFIDSL` | 1.7% | 77.2% | -18.72 |  |

## Zero-shot Qwen2.5-32B-Instruct-AWQ

No existing level is a plain "rewrite concisely" instruction. All `llm_*` candidates come from `src/terser/data/prompts/rewrite_v1.txt` (temperature 0, no examples): six preservation rules (keep the task, every constraint, tone cues, payload verbatim; same language; must be shorter) plus one aggressiveness paragraph per level. `low` removes filler and politeness; `medium` adds direct imperatives and merged repeats; `high` telegram style; `max` drops background and examples; `min` keyword style. A bare "rewrite concisely" prompt is closest in spirit to `medium` / `high` but would lack the preservation rules, so it would likely save more and preserve less; all five levels are reported and C8.4 is checked against each of them.

| Level | Mean output tokens | Run wall / prompt (s) | Batch-16 decode est. (s) | Batch-1 decode est. (s) |
|---|---|---|---|---|
| `llm_low` | 71.2 | n/r | 0.25 | 3.86 |
| `llm_medium` | 64.1 | 0.15 | 0.23 | 3.47 |
| `llm_high` | 52.8 | 0.21 | 0.19 | 2.86 |
| `llm_max` | 35.7 | 0.20 | 0.13 | 1.93 |
| `llm_min` | 66.5 | 0.29 | 0.24 | 3.60 |

Run wall times come from the summaries in `logs/aug_cand.log`, `logs/p2_candidates_2.log`. Estimates divide the mean output length by the measured decode throughput of the 32B AWQ on the RTX 4500 Ada (row 7.6: 18.47 tok/s at batch 1, 281.12 tok/s aggregate at batch 16); prefill is ignored, so they are lower bounds.

## C8.4: vs zero-shot Qwen2.5-32B-AWQ

Pass needs a level the rung preserves at least as much as (at most 2 pp less), a savings gap (level − rung) ≤ 15 pp and a batch-1 latency ratio ≥ 5× (targets revised by ADR-0012; originally ±1 pp preservation, 5 pp and 10× on the primary latency). Latency ratio uses the primary 32B latency (run wall per prompt, else batch-16 estimate) against the rung's batch-16 latency; the batch-1 column divides the 32B batch-1 estimate by the same rung latency.

| Rung | Level | Δ preservation (rung − level) | Savings gap (level − rung) | Latency ratio | Batch-1 ratio | Preserves ≥ level − 2 pp | Pass |
|---|---|---|---|---|---|---|---|
| sft05 | `llm_low` | +9.6 pp | +36.0 pp | 0.66× | 10.0× | yes | fail |
| sft05 | `llm_medium` | +10.9 pp | +43.8 pp | 0.39× | 9.0× | yes | fail |
| sft05 | `llm_high` | +16.0 pp | +51.1 pp | 0.56× | 7.4× | yes | fail |
| sft05 | `llm_max` | +17.7 pp | +62.3 pp | 0.50× | 5.0× | yes | fail |
| sft05 | `llm_min` | +18.4 pp | +48.1 pp | 0.74× | 9.3× | yes | fail |
| sft15 | `llm_low` | +9.5 pp | +29.9 pp | 0.50× | 7.6× | yes | fail |
| sft15 | `llm_medium` | +10.7 pp | +37.7 pp | 0.30× | 6.9× | yes | fail |
| sft15 | `llm_high` | +15.8 pp | +45.0 pp | 0.42× | 5.7× | yes | fail |
| sft15 | `llm_max` | +17.5 pp | +56.2 pp | 0.39× | 3.8× | yes | fail |
| sft15 | `llm_min` | +18.2 pp | +42.0 pp | 0.57× | 7.1× | yes | fail |
| sft3b | `llm_low` | +6.8 pp | +14.0 pp | 0.36× | 5.5× | yes | pass |
| sft3b | `llm_medium` | +8.1 pp | +21.8 pp | 0.21× | 5.0× | yes | fail |
| sft3b | `llm_high` | +13.2 pp | +29.1 pp | 0.31× | 4.1× | yes | fail |
| sft3b | `llm_max` | +14.9 pp | +40.3 pp | 0.28× | 2.8× | yes | fail |
| sft3b | `llm_min` | +15.6 pp | +26.1 pp | 0.41× | 5.2× | yes | fail |
| sft15v2 | `llm_low` | -0.7 pp | +17.5 pp | 0.52× | 8.0× | yes | fail |
| sft15v2 | `llm_medium` | +0.5 pp | +25.3 pp | 0.31× | 7.2× | yes | fail |
| sft15v2 | `llm_high` | +5.6 pp | +32.6 pp | 0.44× | 5.9× | yes | fail |
| sft15v2 | `llm_max` | +7.4 pp | +43.8 pp | 0.40× | 4.0× | yes | fail |
| sft15v2 | `llm_min` | +8.1 pp | +29.6 pp | 0.59× | 7.4× | yes | fail |
| sft3bv2 | `llm_low` | +6.5 pp | +15.9 pp | 0.35× | 5.3× | yes | fail |
| sft3bv2 | `llm_medium` | +7.7 pp | +23.7 pp | 0.21× | 4.8× | yes | fail |
| sft3bv2 | `llm_high` | +12.8 pp | +31.0 pp | 0.30× | 3.9× | yes | fail |
| sft3bv2 | `llm_max` | +14.6 pp | +42.2 pp | 0.27× | 2.7× | yes | fail |
| sft3bv2 | `llm_min` | +15.3 pp | +28.0 pp | 0.39× | 5.0× | yes | fail |
| sft7bv2 | `llm_low` | +3.2 pp | +4.3 pp | 0.20× | 3.1× | yes | fail |
| sft7bv2 | `llm_medium` | +4.4 pp | +12.1 pp | 0.12× | 2.8× | yes | fail |
| sft7bv2 | `llm_high` | +9.5 pp | +19.3 pp | 0.17× | 2.3× | yes | fail |
| sft7bv2 | `llm_max` | +11.2 pp | +30.6 pp | 0.16× | 1.6× | yes | fail |
| sft7bv2 | `llm_min` | +11.9 pp | +16.3 pp | 0.23× | 2.9× | yes | fail |

| Rung | Closest level | 32B savings at rung's preservation (interp.) | C8.4 |
|---|---|---|---|
| sft05 | `llm_low` | outside the 32B range | fail |
| sft15 | `llm_low` | outside the 32B range | fail |
| sft3b | `llm_low` | outside the 32B range | pass |
| sft15v2 | `llm_medium` | 44.6% (gap +22.0 pp) | fail |
| sft3bv2 | `llm_low` | outside the 32B range | fail |
| sft7bv2 | `llm_low` | outside the 32B range | fail |

**C8.4: pass.** sft05: matched levels llm_low, llm_medium, llm_high, llm_max, llm_min, closest `llm_low` (+9.6 pp preservation, +36.0 pp savings gap), best latency ratio 0.74×; sft15: matched levels llm_low, llm_medium, llm_high, llm_max, llm_min, closest `llm_low` (+9.5 pp preservation, +29.9 pp savings gap), best latency ratio 0.57×; sft3b: matched levels llm_low, llm_medium, llm_high, llm_max, llm_min, closest `llm_low` (+6.8 pp preservation, +14.0 pp savings gap), best latency ratio 0.41×; sft15v2: matched levels llm_low, llm_medium, llm_high, llm_max, llm_min, closest `llm_medium` (+0.5 pp preservation, +25.3 pp savings gap), best latency ratio 0.59×; sft3bv2: matched levels llm_low, llm_medium, llm_high, llm_max, llm_min, closest `llm_low` (+6.5 pp preservation, +15.9 pp savings gap), best latency ratio 0.39×; sft7bv2: matched levels llm_low, llm_medium, llm_high, llm_max, llm_min, closest `llm_low` (+3.2 pp preservation, +4.3 pp savings gap), best latency ratio 0.23×. No level of the provisional 32B baseline is a plain "rewrite concisely" prompt, and SFT and 32B latencies come from different batching set-ups, so a batch-1 latency benchmark of both on the same GPU is still needed for a final latency call.

## C8.5: vs LLMLingua-2

Pass needs higher preservation than LLMLingua-2 (not dominated; revised by ADR-0012, originally also equal or better median savings).

| Rung | Δ preservation (rung − LLMLingua-2) | Δ median savings | C8.5 |
|---|---|---|---|
| sft05 | +39.5 pp | -46.6 pp | pass |
| sft15 | +39.3 pp | -40.5 pp | pass |
| sft3b | +36.7 pp | -24.6 pp | pass |
| sft15v2 | +29.1 pp | -28.1 pp | pass |
| sft3bv2 | +36.3 pp | -26.5 pp | pass |
| sft7bv2 | +33.0 pp | -14.9 pp | pass |

**C8.5: pass.** sft05: preservation +39.5 pp, median savings -46.6 pp; sft15: preservation +39.3 pp, median savings -40.5 pp; sft3b: preservation +36.7 pp, median savings -24.6 pp; sft15v2: preservation +29.1 pp, median savings -28.1 pp; sft3bv2: preservation +36.3 pp, median savings -26.5 pp; sft7bv2: preservation +33.0 pp, median savings -14.9 pp.
