# Phase 8 baselines on the rewriter val table (row 8.3.9, checkpoints C8.4 / C8.5)

Same 285 val prompts and metric definitions as [p8_rewriter_val.md](p8_rewriter_val.md) (`val_report.rung_metrics`): median savings on the Qwen2.5 tokenizer over all prompts (fallbacks count 0%); preservation pooled over (prompt, target) for both targets, judged by Qwen2.5-32B-Instruct-AWQ (`pairwise_avg`), fallbacks counted as preserved. Baselines reuse the judged Phase 2 candidates; nothing was regenerated. A candidate that was rejected in row 2.2.1 or is not strictly shorter on Qwen2.5 falls back to the original, like the 8.3.7 safeguard of the SFT rungs.

| System | Source | n | Median savings | Fell back | Preserved | Preserved Qwen7B | Preserved Phi3.5 | Latency (s/prompt) | C8.3 |
|---|---|---|---|---|---|---|---|---|---|
| Original | identity | 285 | 0.0% | 285 | 100.0% | 100.0% | 100.0% | 0.00 | fail |
| Best operator | `ops_FHC` | 285 | 0.3% | 138 | 81.4% | 81.8% | 81.1% | n/r | fail |
| Other operator stack | `ops_CHFIDSL` | 285 | 1.2% | 101 | 75.1% | 75.1% | 75.1% | n/r | fail |
| HMM compressor | row 5.3.6 not judged | pending | pending | pending | pending | pending | pending | pending | — |
| LLMLingua-2 | `llmlingua2` | 285 | 50.7% | 1 | 28.9% | 30.2% | 27.7% | n/r | fail |
| Zero-shot 32B low | `llm_low` | 285 | 40.1% | 24 | 58.8% | 64.9% | 52.6% | 0.25 | fail |
| Zero-shot 32B medium | `llm_medium` | 285 | 47.9% | 26 | 57.5% | 62.8% | 52.3% | 0.15 | fail |
| Zero-shot 32B high | `llm_high` | 285 | 55.2% | 24 | 52.3% | 54.7% | 49.8% | 0.21 | fail |
| Zero-shot 32B max | `llm_max` | 285 | 66.4% | 25 | 50.7% | 55.1% | 46.3% | 0.20 | fail |
| Zero-shot 32B min | `llm_min` | 285 | 52.2% | 19 | 50.0% | 53.0% | 47.0% | 0.29 | fail |
| SFT sft05 | `sft05` | 285 | 7.6% | 98 | 67.4% | 70.9% | 63.9% | 0.33 | fail |
| SFT sft15 | `sft15` | 285 | 7.6% | 101 | 69.3% | 69.8% | 68.8% | 0.50 | fail |
| SFT sft3b | `sft3b` | 285 | 2.1% | 130 | 74.7% | 78.6% | 70.9% | 0.76 | fail |
| SFT sft15v2 | `sft15v2` | 285 | 35.0% | 72 | 57.0% | 61.1% | 53.0% | 0.44 | fail |
| SFT sft3bv2 | `sft3bv2` | 285 | 12.5% | 104 | 71.4% | 75.1% | 67.7% | 0.98 | fail |

Latency: SFT rungs are the mean generation time per prompt in batches of 16 (HF `generate`, RTX 4500 Ada). Zero-shot 32B levels are the measured wall time of the vLLM generation run divided by the prompts it wrote (concurrent requests, continuous batching, same RTX 4500 Ada); `llm_low` has no recorded GPU time (its run was client-bound, [p2_candidates.md](p2_candidates.md)), so it shows the batch-16 decode estimate below. Operators and LLMLingua-2 timings were not recorded (n/r).

Preservation by category:

| System | code | creative | explain | fact | reason | transform |
|---|---|---|---|---|---|---|
| Original | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| Best operator | 78.4% | 70.2% | 84.3% | 80.0% | 97.9% | 78.6% |
| Other operator stack | 64.7% | 64.9% | 77.1% | 73.6% | 96.9% | 74.5% |
| HMM compressor | pending | pending | pending | pending | pending | pending |
| LLMLingua-2 | 7.8% | 27.7% | 47.1% | 24.5% | 49.0% | 24.5% |
| Zero-shot 32B low | 57.8% | 39.4% | 48.6% | 63.6% | 83.3% | 56.1% |
| Zero-shot 32B medium | 52.9% | 34.0% | 60.0% | 65.5% | 77.1% | 55.1% |
| Zero-shot 32B high | 52.0% | 27.7% | 54.3% | 51.8% | 75.0% | 53.1% |
| Zero-shot 32B max | 55.9% | 30.9% | 44.3% | 52.7% | 75.0% | 42.9% |
| Zero-shot 32B min | 56.9% | 26.6% | 47.1% | 54.5% | 70.8% | 41.8% |
| SFT sft05 | 71.6% | 53.2% | 64.3% | 74.5% | 71.9% | 66.3% |
| SFT sft15 | 58.8% | 69.1% | 62.9% | 66.4% | 82.3% | 75.5% |
| SFT sft3b | 57.8% | 75.5% | 85.7% | 80.0% | 82.3% | 70.4% |
| SFT sft15v2 | 48.0% | 54.3% | 65.7% | 68.2% | 44.8% | 62.2% |
| SFT sft3bv2 | 61.8% | 70.2% | 80.0% | 80.0% | 63.5% | 74.5% |

## Best operator (chosen on train)

Phase 2 judged two fixed operator stacks, not single operators, so the plan's "best single operator" is the better stack on the train split (1262 prompts) by the Phase 9 reward `100 * savings - 100 * 1[degraded]` per (prompt, target), fallbacks safe.

| Stack | Train median savings | Train preserved | Train reward | Chosen |
|---|---|---|---|---|
| `ops_FHC` | 0.4% | 83.8% | -13.76 | yes |
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

Pass needs a level within ±1 pp preservation, a savings gap (level − rung) ≤ 5 pp and a latency ratio (level / rung) ≥ 10×. Latency ratio uses the primary 32B latency (run wall per prompt, else batch-16 estimate) against the rung's batch-16 latency; the batch-1 column divides the 32B batch-1 estimate by the same rung latency (different batching, indicative only).

| Rung | Level | Δ preservation (rung − level) | Savings gap (level − rung) | Latency ratio | Batch-1 ratio | Matched ±1 pp | Pass |
|---|---|---|---|---|---|---|---|
| sft05 | `llm_low` | +8.6 pp | +32.5 pp | 0.76× | 11.6× | no | fail |
| sft05 | `llm_medium` | +9.8 pp | +40.3 pp | 0.45× | 10.4× | no | fail |
| sft05 | `llm_high` | +15.1 pp | +47.6 pp | 0.64× | 8.6× | no | fail |
| sft05 | `llm_max` | +16.7 pp | +58.8 pp | 0.58× | 5.8× | no | fail |
| sft05 | `llm_min` | +17.4 pp | +44.6 pp | 0.86× | 10.8× | no | fail |
| sft15 | `llm_low` | +10.5 pp | +32.5 pp | 0.50× | 7.7× | no | fail |
| sft15 | `llm_medium` | +11.8 pp | +40.3 pp | 0.30× | 6.9× | no | fail |
| sft15 | `llm_high` | +17.0 pp | +47.6 pp | 0.43× | 5.7× | no | fail |
| sft15 | `llm_max` | +18.6 pp | +58.8 pp | 0.39× | 3.8× | no | fail |
| sft15 | `llm_min` | +19.3 pp | +44.6 pp | 0.57× | 7.2× | no | fail |
| sft3b | `llm_low` | +16.0 pp | +38.0 pp | 0.33× | 5.1× | no | fail |
| sft3b | `llm_medium` | +17.2 pp | +45.8 pp | 0.20× | 4.6× | no | fail |
| sft3b | `llm_high` | +22.5 pp | +53.1 pp | 0.28× | 3.8× | no | fail |
| sft3b | `llm_max` | +24.0 pp | +64.3 pp | 0.26× | 2.5× | no | fail |
| sft3b | `llm_min` | +24.7 pp | +50.1 pp | 0.38× | 4.8× | no | fail |
| sft15v2 | `llm_low` | -1.8 pp | +5.1 pp | 0.57× | 8.7× | no | fail |
| sft15v2 | `llm_medium` | -0.5 pp | +12.9 pp | 0.34× | 7.9× | yes | fail |
| sft15v2 | `llm_high` | +4.7 pp | +20.2 pp | 0.49× | 6.5× | no | fail |
| sft15v2 | `llm_max` | +6.3 pp | +31.4 pp | 0.44× | 4.4× | no | fail |
| sft15v2 | `llm_min` | +7.0 pp | +17.2 pp | 0.65× | 8.2× | no | fail |
| sft3bv2 | `llm_low` | +12.6 pp | +27.6 pp | 0.26× | 3.9× | no | fail |
| sft3bv2 | `llm_medium` | +13.9 pp | +35.4 pp | 0.15× | 3.6× | no | fail |
| sft3bv2 | `llm_high` | +19.1 pp | +42.6 pp | 0.22× | 2.9× | no | fail |
| sft3bv2 | `llm_max` | +20.7 pp | +53.9 pp | 0.20× | 2.0× | no | fail |
| sft3bv2 | `llm_min` | +21.4 pp | +39.6 pp | 0.29× | 3.7× | no | fail |

| Rung | Closest level | 32B savings at rung's preservation (interp.) | C8.4 |
|---|---|---|---|
| sft05 | `llm_low` | outside the 32B range | fail |
| sft15 | `llm_low` | outside the 32B range | fail |
| sft3b | `llm_low` | outside the 32B range | fail |
| sft15v2 | `llm_medium` | 48.6% (gap +13.6 pp) | fail |
| sft3bv2 | `llm_low` | outside the 32B range | fail |

**C8.4: fail.** sft05: matched levels none, closest `llm_low` (+8.6 pp preservation, +32.5 pp savings gap), best latency ratio 0.86×; sft15: matched levels none, closest `llm_low` (+10.5 pp preservation, +32.5 pp savings gap), best latency ratio 0.57×; sft3b: matched levels none, closest `llm_low` (+16.0 pp preservation, +38.0 pp savings gap), best latency ratio 0.38×; sft15v2: matched levels llm_medium, closest `llm_medium` (-0.5 pp preservation, +12.9 pp savings gap), best latency ratio 0.65×; sft3bv2: matched levels none, closest `llm_low` (+12.6 pp preservation, +27.6 pp savings gap), best latency ratio 0.29×. No level of the provisional 32B baseline is a plain "rewrite concisely" prompt, and SFT and 32B latencies come from different batching set-ups, so a batch-1 latency benchmark of both on the same GPU is still needed for a final latency call.

## C8.5: vs LLMLingua-2

Pass needs higher preservation than LLMLingua-2 and equal or better median savings.

| Rung | Δ preservation (rung − LLMLingua-2) | Δ median savings | C8.5 |
|---|---|---|---|
| sft05 | +38.4 pp | -43.1 pp | fail |
| sft15 | +40.4 pp | -43.1 pp | fail |
| sft3b | +45.8 pp | -48.6 pp | fail |
| sft15v2 | +28.1 pp | -15.7 pp | fail |
| sft3bv2 | +42.5 pp | -38.2 pp | fail |

**C8.5: fail.** sft05: preservation +38.4 pp, median savings -43.1 pp; sft15: preservation +40.4 pp, median savings -43.1 pp; sft3b: preservation +45.8 pp, median savings -48.6 pp; sft15v2: preservation +28.1 pp, median savings -15.7 pp; sft3bv2: preservation +42.5 pp, median savings -38.2 pp.
