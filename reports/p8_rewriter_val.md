# Phase 8 rewriter val report (rows 8.3.3 / 8.3.4, gate C8.3)

Val split, greedy decoding, 8.3.7 safeguard on. Savings on the Qwen2.5 tokenizer; preservation judged by Qwen2.5-32B-Instruct-AWQ (`pairwise_avg`) on every target.

C8.3 as revised by [ADR-0012](../docs/decisions/ADR-0012-feasible-checkpoint-targets.md): median savings ≥ 25%, preservation ≥ 65% overall and ≥ 50% in every category, gate + fallback on (originally ≥ 92% overall and ≥ 85% per category). sft3b passes (26.1%, 65.6%, worst category creative 52.1%); every other rung misses at least one bar.

| Rung | n | Median savings | Oracle median | Fell back | Preserved | Latency (s) | C8.3 |
|---|---|---|---|---|---|---|---|
| sft05 | 285 | 4.1% | 10.0% | 108 | 68.4% | 0.39 | fail |
| sft15 | 285 | 10.2% | 10.0% | 99 | 68.2% | 0.50 | fail |
| sft3b | 285 | 26.1% | 10.0% | 66 | 65.6% | 0.70 | pass |
| sft15v2 | 285 | 22.6% | 10.0% | 73 | 58.1% | 0.49 | fail |
| sft3bv2 | 285 | 24.2% | 10.0% | 70 | 65.3% | 0.72 | fail |
| sft7bv2 | 285 | 35.8% | 10.0% | 63 | 61.9% | 1.24 | fail |

Preservation by category:

| Rung | code | creative | explain | fact | reason | transform |
|---|---|---|---|---|---|---|
| sft05 | 70.6% | 64.9% | 71.4% | 66.4% | 62.5% | 75.5% |
| sft15 | 63.7% | 60.6% | 78.6% | 74.5% | 67.7% | 66.3% |
| sft3b | 57.8% | 52.1% | 72.9% | 64.5% | 82.3% | 66.3% |
| sft15v2 | 52.9% | 56.4% | 72.9% | 70.0% | 41.7% | 57.1% |
| sft3bv2 | 48.0% | 66.0% | 67.1% | 72.7% | 71.9% | 66.3% |
| sft7bv2 | 54.9% | 53.2% | 62.9% | 69.1% | 71.9% | 59.2% |
