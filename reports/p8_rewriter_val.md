# Phase 8 rewriter val report (rows 8.3.3 / 8.3.4, gate C8.3)

Val split, greedy decoding, 8.3.7 safeguard on. Savings on the Qwen2.5 tokenizer; preservation judged by Qwen2.5-32B-Instruct-AWQ (`pairwise_avg`) on every target.

| Rung | n | Median savings | Oracle median | Fell back | Preserved | Latency (s) | C8.3 |
|---|---|---|---|---|---|---|---|
| sft05 | 285 | 7.6% | 10.0% | 98 | 67.4% | 0.33 | fail |
| sft15 | 285 | 7.6% | 10.0% | 101 | 69.3% | 0.50 | fail |
| sft3b | 285 | 2.1% | 10.0% | 130 | 74.7% | 0.76 | fail |
| sft15v2 | 285 | 35.0% | 10.0% | 72 | 57.0% | 0.44 | fail |
| sft3bv2 | 285 | 12.5% | 10.0% | 104 | 71.4% | 0.98 | fail |

Preservation by category:

| Rung | code | creative | explain | fact | reason | transform |
|---|---|---|---|---|---|---|
| sft05 | 71.6% | 53.2% | 64.3% | 74.5% | 71.9% | 66.3% |
| sft15 | 58.8% | 69.1% | 62.9% | 66.4% | 82.3% | 75.5% |
| sft3b | 57.8% | 75.5% | 85.7% | 80.0% | 82.3% | 70.4% |
| sft15v2 | 48.0% | 54.3% | 65.7% | 68.2% | 44.8% | 62.2% |
| sft3bv2 | 61.8% | 70.2% | 80.0% | 80.0% | 63.5% | 74.5% |
