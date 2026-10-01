# 2.2.1 Candidate generation

## Labelled subset (scope)

Judging all 6 candidates of all 7,080 prompts in both A/B orders would take ~85k calls to the 32B judge
(~100M tokens, ~14 GPU-h at the 1.7 calls/s measured in 2.3.3), above the phase2 cap of 60M tokens in
`configs/budget.yaml` (9.8M were already spent on casualisation and the judge-bias study). The
`candidates → respond → judge → label` chain therefore runs on a deterministic subset
(`terser.data.label_subset`, DVC stage `label_subset`): **300 prompts per category = 1,800 prompts**.
The subset takes one casual variant of every source group first, so it covers all **1,665** source
groups. It keeps the frozen splits: train 1,262 / val 285 / test 253. Raising
`label_subset.per_category` extends the set, and the first *n* prompts of a larger subset are the
subset for *n*. Planned phase2 total ≈ 48M tokens.

## Output

`data/interim/candidates.jsonl` (DVC stage `candidates`): 10,800 candidates, 6 per prompt, 0 stand-ins.

| Method | Generator | Candidates | Rejected | Main reason |
| --- | --- | ---: | ---: | --- |
| llm_low | Qwen2.5-32B-Instruct-AWQ, `rewrite_v1`, T = 0 | 1,800 | 75 | payload altered (61) |
| llm_medium | same | 1,800 | 108 | payload altered (90) |
| llm_high | same | 1,800 | 124 | payload altered (119) |
| ops_FHC | operator stack | 1,800 | 0 | |
| ops_CHFIDSL | operator stack | 1,800 | 0 | |
| llmlingua2 | `llmlingua-2-bert-base-multilingual-cased-meetingbank`, rate 0.5, CPU | 1,800 | 0 | |

Rejected rewrites stay in the file (`rejected: true`) and are excluded downstream. LLM wall time:
low ≈ 50 min (client-bound before the ledger fix in `2dce55b`), medium 269 s, high 386 s.
