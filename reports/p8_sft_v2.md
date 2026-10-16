# SFT data v2: augmented rewrite targets (row 8.3.1)

The first SFT set (`data/processed/sft`) capped target savings near 10%: the shortest preserved
candidate per prompt was usually a light operator or LLM rewrite, so every rung learned to save
about 7.6% (see [p8_rewriter_val.md](p8_rewriter_val.md)). `sft_v2` adds two aggressive LLM
rewrite strengths, `llm_max` and `llm_min`, generated with Qwen2.5-32B-Instruct-AWQ
([configs/experiment/p8_target_aug.yaml](../configs/experiment/p8_target_aug.yaml)) for the
train and val prompts only (`exclude_splits: [test]`; the test split is never read).

## Judging the new candidates

Both targets (Qwen2.5-7B-Instruct, Phi-3.5-mini-instruct) answered every new candidate; the
Qwen2.5-32B-AWQ judge labelled them with the same `pairwise_avg` protocol as Phase 2
(`python -m terser.phase2.label_stage`, log `logs/aug_judge.log`).

| | value |
|---|---|
| label records | 5,770 (5,733 judged, 37 identical to the original) |
| judge calls | 15,294 (3,402 from cache) |
| raw A/B disagreements | 3,247 |
| low-margin verdicts | 1,439 |
| wall time | 2 h 35 min |

| strength | preserved | degraded | preserved share |
|---|---|---|---|
| `llm_max` | 1,264 | 1,584 | 44.4% |
| `llm_min` | 1,346 | 1,576 | 46.1% |

## Resulting SFT set

`python -m terser.models.rewriter.sft_data --candidates candidates.jsonl candidates_aug.jsonl
--labels labels.jsonl labels_aug.jsonl ...` keeps, per prompt, the shortest candidate that is
preserved on both targets, else the original prompt (identity target).

| split | prompts | identity targets | median target savings |
|---|---|---|---|
| train | 1,262 | 412 (32.6%) | 21.4% |
| val | 285 | 101 (35.4%) | 17.4% |

Target source on train: original 412, `llm_min` 194, `llm_high` 172, `llm_max` 128,
`ops_CHFIDSL` 108, `llm_low` 98, `llm_medium` 74, `llmlingua2` 51.

**Reading.** The augmentation doubles the median target savings (about 10% to 21.4% on train),
but more than half of the aggressive rewrites degrade at least one target, and a third of the
prompts still have no preserved rewrite. Even a rewriter that reproduced the val targets
exactly would reach a median of 17.4%, below the C8.3 bar of 25%, so the targets, not the model
size, remain the binding limit on C8.3. Both `data/processed/labels_aug.jsonl` and
`data/processed/sft_v2` are tracked with DVC.
