# 2.1.7 DVC pipeline: reproducibility check

Pipeline `collect → casualise → dedupe → split` (`dvc.yaml`, parameters in `params.yaml`).

## Check

1. `dvc repro split` built every stage (casualise through Qwen2.5-32B-Instruct-AWQ on vLLM).
2. `TERSER_LOCAL_LLM_URL=http://127.0.0.1:9/v1 dvc repro -f split` forced **all four stages** to re-run
   with the LLM server unreachable, so casualise could only read `.cache/llm_responses`.
   Wall time 16 s on CPU (log: `logs/dvc_repro_offline.log`).
3. Every output md5 in `dvc.lock` was identical before and after. The only `dvc.lock` change
   was the `split.test_sha256` parameter, which now pins the frozen test split.

## Dataset

| Stage | Result |
| --- | --- |
| collect | 1,680 clean prompts (280 per category); 24 PII-flagged rows dropped |
| casualise | 7,108 accepted, 1,292 rejected by the meaning check |
| dedupe | 7,080 records, 6 clusters removed; near-duplicate rate 0.39% → 0.00% (`reports/p2_duplicates.md`) |
| split | train 4,955 / val 1,133 / test 992 records; 1,165 / 266 / 234 source groups; 0 shared groups; 0 cross-split near-duplicates |

After dedupe, per category: code 1,283 · creative 1,062 · explain 1,108 · fact 1,229 · reason 1,185 · transform 1,213.

Frozen test split: `data/processed/splits/test.jsonl`, sha256
`b4b264a80bcf8b1c415643b5f54156d214841e77088b75ae8b148bf2c9e106c7` (`test.lock.json`). `split` passes
`--expect-test-sha256`, so any rebuild that changes the test split fails.

## Checkpoints

- C2.1 ✅ ≥ 6,000 casual prompts (7,080), ≥ 800 per category (min 1,062), < 1% near-duplicates (0.00%).
- C2.2 ✅ 0 source prompts shared across splits (checked inside the `split` stage and by `tests/test_data_splits.py`).
- C2.9 stays open: it covers the full dataset including the `respond`/`judge`/`label` stages, which are not yet DVC stages.

## Known stale stage: collect

`dvc status` reports `collect` as changed: `src/terser/data/collect.py` was edited after the
lock was written (2.1.3, commit `54616c1`, 2026-10-03: PII is now scrubbed at the collection
boundary instead of the row being dropped), and `pii.py` has had formatting-only edits since.
Re-running the current code gives a different `clean.jsonl` (md5 `6a7c9567…` vs the locked
`261be958…`; 22,272 eligible rows instead of 22,248, because the 24 PII rows are now scrubbed
and kept). Every downstream stage, the frozen test split (sha256 above) and all labels were
built from the locked `clean.jsonl`, so `collect` is deliberately **not** re-run: `split` would
fail its `--expect-test-sha256` check and every label would need regenerating. The locked
dataset dropped all 24 PII-flagged rows, so it carries no unscrubbed PII.
[p2_collect_stats.json](p2_collect_stats.json) is the locked stage's output: its `pii_flagged`
key (those 24 dropped rows) is not produced by the current `collect.py`, which reports
`pii:<type>` counts for scrubbed rows instead.
`dvc repro --dry --allow-missing` (CI) still passes. A future dataset version should re-run the
whole pipeline from `collect` and re-freeze the test split.
