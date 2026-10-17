# C2.9 / 11.1.2: dataset reproducibility from cached LLM responses

Date 2026-10-17. Fresh clone of `~/Terser` at `2442057` in `/home/temp/terser_repro` (deleted
afterwards), CPU only (`CUDA_VISIBLE_DEVICES=""`, `torch.cuda.is_available() == False`), no LLM
server. Every LLM call ran in the new **cache-only mode** (`TERSER_LLM_CACHE_ONLY=1`, see below),
so a cache miss stops the stage instead of reaching a model. As a second guard,
`TERSER_LOCAL_LLM_URL` pointed at an unreachable port, and `HF_HUB_OFFLINE=1` was set.

## Setup

- DVC: the clone uses the same `local` remote as `~/Terser` (`../../terser-dvc-remote`, which
  resolves to `/home/temp/terser-dvc-remote` from both checkouts). `dvc pull -r local` restored
  every pipeline output in `dvc.lock`. `labels.jsonl`, `oracle.csv`, `pair_features.parquet` and
  `span_features.parquet` are **not on the remote**. They were pulled read-only from
  `~/Terser/.dvc/cache`, added to the clone as a second, `--local` remote.
- Raw data (`data/raw/{dolly,oasst1,humaneval,mbpp,gsm8k}`) is in **neither** the remote nor
  `~/Terser/.dvc/cache`, so `dvc pull` fails for those five `.dvc` files. The directories were
  copied from the `~/Terser` workspace. In the clone, `dvc status` then reports no change for the
  five raw `.dvc` files or for the `collect` raw-data deps, so the copied data matches what the
  lock recorded.
- LLM caches: `.cache/llm_responses` (138,289 entries, 4.2 GB), `.cache/llmlingua2`,
  `.cache/token_counts` and `.cache/task_metrics` were copied into the clone. `~/Terser` was only
  read.
- Cached responses are keyed by sha256 of the canonical request (`provider`, `model`, `messages`,
  `params` including `max_tokens`, `system`) in `terser.llm.cache`. The `.cache/` folder is
  git-ignored and not tracked by DVC, so the response cache is local to the workstation. It is the
  one input that `dvc pull` cannot restore (see Gaps).

## Cache-only switch (new)

Until now the only "offline" guard was an unreachable `TERSER_LOCAL_LLM_URL` (used in
`reports/p2_dvc_repro.md`). That guard would not stop a hosted provider, or a vLLM server that
happens to be running. `terser.llm.LLMClient` now has a cache-only mode:
`LLMClient(cache_only=True)`, or by default the env var `TERSER_LLM_CACHE_ONLY=1|true|yes|on`.
In this mode a cache hit is served as before. A miss raises `LLMCacheMissError` before the budget
check, the transport and the ledger are touched. No stage catches the error, so `dvc repro` stops
on the first miss. It is off by default, so existing behaviour is unchanged.

Checked in the clone:

- Negative test: `casualise --limit 1 --temperature 0.81` (a request that cannot be in the cache)
  exits 1 with `LLMCacheMissError`.
- First attempt, before the cache copy had run: `dvc repro -f -s casualise` stopped with
  `LLMCacheMissError: cache miss for local/Qwen/Qwen2.5-32B-Instruct-AWQ (key ad38ad02…)`
  instead of calling a model.

## Per-stage result (`dvc repro -f -s <stage>`, in pipeline order)

The md5 values are the output hashes in `~/Terser/dvc.lock`, compared with the hashes the forced
rerun wrote. After all stages ran, the clone's rewritten `dvc.lock` was **byte-identical** to
`~/Terser/dvc.lock`.

| stage | output | locked md5 | rerun md5 | LLM calls (cache hits) | wall | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| collect | `data/interim/clean.jsonl` | `261be958…` | current code: `6a7c9567…`; locked code: `261be958…` | none | ~1 min | stale by design (below) |
| collect | `reports/p2_collect_stats.json` | `2039173f…` | current code: `ef6a4502…`; locked code: `2039173f…` | none | | stale by design |
| casualise | `data/interim/casual.jsonl` | `11a3c5f1…` | `11a3c5f1…` | all hits, 0 misses | 7 s | identical |
| casualise | `data/interim/casual_rejected.jsonl` | `5e589872…` | `5e589872…` | | | identical |
| casualise | `reports/p2_casualise_stats.json` | `8ef7f0bd…` | `8ef7f0bd…` | | | identical |
| dedupe | `data/interim/casual_dedup.jsonl` | `c501bd42…` | `c501bd42…` | none | 3 s | identical |
| dedupe | `reports/p2_duplicates.md` | `0d4a10e0…` | `0d4a10e0…` | | | identical |
| split | `data/processed/splits` (5 files) | `d14a0978….dir` | `d14a0978….dir` | none | 2 s | identical; the `--expect-test-sha256` frozen test check passed |
| label_subset | `data/interim/label_prompts.jsonl` | `33492387…` | `33492387…` | none | <1 s | identical |
| candidates | `data/interim/candidates.jsonl` | `3a84861b…` | `3a84861b…` | rewrites all hits; LLMLingua-2 from `.cache/llmlingua2` | 26 s | identical |
| token_counts | `data/interim/token_counts.parquet` | `b2fe3fd1…` | `b2fe3fd1…` | none | 1 s | identical |
| register | `reports/p11_registry.md` | | not run | | | out of scope: model registry, not dataset; needs the `models/*` DVC outs |

The rows of 11.1.2 that are **not** DVC stages were rerun by hand in the clone, with the commands
from `reports/p11_flows.md` / MLflow run params (`n_prompts = 1800`, so `--prompts` is
`label_prompts.jsonl`), in cache-only mode:

| step (11.1.2 name) | command | reference md5 | rerun md5 | LLM calls (cache hits) | wall | verdict |
| --- | --- | --- | --- | --- | --- | --- |
| respond (Qwen2.5-7B) | `terser.phase2.respond --target Qwen/Qwen2.5-7B-Instruct` | `30a74339…` (`~/Terser/data/interim/responses/qwen2.5-7b.jsonl`, not DVC-tracked) | `30a74339…` | 13,100 / 13,100 | 4 s | identical |
| respond (Phi-3.5-mini) | `terser.phase2.respond --target microsoft/Phi-3.5-mini-instruct` | `76112b2a…` (workspace file, not DVC-tracked) | `76112b2a…` | 13,100 / 13,100 | 4 s | identical |
| judge + label | `terser.phase2.label_stage --judge-model Qwen/Qwen2.5-32B-Instruct-AWQ --splits-dir data/processed/splits` | `c5a38491…` (`data/processed/labels.jsonl.dvc`) | `c5a38491…` | 41,986 / 41,986 | 44 s | **identical**: 20,986 records, the same counts as `reports/p2_label_stats.json` |
| features (span) | `terser.phase3.build_features --labels <rerun labels>` (CPU) | `0fbf27fc…` (`span_features.parquet.dvc`) | `0fbf27fc…` | none | 263 s (both tables) | identical |
| features (pair) | same run | `3310cfab…` (`pair_features.parquet.dvc`) | `e8e6f6ee…` | none | | the 32 committed columns are **exactly equal**. The current builder adds one column, `overhead_pct` (commit `8dbefd0`, written after the table was last committed in `454fa98`), so the committed table is behind the code. This is not a non-determinism issue |

## collect: stale by design

`dvc status` reports `collect` as changed because `collect.py` (2.1.3 scrub-instead-of-drop,
`54616c1`) and `pii.py` (`db50797`, `6c433db`, `fba61d8`) were edited after the lock was written.
Running collect with the **current** code gives `clean.jsonl` `6a7c9567…` (22,272 eligible rows:
the 24 PII rows are now scrubbed and kept, which changes which rows are sampled). Running it with
the **locked** code (`src/` at `9092ee6`, whose `collect.py` / `pii.py` / `sources.py` md5s are
exactly `f6dd2f96` / `5af62892` / `d054862d` from `dvc.lock`) on the same raw data gives
`clean.jsonl` `261be958…` and `p2_collect_stats.json` `2039173f…`. Both are **identical to the
lock**. So collect itself is deterministic, and the mismatch is fully explained by the deliberate
code change. collect was not rerun through DVC: `split --expect-test-sha256` would reject the new
dataset, and every label would have to be regenerated (see `reports/p2_dvc_repro.md`). Every other
stage was forced to rerun with `-s` on top of the locked `clean.jsonl`.

## Verdicts

- **C2.9 ✅** "`dvc repro` gives an identical dataset hash from cached responses". `dvc repro -f`
  of every dataset stage downstream of the locked `clean.jsonl` (casualise → dedupe → split →
  label_subset → candidates → token_counts) rebuilt every output with the md5 in `dvc.lock`, and
  the rewritten `dvc.lock` is byte-identical. All LLM requests were served from the cache, with
  cache-only mode on and no GPU. The labelled dataset is also identical:
  `respond` ×2 → `label_stage` from cache gives `labels.jsonl` `c5a38491…`, the hash in
  `data/processed/labels.jsonl.dvc`. collect is the one documented exception (stale by design).
  With the locked code it also reproduces its locked hash.
- **11.1.2 ❌ (partial)** "Full DVC pipeline … `dvc repro` end to end". `dvc.yaml` has `collect`,
  `casualise`, `dedupe`, `split`, `label_subset`, `candidates`, `token_counts` (and `register`).
  **`respond`, `judge`, `label`, `features` and `stratify` are not DVC stages.** `respond` and
  `judge`/`label` exist as modules (`terser.phase2.respond`, `terser.phase2.label_stage`) and are
  only chained by the Prefect `data_flow`. Features is `terser.phase3.build_features`, outside
  any pipeline. There is no `stratify` step anywhere: stratified group splitting is done inside
  `split`. The listed order (`features → split → stratify`) also does not match the code, where
  split comes before labelling. Every step that exists reruns from cache with identical outputs. The one
  exception is the pair table, which gains the new `overhead_pct` column; all of its committed
  columns are equal. So the missing piece is wiring, not determinism. `dvc repro` alone does not yet build
  `labels.jsonl`.

## Gaps found (for a fresh machine)

1. `.cache/llm_responses` is not tracked by DVC or pushed anywhere. A fresh clone with
   `dvc pull` gets the dataset files but cannot rerun casualise/candidates/respond/judge without a
   GPU. Fix: `dvc add .cache/llm_responses` (or a tarball) and push.
2. Raw data (`data/raw/*`) is in neither the remote nor `~/Terser/.dvc/cache`, so
   `dvc pull data/raw/*.dvc` fails. Fix: `dvc push` the raw dirs from the workspace.
3. `labels.jsonl`, `oracle.csv`, `pair_features.parquet`, `span_features.parquet` are in
   `~/Terser/.dvc/cache` but not on the `local` remote. Fix: `dvc push`.
4. `pair_features.parquet` needs a refresh (`dvc add`) to pick up `overhead_pct`.
5. `data/interim/responses/*.jsonl` is not DVC-tracked. It reproduces bit-for-bit from cache, but
   it should become a stage output.
6. To close 11.1.2: add `respond_qwen`, `respond_phi`, `label` (and `features`) stages to
   `dvc.yaml` with the commands above, then `dvc repro` from a populated cache.
