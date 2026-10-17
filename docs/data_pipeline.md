# Data pipeline (DVC, rows 2.1.7 and 11.1.2)

`dvc.yaml` builds the dataset and the Phase 3 feature tables from the raw sources. Parameters are
in `params.yaml`; `dvc.lock` pins every dependency and output hash.

| stage | module | what it does | needs |
| --- | --- | --- | --- |
| `collect` (frozen) | `terser.data.collect` | raw sources -> 1,680 clean prompts, PII handled at the boundary | CPU |
| `casualise` | `terser.data.casualise` | 5 casual styles per prompt + meaning check | LLM (Qwen2.5-32B-AWQ) |
| `dedupe` | `terser.data.dedupe` | near-duplicate removal (threshold 0.8) | CPU |
| `split` | `terser.data.splits` | source-grouped train / val / test, **stratification**, frozen test split | CPU |
| `label_subset` | `terser.data.label_subset` | 300 prompts per category for labelling | CPU |
| `candidates` | `terser.data.candidates` | 6 compressed candidates per prompt (LLM rewrites, operators, LLMLingua-2) | LLM + CPU |
| `token_counts` | `terser.data.token_counts` | token counts per target tokenizer | CPU |
| `respond@qwen2.5-7b`, `respond@phi3.5-mini` | `terser.phase2.respond` | target outputs for every prompt and candidate | LLM (target) |
| `label` | `terser.phase2.label_stage` | judge (ADR-0005 `pairwise_avg`) + task metrics -> `labels.jsonl` | LLM (judge) |
| `oracle` | `terser.phase2.oracle` | shortest preserved candidate per (prompt, target) | CPU |
| `features` | `terser.phase3.build_features` | `pair_features.parquet`, `span_features.parquet` | CPU |
| `register` | `terser.phase11.register` | MLflow model registry (not part of the dataset) | CPU |

```bash
dvc repro features                              # whole dataset + feature tables
TERSER_LLM_CACHE_ONLY=1 dvc repro -f features   # CPU rebuild from .cache/llm_responses
```

## Stratification

Row 11.1.2 lists a `stratify` step after `split`. It is not a separate stage, because the
stratification happens inside `split` (`src/terser/data/splits.py`). Whole source groups are
assigned to a split by a seeded hash, which stratifies by category and by source *in
expectation*. It also never moves an existing group when data is added, so prompts that were
once in test cannot leak into training. `splits.json` reports the achieved proportions per
category and per source. `label_subset` then takes an exact per-category quota from the splits. A
no-op `stratify` stage would only duplicate `split`'s outputs, so there is none.

## LLM stages and the response cache

Every LLM call goes through `terser.llm.LLMClient`. Responses are cached under
`.cache/llm_responses`, keyed by sha256 of the full request. With `TERSER_LLM_CACHE_ONLY=1` a
cache miss raises `LLMCacheMissError` instead of calling a model, so a rebuild cannot reach a GPU
server by accident. On the GPU, start the vLLM server named in each stage comment and set
`TERSER_LOCAL_LLM_URL`.

`.cache/` is not tracked by DVC, so a fresh clone needs the response cache (and `llmlingua2`,
`token_counts`, `task_metrics`) copied in before an offline rebuild. Otherwise `dvc repro` of the
LLM stages fails on the first miss.

## Frozen `collect`

`collect` is frozen (`frozen: true`) because its code changed after the dataset was built: 2.1.3
switched from dropping PII rows to scrubbing them. The locked `clean.jsonl` and the frozen test
split are kept, and the locked code (`9092ee6`) reproduces them byte for byte. To start a new
dataset version, unfreeze it (`dvc unfreeze collect`), rerun the whole pipeline and re-freeze the
test split. Evidence: `reports/p2_dvc_repro.md`, `reports/p11_dvc_repro.md`.
