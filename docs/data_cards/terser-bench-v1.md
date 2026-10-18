# Data Card: `terser-bench-v1` (Terser benchmark release)

Casual prompts, compressed candidates, judge labels and rubrics from the Terser project:
tokenizer-aware rewriting of casual prompts into shorter prompts that keep the answer. Built by
`python -m terser.phase12.benchmark build` (plan row **12.11**).

> **Known limitations:**
>
> 1. **PII (row 2.1.3).** The text was scrubbed automatically (regex policy,
>    `docs/pii_policy.md`); the 500-row manual sample (`reports/p2_pii_pilot_sample.csv`) found
>    0 residual PII. Report any residual PII as a repository issue.
> 2. **Labels are judge estimates.** Every label comes from the position-swapped,
>    order-averaged 32B judge; preservation rates inherit its bias.
> 3. **WildChat text is withheld.** The real-slice prompts are user-written chat and are not
>    included (ids and sha256 only, section 7); rebuild them locally with `rebuild-real`.
> 4. **Project licence.** Repository licence file at build time:
>    `LICENSE`.
>
> Updated by hand after commit 38e8e09 changed the card template (licence recorded, PII sample
> done); the build tables below are unchanged.

| Field | Value |
| :--- | :--- |
| Card version | `1.0` |
| Plan row(s) | `12.11` (data from 2.1.1-2.1.8, 2.2.1, 2.3.2, 2.3.3) |
| Owner | `guru-bharadwaj20` |
| Last updated | `2026-10-20` |
| Built (UTC) | `2026-10-20T06:01:24+00:00` |
| Data licence | `CC-BY-SA-4.0` (see `LICENSE`, `NOTICE`) |
| Code | https://github.com/guru-bharadwaj20/Terser @ `32aa11a5091ec388ba034e0d1f1988dcce865f8c`; not copied into the release |
| Code paths | `src/terser/phase12/benchmark.py`, `src/terser/data/`, `src/terser/judge/`, `src/terser/phase2/`, `src/terser/phase12/`, `dvc.yaml` |
| DVC path | inputs: `data/processed/splits/`, `data/interim/candidates.jsonl`, `data/processed/labels.jsonl` |
| DVC hash | see `dvc.lock` at the code commit; sha256 in `MANIFEST.json`, `SHA256SUMS` |

---

## 1. Source

| Upstream | Version / revision | URL | `SOURCES.md` entry |
| :--- | :--- | :--- | :--- |
| `dolly` | `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a` | https://huggingface.co/datasets/databricks/databricks-dolly-15k | dolly |
| `gsm8k` | `740312add88f781978c0658806c59bc2815b9866` | https://huggingface.co/datasets/openai/gsm8k | gsm8k |
| `humaneval` | `7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544` | https://huggingface.co/datasets/openai/openai_humaneval | humaneval |
| `mbpp` | `4bb6404fdc6cacfda99d4ac4205087b89d32030c` | https://huggingface.co/datasets/google-research-datasets/mbpp | mbpp |
| `wildchat` (ids only) | `7d6490e462285cf85d91eabea0f9a954fbddcd1f` | https://huggingface.co/datasets/allenai/WildChat-1M | `wildchat` (evaluation-only section) |

Generated text and labels:

| Model | Role in this release | Licence |
| :--- | :--- | :--- |
| `Qwen/Qwen2.5-32B-Instruct-AWQ` | candidates, casualisation, judge | Apache-2.0 |
| `Qwen/Qwen2.5-7B-Instruct` | target | Apache-2.0 |
| `microsoft/Phi-3.5-mini-instruct` | target | MIT |
| `microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank` | candidates | Apache-2.0 |

Rule operators (`ops_*` candidates) are Terser code. Prompt templates and model settings are in
the code at the commit above (`src/terser/data/casualise.py`, `src/terser/data/candidates.py`,
`src/terser/judge/`).

## 2. Licence and redistribution

| Source | Licence | Research use | Redistribution | Notes / audit link |
| :--- | :--- | :--- | :--- | :--- |
| `dolly` | `CC-BY-SA-3.0` | yes | yes | license_audit.md#dolly |
| `gsm8k` | `MIT` | yes | yes | license_audit.md#gsm8k |
| `humaneval` | `MIT` | yes | yes | license_audit.md#humaneval |
| `mbpp` | `CC-BY-4.0` | yes | yes | license_audit.md#mbpp |
| `wildchat` | `ODC-By-1.0` | yes | ids only | license_audit.md#wildchat |

All data files are released under **CC-BY-SA-4.0** because Dolly's ShareAlike clause
covers the Dolly-derived prompts and candidates (`reports/license_audit.md`, option (b)). The MIT,
CC-BY-4.0 and Apache-2.0 conditions are met by the notices in `NOTICE`. No Llama output is
included.

## 3. Size

| Split | Rows | Source prompts |
| :--- | ---: | ---: |
| train | 4,955 | 1,165 |
| val | 1,133 | 266 |
| test | 992 | 234 |
| **Total** | 7,080 | 1,665 |

| Item | Rows |
| :--- | ---: |
| Candidates (`candidates/candidates.jsonl`) | 10,800 for 1,800 prompts |
| of which rejected by the generator checks (`rejected: true`) | 307 |
| Labels (`labels/labels.jsonl`) | 20,986 |
| Real-slice ids (`prompts/real_ids.jsonl`) | 1,000 |

Candidates per method:

| Method | Candidates |
| :--- | ---: |
| `llm_high` | 1,800 |
| `llm_low` | 1,800 |
| `llm_medium` | 1,800 |
| `llmlingua2` | 1,800 |
| `ops_CHFIDSL` | 1,800 |
| `ops_FHC` | 1,800 |

Token statistics are not part of this card; the repository's `token_counts` stage has them.

Files:

| File | Bytes | Records |
| :--- | ---: | ---: |
| `prompts/train.jsonl` | 5,294,647 | 4,955 |
| `prompts/val.jsonl` | 1,223,336 | 1,133 |
| `prompts/test.jsonl` | 1,191,989 | 992 |
| `prompts/real_ids.jsonl` | 307,347 | 1,000 |
| `candidates/candidates.jsonl` | 6,639,353 | 10,800 |
| `labels/labels.jsonl` | 39,737,551 | 20,986 |
| `rubrics/code.yaml` | 3,294 | - |
| `rubrics/creative.yaml` | 3,054 | - |
| `rubrics/explain.yaml` | 3,595 | - |
| `rubrics/fact.yaml` | 2,773 | - |
| `rubrics/reason.yaml` | 2,946 | - |
| `rubrics/transform.yaml` | 3,097 | - |
| `rubrics/judge_prompts/pairwise_v1.txt` | 777 | - |
| `rubrics/judge_prompts/pointwise_v1.txt` | 652 | - |

## 4. Collection method

Upstream rows were downloaded at the pinned revisions with `python -m terser.data.sources
download`, parsed to a common schema and PII-scrubbed at the collection boundary
(`terser.data.collect`). See `data/SOURCES.md` and `docs/data_pipeline.md` at the code commit.
The DVC stages `collect -> casualise -> dedupe -> split -> label_subset -> candidates -> respond ->
label` produce the inputs of this release.

Record fields:

- `prompts/<split>.jsonl`: `id`, `split`, `source`, `source_id`, `category`, `style`, `prompt`
  (the casual prompt), `clean` (the clean reference prompt), `generator_model`, `prompt_version`.
- `candidates/candidates.jsonl`: the Phase 2 candidate record (`prompt_id`, `candidate_id`, `method`,
  `text`, `generator`, `rejected`, ...) plus `source`, `source_id`, `split` of its prompt.
- `labels/labels.jsonl`: the `label_v1` record of `data/processed/labels.jsonl` (verdicts, `label`,
  `label_int`, judge probabilities and reasons, review flags) plus `source`, `source_id`.
- `prompts/real_ids.jsonl`: `id`, `source`, `source_id`, `category`, `category_source`,
  `pii_scrubbed`, `use`, `split`, `prompt_sha256`; **no text**.

## 5. Casualisation process

Casual prompts were written by the casualisation model in section 1 from the clean prompt, one
record per (source prompt, style), and checked for meaning (`meaning_check`, row 2.1.4). Rows
the check rejected went to `casual_rejected.jsonl`, which feeds neither the splits nor this
release.

| Style | Rows | Share |
| :--- | ---: | ---: |
| rambling | 1,108 | 15.6% |
| hedging | 1,369 | 19.3% |
| texting | 1,503 | 21.2% |
| over_polite | 1,608 | 22.7% |
| repetitive | 1,492 | 21.1% |

## 6. Splits

| Split | Rows | Source prompts | Share of source prompts | Hash |
| :--- | ---: | ---: | ---: | ---: |
| train | 4,955 | 1,165 | 70.0% | file sha256 in MANIFEST.json |
| val | 1,133 | 266 | 16.0% | file sha256 in MANIFEST.json |
| test (frozen, unlocked row 12.1) | 992 | 234 | 14.1% | `b4b264a80bcf8b1c415643b5f54156d214841e77088b75ae8b148bf2c9e106c7` |
| real (WildChat, ids only) | 1,000 | 1,000 | - | `2662cdae1ef0845aba760fd61c2d53226a3010c71d00f7d5aa9471bd722733b4` |

Grouping key: `source_id` (all casual variants of one source prompt share a split, row 2.1.6) ·
Seed: `0` · Leakage test: `tests/test_data_splits.py`; `splits.json`
records `leakage.ok = True` with
0 shared source prompts. The test split was read for this
release through `terser.phase12.unlock.read_locked` (purpose `"12.11 benchmark release"`),
which logs the read in `docs/checkpoints/C12/test_access_log.jsonl`.

Labels per split: test 2,958, train 14,732, val 3,296. Candidates
per split: test 1,518, train 7,572, val 1,710.

## 7. PII handling

| Item | Value |
| :--- | :--- |
| Detector | `terser.data.pii` regex layer, strict policy (row 2.1.3); NER off |
| Action on detection | Typed placeholder replacement at collection |
| Rows affected | See `reports/p2_pii_scrub_smoke.md` and `reports/p2_pii_pilot.md` at the code commit |
| Manual sample size / PII found | 500 rows reviewed by hand, 0 residual PII (`reports/p2_pii_pilot_sample.csv`) |
| Real user prompts included | **No text.** 1,000 WildChat ids only (prompts changed by the PII scrub at slice build: 1) |

**WildChat real slice.** `prompts/real_ids.jsonl` lists the 1,000 real-slice prompts by
WildChat conversation hash (`source_id`). To recreate the text locally from
`allenai/WildChat-1M` @ `7d6490e462285cf85d91eabea0f9a954fbddcd1f` (shard `data/train-00000-of-00014.parquet`, ODC-By-1.0), run from a
checkout of the code commit:

```bash
python -m terser.phase12.benchmark rebuild-real --release <this dir> --out real_prompts.jsonl
```

It downloads the pinned shard if needed, re-runs the row 2.1.8 filters, dedupe and PII scrub
(`terser.data.real_slice`) and checks every prompt against `prompt_sha256`. The hash pins the
text of the frozen slice. When the current scrubber would change a frozen prompt, the command
restores the frozen text and marks the record `scrub_drift: true`. Review those prompts by hand.
Keep the rebuilt file local; it is real user text.

## 8. Known biases

| Bias | Evidence | Likely effect on results |
| :--- | :--- | :--- |
| Labels are one judge's estimates | Judge(s): `Qwen/Qwen2.5-32B-Instruct-AWQ` (20,986); protocol `pairwise_avg`; 7,770 of 20,986 labels flagged `needs_human_review`, 0 position-inconsistent | Preservation rates inherit judge bias (order-averaged protocol, C2.6) |
| Casual styles are model-generated | All casual prompts come from the casualisation model in section 1 | A rewriter can score well by undoing that model's verbosity; the WildChat slice is the check |
| Source mix | Rows by source: dolly 4,612, gsm8k 1,185, humaneval 193, mbpp 1,090 | Each category comes from few sources (section 9), so category and source effects are confounded |
| Contamination | HumanEval, MBPP and GSM8K are public evaluation sets | Affects target-output quality, not the savings / preservation comparison |
| English only | All sources are English | No claims about other languages |

## 9. Task-category balance

| Category | train | val | test | Total | real (ids) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| code | 946 | 220 | 117 | 1,283 | 169 |
| fact | 779 | 235 | 215 | 1,229 | 159 |
| reason | 847 | 175 | 163 | 1,185 | 169 |
| explain | 809 | 122 | 177 | 1,108 | 169 |
| creative | 760 | 169 | 133 | 1,062 | 167 |
| transform | 814 | 212 | 187 | 1,213 | 167 |

Labels per target model:

| Target | preserved | degraded |
| :--- | ---: | ---: |
| `Qwen/Qwen2.5-7B-Instruct` | 6,157 | 4,336 |
| `microsoft/Phi-3.5-mini-instruct` | 5,781 | 4,712 |

Rows per source and category:

| Source | code | fact | reason | explain | creative | transform | Total |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dolly` | 0 | 1,229 | 0 | 1,108 | 1,062 | 1,213 | 4,612 |
| `gsm8k` | 0 | 0 | 1,185 | 0 | 0 | 0 | 1,185 |
| `humaneval` | 193 | 0 | 0 | 0 | 0 | 0 | 193 |
| `mbpp` | 1,090 | 0 | 0 | 0 | 0 | 0 | 1,090 |

## 10. Intended use

- Intended:
  - Research on prompt compression: measuring token savings against answer preservation.
  - Reproducing the Terser Phase 2 labels and the Phase 12 evaluation with the code commit above.
- Out of scope:
  - Redistribution that ignores the known limitations at the top of this card.
  - Treating labels as human ground truth.
  - Training on the test split when reporting results on it.
  - Anything that tries to identify WildChat users or other authors of upstream text.

## 11. Maintenance and version

| Version | Date | DVC hash | Git SHA | Change |
| :--- | :--- | :--- | :--- | :--- |
| 1.0 | 2026-10-20 | see `dvc.lock` | `32aa11a5091ec388ba034e0d1f1988dcce865f8c` | Initial release build (row 12.11) |

Check the files with `python -m terser.phase12.benchmark verify --release <this directory>` or
`sha256sum -c SHA256SUMS`. A new version is built when the inputs change (new DVC version, PII
fixes from the 2.1.3 review, human-validated labels).
