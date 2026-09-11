# Upstream prompt sources

Plan row **2.1.1**. Registry in code: `src/terser/data/sources.py` (`SOURCES`). Licence audit:
[`reports/license_audit.md`](../reports/license_audit.md) (row 2.1.2). One data card per source in
[`docs/data_cards/sources/`](../docs/data_cards/sources/).

All files are fetched from the Hugging Face Hub at a pinned commit and copied to
`data/raw/<source>/` with a `MANIFEST.json` (sha256 and size of every file). Rebuild and check with:

```bash
python -m terser.data.sources download   # all sources, or --source <name>
python -m terser.data.sources verify     # sha256 + size against MANIFEST.json
python -m terser.data.sources stats      # parsed record counts per category hint
```

Retrieved 2026-09-11 (UTC timestamps in each manifest). `verify` reports OK for all five sources.

## Summary

| Source | HF repo | Pinned revision | Licence (SPDX) | Raw rows on disk | Parsed records | Redistribution allowed? |
| :--- | :--- | :--- | :--- | ---: | ---: | :--- |
| `dolly` | [`databricks/databricks-dolly-15k`](https://huggingface.co/datasets/databricks/databricks-dolly-15k) | `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a` | `CC-BY-SA-3.0` | 15,011 | 15,011 | Yes: attribution + ShareAlike (derived prompts must stay CC-BY-SA-3.0 or compatible) |
| `oasst1` | [`OpenAssistant/oasst1`](https://huggingface.co/datasets/OpenAssistant/oasst1) | `fdf72ae0827c1cda404aff25b6603abec9e3399b` | `Apache-2.0` | 88,838 messages | 3,670 | Yes: include licence, keep notices, state changes |
| `humaneval` | [`openai/openai_humaneval`](https://huggingface.co/datasets/openai/openai_humaneval) | `7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544` | `MIT` | 164 | 164 | Yes: keep copyright + permission notice |
| `mbpp` | [`google-research-datasets/mbpp`](https://huggingface.co/datasets/google-research-datasets/mbpp) | `4bb6404fdc6cacfda99d4ac4205087b89d32030c` | `CC-BY-4.0` | 974 | 974 | Yes: attribution, indicate changes |
| `gsm8k` | [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k) | `740312add88f781978c0658806c59bc2815b9866` | `MIT` | 8,792 | 8,792 | Yes: keep copyright + permission notice |
| **Total** | | | | | **28,611** | |

## Files

| Source | File (repository-relative, under `data/raw/<source>/`) | Bytes | Rows |
| :--- | :--- | ---: | ---: |
| `dolly` | `databricks-dolly-15k.jsonl` | 13,085,339 | 15,011 |
| `oasst1` | `data/train-00000-of-00001-b42a775f407cee45.parquet` | 39,516,251 | 84,437 |
| `oasst1` | `data/validation-00000-of-00001-134b8fd0c89408b6.parquet` | 2,080,179 | 4,401 |
| `humaneval` | `openai_humaneval/test-00000-of-00001.parquet` | 83,920 | 164 |
| `mbpp` | `full/train-00000-of-00001.parquet` | 87,223 | 374 |
| `mbpp` | `full/validation-00000-of-00001.parquet` | 25,144 | 90 |
| `mbpp` | `full/test-00000-of-00001.parquet` | 115,824 | 500 |
| `mbpp` | `full/prompt-00000-of-00001.parquet` | 7,878 | 10 |
| `gsm8k` | `main/train-00000-of-00001.parquet` | 2,306,545 | 7,473 |
| `gsm8k` | `main/test-00000-of-00001.parquet` | 419,088 | 1,319 |

The sha256 of each file is in the source's `MANIFEST.json`.

## Records per Terser category hint

Computed with `iter_records` over the files on disk. `—` means no hint (`None`): the source gives no
reliable task-type signal and the category is assigned later (rows 2.1.4, 2.3.5).

| Source | code | fact | reason | explain | creative | transform | — (no hint) | Total |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `dolly` | 0 | 3,768 | 0 | 3,938 | 2,475 | 2,694 | 2,136 | 15,011 |
| `oasst1` | 0 | 0 | 0 | 0 | 0 | 0 | 3,670 | 3,670 |
| `humaneval` | 164 | 0 | 0 | 0 | 0 | 0 | 0 | 164 |
| `mbpp` | 974 | 0 | 0 | 0 | 0 | 0 | 0 | 974 |
| `gsm8k` | 0 | 0 | 8,792 | 0 | 0 | 0 | 0 | 8,792 |
| **Total** | **1,138** | **3,768** | **8,792** | **3,938** | **2,475** | **2,694** | **5,806** | **28,611** |

`code` has 1,138 hinted records, above the C2.1 floor of 800 per category but with little margin
once dedupe and splits are applied; oasst1's 3,670 unlabelled prompts are the
expected top-up.

## Category mapping

| Source | Upstream field / rule | Hint |
| :--- | :--- | :--- |
| `dolly` | `category = creative_writing` (709), `brainstorming` (1,766) | `creative` |
| `dolly` | `summarization` (1,188), `information_extraction` (1,506) | `transform` |
| `dolly` | `closed_qa` (1,773) | `fact` |
| `dolly` | `open_qa` (3,742) with reference ≤ 30 words (1,995) | `fact` |
| `dolly` | `open_qa` with reference > 30 words (1,747) | `explain` |
| `dolly` | `general_qa` (2,191) | `explain` |
| `dolly` | `classification` (2,136) | none (mixes fact lookup and transform) |
| `oasst1` | no task label | none |
| `humaneval`, `mbpp` | all rows | `code` |
| `gsm8k` | all rows | `reason` |

## Parsing and filters

| Source | Filter | `instruction` | `input` | `reference` |
| :--- | :--- | :--- | :--- | :--- |
| `dolly` | blank instruction dropped (0 dropped) | `instruction` | `context` (4,467 non-empty) | `response` |
| `oasst1` | root `prompter` messages, `lang == "en"`, not deleted, not rejected in review (10,364 roots → 3,670); `user_id` and other metadata dropped | `text` | — | best-ranked assistant reply (present for all 3,670) |
| `humaneval` | none | fixed: "Complete the following Python function." | signature + docstring | `canonical_solution` |
| `mbpp` | none (all four splits of the `full` config) | `text` | `test_list` asserts (fix the function name) | `code` |
| `gsm8k` | none (`main` config, train + test) | `question` | — | `answer` |

`source_id` is stable: `dolly:<line>`, `oasst1:<message_id>`, `humaneval:<task_id>`,
`mbpp:<task_id>`, `gsm8k:<split>:<row>`. Splits in Terser are made by source prompt (row 2.1.6), not
by the upstream splits.

**Contamination note.** HumanEval, MBPP (test) and GSM8K (test) are public evaluation benchmarks that
target LLMs have likely seen. Terser uses their *prompts* as material for casualisation and
compression, not to score the target models' task accuracy, so contamination mainly affects the
absolute quality of target outputs, not the savings/preservation comparison.
