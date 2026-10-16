# Data Card: `gsm8k` (upstream source)

Raw upstream source for the Terser prompt pool. Registry entry: `SOURCES["gsm8k"]` in
`src/terser/data/sources.py`; summary in [`data/SOURCES.md`](../../../data/SOURCES.md).

| Field | Value |
| :--- | :--- |
| Card version | `1.0` |
| Plan row(s) | `2.1.1, 2.1.2` |
| Owner | `guru-bharadwaj20` |
| Last updated | `2026-09-11` |
| DVC path | `data/raw/gsm8k/` |
| DVC hash | Pending (`data/raw/gsm8k.dvc` not yet created); file sha256 in `data/raw/gsm8k/MANIFEST.json` |

---

## 1. Source

| Upstream | Version / revision | URL | Retrieved | `SOURCES.md` entry |
| :--- | :--- | :--- | :--- | :--- |
| `openai/gsm8k` | `740312add88f781978c0658806c59bc2815b9866` | https://huggingface.co/datasets/openai/gsm8k | 2026-09-11T01:54:02Z | `gsm8k` |

Files: `main/train-00000-of-00001.parquet` (2,306,545 bytes), `main/test-00000-of-00001.parquet` (419,088 bytes).

## 2. Licence and redistribution

| Source | Licence | Research use | Redistribution | Notes / audit link |
| :--- | :--- | :---: | :---: | :--- |
| `gsm8k` | `MIT` | yes | yes | Keep copyright + permission notice. Decision: include. `reports/license_audit.md#gsm8k` |

No LLM-generated text in this raw source. Licences of the models that will casualise and rewrite it
are covered in `reports/license_audit.md` § Generating models ([ADR-0003](../../decisions/ADR-0003-training-on-model-outputs.md)).

## 3. Size

Tokenizer statistics are computed on the casualised dataset (row 2.1.4); for the raw source the
median instruction length is given in whitespace words.

| Stage | Rows | Unique source prompts | Median instruction words | p95 tokens |
| :--- | ---: | ---: | ---: | ---: |
| Raw | 8,792 (train 7,473, test 1,319) | — | — | — |
| Parsed (`iter_records`) | 8,792 | 8,792 | 42 | — |
| After PII scrub | pending (2.1.3) | | | |
| After dedupe | pending (2.1.5) | | | |
| Final | pending | | | |

## 4. Collection method

`python -m terser.data.sources download --source gsm8k`; parser `parse_gsm8k`. `question` is the instruction, `answer` (worked solution ending in `#### <number>`) is the reference. Both upstream splits are used. Files are downloaded with `huggingface_hub.hf_hub_download` at the pinned revision,
copied to `data/raw/gsm8k/`, and checked with `python -m terser.data.sources verify` (sha256 + size;
OK on 2026-09-11). No sampling or random seed at this stage.

## 5. Casualisation process

Not applicable at this stage: this card covers the raw upstream source. Casualisation (row 2.1.4) produces a separate carded dataset that links back here.

## 6. Splits

Not applicable yet. Upstream splits are ignored; Terser's source-grouped 70/15/15 split (row 2.1.6) is made after casualisation and dedupe, grouping key `source_id`. Filled in when row 2.1.6 runs.

## 7. PII handling

| Item | Value |
| :--- | :--- |
| Detector | Pending row 2.1.3 (runs before `data/interim/`) |
| Action on detection | Pending row 2.1.3 |
| Rows affected | Pending |
| Manual sample size / PII found | Pending (500-row sample across all sources) |
| Real user prompts included | no |

Crowd-written grade-school maths word problems with fictional names; no personal data expected.

## 8. Known biases

| Bias | Evidence | Likely effect on results |
| :--- | :--- | :--- |
| All `reason` prompts are grade-school arithmetic | Upstream card | `reason` over-represents one narrow task type |
| 8,792 records vs 1,138 for code | Category table | Balanced sampling must down-sample GSM8K |
| Numbers are load-bearing | Every answer is numeric | Operators that drop quantities break preservation; good stress test |
| Public benchmark, likely seen by target models | Public since 2021 | Inflated target output quality |

## 9. Task-category balance

Category *hints* from the parser on the raw source (not final labels; splits pending row 2.1.6).

| Category | train | val | test | Total |
| :--- | ---: | ---: | ---: | ---: |
| code | — | — | — | 0 |
| fact | — | — | — | 0 |
| reason | — | — | — | 8,792 |
| explain | — | — | — | 0 |
| creative | — | — | — | 0 |
| transform | — | — | — | 0 |
| no hint | — | — | — | 0 |

## 10. Intended use

- Intended: `reason` prompts with checkable numeric answers.
- Out of scope: Reporting target-model GSM8K accuracy as a Terser result.

## 11. Maintenance and version

A new version is cut when the pinned revision changes or the parser's filters or category mapping
change. Owner: `guru-bharadwaj20`.

| Version | Date | DVC hash | Git SHA | Change |
| :--- | :--- | :--- | :--- | :--- |
| 1.0 | 2026-09-11 | pending | pending | Initial download at revision `740312add88f` |
