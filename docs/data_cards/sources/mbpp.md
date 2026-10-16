# Data Card: `mbpp` (upstream source)

Raw upstream source for the Terser prompt pool. Registry entry: `SOURCES["mbpp"]` in
`src/terser/data/sources.py`; summary in [`data/SOURCES.md`](../../../data/SOURCES.md).

| Field | Value |
| :--- | :--- |
| Card version | `1.0` |
| Plan row(s) | `2.1.1, 2.1.2` |
| Owner | `guru-bharadwaj20` |
| Last updated | `2026-09-11` |
| DVC path | `data/raw/mbpp/` |
| DVC hash | Pending (`data/raw/mbpp.dvc` not yet created); file sha256 in `data/raw/mbpp/MANIFEST.json` |

---

## 1. Source

| Upstream | Version / revision | URL | Retrieved | `SOURCES.md` entry |
| :--- | :--- | :--- | :--- | :--- |
| `google-research-datasets/mbpp` | `4bb6404fdc6cacfda99d4ac4205087b89d32030c` | https://huggingface.co/datasets/google-research-datasets/mbpp | 2026-09-11T01:53:57Z | `mbpp` |

Files: `full/train-00000-of-00001.parquet` (87,223 bytes), `full/validation-00000-of-00001.parquet` (25,144), `full/test-00000-of-00001.parquet` (115,824), `full/prompt-00000-of-00001.parquet` (7,878).

## 2. Licence and redistribution

| Source | Licence | Research use | Redistribution | Notes / audit link |
| :--- | :--- | :---: | :---: | :--- |
| `mbpp` | `CC-BY-4.0` | yes | yes (BY) | Attribution to Google Research (Austin et al., 2021), indicate changes. Decision: include. `reports/license_audit.md#mbpp` |

No LLM-generated text in this raw source. Licences of the models that will casualise and rewrite it
are covered in `reports/license_audit.md` § Generating models ([ADR-0003](../../decisions/ADR-0003-training-on-model-outputs.md)).

## 3. Size

Tokenizer statistics are computed on the casualised dataset (row 2.1.4); for the raw source the
median instruction length is given in whitespace words.

| Stage | Rows | Unique source prompts | Median instruction words | p95 tokens |
| :--- | ---: | ---: | ---: | ---: |
| Raw | 974 (train 374, validation 90, test 500, prompt 10) | — | — | — |
| Parsed (`iter_records`) | 974 | 974 | 14 | — |
| After PII scrub | pending (2.1.3) | | | |
| After dedupe | pending (2.1.5) | | | |
| Final | pending | | | |

## 4. Collection method

`python -m terser.data.sources download --source mbpp`; parser `parse_mbpp`. `text` is the instruction, the `test_list` asserts are the `input` payload (they fix the function name), `code` is the reference. All four upstream splits are used. Files are downloaded with `huggingface_hub.hf_hub_download` at the pinned revision,
copied to `data/raw/mbpp/`, and checked with `python -m terser.data.sources verify` (sha256 + size;
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

Crowd-sourced programming problems written inside Google; no personal data expected.

## 8. Known biases

| Bias | Evidence | Likely effect on results |
| :--- | :--- | :--- |
| Entry-level Python problems | Upstream card | Code prompts are simpler than real developer requests |
| Public benchmark, likely seen by target models | Public since 2021 | Inflated target output quality |
| One-sentence instructions | Median 14 words | Casualisation supplies most of the compressible text |

## 9. Task-category balance

Category *hints* from the parser on the raw source (not final labels; splits pending row 2.1.6).

| Category | train | val | test | Total |
| :--- | ---: | ---: | ---: | ---: |
| code | — | — | — | 974 |
| fact | — | — | — | 0 |
| reason | — | — | — | 0 |
| explain | — | — | — | 0 |
| creative | — | — | — | 0 |
| transform | — | — | — | 0 |
| no hint | — | — | — | 0 |

## 10. Intended use

- Intended: Main source of `code` prompts with assert-based checks.
- Out of scope: Reporting target-model MBPP accuracy as a Terser result.

## 11. Maintenance and version

A new version is cut when the pinned revision changes or the parser's filters or category mapping
change. Owner: `guru-bharadwaj20`.

| Version | Date | DVC hash | Git SHA | Change |
| :--- | :--- | :--- | :--- | :--- |
| 1.0 | 2026-09-11 | pending | pending | Initial download at revision `4bb6404fdc6c` |
