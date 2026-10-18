# Data Card: `oasst1` (upstream source)

Raw upstream source for the Terser prompt pool. Registry entry: `SOURCES["oasst1"]` in
`src/terser/data/sources.py`; summary in [`data/SOURCES.md`](../../../data/SOURCES.md).

| Field | Value |
| :--- | :--- |
| Card version | `1.0` |
| Plan row(s) | `2.1.1, 2.1.2` |
| Owner | `guru-bharadwaj20` |
| Last updated | `2026-09-11` |
| DVC path | `data/raw/oasst1/` |
| DVC hash | Pending (`data/raw/oasst1.dvc` not yet created); file sha256 in `data/raw/oasst1/MANIFEST.json` |

---

## 1. Source

| Upstream | Version / revision | URL | Retrieved | `SOURCES.md` entry |
| :--- | :--- | :--- | :--- | :--- |
| `OpenAssistant/oasst1` | `fdf72ae0827c1cda404aff25b6603abec9e3399b` | https://huggingface.co/datasets/OpenAssistant/oasst1 | 2026-09-11T01:53:46Z | `oasst1` |

Files: `data/train-00000-of-00001-b42a775f407cee45.parquet` (39,516,251 bytes), `data/validation-00000-of-00001-134b8fd0c89408b6.parquet` (2,080,179 bytes).

## 2. Licence and redistribution

| Source | Licence | Research use | Redistribution | Notes / audit link |
| :--- | :--- | :---: | :---: | :--- |
| `oasst1` | `Apache-2.0` | yes | yes | Keep licence text + notices, mark changes. Decision: include. `reports/license_audit.md#oasst1` |

No LLM-generated text in this raw source. Licences of the models that will casualise and rewrite it
are covered in `reports/license_audit.md` § Generating models ([ADR-0003](../../decisions/ADR-0003-training-on-model-outputs.md)).

## 3. Size

Tokenizer statistics are computed on the casualised dataset (row 2.1.4); for the raw source the
median instruction length is given in whitespace words.

| Stage | Rows | Unique source prompts | Median instruction words | p95 tokens |
| :--- | ---: | ---: | ---: | ---: |
| Raw | 88,838 messages (10,364 root prompts) | — | — | — |
| Parsed (`iter_records`) | 3,670 | 3,670 | 15 | — |
| After PII scrub | pending (2.1.3) | | | |
| After dedupe | pending (2.1.5) | | | |
| Final | pending | | | |

## 4. Collection method

`python -m terser.data.sources download --source oasst1`; parser `parse_oasst1`. Keeps root `prompter` messages with `lang == "en"`, not deleted, not rejected in review; the best-ranked (lowest `rank`) assistant reply is kept as `reference` (present for all 3,670). `user_id`, timestamps, toxicity scores and all other metadata are dropped at parse time. Files are downloaded with `huggingface_hub.hf_hub_download` at the pinned revision,
copied to `data/raw/oasst1/`, and checked with `python -m terser.data.sources verify` (sha256 + size;
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
| Real user prompts included | yes: volunteer contributions under the Open Assistant terms, released by the project under Apache-2.0; `user_id` is dropped |

Messages were typed by volunteers on open-assistant.io and may contain personal details despite moderation (the message schema carries a `pii` label from reviewers). Highest PII risk of the five sources: row 2.1.3 scrubbing must run before any oasst1 text enters `data/interim/`.

## 8. Known biases

| Bias | Evidence | Likely effect on results |
| :--- | :--- | :--- |
| Volunteer, tech-leaning contributors; many coding and ChatGPT-style requests | Upstream card; manual skim | Likely over-represents code/explain once labelled |
| English filter keeps 3,670 of 10,364 roots (es 3,910 > en) | Parsed counts | No claims about non-English prompts |
| Prompts already somewhat casual | Median 15 words, conversational register | Closest to real user traffic; useful as a realism check against casualised Dolly |

## 9. Task-category balance

Category *hints* from the parser on the raw source (not final labels; splits pending row 2.1.6).

| Category | train | val | test | Total |
| :--- | ---: | ---: | ---: | ---: |
| code | — | — | — | 0 |
| fact | — | — | — | 0 |
| reason | — | — | — | 0 |
| explain | — | — | — | 0 |
| creative | — | — | — | 0 |
| transform | — | — | — | 0 |
| no hint | — | — | — | 3,670 |

## 10. Intended use

- Intended: Realistic user-style prompts; labelled later (row 2.1.4) to top up under-filled categories.
- Out of scope: Any use that keeps or tries to recover contributor identity (`user_id`).

## 11. Maintenance and version

A new version is cut when the pinned revision changes or the parser's filters or category mapping
change. Owner: `guru-bharadwaj20`.

| Version | Date | DVC hash | Git SHA | Change |
| :--- | :--- | :--- | :--- | :--- |
| 1.0 | 2026-09-11 | pending | pending | Initial download at revision `fdf72ae0827c` |
