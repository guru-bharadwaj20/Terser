# Data Card: `dolly` (upstream source)

Raw upstream source for the Terser prompt pool. Registry entry: `SOURCES["dolly"]` in
`src/terser/data/sources.py`; summary in [`data/SOURCES.md`](../../../data/SOURCES.md).

| Field | Value |
| :--- | :--- |
| Card version | `1.0` |
| Plan row(s) | `2.1.1, 2.1.2` |
| Owner | `guru-bharadwaj20` |
| Last updated | `2026-09-11` |
| DVC path | `data/raw/dolly/` |
| DVC hash | Pending (`data/raw/dolly.dvc` not yet created); file sha256 in `data/raw/dolly/MANIFEST.json` |

---

## 1. Source

| Upstream | Version / revision | URL | Retrieved | `SOURCES.md` entry |
| :--- | :--- | :--- | :--- | :--- |
| `databricks/databricks-dolly-15k` | `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a` | https://huggingface.co/datasets/databricks/databricks-dolly-15k | 2026-09-11T01:53:29Z | `dolly` |

Files: `databricks-dolly-15k.jsonl` (13,085,339 bytes).

## 2. Licence and redistribution

| Source | Licence | Research use | Redistribution | Notes / audit link |
| :--- | :--- | :---: | :---: | :--- |
| `dolly` | `CC-BY-SA-3.0` | yes | yes (BY + SA) | Attribution to Databricks and Wikipedia contributors; derived prompts inherit CC-BY-SA-3.0. Decision: include. `reports/license_audit.md#dolly` |

No LLM-generated text in this raw source. Licences of the models that will casualise and rewrite it
are covered in `reports/license_audit.md` § Generating models ([ADR-0003](../../decisions/ADR-0003-training-on-model-outputs.md)).

## 3. Size

Tokenizer statistics are computed on the casualised dataset (row 2.1.4); for the raw source the
median instruction length is given in whitespace words.

| Stage | Rows | Unique source prompts | Median instruction words | p95 tokens |
| :--- | ---: | ---: | ---: | ---: |
| Raw | 15,011 | — | — | — |
| Parsed (`iter_records`) | 15,011 | 15,011 | 10 | — |
| After PII scrub | pending (2.1.3) | | | |
| After dedupe | pending (2.1.5) | | | |
| Final | pending | | | |

## 4. Collection method

`python -m terser.data.sources download --source dolly`; parser `parse_dolly`. Rows with a blank instruction are dropped (0 dropped). `context` becomes `input` (4,467 rows), `response` becomes `reference`. The Dolly `category` field is mapped to a Terser hint (`data/SOURCES.md` § Category mapping). Files are downloaded with `huggingface_hub.hf_hub_download` at the pinned revision,
copied to `data/raw/dolly/`, and checked with `python -m terser.data.sources verify` (sha256 + size;
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
| Real user prompts included | no (employee-written instructions, not end-user traffic) |

Upstream card: "To our knowledge, there are no private person's personal identifiers or sensitive information." Text was written by Databricks employees; some `context` passages are copied from Wikipedia and name public figures.

## 8. Known biases

| Bias | Evidence | Likely effect on results |
| :--- | :--- | :--- |
| Register: instructions are short and clean (median 10 words) | Median instruction length on disk | Casual variants must add the verbosity; savings on raw Dolly would be near zero |
| Wikipedia-heavy `context` for closed_qa / summarization / extraction | Upstream card § License/Attribution | Over-represents encyclopaedic payloads in `transform` and `fact` |
| Written by Databricks employees in 2023, English only | Upstream card | No claims about non-English or domain-expert prompts |
| `open_qa` → fact/explain split uses a 30-word reference threshold | `DOLLY_OPEN_QA_FACT_MAX_WORDS` | Hint noise between `fact` and `explain`; final labels come later |

## 9. Task-category balance

Category *hints* from the parser on the raw source (not final labels; splits pending row 2.1.6).

| Category | train | val | test | Total |
| :--- | ---: | ---: | ---: | ---: |
| code | — | — | — | 0 |
| fact | — | — | — | 3,768 |
| reason | — | — | — | 0 |
| explain | — | — | — | 3,938 |
| creative | — | — | — | 2,475 |
| transform | — | — | — | 2,694 |
| no hint | — | — | — | 2,136 |

## 10. Intended use

- Intended: Clean instructions for casualisation (row 2.1.4) across fact, explain, creative and transform.
- Out of scope: Releasing derived prompts under a licence other than CC-BY-SA-3.0 (or a compatible later CC-BY-SA); mixing into a permissively licensed release file.

## 11. Maintenance and version

A new version is cut when the pinned revision changes or the parser's filters or category mapping
change. Owner: `guru-bharadwaj20`.

| Version | Date | DVC hash | Git SHA | Change |
| :--- | :--- | :--- | :--- | :--- |
| 1.0 | 2026-09-11 | pending | pending | Initial download at revision `bdd27f4d94b9` |
