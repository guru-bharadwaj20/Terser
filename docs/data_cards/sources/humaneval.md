# Data Card: `humaneval` (upstream source)

Raw upstream source for the Terser prompt pool. Registry entry: `SOURCES["humaneval"]` in
`src/terser/data/sources.py`; summary in [`data/SOURCES.md`](../../../data/SOURCES.md).

| Field | Value |
| :--- | :--- |
| Card version | `1.0` |
| Plan row(s) | `2.1.1, 2.1.2` |
| Owner | `guru-bharadwaj20` |
| Last updated | `2026-09-11` |
| DVC path | `data/raw/humaneval/` |
| DVC hash | Pending (`data/raw/humaneval.dvc` not yet created); file sha256 in `data/raw/humaneval/MANIFEST.json` |

---

## 1. Source

| Upstream | Version / revision | URL | Retrieved | `SOURCES.md` entry |
| :--- | :--- | :--- | :--- | :--- |
| `openai/openai_humaneval` | `7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544` | https://huggingface.co/datasets/openai/openai_humaneval | 2026-09-11T01:53:48Z | `humaneval` |

Files: `openai_humaneval/test-00000-of-00001.parquet` (83,920 bytes).

## 2. Licence and redistribution

| Source | Licence | Research use | Redistribution | Notes / audit link |
| :--- | :--- | :---: | :---: | :--- |
| `humaneval` | `MIT` | yes | yes | Keep copyright + permission notice. Decision: include. `reports/license_audit.md#humaneval` |

No LLM-generated text in this raw source. Licences of the models that will casualise and rewrite it
are covered in `reports/license_audit.md` § Generating models (ADR-0003, pending).

## 3. Size

Tokenizer statistics are computed on the casualised dataset (row 2.1.4); for the raw source the
median instruction length is given in whitespace words.

| Stage | Rows | Unique source prompts | Median instruction words | p95 tokens |
| :--- | ---: | ---: | ---: | ---: |
| Raw | 164 | — | — | — |
| Parsed (`iter_records`) | 164 | 164 | 5 (fixed instruction; payload in `input`) | — |
| After PII scrub | pending (2.1.3) | | | |
| After dedupe | pending (2.1.5) | | | |
| Final | pending | | | |

## 4. Collection method

`python -m terser.data.sources download --source humaneval`; parser `parse_humaneval`. The instruction is the fixed string "Complete the following Python function."; the signature and docstring are the `input` payload, `canonical_solution` the reference. Files are downloaded with `huggingface_hub.hf_hub_download` at the pinned revision,
copied to `data/raw/humaneval/`, and checked with `python -m terser.data.sources verify` (sha256 + size;
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

Hand-written programming problems; no personal data expected.

## 8. Known biases

| Bias | Evidence | Likely effect on results |
| :--- | :--- | :--- |
| Only 164 problems, Python only | Row count | Small contribution to `code`; payload-heavy prompts |
| Widely used benchmark, likely in target models' training data | Public since 2021 | Target outputs may be better than on unseen code tasks |
| Payload dominates the prompt | Docstring + signature in `input` | Masking (payload protection) matters; little instruction text to compress |

## 9. Task-category balance

Category *hints* from the parser on the raw source (not final labels; splits pending row 2.1.6).

| Category | train | val | test | Total |
| :--- | ---: | ---: | ---: | ---: |
| code | — | — | — | 164 |
| fact | — | — | — | 0 |
| reason | — | — | — | 0 |
| explain | — | — | — | 0 |
| creative | — | — | — | 0 |
| transform | — | — | — | 0 |
| no hint | — | — | — | 0 |

## 10. Intended use

- Intended: Code prompts with executable reference solutions for casualisation and preservation checks.
- Out of scope: Reporting target-model pass@k as a Terser result.

## 11. Maintenance and version

A new version is cut when the pinned revision changes or the parser's filters or category mapping
change. Owner: `guru-bharadwaj20`.

| Version | Date | DVC hash | Git SHA | Change |
| :--- | :--- | :--- | :--- | :--- |
| 1.0 | 2026-09-11 | pending | pending | Initial download at revision `7dce6050a7d6` |
