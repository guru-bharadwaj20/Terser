# Data Card: `<dataset-name>`

<!--
Copy this file to docs/data_cards/<dataset-name>.md and fill every section.
Delete the guidance comments when done. Write "Unknown" or "Not applicable" with a reason
rather than leaving a section empty. A dataset without a completed card is not used downstream.
-->

| Field | Value |
| :--- | :--- |
| Card version | `<e.g. 1.0>` |
| Plan row(s) | `<e.g. 2.1.1, 2.1.4>` |
| Owner | `<name>` |
| Last updated | `<YYYY-MM-DD>` |
| DVC path | `<e.g. data/processed/casual_prompts/>` |
| DVC hash | `<md5 from the .dvc file or dvc.lock>` |

---

## 1. Source

<!--
Where the data comes from. For each upstream source: name, URL, version or commit/revision,
download date, and the entry in data/SOURCES.md. For generated data, name the generating model
(local open-source model, e.g. the Hugging Face id and revision), its quantisation, the serving stack
(vLLM or Ollama / llama.cpp) and version, the prompt template path, and the sampling parameters.
Outputs from the CPU smoke-test stand-in (`stand_in: true`, row 2.2.4) never enter a carded dataset.
-->

| Upstream | Version / revision | URL | Retrieved | `SOURCES.md` entry |
| :--- | :--- | :--- | :--- | :--- |
| | | | | |

## 2. Licence and redistribution

<!--
Licence of each upstream source (SPDX id where possible), whether research use is permitted, and
whether redistribution is permitted (needed for the benchmark release, row 12.11). Unknown terms
are treated as non-redistributable (row 2.1.2). For LLM-generated text, record the licence of the
generating open model (e.g. Llama 3.1 Community Licence, Qwen2.5 licence) and what it allows for
training on its outputs, and link ADR-0003.
-->

| Source | Licence | Research use | Redistribution | Notes / audit link |
| :--- | :--- | :---: | :---: | :--- |
| | | yes / no | yes / no / unknown | `reports/license_audit.md#...` |

## 3. Size

<!--
Counts after every filtering step, so the reader can see what was dropped. Give rows, unique source
prompts, and token statistics on at least one target tokenizer (state which).
-->

| Stage | Rows | Unique source prompts | Median tokens (`<tokenizer>`) | p95 tokens |
| :--- | ---: | ---: | ---: | ---: |
| Raw | | | | |
| After PII scrub | | | | |
| After dedupe | | | | |
| Final | | | | |

## 4. Collection method

<!--
How the raw data was obtained: download script and its path, filters (language, length, task type),
sampling procedure and seed. Link the DVC stage that produces it (e.g. `collect`).
-->

## 5. Casualisation process

<!--
Only for casualised data (row 2.1.4); otherwise "Not applicable". Record: model (Hugging Face id, revision,
quantisation) and serving stack, prompt template path, the styles used (rambling, hedging, texting, over-polite, repetitive) and their share,
temperature/seed, how the clean reference is kept, validation checks applied to outputs (e.g. meaning
preserved, no added requirements), rejection rate, and compute used (generated tokens and
GPU-hours).
-->

| Style | Rows | Share | Rejected by checks |
| :--- | ---: | ---: | ---: |
| rambling | | | |
| hedging | | | |
| texting | | | |
| over-polite | | | |
| repetitive | | | |

## 6. Splits

<!--
Splits are grouped by source prompt (row 2.1.6): all casual variants and paraphrase groups of one source
prompt land in the same split. State the grouping key, split ratios, seed, freeze date and hash of the test
split, and the result of the automated leakage test (must be 0 shared source prompts).
-->

| Split | Rows | Source prompts | Share | Hash |
| :--- | ---: | ---: | ---: | :--- |
| train | | | 70% | |
| val | | | 15% | |
| test (frozen `<date>`) | | | 15% | |

Grouping key: `<field>` · Seed: `<n>` · Leakage test: `<test path>` → `<result>`

## 7. PII handling

<!--
Detector used (name, version, entity types), where in the pipeline it runs (before data/interim/),
what happens to detections (redact, replace with placeholder, drop row), counts, and the manual
sample result (row 2.1.3 requires 0 PII in a 500-row sample). State whether any real user prompts are
included and, if so, the consent basis and retention (≤ 30 days for raw text).
-->

| Item | Value |
| :--- | :--- |
| Detector | |
| Action on detection | |
| Rows affected | |
| Manual sample size / PII found | |
| Real user prompts included | no / yes (consent basis: ...) |

## 8. Known biases

<!--
Concrete, checkable statements. Examples: language (English only), register of source instructions,
over-representation of coding tasks, casualisation style reflecting the generating model's idea of
"casual", cultural assumptions in hedging and politeness, length distribution vs real user prompts.
Say how each bias could affect savings or preservation estimates.
-->

| Bias | Evidence | Likely effect on results |
| :--- | :--- | :--- |
| | | |

## 9. Task-category balance

<!--
Counts per category (code, fact, reason, explain, creative, transform) per split. C2.1 requires
≥ 800 per category overall. If Phase 6 clusters are available, add category × cluster counts or link them.
-->

| Category | train | val | test | Total |
| :--- | ---: | ---: | ---: | ---: |
| code | | | | |
| fact | | | | |
| reason | | | | |
| explain | | | | |
| creative | | | | |
| transform | | | | |

## 10. Intended use

<!--
What the data is for (e.g. training the quality gate, evaluating the rewriter) and what it must not be
used for (e.g. tuning on the test split, claims about non-English prompts, redistribution when the
licence forbids it).
-->

- Intended:
- Out of scope:

## 11. Maintenance and version

<!--
Version history with DVC hash per version, the git SHA of the pipeline that produced it, and what changed.
Name the owner responsible for updates and the trigger for a new version (new source, CT cycle, fix).
-->

| Version | Date | DVC hash | Git SHA | Change |
| :--- | :--- | :--- | :--- | :--- |
| 1.0 | | | | Initial release |
