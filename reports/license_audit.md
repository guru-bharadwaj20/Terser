# Licence audit: upstream sources and generating models

Plan row **2.1.2**. Audited 2026-09-11 against the dataset cards at the pinned revisions in
[`data/SOURCES.md`](../data/SOURCES.md). Each card (`README.md`) and, where present, licence file was
fetched with `huggingface_hub.hf_hub_download(repo_id, "README.md", repo_type="dataset",
revision=<pinned>)`; the quotes below are verbatim from those files.

**Rule.** Unknown or unstated terms are treated as non-redistributable. Decisions:
`include` (use and redistribute derived data under the stated conditions),
`include-no-redistribute` (use for research, never ship text), `exclude`.

This is an engineering record for the project, not legal advice.

## Summary

| Source | SPDX | Research use | Redistribution | Conditions | Decision |
| :--- | :--- | :---: | :---: | :--- | :--- |
| [`dolly`](#dolly) | `CC-BY-SA-3.0` | yes | yes | Attribution; **ShareAlike** on adaptations; no extra restrictions | include |
| [`oasst1`](#oasst1) | `Apache-2.0` | yes | yes | Licence copy; retain notices; state changes | include |
| [`humaneval`](#humaneval) | `MIT` | yes | yes | Retain copyright + permission notice | include |
| [`mbpp`](#mbpp) | `CC-BY-4.0` | yes | yes | Attribution; indicate changes; no extra restrictions | include |
| [`gsm8k`](#gsm8k) | `MIT` | yes | yes | Retain copyright + permission notice | include |
| [`wildchat`](#wildchat) (eval only, row 2.1.8) | `ODC-By-1.0` | yes | yes | Attribution; user turns only | include (evaluation only) |

No source has unknown terms, so none is downgraded to `include-no-redistribute`. The binding
constraint on the benchmark release (row 12.11) is Dolly's ShareAlike clause, plus the model licence
notices in [§ Generating models](#generating-models).

---

## dolly

| Item | Value |
| :--- | :--- |
| Repo / revision | `databricks/databricks-dolly-15k` @ `bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a` |
| Licence source | Dataset card https://huggingface.co/datasets/databricks/databricks-dolly-15k/blob/bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a/README.md (no separate LICENSE file in the repo) |
| Card licence field | `license: cc-by-sa-3.0` |
| SPDX | `CC-BY-SA-3.0` |
| Research use | yes |
| Redistribution | yes, under CC-BY-SA-3.0 |
| Decision | **include** |

Card text:

> This dataset was developed at Databricks (https://www.databricks.com) and its use is subject to
> the CC BY-SA 3.0 license.
>
> Certain categories of material in the dataset include materials from the following sources,
> licensed under the CC BY-SA 3.0 license:
>
> Wikipedia (various pages) - https://www.wikipedia.org/
> Copyright © Wikipedia editors and contributors.

**Conditions.** Attribution to Databricks, Inc. (and Wikipedia editors and contributors for `context`
passages), a link to the licence, and a note of changes. No additional technical or legal
restrictions may be applied to the shared material.

**ShareAlike implications.**

1. *Casual prompts derived from Dolly (row 2.1.4) are adaptations.* An LLM rewrite of a Dolly
   instruction into a rambling/hedging/texting/over-polite/repetitive style keeps its meaning and
   usually most of its wording, so it is treated as an adaptation of CC-BY-SA material. The
   operator- and model-generated compressed candidates (row 2.2.1) derived from those prompts are
   treated the same way. If distributed, they must be released under CC-BY-SA-3.0 or a later
   compatible CC-BY-SA version (e.g. CC-BY-SA-4.0).
2. *Benchmark release (row 12.11).* The release must either (a) put every Dolly-derived file
   (clean reference, casual variants, candidates, and target outputs that quote the prompt) under
   CC-BY-SA, separate from permissively licensed files, or (b) license the whole prompt/candidate
   release under CC-BY-SA-4.0. Option (b) is simpler: CC-BY-SA-4.0 is compatible with the
   MIT/Apache-2.0/CC-BY-4.0 inputs (those only require attribution and notices, which the release
   carries). Each record keeps `source` and `source_id` so per-record provenance and attribution
   are machine-readable.
3. *Labels, rubrics and code.* Binary quality labels, judge rubrics written by the project, and the
   evaluation code are not adaptations of Dolly text and can stay under the project's own licence.
   A file that embeds Dolly-derived text next to labels is covered by point 2.
4. *Models.* Training a quality gate or rewriter on Dolly-derived data is research use; whether
   model weights are an "adaptation" is unsettled. The project records the Dolly share of training
   data in the model card and does not claim a non-ShareAlike licence for anything that
   reproduces Dolly text.
5. No DRM, access gating, or terms that restrict recipients may be added to the released
   Dolly-derived files.

## oasst1

| Item | Value |
| :--- | :--- |
| Repo / revision | `OpenAssistant/oasst1` @ `fdf72ae0827c1cda404aff25b6603abec9e3399b` |
| Licence source | Dataset card https://huggingface.co/datasets/OpenAssistant/oasst1/blob/fdf72ae0827c1cda404aff25b6603abec9e3399b/README.md and `LICENSE` file in the same repo (full Apache License 2.0 text) |
| Card licence field | `license: apache-2.0` |
| LICENSE file header | "Apache License / Version 2.0, January 2004 / http://www.apache.org/licenses/" |
| SPDX | `Apache-2.0` |
| Research use | yes |
| Redistribution | yes |
| Decision | **include** |

**Conditions.** Give recipients a copy of the licence, keep attribution notices, and mark modified
files as changed (§4 of Apache-2.0). No ShareAlike: derived prompts may be released under
CC-BY-SA-4.0 alongside Dolly material.

**Other considerations (not licence).** Messages were written by volunteers on open-assistant.io and
may contain personal information. The parser drops `user_id` and metadata; PII scrubbing (row 2.1.3)
must run before any release. The `synthetic` flag is `False` for all 84,437 train-split
messages checked, and Terser keeps only root prompter messages.

## humaneval

| Item | Value |
| :--- | :--- |
| Repo / revision | `openai/openai_humaneval` @ `7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544` |
| Licence source | Dataset card https://huggingface.co/datasets/openai/openai_humaneval/blob/7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544/README.md (no LICENSE file in the repo; upstream repo https://github.com/openai/human-eval is MIT) |
| Card licence field | `license:` / `- mit`; § Licensing Information: "MIT License" |
| SPDX | `MIT` |
| Research use | yes |
| Redistribution | yes |
| Decision | **include** |

**Conditions.** Keep the copyright and permission notice in copies or substantial portions. The
HF repo carries no copyright line; the upstream GitHub `LICENSE` reads "Copyright (c) OpenAI
(https://openai.com)", which the release reproduces, citing Chen et al. (2021).

## mbpp

| Item | Value |
| :--- | :--- |
| Repo / revision | `google-research-datasets/mbpp` @ `4bb6404fdc6cacfda99d4ac4205087b89d32030c` |
| Licence source | Dataset card https://huggingface.co/datasets/google-research-datasets/mbpp/blob/4bb6404fdc6cacfda99d4ac4205087b89d32030c/README.md (no LICENSE file in the repo) |
| Card licence field | `license:` / `- cc-by-4.0`; § Licensing Information: "CC-BY-4.0" |
| SPDX | `CC-BY-4.0` |
| Research use | yes |
| Redistribution | yes |
| Decision | **include** |

**Conditions.** Attribute Google Research (Austin et al., 2021, "Program Synthesis with Large
Language Models"), link the licence, indicate changes. No ShareAlike; compatible with a CC-BY-SA-4.0
release.

## gsm8k

| Item | Value |
| :--- | :--- |
| Repo / revision | `openai/gsm8k` @ `740312add88f781978c0658806c59bc2815b9866` |
| Licence source | Dataset card https://huggingface.co/datasets/openai/gsm8k/blob/740312add88f781978c0658806c59bc2815b9866/README.md (no LICENSE file in the repo; upstream repo https://github.com/openai/grade-school-math is MIT) |
| Card licence field | `license:` / `- mit`; § Licensing Information: "The GSM8K dataset is licensed under the [MIT License](https://opensource.org/licenses/MIT)." |
| SPDX | `MIT` |
| Research use | yes |
| Redistribution | yes |
| Decision | **include** |

**Conditions.** Keep the copyright and permission notice (upstream GitHub `LICENSE`: "Copyright (c) 2021
OpenAI"); cite Cobbe et al. (2021).

---

## wildchat

Audited 2026-09-30. This source feeds only the real-prompt evaluation slice (row 2.1.8,
[`docs/data_cards/real_slice.md`](../docs/data_cards/real_slice.md)), never the training pool.

| Item | Value |
| :--- | :--- |
| Repo / revision | `allenai/WildChat-1M` @ `7d6490e462285cf85d91eabea0f9a954fbddcd1f` |
| Licence source | Dataset card `README.md` and `LICENSE.md` at the pinned revision |
| Card licence field | `license: odc-by` |
| LICENSE.md header | "# ODC Attribution License (ODC-By)" |
| Card note | "**2024-06-26: License Change.** We have updated the license of WildChat to ODC-BY … This change is retroactively applied to any previous downloads under the ImpACT license." |
| SPDX | `ODC-By-1.0` |
| Gated | no |
| Research use | yes |
| Redistribution | yes (attribution) |
| Decision | **include (evaluation only)** |

**Conditions.** Attribute the database (cite Zhao et al., 2024, "WildChat: 1M ChatGPT Interaction
Logs in the Wild") in any release that contains slice prompts.

**Other considerations (not licence).**

- Real user text. The authors de-identified it with Presidio plus hand-written rules, and removed
  toxic and flagged conversations.
- Terser adds three safeguards:
  - it excludes every row the authors marked `redacted`;
  - it drops all user metadata at parse time;
  - it re-runs its own PII scrub.
- Only user turns are used, so no ChatGPT outputs (and no OpenAI terms on outputs) are involved.

## Generating models

Casualisation (2.1.4), candidate generation (2.2.1), target outputs (2.2.3) and judging (2.3.3) produce
text with local open models. Their licences attach conditions to that output and to models trained on
it; [ADR-0003](../docs/decisions/ADR-0003-training-on-model-outputs.md) (training on model outputs)
records the final policy. Licence fields below were
read from each model's Hugging Face `README.md` on 2026-09-11.

| Model (plan use) | Card licence field | Output use for training | Conditions on release |
| :--- | :--- | :--- | :--- |
| Qwen2.5-32B-Instruct-AWQ, 14B-Instruct-AWQ (casualisation, candidates, judge) | `license: apache-2.0` | Permitted | Apache-2.0 notices if weights are shipped; no conditions on outputs |
| Qwen2.5-7B-Instruct (target) | `license: apache-2.0` | Permitted | As above |
| Qwen2.5-0.5B / 1.5B-Instruct (CPU stand-in, rewriter SFT) | `license: apache-2.0` | Permitted | As above; stand-in outputs never enter carded data |
| **Qwen2.5-3B-Instruct** (rewriter LoRA candidate, row 8) | `license: other`, `license_name: qwen-research` | Research only | Non-commercial; see below |
| Llama-3.1-8B-Instruct (target, small judge) | `license: llama3.1` | Permitted with conditions | "Built with Llama"; model naming; AUP; see below |

### Qwen2.5 licences

Most Qwen2.5 sizes used here are Apache-2.0, which puts no restriction on generated text. The
exception is **Qwen2.5-3B** whose `LICENSE` is the *Qwen RESEARCH LICENSE AGREEMENT*:

> "Non-Commercial" shall mean for research or evaluation purposes only.
>
> You are granted a non-exclusive, worldwide, non-transferable and royalty-free limited license [...]
> to use, reproduce, distribute, copy, create derivative works of, and make modifications to the
> Materials FOR NON-COMMERCIAL PURPOSES ONLY.

Implication: a rewriter fine-tuned from Qwen2.5-3B may be released for research only, with the
Qwen Research licence and its notice ("Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT,
Copyright (c) Alibaba Cloud. All Rights Reserved."). Prefer the Apache-2.0 0.5B/1.5B bases for any
artifact meant for unrestricted release. The 3B model is not used to generate dataset text.

### Llama 3.1 Community Licence

Quoted from the `extra_gated_prompt` in the Llama-3.1-8B-Instruct card:

> If you distribute or make available the Llama Materials (or any derivative works thereof), or a
> product or service (including another AI model) that contains any of them, you shall (A) provide a
> copy of this Agreement [...]; and (B) prominently display "Built with Llama" [...]. If you use the Llama Materials or any outputs or results of the Llama
> Materials to create, train, fine tune, or otherwise improve an AI model, which is distributed or
> made available, you shall also include "Llama" at the beginning of any such AI model name.

> If, on the Llama 3.1 version release date, the monthly active users [...] is greater than 700
> million monthly active users in the preceding calendar month, you must request a license from Meta

Implications:

1. **Target outputs** from Llama-3.1-8B (2.2.3) and **judge verdicts** from the Llama small judge
   (2.2.5, 8.4) may be used to train Terser's quality gate. If a model trained on those outputs is
   distributed, its name must begin with "Llama" and the release must carry the Llama 3.1 licence
   and a "Built with Llama" notice. To avoid this, train release models only on labels derived from
   Qwen (Apache-2.0) judge verdicts and keep Llama-derived signals for evaluation.
2. **Releasing Llama outputs** in the benchmark (12.11) requires shipping a copy of the Llama 3.1
   Community Licence with those files and complying with the Acceptable Use Policy; those files are
   kept separate and labelled. The 700M-MAU clause does not apply to this project.
3. Llama-generated text is not used as casual prompts or as candidate rewrites, so the prompt pool
   itself carries no Llama conditions.

### Combined effect on the benchmark release (row 12.11)

| Release component | Licence it must carry |
| :--- | :--- |
| Prompts and candidates derived from Dolly | CC-BY-SA-4.0 (ShareAlike), attribution to Databricks + Wikipedia contributors |
| Prompts and candidates derived from oasst1, HumanEval, MBPP, GSM8K | CC-BY-SA-4.0 acceptable (if released together) with Apache-2.0 / MIT / CC-BY-4.0 notices retained |
| Qwen2.5 (Apache-2.0) generated text | No extra condition |
| Llama-3.1 target outputs and verdicts | Separate files with the Llama 3.1 Community Licence + AUP |
| Labels, rubrics, eval code | Project licence |
| Models trained on Llama outputs | Name prefixed "Llama", Llama licence, "Built with Llama" |
| Models fine-tuned from Qwen2.5-3B | Qwen Research licence, non-commercial |

Recommendation: release prompts + candidates as one CC-BY-SA-4.0 bundle with a `NOTICE` listing every
upstream source and its licence, Llama outputs as a separate bundle under the Llama 3.1 licence,
and code under the project licence.
