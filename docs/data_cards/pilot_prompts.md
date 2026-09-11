# Data Card: `pilot_prompts`

Pilot set of casual prompts for Phase 1 (row 1.4.1). Used for the tokenizer economics study
(rows 1.4.2–1.4.4) and as the prompt pool for pilot labelling (row 1.3.6, GPU run pending).

## 1. Source

| Part | Rows | Origin |
| :--- | :---: | :--- |
| Seeds (`origin: "seed"`) | 60 | Hand-written casual prompts drafted for the project, 10 per category |
| Authored (`origin: "authored"`) | 440 | Written for this project by Claude Code agents under the project owner's direction, following per-category briefs (topic mix, style mix, length range, payload share) |

No text was copied from existing datasets (Dolly, OpenAssistant, GSM8K, HumanEval, etc.), and no
text was produced by an LLM casualisation run: the plan's casualisation step (row 2.1.4) needs the
GPU and was not used for this pilot set.

## 2. Licence and redistribution

Original project content; redistributable with the repository under the project's licence. No
third-party licence applies. Because the text was authored by an AI assistant rather than collected
from people, it may be less varied than real user prompts (see §8).

## 3. Size

- **500 prompts** in `data/pilot/prompts.jsonl` (merged, validated by `terser.phase1.prompts.load_prompt_dir`), also kept per category in `data/pilot/prompts/<category>.jsonl`.
- Prompt length: median 35 words, maximum 203 words.
- 147 prompts (29.4%) carry a payload (code, data or quoted text).

## 4. Collection method

Each category file was written against a brief that fixed the style counts, topic coverage, length
range and payload share, then machine-validated: JSON schema and field types, unique sequential ids,
exact counts per style, every `constraints` entry (and `tone_cues` for creative) is a verbatim
substring of its prompt, LF line endings. For `reason`, every `answer` was recomputed by a Python
script (83/83 verified, including uniqueness checks for logic puzzles).

Fields: `id`, `category`, `prompt`, `origin`, `style`, `has_payload`, `constraints`; plus `answer`
(reason only) and `tone_cues` (creative only).

## 5. Casualisation process

None by model. Casual register was written directly, in six styles:

| Style | Rows | Meaning |
| :--- | :---: | :--- |
| rambling | 96 | Long, digressive framing around the request |
| hedging | 96 | Uncertainty and softeners ("maybe", "if possible", "sorry if this is dumb") |
| repetitive | 84 | Requirements restated in different words |
| texting | 84 | Lowercase, abbreviations, typos |
| over_polite | 80 | Heavy politeness and thanks |
| efficient | 60 | Already concise controls (10 per category) |

## 6. Splits

None. This is a pilot set for measuring operator savings and building the labelling pipeline; it
is not used to train or evaluate models. Phase 2 builds the source-grouped train/val/test splits
(row 2.1.6) from its own corpus.

## 7. PII handling

No real user prompts are included. Authors were instructed to use fictional names only and no
personal data. No PII detector has been run and no human has yet read all 500 rows; that review is
an open item before any public release.

## 8. Known biases

- Written by one AI assistant: vocabulary, phrasing and "casual" patterns are narrower than real
  users' writing, and likely easier for rule-based operators than genuine prompts. Savings measured
  on this set are an optimistic pilot estimate, to be re-measured on the Phase 2 corpus.
- English only (translation prompts contain other languages in their payloads).
- Style counts are fixed by design, not by real-world frequency.
- Seed metadata (`style`, `has_payload`, `constraints`) was assigned after the fact; a few seed
  labels are debatable (e.g. `code-007` asks to explain a regex and could be `explain`).

## 9. Task-category balance

| Category | Rows | Payload rows |
| :--- | :---: | :---: |
| code | 84 | 34 |
| fact | 83 | 13 |
| reason | 83 | 7 |
| explain | 83 | 10 |
| creative | 84 | 0 |
| transform | 83 | 83 |

## 10. Intended use

- Rows 1.4.2–1.4.4: per-operator token savings, applicability and cross-tokenizer agreement.
- Row 1.3.6: prompt pool for pilot labels once target and judge models run on the GPU.
- Not for training, not for final benchmark claims.

## 11. Maintenance and version

| Version | Date | Change |
| :--- | :--- | :--- |
| 1 | 2026-09-11 | Initial 500 prompts (60 seeds + 440 authored) |

Tracked in git (small, ~0.3 MB). Owner: project maintainer. A new version is made when prompts are
added or labels corrected; the tokenizer study must be re-run after any change.
