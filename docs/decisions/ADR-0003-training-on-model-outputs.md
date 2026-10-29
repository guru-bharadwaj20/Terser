# ADR-0003: Training on model outputs: licence check for generating and base models

| Field | Value |
| :--- | :--- |
| Status | Accepted (2026-10-28, by the owner; ADRs no longer wait for a second contributor's review) |
| Date | 2026-10-16 |
| Plan row(s) | Data rule "record the licence check in ADR-0003 before Phase 8"; 8.3.1, 8.3.3, 8.3.4; risk R11 |
| Deciders | guru-bharadwaj20 |
| Related | [reports/license_audit.md](../../reports/license_audit.md) § Generating models, ADR-0001 (local open-source models), ADR-0011 (Phi-3.5 replaces Llama-3.1), [docs/risks.md](../risks.md) (R11), [docs/model_cards/rewriter.md](../model_cards/rewriter.md) |

## Context

The project's data rules require that training on model outputs comply with each open model's licence and
that the check be recorded here **before Phase 8**. This ADR was written late. Phase 8 code landed
on 2026-10-01 (8.3.2 tiny CPU smoke test, no model-output data). The first training on
model-generated data was on 2026-10-11 (8.3.1 SFT data `e083b5a`, rung 1 Qwen2.5-0.5B `1995649`,
rung 2 Qwen2.5-1.5B `8fba5a1`). Rung 3 Qwen2.5-3B was trained on 2026-10-16 (`7070e76`). The
quality gate and span tagger (Phases 3–5) were trained earlier on the same judge labels. **This
ADR records the check retroactively**, against the data those runs actually used.

**What the training data contains.** Model ids come from the provenance fields in the data files
(`generator`, `generator_model`, `target_model`, `judge_model`):

| Data | Model(s) that produced it | Rows | Used to train |
| :--- | :--- | ---: | :--- |
| Casual prompts, the SFT inputs (2.1.4, `data/interim/casual_dedup.jsonl`) | Qwen2.5-32B-Instruct-AWQ (`generator_model`) | 7,080 | All components (inputs) |
| Rewrite candidates `llm_low/medium/high` (2.2.1, `data/interim/candidates.jsonl`) | Qwen2.5-32B-Instruct-AWQ, prompt `rewrite_v1` | 3 × 1,800 | Rewriter targets, gate pairs |
| Rewrite candidates `llmlingua2` (2.2.1) | `microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank`, rate 0.5 (token deletion; output is a subsequence of the prompt) | 1,800 | Same |
| Rewrite candidates `ops_FHC`, `ops_CHFIDSL` (2.2.1) | Project rule operators, no model | 2 × 1,800 | Same |
| Rewrite candidates `llm_max`, `llm_min` (8.3.4 augmentation, [p8_target_aug.yaml](../../configs/experiment/p8_target_aug.yaml), `candidates_aug.jsonl`) | Qwen2.5-32B-Instruct-AWQ | 2 × 1,800 | `sft_v2` rewriter targets |
| Target responses (2.2.3, `data/interim/responses/`) | Qwen2.5-7B-Instruct, Phi-3.5-mini-instruct | 16,483 each (+ 13,747 each for the augmentation) | Not trained on as text; they are judged |
| Judge verdicts and labels (2.3.3, `labels.jsonl`, `labels_aug.jsonl`) | Qwen2.5-32B-Instruct-AWQ (`pairwise_avg`, ADR-0005) | 20,986 + 5,770 | Gate labels; selection of SFT targets |
| Second-judge verdicts (2.2.5, `labels_judge2.jsonl`) | Phi-3.5-mini-instruct | 1,000 | Not trained on (read only by `terser.phase2.self_preference`) |

[`sft_data.py`](../../src/terser/models/rewriter/sft_data.py) keeps, per train / val prompt, the
shortest candidate labelled `preserved` on both targets, else the original prompt. The SFT
targets therefore come from three sources: Qwen2.5-32B rewrites, LLMLingua-2 deletions and
operator output.

| SFT set (`stats.json`) | Split | Pairs | Qwen2.5-32B rewrite targets | LLMLingua-2 | Operators | Identity (original prompt) | Median target savings |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `data/processed/sft` | train | 1,262 | 564 | 65 | 163 | 470 | 10.1% |
| `data/processed/sft` | val | 285 | 135 | 14 | 27 | 109 | 10.0% |
| `data/processed/sft_v2` | train | 1,262 | 666 | 51 | 133 | 412 | 21.4% |
| `data/processed/sft_v2` | val | 285 | 151 | 12 | 21 | 101 | 17.4% |

**Models fine-tuned.** `Qwen/Qwen2.5-0.5B-Instruct` (full FT, registry v4, Archived),
`Qwen/Qwen2.5-1.5B-Instruct` (LoRA, v5, Archived) and `Qwen/Qwen2.5-3B-Instruct` (LoRA, v6,
Staging) ([p11_registry.md](../../reports/p11_registry.md)). The 7B QLoRA rung has not been
trained.

**Llama 3.1 was not used for any output.** ADR-0011 replaced Llama-3.1-8B-Instruct with
Phi-3.5-mini-instruct as the second target and judge because the gated repository is not
accessible. A search for `meta-llama` / `"llama-3.1-8b` over all 20 JSONL files in
`data/interim/`, `data/interim/responses/`, `data/processed/` and `data/processed/sft*/` found
0 matches. The only target models in the label files are Qwen2.5-7B-Instruct and
Phi-3.5-mini-instruct. Llama appears in the pipeline only as a **tokenizer**: candidates longer
than the original on `llama-3.1` are rejected (`length_targets`). The tokenizer is loaded from
the ungated mirror `unsloth/Meta-Llama-3.1-8B-Instruct` when `HF_TOKEN` has no access
([count.py](../../src/terser/tokenizers/count.py)). Counting tokens does not train on Llama
output.

**Licence terms** (card fields read 2026-09-11, [license_audit.md](../../reports/license_audit.md);
Phi-3.5 from ADR-0011):

| Model | Licence | Training on its outputs | Condition on what we train or release |
| :--- | :--- | :--- | :--- |
| Qwen2.5-32B-Instruct-AWQ, 14B-Instruct-AWQ | Apache-2.0 | Permitted | None on outputs; Apache-2.0 notices if weights are shipped |
| Qwen2.5-7B-Instruct | Apache-2.0 | Permitted | As above |
| Qwen2.5-0.5B / 1.5B-Instruct (bases) | Apache-2.0 | n/a (base weights) | Apache-2.0 notices on the fine-tuned weights |
| **Qwen2.5-3B-Instruct** (base) | Qwen Research Licence (`license: other`, `license_name: qwen-research`) | n/a (base weights) | Derivatives "FOR NON-COMMERCIAL PURPOSES ONLY" ("research or evaluation purposes only"), with the Qwen Research licence notice |
| Phi-3.5-mini-instruct | MIT | Permitted | Keep the copyright and permission notice if its outputs are redistributed |
| Llama-3.1-8B-Instruct | Llama 3.1 Community Licence | Permitted with conditions | A model trained on Llama outputs and distributed must have a name starting with "Llama", ship the licence and show "Built with Llama". **Not triggered**: no Llama output is used |

No generating model used so far restricts training on its outputs. The one restrictive term is
on a **base model**: Qwen2.5-3B's research licence limits what can be done with the v6 rewriter.

## Decision

Training Terser's components on the outputs listed above is **compliant**. The 3B rewriter is
**research / non-commercial only**. Specifically:

1. **Permitted generators.** The gate, tagger, savings predictor and rewriter may be trained on
   text and verdicts from Qwen2.5-32B-Instruct-AWQ, Qwen2.5-14B-Instruct-AWQ, Qwen2.5-7B-Instruct
   (all Apache-2.0) and Phi-3.5-mini-instruct (MIT), and on operator and LLMLingua-2 output.
   Qwen2.5-3B is never used to *generate* dataset text. CPU stand-ins (Qwen2.5-0.5B / 1.5B via
   Ollama) never enter training data.
2. **Qwen2.5-3B derivatives** (v6 and any `sft_qwen3b_lora_v2` run) are for research and
   evaluation only. They may be released only under the Qwen Research Licence, with the notice
   "Qwen is licensed under the Qwen RESEARCH LICENSE AGREEMENT, Copyright (c) Alibaba Cloud. All
   Rights Reserved.". They must not be served in Production for a commercial deployment. A
   rewriter meant for unrestricted release must use an Apache-2.0 base (Qwen2.5-0.5B, 1.5B or 7B).
3. **Llama.** Llama-3.1 outputs stay out of training data. If ADR-0011 is superseded and Llama
   outputs are generated, they are kept for evaluation only, unless a new ADR accepts the naming,
   licence and "Built with Llama" conditions for every model trained on them.
4. **Provenance.** Every generated row keeps the model id that produced it (`generator`,
   `generator_model`, `target_model`, `judge_model`, `stand_in`). Stand-in rows never enter
   training data (`sft_data.py` drops `stand_in` rows).
5. **Data licences are separate.** Dolly's CC-BY-SA-3.0 ShareAlike terms on prompts, candidates
   and anything that reproduces them are governed by
   [license_audit.md](../../reports/license_audit.md) § dolly, not by this ADR.

## Alternatives considered

| Alternative | Pros | Cons | Why not chosen |
| :--- | :--- | :--- | :--- |
| Drop the 3B rung; train only on Apache-2.0 bases | Every rewriter could be released without restriction | Loses the best-preserving rung (74.7% preserved vs 67.4% / 69.3% for rungs 1 / 2; all fail C8.3) and the 8.3.4 escalation ladder | No commercial release is planned; recording the restriction is enough |
| Train the rewriter only on operator candidates (R11 contingency) | No model-output licence question at all | Operators supply 163 of 1,262 v1 targets (133 in v2); savings are capped by hand-written rules | Not needed: every generating model's licence permits training on its outputs |
| Mark the requirement not met and pause Phase 8 | Follows the "before Phase 8" order literally | Training already ran; pausing does not change which outputs were used | A retroactive check against the actual data gives the same assurance |

## Consequences

- Positive:
  - The licence position of every trained component is now written down. Gate, tagger and the
    0.5B / 1.5B rewriters depend only on Apache-2.0 / MIT models; the Llama 3.1 naming and
    "Built with Llama" conditions do not apply anywhere.
  - Risk R11's trigger (a licence clause restricting training on outputs) did not fire for any
    generating model.
- Negative:
  - The check came after the Phase 8 training it governs. Compliance was confirmed against the
    data, not ensured in advance.
  - The current Staging rewriter (v6, Qwen2.5-3B) cannot be used commercially. Promoting it for
    such use needs retraining on an Apache-2.0 base.
  - Model revisions are not pinned (base models, generator, targets and judge; see the
    rewriter card). R11's mitigation "pin model revisions" is not met, and the licence fields
    were read from whatever revision was current on 2026-09-11.
  - The LLMLingua-2 checkpoint's licence is not recorded in the repo, and Phi-3.5-mini-instruct
    (MIT, from ADR-0011) has no row in `license_audit.md` § Generating models.
- Follow-up:
  - Add Phi-3.5-mini-instruct and the LLMLingua-2 checkpoint to `license_audit.md`
    § Generating models, and change "ADR-0003 (… reserved)" to a link to this file.
  - Replace the "ADR-0003 pending / does not exist" notes with a link to this ADR in the model cards
    (`rewriter.md`, `quality_gate.md`, `span_tagger.md`, `textcnn.md`) and the source data cards.
  - Update R11 in `docs/risks.md` once this ADR is accepted.
  - Pin model revisions in the generation, respond and SFT configs before data is regenerated.
  - Open question: whether loading the Llama-3.1 tokenizer from the ungated mirror
    (token counting only) is consistent with ADR-0011's rule against bypassing the gate.
  - Verification: re-run the provenance counts above (model-id fields in the data
    files, Llama grep), and the release checklist (row 12.11) checks that any 3B-derived artifact
    ships under the Qwen Research Licence.
