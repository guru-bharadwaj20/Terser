# Checkpoint C1 — Framing and Operators

| Field | Value |
| :--- | :--- |
| Date | 2026-09-11 |
| Git SHA at measurement | `7ae215d` |
| Machine | Windows 11, Intel Core i3-5005U, 8 GB RAM, **no GPU** |
| Data | `data/pilot/prompts.jsonl` (500 prompts, data card `docs/data_cards/pilot_prompts.md`); raw study rows DVC-tracked at `reports/p1_tokenizer_study.csv` |
| Seeds | bootstrap seed 0 (1,000 resamples) |

## Checks

| # | Check | Target | Measured | Result |
| :---: | :--- | :--- | :--- | :---: |
| C1.1 | ADR-0002 merged | Reviewed by ≥ 1 other contributor | `docs/decisions/ADR-0002-problem-framing.md` is committed with Status **Proposed**. No human review yet | ⏳ pending review |
| C1.2 | Operator safety | Every operator is a no-op on 50 already-efficient prompts (100%) | `tests/operators/test_registry.py` on `tests/fixtures/efficient_prompts.jsonl`: F, H, D, I, L, C, S each **50/50**; full stack **50/50**; S and full stack also 50/50 on each of gpt-4o, qwen2.5, llama-3.1. Full-stack idempotence holds on all 500 pilot prompts | ✅ |
| C1.3 | Version space | Non-empty for ≥ 4 of 6 categories; collapses explained | Runner `terser.phase1.version_spaces` is implemented and tested on synthetic labels. Real pilot labels (row 1.3.6) need target and judge models on the GPU | ⏳ blocked on GPU |
| C1.4 | Savings exist | Median best-single-operator savings ≥ 10% | **1.7%** (gpt-4o), **1.6%** (qwen2.5), **1.6%** (llama-3.1). Full stack `FHCIDSL`: median 2.0% / 1.9% / 1.9%, mean 5.1% / 4.9% / 4.9%, p75 6.7% / 6.3% / 6.4%. 217 of 500 prompts are unchanged by every single operator | ❌ **FAIL** |
| C1.5 | Tokenizer dependence | ρ < 0.9 for ≥ 1 pair, or documented agreement | Spearman ρ of per-(prompt, operator) savings: gpt-4o vs qwen2.5 **0.991** (95% CI 0.975–1.000), gpt-4o vs llama-3.1 0.991, qwen2.5 vs llama-3.1 1.000. On changed pairs only (n = 1,843): 0.976 (0.943–0.995). The best operator differs across tokenizers for only 4 prompts. Documented agreement | ✅ |

## Findings

1. **Rule-based operators recover little.** Conservative, meaning-safe regex operators remove a median of about 2% of tokens. Savings concentrate in over-polite prompts (full-stack median 8.3% on gpt-4o) and are zero at the median for `texting`, `efficient`, `fact` and `reason` prompts. `L` (list-ify) never saves tokens under real tokenizers, because bullets cost about as much as the removed lead-ins. `S` changes almost nothing.
2. **Tokenizers agree.** Across gpt-4o (`o200k_base`), Qwen2.5 and Llama-3.1, operator savings rank almost identically; Qwen2.5 and Llama-3.1 are indistinguishable. Per-target optimisation of *these operators* is not worth it. This says nothing yet about learned rewriters, which can make larger, tokenizer-sensitive edits.
3. **This is a pilot on assistant-authored prompts** (see the data card's bias section). Real casual prompts may carry more removable filler; they may also be harder to parse.

## What the failure means for the plan

C1.4 is not a hard gate. By the checkpoint protocol it cannot be passed by lowering the target without an ADR. The result is consistent with the idea document's §3.2 limitation ("these are largely heuristic"), and it strengthens the case for the learned and LLM-based candidates that Phase 2 measures at its real go/no-go gate (**C2.7**, oracle savings ≥ 25% from LLM candidates, GPU needed).

Options for the project owner:

| Option | Effect |
| :--- | :--- |
| A. Accept the FAIL as a negative result, keep the 10% target, proceed with Phase 2 (recommended) | Operators stay as baselines, safe fallbacks and RL actions (Phase 9); headroom is decided by C2.7 |
| B. Write an ADR revising C1.4 (e.g. "operators alone" becomes informative, not a gate) | Same practical path as A, with the plan text updated |
| C. Make operators more aggressive and re-measure | Likely raises savings but also meaning-loss risk; would need the Phase 2 judge to validate, so it cannot be judged safely on CPU today |

## Decision

`FAIL` (C1.4), with C1.1 pending human review and C1.3 blocked on the GPU. Phase 2 CPU work continues in parallel as it does not depend on C1.4; the owner should pick option A, B or C.
