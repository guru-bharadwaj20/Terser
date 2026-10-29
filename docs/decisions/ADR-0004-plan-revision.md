# ADR-0004: Plan revision after checkpoint C1 and GPU availability

| Field | Value |
| :--- | :--- |
| Status | Accepted (2026-10-28, by the owner; ADRs no longer wait for a second contributor's review) |
| Date | 2026-09-30 |
| Plan row(s) | Whole plan; checkpoints C1.4, C2.9, C7.3, C8.6, C8.7, C9.3–C9.5; headline S5 |
| Deciders | guru-bharadwaj20 |
| Related | ADR-0002 (problem framing), [C1 checkpoint](../checkpoints/C1/README.md), [hardware.md](../hardware.md) |

## Context

A checkpoint target changes only through an ADR that shows the target was wrong.
Three pieces of evidence arrived after the plan was written:

1. **C1.4 failed for a structural reason.** Conservative rule operators save a median of 1.7% (gpt-4o),
   1.6% (qwen2.5) and 1.6% (llama-3.1); the full stack saves about 2%. 217 of 500 pilot prompts are
   untouched by every operator. The target (≥ 10%) assumed the operators would be the main source of
   savings. They are not: any meaningful savings must come from generative rewriting, which C2.7 already
   gates. C1.4 measured the wrong thing to be a gate.
2. **Tokenizers agree.** Spearman ρ of per-(prompt, operator) savings is 0.991 (gpt-4o vs qwen2.5),
   0.991 (gpt-4o vs llama-3.1) and 1.000 (qwen2.5 vs llama-3.1); the best operator differs for 4 of 500
   prompts. Work built on tokenizer-specific rewriting (target-conditioned rewriter 8.3.11 / C8.6,
   cross-model matrix 12.5 / S5, a token-cost MLE prior for `S` 5.2.2) has no effect left to measure.
3. **GPU is no longer the bottleneck.** Two 24 GB RTX 4090 workers with the same compute capability as
   the RTX 4500 Ada are available now. Phase 7 no longer needs to wait, and throughput savers designed
   for a single scarce card (small judge 8.4, C8.7) are not needed.

Separately, a review of every row against two tests — *does the syllabus require it?* and *does the
product or its evaluation need it?* — found rows that pass neither and rows that duplicate each other.

## Decision

Revise the project plan as follows. Row ids are kept stable; removed ids are not reused.

**Checkpoint changes**

| Check | Change | Reason |
| :--- | :--- | :--- |
| C1.4 | Non-blocking; stays ❌ as a recorded negative result | Evidence 1 |
| C2.7 | Also reported on the real-prompt slice (2.1.8) | Synthetic-prompt circularity (below) |
| C2.9 | "Spend within ±20% of estimate" dropped; dataset-hash reproducibility kept | No priced API; token caps already bound compute (0.3.6) |
| C7.3 | Removed (CPU/GPU parity of the MLP) | Not needed by any downstream claim |
| C8.6, C8.7 | Removed | Evidence 2 and 3 |
| C9.3–C9.5 | Moved to the Stretch section with 9.3 | GRPO is outside the syllabus and not on the path to the headline claim |
| S5 | Redefined: S1–S3 hold on each target model separately | Evidence 2; the single rewriter still has to work for both targets |

**Rows removed** (merged content noted): 2.4.3 (into 12.10) · 3.4.4 (into 3.5.3 figure) · 3.6.2 (into
3.6.1) · 4.1.6 and 4.5.1–4.5.2 (into 10.2) · 4.2.2–4.2.5 (into 4.2.1) · 4.3.2–4.3.6 (into 4.3.1) · 4.4.9 ·
5.1.2 (into 5.1.1) · 5.1.4 (into 5.1.6) · 5.2.2 · 6.10 · 7.8 · 7.10 (into C7.4) · 8.3.5–8.3.6 (into 8.3.4) ·
8.3.11 · 8.4.1–8.4.2 · 9.1.3 (into 9.1.2) · 9.1.5–9.1.6 (into 9.1.1) · 10.3–10.4 (into 10.2) · 10.6 ·
11.2.1 · 12.5 · 13.3 · 13.5 (into 11.3.1) · 13.7 (into 13.6).

**Rows moved to Stretch:** 9.3.1–9.3.5 (GRPO).

**Rows added:** 2.1.8 real-prompt slice, 13.1 widened to include the README.

## Alternatives considered

| Alternative | Pros | Cons | Why not chosen |
| :--- | :--- | :--- | :--- |
| Keep the plan unchanged | No governance cost | C1.4 can never pass, so Phase 1 can never close; effort goes to effects already shown to be absent | Blocks the plan on a mis-specified check |
| Lower C1.4 to 2% | Phase 1 closes | "We did not reach it" is exactly what that rule forbids | Invalid by the plan's own rules |
| Drop the classical phases, keep only the rewriter | Faster to a product | Loses the syllabus coverage the project exists to demonstrate | Out of scope |

## Consequences

- Positive: 256 → 214 task rows and 66 → 60 checks with every syllabus row still covered; the critical
  path no longer waits on hardware; the headline claim gains an external-validity slice.
- Negative: fewer tokenizer-level results to report; the RL story ends at tabular Q-learning unless the
  stretch rows are done.
- New risk: **synthetic-prompt circularity.** Casual prompts are produced by an LLM from clean
  instructions, so a rewriter can score well by undoing the verbosity the pipeline added. Mitigation:
  row 2.1.8 and reporting S1–S3 separately on real prompts.
- Follow-up: the removed-row list is checked against the syllabus map ([syllabus_map.md](../syllabus_map.md)).
