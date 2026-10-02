# ADR-0011: Phi-3.5-mini-instruct replaces Llama-3.1-8B-Instruct as the second model family

| Field | Value |
| :--- | :--- |
| Status | Accepted (decision by the project owner) |
| Date | 2026-10-02 |
| Plan row(s) | 2.2.3 (second target), 2.2.5 (second judge) |
| Deciders | guru-bharadwaj20 |
| Related | [reports/p2_blockers.md](../../reports/p2_blockers.md), ADR-0005 (judge protocol) |

## Context

Rows 2.2.3 and 2.2.5 use `meta-llama/Llama-3.1-8B-Instruct` as the second, non-Qwen model family:
a second target (so labels are not specific to one target family) and a second judge (so the
Qwen judge's self-preference can be measured). The official repository is gated and this
machine's Hugging Face token has no access (`GatedRepoError`, checked 2026-10-01). Ungated
re-uploads of the weights exist but are not used: bypassing the gate through a mirror is not
acceptable for the project.

## Decision

Use `microsoft/Phi-3.5-mini-instruct` (3.8B, MIT licence, ungated) for both roles:

- **2.2.3 second target**: vLLM bf16, temperature 0 (creative: 3 samples at 0.7), same
  1,800-prompt labelled subset and candidates as the Qwen2.5-7B target, via
  `python -m terser.phase2.respond`. Output `data/interim/responses/phi3.5-mini.jsonl`.
- **2.2.5 second judge**: Phi-3.5 judges a sample of 1,000 judged pairs (500 per target family)
  with the same `pairwise_avg` protocol as the Qwen judge; agreement with the Qwen judge is
  compared between Qwen-target and Phi-target outputs.

## Consequences

- Phi-3.5 is smaller (3.8B vs 8B) and likely a weaker judge than Llama-3.1-8B; its absolute
  agreement with the 32B judge is expected to be lower. The 2.2.5 check is about the *gap* between
  target families, which stays meaningful.
- Phi-3.5 is a different family (Microsoft; different pre-training data and tokenizer) from Qwen,
  which is what the self-preference control needs.
- If Llama-3.1 access is granted later, the same commands can be rerun with the Llama model id
  and this ADR superseded.
