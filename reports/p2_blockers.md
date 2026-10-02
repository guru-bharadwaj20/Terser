# Phase 2 blockers

## 2.2.5 Judge self-preference control: blocked on Llama-3.1 access

The row needs a second-family judge (Llama-3.1-8B-Instruct) and outputs from a second-family target
(Llama-3.1-8B-Instruct, row 2.2.3), so that the Qwen judge's agreement can be compared on Qwen-target
vs Llama-target outputs.

- `meta-llama/Llama-3.1-8B-Instruct` returns `GatedRepoError` for this machine's Hugging Face token
  (checked 2026-10-01), so 2.2.3 runs with the Qwen2.5-7B-Instruct target only.
- Re-uploads of the weights (e.g. `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4`) can be
  downloaded without the gate. They were **not** used: Llama was dropped as a target because the
  official repo is gated, and that decision should not be bypassed through a mirror.

With one target family, the "agreement gap between target families" in the Definition of Done
cannot be measured. **Unblock** with either:

1. access granted on `meta-llama/Llama-3.1-8B-Instruct`: run `terser.phase2.respond` with the Llama
   target on the same 1,800-prompt subset, then the 2.2.5 comparison; or
2. a project decision to substitute another ungated, non-Qwen family for both the second target and
   the second judge (e.g. `microsoft/Phi-3.5-mini-instruct`, MIT; `google/gemma-2-9b-it` is gated
   too), recorded as an ADR.

2.2.3 is marked done for Qwen2.5-7B only, with the Llama target listed as outstanding in its report.
