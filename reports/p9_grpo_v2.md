# GRPO v2: preservation-aware reward (follow-up negative-result ablation)

Follow-up to S-9.3 (`reports/p9_rl_vs_sft.md`, `reports/p9_grpo_audit.md`). Not a plan row; no row or check is flipped here. Full generated tables: `reports/p9_grpo_v2_val.md` (val comparison) and `reports/p9_grpo_v2_audit.md` (per-step training audit).

## Why

v1 used `reward = savings_pct x gate_pass` (served gate threshold 0.2646) and reward-hacked: on val, the rewrites the gate passed were judged preserved only 42.4% of the time, against the gate's Phase 2 precision of 91.0% (C9.3 fail). Its val savings (44.6%) and preservation (48.6%) both came in below its hpo08 SFT start (54.3% / 49.1%). The gate passed only about 3% of samples, so 75% of groups had a flat reward and the few non-zero rewards went to rewrites that fooled the gate.

## Design (`reward: v2`, `terser.models.rl.grpo.make_reward_v2`)

`reward = min(savings_pct, 70) x (1 - P(degraded))^4 x ramp(cos; 0.85 -> 0.97) x keep^2` (a fallback earns 0)

* **Soft gate.** `(1 - P)^4` replaces the hard pass, so the gradient stays dense. At the served threshold P = 0.26 the factor is 0.30, and at P = 0.5 it is 0.06.
* **Embedding similarity.** Cosine similarity between the original and the rewrite, from `BAAI/bge-small-en-v1.5` running on the CPU (it does not compete for the 24 GB GPU). It ramps linearly from 0 at 0.85 to 1 at 0.97. The ramp was set from the hpo08 val rewrites, whose median cosine is 0.906.
* **Protected-item retention.** `keep` is the share of the original's numbers, quoted and code spans, URLs, identifiers (snake_case, camelCase, ACRONYMS) and constraint words (not / only / exactly / words / json / bullet ...) that also appear in the rewrite. It is squared.
* **Length floor.** Savings above 70% earn nothing extra.
* Calibration check on judged hpo08 val rewrites (n = 259): ROC AUC for predicting judged preservation is 0.62 for cosine, 0.62 for retention and 0.65 for the combined term. These are weak but positive proxies.
* Everything else matches v1: start from hpo08 merged (also the KL reference), LoRA r16, group 8, 4 prompts per step, beta 0.04, lr 1e-5. `gate_pass` in the audit and in C9.3 is still the served-threshold pass, so v1 and v2 are measured the same way. v1 remains the default (`reward: v1`). Config: `configs/experiment/grpo_qwen15b_lora_v2.yaml`. Queue spec: `gpu_queue/grpo-qwen1.5b-v2.yaml`. Budget cap: `phase9_3_v2` (local only, $0).

The run hit the 3000-step cap after 18.2 h (2026-10-22 06:59 to 2026-10-23 01:11 IST). The audit, val responses (Qwen2.5-7B, Phi-3.5) and judging (Qwen2.5-32B-AWQ via vLLM) were the same as v1.

## Results (val split, 285 prompts)

| Rung | Median savings | Fell back | Preserved | Gate-passed | Gate precision (judged pairs) | Wilson 95% CI |
|---|---|---|---|---|---|---|
| hpo08 (SFT start) | 54.3% | 26 | 49.1% | 12 | 70.8% (24) | 50.8-85.1% |
| sft15v2 | 35.0% | 72 | 57.0% | 12 | 50.0% (24) | 31.4-68.6% |
| grpo v1 | 44.6% | 36 | 48.6% | 46 | 42.4% (92) | 32.8-52.6% |
| **grpo v2** | 29.1% | 8 | 54.6% | 35 | **71.4%** (70) | 59.9-80.7% |

Preservation by category for v2 vs hpo08: code 52.9 / 52.0, creative 46.8 / 34.0, explain 58.6 / 52.9, fact 49.1 / 52.7, reason 78.1 / 61.5, transform 43.9 / 41.8. The worst drop is 3.6 pp (fact). Degenerate val rewrites: 1.1% (hpo08 1.8%).

Training audit (judge on up to 16 samples per audited step): gate-passed samples judged preserved **47.1% (65/138)** for v2 vs 24.8% (31/125) for v1. Training means, first 100 vs last 100 steps: cosine 0.894 -> 0.953, retention 0.746 -> 0.916, savings 43.7% -> 32.5%, gate pass 4.1% -> 7.7%, degenerate 12.2% -> 2.5%, KL 0.005 -> 0.51.

## Reading

* **Hacking is reduced but not removed.** Gate precision on val rose from 42.4% to 71.4%. That is now level with the hpo08 start (70.8%) but still 19.6 pp below the Phase 2 91.0%, so C9.3 would still fail its original 5 pp target; it passes the revised ≥ 70% target ([ADR-0012](../docs/decisions/ADR-0012-feasible-checkpoint-targets.md)). The gap at hpo08 suggests part of the shortfall comes from the gate generalising poorly to model rewrites, not from RL.
* **The policy traded savings for preservation.** It learned to rewrite more conservatively: higher cosine, more items kept, far fewer fallbacks (8 vs 26). Preservation is 54.6% (+5.4 pp over hpo08 and +6.0 pp over v1), but median savings fell to 29.1% (-25.2 pp). It does not beat sft15v2, which preserves more (57.0%) and saves more (35.0%). C9.5 fails. The SFT champion is unchanged.
* **The proxies are weak.** With AUC around 0.65, the preservation proxies mostly push towards copying the prompt. They are not a faithful preservation signal, so the result is a sit-on-the-fence policy rather than one that compresses well. Next steps would be a learned preservation reward model trained on the Phase 2 labels, or periodic judge-in-the-loop rewards.

Adapter: `models/rewriter/grpo-qwen1.5b-v2` (DVC). Status log: `logs/p9_grpo_v2.status`.