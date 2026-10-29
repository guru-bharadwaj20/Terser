# Judge verification of the tabular policy (row 9.2.4, gate C9.2)

300 val prompts (seed 0), labels `data/interim/rewrites/val_labels_rl.jsonl` (Qwen2.5-32B-Instruct-AWQ pairwise judge, label_v1). Savings = 1 - tokens(rewrite) / tokens(original) on the Qwen2.5 tokenizer; preservation pooled over (prompt, target) for Qwen2.5-7B-Instruct, Phi-3.5-mini-instruct.

Methods: `rl_greedy` is the greedy rollout of the 9.2.3 learner (alpha = 1 / N, gamma 0.9, retrained deterministically on CPU) until `STOP`; `fixed_order` is the single operator subset with the best mean gate-reward return on the 300 training prompts (chosen before judging, no val labels).

| method | n | median savings | mean savings | median savings if preserved | unchanged | preserved | preserved Qwen2.5-7B-Instruct | preserved Phi-3.5-mini-instruct |
|---|---|---|---|---|---|---|---|---|
| fixed order `FHIS` | 300 | 1.4% | 3.0% | 0.0% | 103 | 77.8% | 78.3% | 77.3% |
| RL greedy (9.2.3 Q-table) | 300 | 0.0% | 1.7% | 0.0% | 191 | 89.3% | 90.0% | 88.7% |

Preservation by category (pooled over targets; n prompts in brackets):

| method | code | creative | explain | fact | reason | transform |
|---|---|---|---|---|---|---|
| fixed order `FHIS` | 78.8% (52) | 67.8% (45) | 76.8% (28) | 75.4% (63) | 95.5% (44) | 75.0% (68) |
| RL greedy (9.2.3 Q-table) | 94.2% (52) | 75.6% (45) | 92.9% (28) | 93.7% (63) | 95.5% (44) | 85.3% (68) |

Preservation by category on Qwen2.5-7B-Instruct:

| method | code | creative | explain | fact | reason | transform |
|---|---|---|---|---|---|---|
| fixed order `FHIS` | 80.8% | 71.1% | 75.0% | 74.6% | 97.7% | 73.5% |
| RL greedy (9.2.3 Q-table) | 94.2% | 77.8% | 92.9% | 93.7% | 97.7% | 85.3% |

Preservation by category on Phi-3.5-mini-instruct:

| method | code | creative | explain | fact | reason | transform |
|---|---|---|---|---|---|---|
| fixed order `FHIS` | 76.9% | 64.4% | 78.6% | 76.2% | 93.2% | 76.5% |
| RL greedy (9.2.3 Q-table) | 94.2% | 73.3% | 92.9% | 93.7% | 93.2% | 85.3% |

## C9.2 Tabular policy value

Definition: *equal preservation* means the RL policy's pooled preservation is at most 2% (absolute) below the fixed order's. C9.2 passes iff the RL policy is not dominated by the fixed order: better on median savings or on preservation (revised by [ADR-0012](../docs/decisions/ADR-0012-feasible-checkpoint-targets.md); originally equal preservation and savings >= the fixed order's).

* Preservation RL - fixed: +11.5 pp (equal: **yes**)
* Median savings RL - fixed: -1.4 pp (>= 0: **no**)
* **C9.2: pass**

Pre-judge context (from `prep`): gate-estimated degraded share RL 0.0%, fixed 0.0%; RL prompts that reached a state unseen in training: 57 (all Q = 0 there, so the greedy tie break applies the next operator in `F H D I L C S` order). Fixed-order train mean return 3.303 (rule: best mean gate-reward return over all 127 non-empty operator subsets).
