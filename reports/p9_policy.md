# Learned greedy policy per category (row 9.2.3)

Greedy rollout of the best learner on the 300 training prompts. Savings on the qwen2.5 tokenizer; degradation is the **gate's** estimate. Row 9.2.4 checks it with the real judge.

| category | n | median_savings | gate_degraded |
|---|---|---|---|
| code | 53 | 0 | 0 |
| creative | 40 | 0.01905 | 0 |
| explain | 51 | 0 | 0 |
| fact | 47 | 0 | 0 |
| reason | 56 | 0.02062 | 0 |
| transform | 53 | 0 | 0 |

Most frequent greedy operator sequences (operator codes F H D I L C S, then STOP):

* **code**: `STOP` (38), `HDIFCSTOP` (10), `DSTOP` (2)
* **creative**: `HLFCISTOP` (12), `STOP` (12), `ILFSTOP` (7)
* **explain**: `STOP` (22), `ISFSTOP` (9), `IHLFDSC` (8)
* **fact**: `STOP` (27), `HFCISDSTOP` (10), `HSTOP` (4)
* **reason**: `HSFLCISTOP` (9), `DLIHFSSTOP` (8), `FLHSTOP` (6)
* **transform**: `STOP` (19), `HFSDILC` (7), `FDHISTOP` (6)

**Reading.** The best learner (alpha = 1 / N, gamma 0.9) still stops immediately on many
prompts (`STOP` is the most frequent sequence in 4 of 6 categories), and median gate-estimated
savings are 0-2%. The deterministic operators save little on these prompts (C1.4: 1.7% median
for the best single operator), while the gate's degradation risk costs 100 points, so doing
little is the correct response to that reward. Longer chains (`H F C I S`, `H S F L C I`) appear
in `creative` and `reason`, where filler and hedging are dense, and those are the only
categories with non-zero median savings (1.9% and 2.1%). Sequences without a final `STOP` hit
the episode's step limit.
