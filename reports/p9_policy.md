# Learned greedy policy per category (row 9.2.3)

Greedy rollout of the best learner on the 300 training prompts. Savings on the qwen2.5 tokenizer; degradation is the **gate's** estimate. Row 9.2.4 checks it with the real judge.

| category | n | median_savings | gate_degraded |
|---|---|---|---|
| code | 53 | 0 | 0 |
| creative | 40 | 0.02283 | 0 |
| explain | 51 | 0 | 0 |
| fact | 47 | 0 | 0 |
| reason | 56 | 0 | 0 |
| transform | 53 | 0 | 0 |

Most frequent greedy operator sequences (operator codes F H D I L C S, then STOP):

* **code**: `STOP` (23), `HSTOP` (12), `ISTOP` (11)
* **creative**: `IDFSTOP` (11), `STOP` (8), `ILFSTOP` (7)
* **explain**: `STOP` (22), `IHSTOP` (11), `IHFSTOP` (8)
* **fact**: `STOP` (25), `HFISTOP` (11), `HSTOP` (6)
* **reason**: `FSTOP` (30), `STOP` (18), `FISTOP` (4)
* **transform**: `STOP` (22), `HSTOP` (18), `FSTOP` (6)

**Reading.** The greedy policy mostly stops immediately (`STOP` is the most frequent sequence
in 4 of 6 categories), and median gate-estimated savings are about 0%. The deterministic
operators save little on these prompts (C1.4: 1.7% median for the best single operator), while
the gate's degradation risk costs 100 points. Learning to do nothing is the correct response to
that reward. Short `F` / `H` / `I` chains appear only where filler and hedging are dense
(`reason`: `F` in 30 of 56 prompts; `creative`: `I D F`).
