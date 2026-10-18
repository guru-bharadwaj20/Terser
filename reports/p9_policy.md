# Learned greedy policy per category (row 9.2.3)

Greedy rollout of the best learner on the 300 training prompts. Savings on the qwen2.5 tokenizer; degradation is the **gate's** estimate. Row 9.2.4 checks it with the real judge.

| category | n | median_savings | gate_degraded |
|---|---|---|---|
| code | 53 | 0 | 0 |
| creative | 40 | 0.03393 | 0 |
| explain | 51 | 0 | 0 |
| fact | 47 | 0 | 0 |
| reason | 56 | 0.004348 | 0 |
| transform | 53 | 0 | 0 |

Most frequent greedy operator sequences (operator codes F H D I L C S, then STOP):

* **code**: `STOP` (21), `HLFDCSI` (10), `HFSCDLSTOP` (6)
* **creative**: `DHFCLISTOP` (11), `STOP` (8), `LIFSTOP` (7)
* **explain**: `STOP` (18), `ISHFDLC` (9), `ILHFDCS` (8)
* **fact**: `STOP` (27), `HIDLFSTOP` (11), `FDHCILS` (5)
* **reason**: `STOP` (19), `FDLHISC` (14), `IFDHSLSTOP` (11)
* **transform**: `STOP` (32), `FIHDCSTOP` (6), `DSSTOP` (5)
