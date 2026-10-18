# Phase 6 per-cluster risk (row 6.9, C6.3)

K-means k = 9 on 4,955 train prompts (bge-small). Degradation rate over 14,732 train label records (both targets); savings = mean oracle savings over targets. High risk = degradation rate >= 1.5x overall; high-risk clusters become eval slices in row 12.3.

> **Note (post-audit):** `median_savings` below was taken over label records, with prompts lacking an oracle row counted as 0 savings. `phase6.studies` now takes it per prompt and leaves missing savings out; the table is unchanged until the study is re-run (the degradation rates and C6.3 are unaffected).

| cluster | n | median_savings | degradation_rate | risk_ratio | high_risk | top_category |
|---|---|---|---|---|---|---|
| 6 | 1214 | 0.382 | 0.515 | 1.195 | False | explain |
| 5 | 994 | 0.303 | 0.480 | 1.114 | False | fact |
| 3 | 2036 | 0.418 | 0.478 | 1.110 | False | creative |
| 8 | 1520 | 0.390 | 0.467 | 1.084 | False | fact |
| 2 | 1768 | 0.317 | 0.465 | 1.080 | False | transform |
| 4 | 2632 | 0.203 | 0.457 | 1.061 | False | code |
| 0 | 1336 | 0.388 | 0.450 | 1.044 | False | explain |
| 1 | 736 | 0.440 | 0.448 | 1.041 | False | creative |
| 7 | 2496 | 0.574 | 0.242 | 0.562 | False | reason |
