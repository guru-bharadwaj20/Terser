# Phase 6 per-cluster risk (row 6.9, C6.3)

K-means k = 9 on 4,955 train prompts (bge-small). Degradation rate over 14,708 train label records (both targets); savings = mean oracle savings over targets. High risk = degradation rate >= 1.5x overall; high-risk clusters become eval slices in row 12.3.

| cluster | n | median_savings | degradation_rate | risk_ratio | high_risk | top_category |
|---|---|---|---|---|---|---|
| 6 | 1214 | 0.375 | 0.515 | 1.195 | False | explain |
| 5 | 990 | 0.296 | 0.482 | 1.118 | False | fact |
| 3 | 2032 | 0.413 | 0.478 | 1.110 | False | creative |
| 8 | 1518 | 0.381 | 0.466 | 1.083 | False | fact |
| 2 | 1768 | 0.302 | 0.465 | 1.079 | False | transform |
| 4 | 2622 | 0.203 | 0.458 | 1.062 | False | code |
| 0 | 1334 | 0.377 | 0.450 | 1.044 | False | explain |
| 1 | 734 | 0.435 | 0.448 | 1.040 | False | creative |
| 7 | 2496 | 0.573 | 0.242 | 0.562 | False | reason |
