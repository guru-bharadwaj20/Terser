### Input drift: train_vs_real_slice

- Verdict: **DRIFT** (PSI pc_1 = 1.167 > 0.2; PSI pc_2 = 0.469 > 0.2; PSI pc_7 = 0.306 > 0.2; PSI pc_8 = 0.678 > 0.2; PSI pc_9 = 0.433 > 0.2; PSI pc_11 = 0.225 > 0.2; PSI pc_15 = 0.652 > 0.2; PSI length_words = 0.698 > 0.2; MMD p = 0.0020 < 0.01)
- Rows: reference 4,955, current 1,000
- PCA: 16 components fitted on the reference, 36.0% of embedding variance
- MMD² (RBF, σ = 0.549, 500 rows per side): 0.04687, permutation p = 0.0020
- Evidently DataDriftPreset (PSI, threshold 0.2): 11 of 18 columns drifted

| Feature | PSI | PSI > threshold | Evidently PSI | Evidently > threshold |
|---|---:|:---:|---:|:---:|
| pc_1 | 1.167 | yes | 1.464 | yes |
| pc_2 | 0.469 | yes | 0.472 | yes |
| pc_3 | 0.188 |  | 0.243 | yes |
| pc_4 | 0.078 |  | 0.147 |  |
| pc_5 | 0.014 |  | 0.028 |  |
| pc_6 | 0.027 |  | 0.045 |  |
| pc_7 | 0.306 | yes | 0.363 | yes |
| pc_8 | 0.678 | yes | 0.677 | yes |
| pc_9 | 0.433 | yes | 0.458 | yes |
| pc_10 | 0.051 |  | 0.052 |  |
| pc_11 | 0.225 | yes | 0.229 | yes |
| pc_12 | 0.193 |  | 0.233 | yes |
| pc_13 | 0.013 |  | 0.015 |  |
| pc_14 | 0.036 |  | 0.047 |  |
| pc_15 | 0.652 | yes | 0.699 | yes |
| pc_16 | 0.182 |  | 0.223 | yes |
| length_words | 0.698 | yes | 0.714 | yes |
| category | 0.004 |  | 0.004 |  |

| Length (words) | median | p90 | mean |
|---|---:|---:|---:|
| reference | 62 | 172 | 81.3 |
| current | 29 | 161 | 58.6 |

| Category | reference share | current share |
|---|---:|---:|
| code | 19.1% | 16.9% |
| creative | 15.3% | 16.7% |
| explain | 16.3% | 16.9% |
| fact | 15.7% | 15.9% |
| reason | 17.1% | 16.9% |
| transform | 16.4% | 16.7% |

Artifacts: `reports/monitoring/input_drift_train_vs_real_slice.evidently.json`, `reports/monitoring/input_drift_train_vs_real_slice.html`, `reports/monitoring/input_drift_train_vs_real_slice.json`, `reports/monitoring/input_drift_train_vs_real_slice.md`
