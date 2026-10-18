### Input drift: train_vs_val_code

- Verdict: **DRIFT** (PSI pc_1 = 6.854 > 0.2; PSI pc_2 = 1.945 > 0.2; PSI pc_3 = 1.776 > 0.2; PSI pc_4 = 1.998 > 0.2; PSI pc_5 = 0.204 > 0.2; PSI pc_6 = 1.218 > 0.2; PSI pc_7 = 2.476 > 0.2; PSI pc_8 = 1.205 > 0.2; PSI pc_9 = 1.018 > 0.2; PSI pc_10 = 0.521 > 0.2; PSI pc_13 = 0.280 > 0.2; PSI pc_15 = 0.266 > 0.2; PSI pc_16 = 1.222 > 0.2; PSI length_words = 0.460 > 0.2; PSI category = 7.315 > 0.2; MMD p = 0.0020 < 0.01)
- Rows: reference 4,955, current 220
- PCA: 16 components fitted on the reference, 36.0% of embedding variance
- MMD² (RBF, σ = 0.600, 220 rows per side): 0.27024, permutation p = 0.0020
- Evidently DataDriftPreset (PSI, threshold 0.2): 17 of 18 columns drifted

| Feature | PSI | PSI > threshold | Evidently PSI | Evidently > threshold |
|---|---:|:---:|---:|:---:|
| pc_1 | 6.854 | yes | 7.184 | yes |
| pc_2 | 1.945 | yes | 2.071 | yes |
| pc_3 | 1.776 | yes | 1.865 | yes |
| pc_4 | 1.998 | yes | 1.820 | yes |
| pc_5 | 0.204 | yes | 0.381 | yes |
| pc_6 | 1.218 | yes | 1.239 | yes |
| pc_7 | 2.476 | yes | 1.959 | yes |
| pc_8 | 1.205 | yes | 1.092 | yes |
| pc_9 | 1.018 | yes | 0.977 | yes |
| pc_10 | 0.521 | yes | 0.741 | yes |
| pc_11 | 0.106 |  | 0.177 |  |
| pc_12 | 0.188 |  | 0.384 | yes |
| pc_13 | 0.280 | yes | 0.455 | yes |
| pc_14 | 0.089 |  | 0.269 | yes |
| pc_15 | 0.266 | yes | 0.237 | yes |
| pc_16 | 1.222 | yes | 1.006 | yes |
| length_words | 0.460 | yes | 0.503 | yes |
| category | 7.315 | yes | 7.315 | yes |

| Length (words) | median | p90 | mean |
|---|---:|---:|---:|
| reference | 62 | 172 | 81.3 |
| current | 68 | 112 | 72.2 |

| Category | reference share | current share |
|---|---:|---:|
| code | 19.1% | 100.0% |
| creative | 15.3% | 0.0% |
| explain | 16.3% | 0.0% |
| fact | 15.7% | 0.0% |
| reason | 17.1% | 0.0% |
| transform | 16.4% | 0.0% |

Artifacts: `reports/monitoring/input_drift_train_vs_val_code.evidently.json`, `reports/monitoring/input_drift_train_vs_val_code.html`, `reports/monitoring/input_drift_train_vs_val_code.json`, `reports/monitoring/input_drift_train_vs_val_code.md`
