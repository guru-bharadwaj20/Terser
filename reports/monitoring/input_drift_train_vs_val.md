### Input drift: train_vs_val

- Verdict: **no drift**
- Rows: reference 4,955, current 1,133
- PCA: 16 components fitted on the reference, 36.0% of embedding variance
- MMD² (RBF, σ = 0.588, 500 rows per side): 0.00025, permutation p = 0.2196
- Evidently DataDriftPreset (PSI, threshold 0.2): 0 of 18 columns drifted

| Feature | PSI | Evidently PSI | > threshold |
|---|---:|---:|:---:|
| pc_1 | 0.035 | 0.035 |  |
| pc_2 | 0.017 | 0.028 |  |
| pc_3 | 0.008 | 0.034 |  |
| pc_4 | 0.019 | 0.044 |  |
| pc_5 | 0.019 | 0.023 |  |
| pc_6 | 0.009 | 0.048 |  |
| pc_7 | 0.003 | 0.014 |  |
| pc_8 | 0.007 | 0.020 |  |
| pc_9 | 0.028 | 0.042 |  |
| pc_10 | 0.033 | 0.045 |  |
| pc_11 | 0.027 | 0.027 |  |
| pc_12 | 0.005 | 0.038 |  |
| pc_13 | 0.014 | 0.017 |  |
| pc_14 | 0.042 | 0.070 |  |
| pc_15 | 0.016 | 0.013 |  |
| pc_16 | 0.024 | 0.056 |  |
| length_words | 0.009 | 0.027 |  |
| category | 0.042 | 0.042 |  |

| Length (words) | median | p90 | mean |
|---|---:|---:|---:|
| reference | 62 | 172 | 81.3 |
| current | 64 | 176 | 82.8 |

| Category | reference share | current share |
|---|---:|---:|
| code | 19.1% | 19.4% |
| creative | 15.3% | 14.9% |
| explain | 16.3% | 10.8% |
| fact | 15.7% | 20.7% |
| reason | 17.1% | 15.4% |
| transform | 16.4% | 18.7% |

Artifacts: `reports/monitoring/input_drift_train_vs_val.evidently.json`, `reports/monitoring/input_drift_train_vs_val.html`, `reports/monitoring/input_drift_train_vs_val.json`, `reports/monitoring/input_drift_train_vs_val.md`
