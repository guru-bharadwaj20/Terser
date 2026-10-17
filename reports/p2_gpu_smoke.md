# Phase 2 hardware smoke — 2026-10-03

Rows 2.2.1–2.2.3 and 2.3.3; C7.4 hardware follow-up. The smoke completed on worker-1. No full-scale run started. All project servers and sampling processes were stopped; final GPU compute inventory is empty and device memory returned to 43 MiB. The original queued 32B configuration failed startup and exceeds the queue's memory ceiling. This is operational smoke evidence, not a Phase 2 quality or C7 checkpoint pass.

## Reproducibility and scope

Selection and serving seed: 42. Creative common-random-number seeds: 1000, 1001, 1002. The selected pilot contains 20 prompts: code/creative four each; explain/fact/reason/transform three each. Selected IDs and the clean PII scan are in [prepare_stats.json](../data/raw/p2_smoke_20261003/prepare_stats.json). No stand-ins were used. Raw outputs/caches remain in `data/raw/`; intermediate files were scanned before entering `data/interim/` and carry `smoke_only=true`. The regex scan found all records clean; that does not replace production human review.

Worker code was captured before concurrent local edits: base SHA `7bfce9f364d02ac0ff52eebf3e507b6fdfa5812d`, `git_dirty=true`. The [source archive](../data/raw/p2_smoke_20261003/code_snapshot.tar.gz) SHA256 is `9fbe0e454cf6394a5933fda0215c6b09c9596e6ba500545e384de6c1f11e1476`. The archive and recorded wrapper hashes establish the executed code; the pre-provider-fix wrapper was reconstructed from the one-line change and verified against its recorded SHA256, and both versions are retained; the current local checkout has subsequent independent commits. Prompt SHA256 is `620bae63a58ba7f0a5364316ef79c93ea8fac57e02c27be2d593d7de8571bac0`. Input MD5/DVC leaf hash is `867d12bea7893564ea5821759ff67c19`.

Serving was restricted to localhost port 8012, launched with nohup/setsid, with logs under `~/terser/logs/`. Both target models used bf16, length 4096, utilization 0.90, batched tokens 2048, sequences 64. The 32B model used `awq_marlin`, length 4096, utilization 0.90, batched tokens 1024, sequences 64. Client workers: 16. Exact executed scripts: [serve.sh](../data/raw/p2_smoke_20261003/tools/serve.sh), [stage wrapper](../data/raw/p2_smoke_20261003/tools/smoke_stage.py), [stage sequence](../data/raw/p2_smoke_20261003/tools/run_smoke.sh), [judge resume](../data/raw/p2_smoke_20261003/tools/run_judge.sh). These are smoke scaffolding; the repository's production DVC graph currently ends at token counting.

## Environment and downloads

[Environment report](p2_worker_environment.md), [versions](../data/raw/p2_smoke_20261003/versions.json), and [dependency freeze](../data/raw/p2_smoke_20261003/pip_freeze.txt) retain exact versions. Worker-1: RTX 4090, 24,564 MiB device memory, driver 595.84, advertised CUDA capability 13.2. Python 3.11.17; vLLM 0.30.0; Torch 2.13.0+cu130; CUDA runtime 13.0; Transformers 5.18.0; MLflow 3.16.1; project 0.1.0. Environment path: `~/terser-env`; editable source: `~/terser`. The compiler packages NVCC/CRT/NVVM are pinned to 13.0.88 and `CUDA_HOME` selects that environment's toolkit.

All three models fit on worker-1. Each download checked actual free bytes before starting and reserved 6 GiB. Download records and measured snapshot sizes: [downloads_verified.json](../data/raw/p2_smoke_20261003/downloads_verified.json). Snapshot sizes include tokenizer merges loaded during serving; weights were not changed.

| Model | Pinned revision | Snapshot bytes measured | Free bytes before / after download |
| :--- | :--- | ---: | :--- |
| Qwen/Qwen2.5-7B-Instruct | `a09a35458c702b33eeacc393d103063234e8bc28` | 15,242,788,168 | 63,421,341,696 / 45,418,160,128 |
| microsoft/Phi-3.5-mini-instruct | `2fe192450127e6a83f7441aef6e3ca586c338b77` | 7,644,642,041 | 45,397,053,440 / 36,510,236,672 |
| Qwen/Qwen2.5-32B-Instruct-AWQ | `5c7cb76a268fc6cfbb9c4777eb24ba6e27f9ee6c` | 19,340,625,805 | 36,510,236,672 / 68,953,518,080 |

The filesystem's available space increased during the final download; that external change was observed without investigating other users' files. Final free space was 68,420,648,960 bytes ([inventory](../data/raw/p2_smoke_20261003/commands/53_final_inventory.log)). Worker-2 was used only for inventory and a project-local transfer probe; its judge deployment remains proposed. Worker-3 was not used.

## Measured stage results

The denominator is first uncached stage wall time, including client coordination and output scanning, excluding server startup. Tokens are generated completion tokens across actual unique requests. These are aggregate smoke rates, not single-request decoding rates. Stage GPU-hours are stage seconds/3600 on one reserved card; separately logged server-reserved GPU-hours include startup and shutdown. CPU counting logs zero GPU-hours.

| Stage | Stage seconds | Generated tokens | Generated tok/s | Stage GPU-hours | Tracking run |
| :--- | ---: | ---: | ---: | ---: | :--- |
| Candidates | 19.200 | 1,892 | 98.54 | 0.005333 | `9956ec544f2d41c29f8ffa3cbefb36a0` |
| Token counts (CPU) | 5.689 | 0 | 0.00 | 0.000000 | `be1268cb75ee4fc0895e968be2a2c729` |
| Qwen 7B responses | 60.839 | 45,817 | 753.09 | 0.016900 | `6f6aa31b52dd4ecc8032ca041d6ebf79` |
| Phi responses | 45.319 | 59,906 | 1321.88 | 0.012589 | `57aa35c9fcc542f89fb20839e032124b` |
| 32B judge | 264.625 | 22,698 | 85.77 | 0.073507 | `15c1a4b59b604a85bd80c54ce06fef56` |

All run IDs above resolve in [tracking export](../data/raw/p2_smoke_20261003/mlflow_runs.json); raw stats are beside it. Run names follow `p2-<task>-<model>-smoke-s42`; all required tags include `hardware=rtx4090`, `host=worker-1`, row, phase, task, input hash, SHA and dirty-tree flag. [All runs](../data/raw/p2_smoke_20261003/mlflow_all_runs.json) include retained failed attempts. Local `mlflow.db` has artifact paths rebased to this workspace; `mlflow_worker1.db` preserves the original database and remote paths. Both share the same run IDs and metrics.

Observed coverage:

- 120 candidates, six per prompt; 60 real rewrite requests. Four candidates rejected (low one, medium one, high two), leaving 116 accepted candidates.
- 280 token-count cells: 140 original/candidate texts × two actual target tokenizers; no missing values. The default tokenizer registry lacks Phi, so the smoke explicitly used its real tokenizer.
- 192 logical response records per target, each from 157 unique uncached requests. Creative originals and rewrites have three samples. Completion caps were reached by six Qwen responses and ten Phi responses.
- 232 labels, one for every accepted candidate and target. The judge used position-swapped `pairwise_avg`, greedy decoding, 20 top log-probabilities, two retries, and 256-token completion cap. It made 494 uncached requests and generated 22,698 tokens. There were 182 judged pairs and 50 identical pairs.
- Label values: 147 preserved, 85 degraded; 95 records flagged for human review. Raw position disagreement occurred in 147 of 274 parsed comparisons; 57 low-margin comparisons were recorded. These diagnostics require rubric/position-consistency review before production. The pilot has no reference answers/tests for reference-based code, fact or reasoning metrics, so this smoke cannot validate those metrics or label quality.

Coverage, schema, unique keys, seeds, model snapshot bytes, tracking tags, statuses and memory samples were checked by [verification](../data/raw/p2_smoke_20261003/commands/51_verify.log). No production tables or frozen test data were changed.

## C7.4 re-measurement

The [unaltered queue configuration](../data/raw/p2_smoke_20261003/queued_config.yaml) has 1,073,741,824 KV bytes. On this vLLM build that supports only 4,080 tokens, so length 4,096 fails startup. Its measured startup maximum was 22,969 MiB device-wide (22.430664 GiB), 22.388672 GiB after subtracting the 43 MiB desktop baseline. That is not a serving throughput measurement. Tracking run: `fae1068486234367b9ac69216905d711`; [failure measurements](../data/raw/p2_smoke_20261003/startup_failures.json).

For the smoke only, KV allocation increased by one 4 MiB block to 1,077,936,128 bytes. That advertises 4,112 cache tokens and allows length 4,096. Other model limits remained as above. The working judge also reached 22,969 MiB during startup/serving. Its net peak exceeds the numeric 22 GiB queue ceiling by 0.388672 GiB; the queue's 21.8 GiB estimate is not verified. The queue file was not changed and no checkpoint/status was promoted.

| Server / workload | Peak device MiB | Peak device GiB | Peak net GiB | Startup seconds | Server-reserved GPU-hours | Samples |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| candidates32aligned | 22,969 | 22.430664 | 22.388672 | 75.122 | 0.029090 | 524 |
| qwen7 | 21,191 | 20.694336 | 20.652344 | 35.052 | 0.027273 | 491 |
| phi35 | 21,937 | 21.422852 | 21.380859 | 35.052 | 0.022992 | 414 |
| judge32final | 22,969 | 22.430664 | 22.388672 | 30.045 | 0.082503 | 1484 |

[Memory summary](../data/raw/p2_smoke_20261003/vram_summary.json) ties each row to its tracking run; complete timestamped CSVs and server logs are under [server_logs](../data/raw/p2_smoke_20261003/server_logs). Sampling requested a 200 ms interval; these are sampled maxima and may miss shorter transients. GiB means MiB/1024; the queue's field uses a numeric “GB” ceiling following its documented GiB convention.

Startup recovery was scoped to project processes and paths: missing conda executable led to uv; package transfer retries used a larger timeout; the system 11.5 compiler did not support the required flag; initially resolved 13.4 compiler packages mismatched runtime headers and were pinned to 13.0.88. A missing explicit local provider stopped the first judge wrapper before judge requests; it was corrected and only judging resumed. All failed commands and their real output are retained. A verification assertion initially omitted the Qwen tokenizer's added `merges.txt`; its exact 1,671,839-byte size was measured and included before verification passed.

## Handoff

[Proposed full-scale plan](p2_full_scale_plan.md) specifies configured counts, host assignments and measured-smoke timing projections. Launch remains blocked by the unsafe original queue configuration, memory-ceiling overshoot and label-quality diagnostics. No additional GPU experiment was run after the completed smoke. Final [inventory and PID checks](../data/raw/p2_smoke_20261003/commands/53_final_inventory.log) show all eight started server PIDs and all eight sampler PIDs absent; [port check](../data/raw/p2_smoke_20261003/commands/51_verify.log) returned connection refused. No servers were started on worker-2 or worker-3.

[Command ledger](p2_smoke_commands.md) provides exact remote scripts, exit codes and trimmed real output. Download weights and the environment remain under the authorized project paths on worker-1; only smoke outputs, source archive, logs and tracking artifacts were copied locally.

The local `.git` directory is mounted read-only. `git add reports/p2_worker_environment.md` failed with `Unable to create .../.git/index.lock: Read-only file system`; therefore no task commit could be created and nothing was pushed. The prepared changes were committed later from the main checkout; the handoff note that listed them was not kept. CONTRIBUTING statuses remain unchanged.

## Local DVC retention

Raw directory DVC hash: `64c062cc02287462965b805ec811e76b.dir`, 1,162 files, 37,545,731 bytes; [pointer](../data/raw/p2_smoke_20261003.dvc).
Intermediate directory DVC hash: `8fa2095930f65ac81fc2f01591cb9bc5.dir`, six files, 1,130,068 bytes; [pointer](../data/interim/p2_smoke_20261003.dvc).
The prompt leaf MD5 matches every tracking run's `data_hash`. DVC status reports “Data and pipelines are up to date.” Local content verification checked each retained file against its DVC MD5, all 868 cached real calls against the ledger, both databases' integrity, required tracking tags/metrics, and both recorded wrapper hashes. [Local check output](p2_smoke_local_checks.txt) retains actual output. No code changes required a new unit-test suite; the executed pipeline and saved-artifact audits provide this smoke's validation.
