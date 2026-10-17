# Quality canary (row 11.5.4)

Every night a fixed set of 200 prompts goes through the **deployed** rewrite service. Both target
LLMs answer the original prompt and the rewrite, and the judge compares the two answers. The run's
preservation and the 7-run rolling preservation go to the Prometheus Pushgateway, where the
"Terser serving" Grafana dashboard and the `< 92%` alert (row 11.5.5,
[p11_containers_dashboard.md](p11_containers_dashboard.md)) read them.

The generated sections below are rewritten by `python -m terser.monitoring.canary finalize`
(latest run and history) and `... verify` (dashboard check). Text outside the markers is kept.

## Canary set `canary_v1`

| | |
|---|---|
| File | `data/canary/canary_v1.jsonl` (DVC-tracked; manifest `data/canary/canary_v1.json`) |
| Source | `data/processed/splits/val.jsonl` only. The test split is never read: the CLI refuses any path named `test` |
| Size | 200 prompts, 200 distinct source items |
| By category | code 39, creative 30, explain 22, fact 41, reason 31, transform 37 |
| Selection rule | Rejected and stand-in records are dropped. Each category is sorted by `id`, and the 200 slots are allocated in proportion to category size (largest remainder, ties by category name). One `random.Random(0)` walks the categories in name order and shuffles each one. From the shuffle, the first record of every distinct `source_id` is taken before any second casual variant of the same item. The output is sorted by `id` |

Changing the set means a new version (`canary_v2.jsonl`). The rolling value only pools runs of
the same version, so a new set starts a new window.

## What one run measures

1. **Service.** Each prompt goes to `POST /rewrite` once per target: `target` is `qwen2.5` or
   `phi3.5`, and `mode` is `quality` by default (`--mode fast` is also supported). Requests are
   sent one at a time, after two warm-up requests with a non-canary prompt. HTTP errors are
   retried twice.
2. **Targets.** Qwen2.5-7B-Instruct and Phi-3.5-mini-instruct answer the original and the rewrite
   (`terser.phase2.respond`: greedy, or 3 seeded samples for `creative`; 768 max tokens).
3. **Judge.** Qwen2.5-32B-Instruct-AWQ judges each pair with `pairwise_avg`, plus the category
   task metric for code, fact and reason (`terser.phase2.label_stage.run_labels`, label_v1 rule).

### Counting rule

- **Unchanged rewrites count as preserved.** This covers gate rejections, fallbacks that return
  the original, and rewrites with no edit. The target then receives the original prompt, so
  nothing can degrade. These pairs are not sent to the targets or the judge. This is the same
  convention as [p8_rewriter_val.md](p8_rewriter_val.md), where a fallback is preserved by
  construction.
- A changed rewrite is preserved only if its label is `preserved`. An unlabelled pair (the
  judge's verdict could not be parsed) counts as **not** preserved.
- A request the service failed after retries is left out of the ratio and reported as an
  error. If more than 5% of requests fail, the run fails.
- Pushed run value = preserved / (prompt, target) pairs, pooled over both targets (at most 400
  pairs). Rolling value = counts pooled over the last 7 runs of the same canary version.

The run therefore measures what users receive: a service that sends almost everything back
unchanged scores close to 100%. Read the run's path, gate-rejection and fallback counts next to
the ratio.

## How it runs

- **Nightly:** Prefect deployment `canary_flow/canary_flow` (`0 2 * * *` UTC, tags `monitoring`,
  `row-11.5.4`). Its steps are rewrite → respond per target → judge → finalize, and the endpoints
  are flow parameters.
- **Single-GPU workstation:** `/home/temp/canary_chain.sh` runs the same CLI stages and serves one
  vLLM model at a time. It also starts the compose Pushgateway, Prometheus and Grafana, and
  checks the dashboard.
- **Outputs:** `logs/canary/<run_id>/` holds `run.json`, `rewrites.jsonl`, `candidates_*`,
  `responses_*`, `labels.jsonl`, `result.json`, `dashboard.json` and the run's own LLM budget
  ledger. The history is `logs/canary_history.jsonl`.

## Results

First run on 2026-10-18 with the 1.5B LoRA v2 rewriter in `quality` mode. The canary pipeline
works end to end: rewrite, both targets, the judge, the Pushgateway push, and every canary panel
on the dashboard returned HTTP 200 with the pushed value. Row 11.5.4 is met.

**Read the 98.0% with the path counts.** 380 of 400 pairs came back unchanged:

- 263 requests hit the 0.4 s quality timeout and fell back to the fast path.
- The gate rejected 372 rewrites in total.

Only 20 pairs carried an edited prompt, and 12 of those 20 (60%) were preserved. That matches
the gate-on numbers in [p8_rewriter_val.md](p8_rewriter_val.md). The high ratio therefore says
the service rarely harms users, not that the rewriter is good. C8.3 stays a fail. The "Per-category
bar 85%" line below is computed over every pair, including the unchanged ones, so it is not
evidence for C8.3.

<!-- canary:begin (generated by python -m terser.monitoring.canary) -->
### Latest run: `2026-10-18` (2026-10-18 08:14 IST)

- Service: `http://localhost:8090`, mode `quality`; canary `canary_v1` (sha256 `e1a9c582e9ff`)
- Targets: `Qwen/Qwen2.5-7B-Instruct`, `microsoft/Phi-3.5-mini-instruct`
- Judge: `Qwen/Qwen2.5-32B-Instruct-AWQ` (`pairwise_avg` + task metric, label_v1 rule)
- **Run preservation 98.0%** (392/400 pairs); **rolling 98.0%** over 1 run(s); alert below 92%: **ok**
- Unchanged (preserved by construction) 380, judged 20 (degraded 8, unlabelled 0), service errors 0
- Service paths {"fast": 263, "quality": 137}; gate rejected 372; fallbacks {"timeout": 263}
- Pushed: `localhost:9091` at 2026-10-18 19:44 IST

| Target | code | creative | explain | fact | reason | transform | Overall |
|---|---|---|---|---|---|---|---|
| phi3.5-mini | 100.0% (39/39) | 96.7% (29/30) | 95.5% (21/22) | 92.7% (38/41) | 96.8% (30/31) | 100.0% (37/37) | 97.0% (194/200) |
| qwen2.5-7b | 100.0% (39/39) | 96.7% (29/30) | 100.0% (22/22) | 97.6% (40/41) | 100.0% (31/31) | 100.0% (37/37) | 99.0% (198/200) |
| **both** | 100.0% | 96.7% | 97.7% | 95.1% | 98.4% | 100.0% | **98.0%** |

Per-category bar 85% (C8.3): met.

### Run history

| Run | Time | Mode | Pairs | Run preservation | Rolling (runs) | Alert |
|---|---|---|---|---|---|---|
| `2026-10-18` | 2026-10-18 08:14 IST | quality | 400 | 98.0% | 98.0% (1) | ok |
<!-- canary:end -->

<!-- canary-dashboard:begin (generated by python -m terser.monitoring.canary) -->
### Dashboard check (2026-10-18 19:44 IST): **metric on dashboard**

Prometheus instant queries: `terser_canary_rolling_preservation_ratio` = 0.98, `terser_canary_preservation_ratio` = 0.98, `terser_canary_prompts` = 400.0; Pushgateway scrape target health ['up'].

Grafana dashboard `terser-serving`, canary panels through `/api/ds/query`:

| Panel | Expression | HTTP | Value |
|---|---|---|---|
| Rolling preservation (canary) | `min(terser_canary_rolling_preservation_ratio)` | 200 | 0.98 |
| Last canary run | `min(terser_canary_preservation_ratio)` | 200 | 0.98 |
| Canary age | `time() - max(terser_canary_last_run_timestamp_seconds)` | 200 | 11.583044290542603 |
| Canary pairs (last run) | `max(terser_canary_prompts)` | 200 | 400.0 |
| Canary preservation over time | `min(terser_canary_rolling_preservation_ratio)` | 200 | 0.98 |
| Canary preservation over time | `min(terser_canary_preservation_ratio)` | 200 | 0.98 |
<!-- canary-dashboard:end -->
