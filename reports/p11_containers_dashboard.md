# Containers and dashboard (rows 11.3.7, 11.5.5)

Run on 2026-10-17, 13:10 to 13:35 UTC, Docker Desktop 29.8.0 (Compose v5.5.1) on the RTX 4500 Ada
workstation. The GPU was busy with training and judge jobs the whole time, so nothing below
used it. The quality-path container was built and checked statically but not started. What
still needs the GPU is listed in [Pending (needs the GPU)](#pending-needs-the-gpu).

## What changed

| File | Change |
|---|---|
| `docker/Dockerfile.cpu` | Bakes the ONNX fast path (`models/fast/{gate,tagger}.onnx`) into the image. The default config is now `configs/serving/fast_onnx.yaml`; it was the rule-based `default.yaml`. **Editable install** (`pip install -e`): `terser.serving.api.resolve_path` resolves model paths against the source tree, so a regular install looks under site-packages and cannot find `models/fast`. Also warms the `phi3.5` tokenizer. It is a priced target, but the image runs `HF_HUB_OFFLINE=1`, so `phi3.5` requests would have failed |
| `.dockerignore` | `!models/fast` re-includes the 135 KB ONNX models while `models/` (multi-GB rewriter checkpoints) stays out. Also excludes `experiments/`, `logs/`, `gpu_queue/`, `*.egg-info` |
| `docker/Dockerfile.gpu` | Base `vllm/vllm-openai:v0.30.0` (CUDA base), replacing `v0.6.6`. It now matches the host's vLLM 0.30.0 ([p2_gpu_smoke.md](p2_gpu_smoke.md)). The old entrypoint passed `--model /models/rewriter`, which is the directory holding every rung, not a model. No rung is a merged model either: v5 and v6 are LoRA adapters. The new entrypoint serves the Staging rung, v6 (`Qwen/Qwen2.5-3B-Instruct` + `sft-qwen3b-lora`, [p11_registry.md](p11_registry.md)), as `terser-rewriter` |
| `docker/gpu-entrypoint.sh` (new) | Runs `vllm serve BASE --enable-lora --max-lora-rank 16 --lora-modules terser-rewriter=PATH` for a LoRA rung, or `--served-model-name` for a full checkpoint (`TERSER_LORA_PATH=` empty plus `TERSER_REWRITER_MODEL`). If the adapter is missing it exits 2 with a `dvc pull` hint |
| `docker/serving.compose.yaml` | Was the rule-based config. It now matches `fast_onnx.yaml` (ONNX tagger, ONNX gate at 0.264564) plus `quality.base_url: http://terser-gpu:8000/v1` |
| `compose.yaml` | Removed the `gpu` profile, so plain `docker compose up` starts **both paths** (row 11.3.7 acceptance). Added `prometheus` (v3.7.0), `pushgateway` (v1.11.1) and `grafana` (12.2.0), with monitoring ports bound to 127.0.0.1. On a CPU-only host: `docker compose up terser-cpu prometheus pushgateway grafana` |
| `docker/monitoring/prometheus.yml` (new) | Scrapes `terser-cpu:8080/metrics`, `terser-gpu:8000/metrics` (vLLM) and `pushgateway:9091` (`honor_labels`) |
| `docker/monitoring/grafana/provisioning/{datasources,dashboards,alerting,plugins}/*.yaml` (new) | Datasource uid `prometheus`, a dashboard provider for folder `Terser`, the alert rule and an empty plugin list |
| `docker/monitoring/grafana/dashboards/terser-serving.json` (new) | Dashboard `terser-serving`: 18 panels in 3 rows, listed below |
| `src/terser/monitoring/canary_metrics.py` (new) | The metric contract for the row 11.5.4 canary: the gauges, the pooled 7-run rolling value, and push/delete to the Pushgateway. Has a CLI |
| `tests/monitoring/test_canary_metrics.py`, `tests/monitoring/test_compose_stack.py` (new) | 16 tests, listed below |

### Canary metric contract (row 11.5.4 → 11.5.5)

Row 11.5.4 (the canary itself) is not built yet, so this row defines the metric it will push.
After each nightly run the canary calls `terser.monitoring.canary_metrics.push_canary(gateway,
preserved, total, history)`. That pushes job `terser_canary` with these gauges:

| Gauge | Meaning |
|---|---|
| `terser_canary_preservation_ratio` | Preserved / judged pairs in this run |
| `terser_canary_rolling_preservation_ratio` | **Pooled** over the last 7 runs (counts summed, not ratios averaged). **Alert input** |
| `terser_canary_prompts` | Judged (prompt, target) pairs in this run |
| `terser_canary_rolling_runs` | Runs inside the window |
| `terser_canary_last_run_timestamp_seconds` | Push time, used by the staleness panel |

### Dashboard panels

| Row | Panels |
|---|---|
| Quality | Rolling preservation (stat; red below 0.92), last run, canary age (orange after 36 h), canary pairs, preservation over time (threshold line and area at 0.92) |
| Traffic and latency | Requests/s, gate rejection rate, fallback rate, cache hit rate, p95 latency, targets up (stats). Requests/s by path and mode, latency p50/p95/p99 by path, gate rejections/s by path, fallbacks/s by reason (time series) |
| Savings and GPU | Tokens saved/s by target, net savings USD/h (saved and lost) by target, GPU (vLLM KV-cache use, running requests, NVML utilisation) |

### Alert rule

`Rolling preservation below 92%` (uid `terser-rolling-preservation`, folder Terser, group
`terser-quality`):

- **Query:** `A = min(terser_canary_rolling_preservation_ratio)`, instant.
- **Condition:** `C = threshold(A) lt 0.92`.
- **Timing:** evaluated every 1 m, `for: 0s`.
- **No data → OK:** the rule does not page before the canary's first run.
- **Labels:** `severity=page`, `service=terser`, `row=11.5.5`.

## Verified on the CPU

### Images (row 11.3.7)

| Image | Base | Size (`docker image inspect`) | Build |
|---|---|---|---|
| `terser-cpu:latest` | `python:3.11-slim` | **1,671,541,462 B (1.67 GB)** | 173 s, cold pip layer |
| `terser-gpu:latest` | `vllm/vllm-openai:v0.30.0` (CUDA) | **30,729,679,975 B (30.7 GB)** | 4 s once the base was pulled; the base accounts for almost all of the size |

Building the GPU image needed no GPU. The CPU image is dominated by the package's base
dependencies (mlflow, lightgbm, pandas, scipy, matplotlib, hydra), which `pip install ".[serve]"`
pulls in through `[project].dependencies`. A slimmer serving-only dependency set would shrink it,
but would need a packaging change.

### CPU container serves the fast path

`docker compose up -d terser-cpu prometheus pushgateway grafana`. terser-cpu was `healthy` (the
Dockerfile healthcheck) within 36 s.

```text
GET /healthz -> {"status":"ok","quality_path":true}
POST /rewrite {"prompt":"hey so um, could you maybe write a Python function that checks if a string
  is a palindrome? It should ignore case. Thanks so much!!","target":"gpt-4o","mode":"fast"}
-> {"tokens_before":29,"tokens_after":29,"gate_score":0.5444,"fallback_used":false,"path":"fast",
    "mode":"fast","target":"gpt-4o","gate_rejected":true,"fallback_reason":null,"cached":false,
    "net_savings_usd":0.0,"latency_ms":474.4}       # first request: lazy app build
```

`gate_score` comes from the trained ONNX gate, so the ONNX fast path is the one being served.

Traffic was run inside the container: 600 distinct train prompts, then the first 100 of them,
then all 600 twice more. Targets rotated over gpt-4o, qwen2.5 and phi3.5; 10% of requests were
quality mode; requests were spaced 50 ms apart.

| Run | Requests | Errors | Path/mode | Gate rejected | Fallbacks | Cached | Client p50 / p95 ms |
|---|---|---|---|---|---|---|---|
| 600 distinct | 600 | 0 | fast/fast 540, fast/quality 60 | 595 | timeout 60 | 0 | 10.9 / 409.8 |
| first 100 again | 100 | 0 | 90 / 10 | 98 | timeout 10 | 90 | 2.2 / 405.9 |
| 600 again (×2) | 1,200 | 0 | 1,080 / 120 | 1,190 | timeout 120 | 1,080 | 3.1 / 409.9 and 2.8 / 409.1 |

Total: **1,901 requests, 0 errors**.

- The p95 values around 410 ms are quality-mode requests. They wait the 0.4 s
  `quality_timeout_s` budget for the absent `terser-gpu`, then fall back to the fast path
  (`fallback_reason: timeout`). That is row 11.3.5 behaving as designed.
- Fallbacks are never cached, which is why only fast-mode repeats hit the cache.

Raw `/metrics` after the first 701 requests:

```text
terser_requests_total{mode="fast",path="fast"} 631.0
terser_requests_total{mode="quality",path="fast"} 70.0
terser_request_latency_seconds_count{path="fast"} 701.0
terser_fallbacks_total{reason="timeout"} 70.0
terser_gate_rejections_total{path="fast"} 694.0
terser_cache_hits_total 90.0
terser_tokens_saved_total{target="phi3.5"} 164.0
terser_tokens_saved_total{target="gpt-4o"} 33.0
```

**Finding (not fixed here): the ONNX gate rejects 99.2% of fast-path rewrites on this traffic**
(595 of 600 distinct train prompts). So the deployed fast path returns the original prompt almost
always. [p11_fast_path.md](p11_fast_path.md) reports 23.7% of rewrites passing at the same
threshold, 0.264564, but that figure is on the OOF candidate rewrites, not on the tagger's own
outputs. Before the fast path's savings are reported, check whether the gate is mis-calibrated
on tagger outputs or the tagger's deletions really do degrade prompts.

### Dashboard live (row 11.5.5)

Checked through Grafana's HTTP API (`/api/search`, `/api/dashboards/uid/terser-serving`,
`/api/ds/query` for every panel target over the last 20 min, `/api/v1/provisioning/alert-rules`,
`/api/prometheus/grafana/api/v1/rules`) and Prometheus `/api/v1/targets`:

```json
"health":    {"database": "ok", "version": "12.2.0"},
"dashboard": {"title": "Terser serving", "uid": "terser-serving", "folder": "Terser",
              "provisioned": true, "panels": 18},
"prometheus_targets": [
  {"job": "pushgateway", "health": "up"},
  {"job": "terser-cpu",  "health": "up"},
  {"job": "terser-gpu",  "health": "down",
   "lastError": "dial tcp: lookup terser-gpu on 127.0.0.11:53: no such host"}]
```

Panel query results with traffic running and the drill value from the next section pushed:
**17 of 18 panels returned data**. The GPU panel is empty because terser-gpu was not running.

| Panel | Series | Last value |
|---|---|---|
| Rolling preservation (canary) | 1 | 0.9186 |
| Last canary run | 1 | 0.88 |
| Canary age | 1 | 103 s |
| Canary pairs (last run) | 1 | 200 |
| Canary preservation over time | 2 | 0.88 |
| Requests / s | 1 | 10.37 |
| Gate rejection rate | 1 | 0.991 |
| Fallback rate | 1 | 0.100 |
| Cache hit rate | 1 | 0.900 |
| p95 latency (all paths) | 1 | 0.50 s |
| Targets up | 1 | 2 (of 3) |
| Requests / s by path and mode | 2 | — |
| Latency p50 / p95 / p99 by path | 3 | — |
| Gate rejections / s by path | 1 | — |
| Fallbacks / s by reason | 1 | — |
| Tokens saved / s by target | 2 | — |
| Net savings (USD / h) by target | 3 | — |
| GPU (vLLM on terser-gpu) | **0** | pending the GPU |

### Alert drill

The rule loaded from file provisioning:

```json
{"uid": "terser-rolling-preservation", "title": "Rolling preservation below 92%",
 "ruleGroup": "terser-quality", "condition": "C",
 "expr": "min(terser_canary_rolling_preservation_ratio)",
 "evaluator": {"type": "lt", "params": [0.92]}, "noDataState": "OK", "provenance": "file"}
```

The drill pushed **synthetic** values; these are not canary results. Each push ran inside
terser-cpu with
`python -m terser.monitoring.canary_metrics --gateway pushgateway:9091 ... --history 185/200 ×6`:

| Time (UTC) | Push | Rolling | Rule state |
|---|---|---|---|
| 13:20:20 | (nothing) | no data | `Normal (NoData)` |
| ~13:23:50 | 176/200 | 1,286/1,400 = **0.9186** | **firing** at 13:24:20, summary "Canary rolling preservation 91.86% is below 92%" |
| ~13:25:50 | 190/200 | 1,300/1,400 = **0.9286** | `Normal` at 13:26:20 |
| ~13:26:40 | `--delete` | no data | `Normal (NoData)` |
| 13:32:29 to 13:34:20 | Repeat after the template fix: 176/200, then `--delete` | 0.9186, then no data | firing at 13:33:20, `Normal (NoData)` at 13:34:20 |

The first run logged an "Error in expanding template" for the NoData instance, because the
summary template dereferenced a missing `A`. The template now uses
`{{ with .Values.A }}…{{ else }}(no data){{ end }}`, and the repeat run logged no template error.
Grafana's file provisioning expands `$VAR`, so the templates avoid `$values`; a test checks this.

The Pushgateway group was deleted after the drill, so no synthetic value is left behind.

**Not verified: alert delivery.** No contact point is configured, so Grafana's default email
receiver failed with "SMTP not configured". The rule fires; routing it to a real receiver (email,
Slack or webhook) is a deployment setting.

### Static checks of the GPU image (no GPU)

| Check | Result |
|---|---|
| `docker run terser-gpu` without the adapter mounted | exit 2, `terser-gpu: no adapter_config.json under /models/rewriter/sft-qwen3b-lora (dvc pull models/rewriter?)` |
| vLLM 0.30.0 `EngineArgs` has `enable_lora`, `max_lora_rank`, `served_model_name`, `max_model_len`, `gpu_memory_utilization`, `dtype` | all True |
| `--lora-modules` accepts `name=path` (`LoRAParserAction`, "Old format: name=path") | yes |
| `PEFTHelper.from_local_dir('/models/rewriter/sft-qwen3b-lora')`, with the adapter config mounted | ok: r = 16, alpha = 32, 7 target modules. The base in `adapter_config.json` is `Qwen/Qwen2.5-3B-Instruct`, matching `TERSER_BASE_MODEL` |
| `sh -n docker/gpu-entrypoint.sh` | ok |

`vllm serve --help` itself could not run: vLLM 0.30 builds its argument parser from `VllmConfig`,
which needs a device, and failed with "Failed to infer device type".

### Teardown

`docker compose down -v` removed all containers, the network and the volumes. `docker ps -a`
is empty. The images are kept.

## Pending (needs the GPU)

Run these once the card is free; the hardware rules forbid running vLLM while a training job
holds it. On this workstation Docker runs from Windows, so use a Windows copy of the repo or of
the build context. A WSL UNC path does not work as a bind-mount source. On a Linux GPU host, run
from the repo root.

```bash
dvc pull models/fast models/rewriter/sft-qwen3b-lora
docker compose up -d --build          # all five services; the first start downloads
                                      # Qwen/Qwen2.5-3B-Instruct (~6 GB) into the hf-cache volume
docker compose ps                     # wait for terser-gpu "healthy" (start period 600 s)
curl -s localhost:8000/v1/models      # expect ids "Qwen/Qwen2.5-3B-Instruct" and "terser-rewriter"
curl -s localhost:8080/healthz        # {"status":"ok","quality_path":true}
curl -s -X POST localhost:8080/rewrite -H 'content-type: application/json' \
  -d '{"prompt":"hey so um, could you maybe write a Python function that checks if a string is a palindrome? It should ignore case. Thanks so much!!","target":"qwen2.5","mode":"quality"}'
                                      # pass: "path":"quality","fallback_used":false
curl -s localhost:9090/api/v1/targets | grep -o '"job":"terser-gpu"[^}]*"health":"[a-z]*"'  # up
docker compose down                   # free the GPU again
```

What these establish, all pending:

1. **`docker compose up` serves both paths** (row 11.3.7 acceptance). Requires a `quality` mode
   response with `path: "quality"`.
2. **GPU panel data.** `vllm:*` metrics on the dashboard.
3. **Quality-path latency under the 0.4 s budget.** Served quality-path latency has never been
   measured (rewriter model card, section 5.3). If the 3B LoRA misses the budget, quality requests
   fall back with `timeout`: the stack is correct, but C11.3 is not met. Raise
   `quality_timeout_s` only if C11.3 is relaxed.

Also open: row 11.5.4 must call `push_canary` (pushgateway `:9091`) after each nightly run for
the canary panels and the alert to carry real data, and the alert needs a contact point.

## Tests and lint

- **Tests:** `tests/monitoring/test_canary_metrics.py` (pooling, validation, gauges,
  push/delete through a fake handler) and `tests/monitoring/test_compose_stack.py`. The stack
  tests check that no service has a profile (so `up` starts both paths), that bind-mount sources
  exist, that the compose serving config equals `fast_onnx.yaml` apart from `quality`, the
  served-model-name agreement, that the default LoRA's base matches its adapter config, the
  Prometheus targets, panel datasources, and that every `terser_*` metric on the dashboard is one
  `ServiceMetrics` or the canary actually exports, plus the alert threshold of 0.92. **16
  passed**.
- **black and ruff:** clean on the new Python files.
- **mypy src:** the new module is clean. The one error is in `src/terser/phase12/candidates.py`,
  an untracked file from another in-progress task.
