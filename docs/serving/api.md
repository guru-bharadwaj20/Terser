# Terser inference API

Plan rows 11.3.1 (API), 11.3.4 (cache), 11.3.5 (timeout fallback), 11.3.6 (net savings) and
11.5.1 (`/metrics`). Code: [`src/terser/serving/`](../../src/terser/serving/). Machine-readable
schema: [`openapi.json`](openapi.json). Regenerate it after any API change with
`python -m terser.serving.api --openapi docs/serving/openapi.json`. A test fails if the committed
file is stale.

```bash
pip install -e ".[serve]"
uvicorn terser.serving.api:app --host 0.0.0.0 --port 8080
# Swagger UI: http://localhost:8080/docs   ReDoc: /redoc   schema: /openapi.json
```

Configuration: [`configs/serving/default.yaml`](../../configs/serving/default.yaml), or the file
named by `TERSER_SERVING_CONFIG`. The quality path is off until `quality.base_url` and
`quality.model` point at a vLLM server running the trained rewriter (row 11.3.3).

## Endpoints

| Method | Path | Purpose |
| :--- | :--- | :--- |
| `POST` | `/rewrite` | Rewrite one prompt for one target |
| `GET` | `/metrics` | Prometheus exposition format (row 11.5.1) |
| `GET` | `/healthz` | Liveness, and whether the quality path is configured |

## `POST /rewrite`

### Request

| Field | Type | Default | Meaning |
| :--- | :--- | :--- | :--- |
| `prompt` | string, 1 to 32,000 chars | required | The casual prompt |
| `target` | string | `gpt-4o` | Target model. Selects the tokenizer used for counting and the price used for accounting |
| `mode` | `fast` \| `quality` | `fast` | `fast`: CPU span path. `quality`: generative rewriter within the latency budget, falling back to the fast path |
| `consent_raw_text` | bool | `false` | Allow raw prompt and rewrite text in the request log. Hashes only otherwise ([logging.md](logging.md)) |

### Response

| Field | Meaning |
| :--- | :--- |
| `rewrite` | Text to send to the target. Equals `prompt` when the gate rejected the candidate or every rewriter failed |
| `tokens_before`, `tokens_after` | Ground-truth token counts of `prompt` and `rewrite` on `target` |
| `gate_score` | Gate `P(degraded)` of the candidate rewrite (0 when the candidate equals the prompt) |
| `fallback_used` | `true` when the requested path was not the one that produced the rewrite |
| `fallback_reason` | `timeout` (quality path over `quality_timeout_s`), `error` (quality path raised), `unavailable` (no quality path configured), `fast_error` (fast path raised too, so the original is returned), or `null` |
| `path` | Path that produced the result: `fast`, `quality` or `original` |
| `gate_rejected` | `true` when `gate_score > gate_threshold` and the original prompt was returned |
| `cached` | `true` for an LRU hit (key: sha256 of target, mode and the normalised prompt) |
| `net_savings_usd` | `price_T × (tokens_before − tokens_after) − rewriter_cost` for this request |
| `request_id` | Correlates with the structured log line |
| `latency_ms` | Service time, without HTTP overhead |

`422` is returned for a schema violation, an unknown `mode`, or a `target` with no configured price.

### Examples

The responses below come from a real run of the fast path (rule-based operators and the
`ConstraintGate` stand-in, the default config) on the CPU dev laptop. The first request includes
loading the tokenizer, which is why its latency is high.

**Fast path**

```http
POST /rewrite
Content-Type: application/json

{"prompt": "hey so um, could you maybe write a Python function that checks if a string is a palindrome? It should ignore case. Thanks so much!!", "target": "gpt-4o", "mode": "fast"}
```

```json
{
  "rewrite": "Write a Python function that checks if a string is a palindrome. It should ignore case.",
  "tokens_before": 29,
  "tokens_after": 18,
  "gate_score": 0.0,
  "fallback_used": false,
  "path": "fast",
  "mode": "fast",
  "target": "gpt-4o",
  "gate_rejected": false,
  "fallback_reason": null,
  "cached": false,
  "net_savings_usd": 2.75e-05,
  "request_id": "41ec4d73f4e345048c4a6ace4b88c534",
  "latency_ms": 498.02
}
```

`net_savings_usd` = 2.50 USD / 1M tokens × 11 tokens − 0 (the fast path's configured cost).

**Quality mode with no quality path configured** falls back to the fast path:

```json
{
  "rewrite": "Write a Python function that checks if a string is a palindrome. It should ignore case.",
  "tokens_before": 29,
  "tokens_after": 18,
  "gate_score": 0.0,
  "fallback_used": true,
  "path": "fast",
  "mode": "quality",
  "target": "gpt-4o",
  "gate_rejected": false,
  "fallback_reason": "unavailable",
  "cached": false,
  "net_savings_usd": 2.75e-05,
  "request_id": "c314026ec5cd4e1b8a4c248f3d388f43",
  "latency_ms": 1.97
}
```

**Unknown target**

```json
{"detail": "no input price configured for target 'claude-9'"}
```

```bash
curl -s localhost:8080/rewrite -H 'content-type: application/json' \
  -d '{"prompt": "could you maybe sort this list for me please", "mode": "fast"}'
```

## Caching

The LRU cache (11.3.4) keys on `sha256(target \0 mode \0 normalise(prompt))`. Normalisation is
NFC, newline unification and trimming of trailing whitespace only. Case and inner spacing are kept
because they can change token counts. Fallback results are never cached. Measured hit and miss
latency: [`reports/p11_cache_latency.md`](../../reports/p11_cache_latency.md).

## Metrics

See [`src/terser/serving/metrics.py`](../../src/terser/serving/metrics.py) for the mapping from the
plan's quantities (RPS, p50/p95/p99 per path, fallback rate, gate rejections, tokens saved, net
savings, GPU utilisation and VRAM) to Prometheus series and queries.
