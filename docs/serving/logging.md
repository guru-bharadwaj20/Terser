# Request log schema (row 11.5.6)

Every `POST /rewrite` emits exactly one JSON object, at `INFO`, on the `terser.requests` logger
([`src/terser/serving/request_log.py`](../../src/terser/serving/request_log.py)). Keys are sorted.
Fields that do not apply are `null`, never missing. `build_record` raises `KeyError` for any field
not in this schema, so personal data cannot be added to the log by accident.

**Privacy default.** The prompt and the rewrite are logged only as SHA-256 hashes. The raw text
fields `prompt` and `rewrite` appear only when the request set `consent_raw_text: true`. This
follows CONTRIBUTING, *Data, privacy and LLM usage*: hashes and metrics by default, raw text only
when the user opted in, and retained for no more than 30 days. The log sink must enforce that
retention, for example with a 30-day index lifecycle.

| Field | Type | Meaning |
| :--- | :--- | :--- |
| `schema_version` | int | `1`. Bump on any breaking change to this table |
| `ts` | float | Unix time (seconds) when the record was built |
| `request_id` | string | uuid4 hex. Returned in the response so a client can quote it |
| `target` | string | Target model |
| `mode` | `fast` \| `quality` | Requested mode |
| `path` | `fast` \| `quality` \| `original` | Path that produced the result |
| `prompt_sha256` | hex string | SHA-256 of the UTF-8 prompt |
| `rewrite_sha256` | hex string | SHA-256 of the returned text |
| `tokens_before` / `tokens_after` | int | Token counts on `target` |
| `tokens_saved` | int | `tokens_before − tokens_after` (can be 0) |
| `gate_score` | float | Gate `P(degraded)` |
| `gate_rejected` | bool | Original returned because of the gate |
| `fallback_used` | bool | Requested path not used |
| `fallback_reason` | string \| null | `timeout`, `error`, `unavailable`, `fast_error` |
| `cached` | bool | LRU hit |
| `latency_ms` | float | Service time |
| `rewriter_cost_usd` | float | Cost of producing the rewrite (0 for a cache hit) |
| `net_savings_usd` | float | `price_T × tokens_saved − rewriter_cost_usd` (row 11.3.6) |
| `consent_raw_text` | bool | Whether the raw text fields are present |
| `prompt`, `rewrite` | string | **Only** when `consent_raw_text` is `true` |

Example (no consent):

```json
{"cached": false, "consent_raw_text": false, "fallback_reason": null, "fallback_used": false,
 "gate_rejected": false, "gate_score": 0.0, "latency_ms": 1.71, "mode": "fast",
 "net_savings_usd": 2.75e-05, "path": "fast",
 "prompt_sha256": "<64 hex>", "request_id": "41ec4d73f4e345048c4a6ace4b88c534",
 "rewrite_sha256": "<64 hex>", "rewriter_cost_usd": 0.0, "schema_version": 1,
 "target": "gpt-4o", "tokens_after": 18, "tokens_before": 29, "tokens_saved": 11,
 "ts": 1790870480.3}
```

Request IDs join this log with the Prometheus series (aggregates) and with the nightly canary
(row 11.5.4) and drift jobs, which read only hashes, counts and categories from it.
