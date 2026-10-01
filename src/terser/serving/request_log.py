"""Structured per-request logging (plan row **11.5.6**; schema in ``docs/serving/logging.md``).

One JSON object per request on the ``terser.requests`` logger. Prompts and rewrites are logged as
sha256 hashes by default; raw text appears only when the caller set ``consent_raw_text`` (the
CONTRIBUTING privacy rule: hashes and metrics by default, raw text only when opted in).
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from collections.abc import Mapping
from typing import Any

__all__ = ["LOG_FIELDS", "LOGGER_NAME", "build_record", "log_request", "new_request_id", "sha256"]

LOGGER_NAME = "terser.requests"
SCHEMA_VERSION = 1

LOG_FIELDS: tuple[str, ...] = (
    "schema_version",
    "ts",
    "request_id",
    "target",
    "mode",
    "path",
    "prompt_sha256",
    "rewrite_sha256",
    "tokens_before",
    "tokens_after",
    "gate_score",
    "gate_rejected",
    "fallback_used",
    "fallback_reason",
    "cached",
    "latency_ms",
    "tokens_saved",
    "rewriter_cost_usd",
    "net_savings_usd",
    "consent_raw_text",
)
RAW_FIELDS: tuple[str, ...] = ("prompt", "rewrite")


def new_request_id() -> str:
    """A random request id (uuid4 hex)."""
    return uuid.uuid4().hex


def sha256(text: str) -> str:
    """Hex sha256 of UTF-8 text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_record(
    *, prompt: str, rewrite: str, consent_raw_text: bool = False, **fields: Any
) -> dict[str, Any]:
    """Log record with the schema's fields; raw text included only with consent."""
    rec: dict[str, Any] = {k: None for k in LOG_FIELDS}
    rec.update(
        schema_version=SCHEMA_VERSION,
        ts=time.time(),
        prompt_sha256=sha256(prompt),
        rewrite_sha256=sha256(rewrite),
        consent_raw_text=bool(consent_raw_text),
    )
    unknown = set(fields) - set(LOG_FIELDS)
    if unknown:
        raise KeyError(f"fields not in the log schema: {sorted(unknown)}")
    rec.update(fields)
    if consent_raw_text:
        rec["prompt"], rec["rewrite"] = prompt, rewrite
    return rec


def log_request(record: Mapping[str, Any], logger: logging.Logger | None = None) -> None:
    """Emit ``record`` as one JSON line at INFO."""
    (logger or logging.getLogger(LOGGER_NAME)).info(json.dumps(dict(record), sort_keys=True))
