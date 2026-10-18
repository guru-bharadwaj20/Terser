"""Structured per-request logging (plan row **11.5.6**; schema in ``docs/serving/logging.md``).

One JSON object per request on the ``terser.requests`` logger. Prompts and rewrites are logged as
sha256 hashes by default; raw text appears only when the caller set ``consent_raw_text`` (the
CONTRIBUTING privacy rule: hashes and metrics by default, raw text only when opted in).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

__all__ = [
    "LOG_FIELDS",
    "LOGGER_NAME",
    "build_record",
    "configure_request_log",
    "log_request",
    "new_request_id",
    "sha256",
]

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


def configure_request_log(path: str | Path | None = None) -> logging.Logger:
    """Give the request logger a sink if it has none (the serving app calls this at start-up).

    Without a handler, INFO records go nowhere under the default logging config, so the request
    log wrote nothing in production. The sink is a JSON-lines file at ``path`` (default
    ``$TERSER_REQUEST_LOG`` if set) or else stdout, one record per line, at INFO.
    An existing handler (tests, a deployment's own logging config) is left alone.
    """
    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:
        return logger
    path = path or os.environ.get("TERSER_REQUEST_LOG")
    handler: logging.Handler
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(path, encoding="utf-8")
    else:
        handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def log_request(record: Mapping[str, Any], logger: logging.Logger | None = None) -> None:
    """Emit ``record`` as one JSON line at INFO."""
    (logger or logging.getLogger(LOGGER_NAME)).info(json.dumps(dict(record), sort_keys=True))
