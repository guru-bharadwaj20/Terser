"""HTTP API (11.3.1) and Prometheus /metrics (11.5.1) tests; need the ``serve`` extra."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("prometheus_client")

from fastapi.testclient import TestClient

from terser.serving.accounting import Pricing
from terser.serving.api import RewriteIn, RewriteOut, create_app, load_config
from terser.serving.metrics import ServiceMetrics
from terser.serving.paths import RewriteResult
from terser.serving.service import RewriteService

DOCS = Path(__file__).resolve().parents[2] / "docs" / "serving"


def _client(quality: object = None) -> TestClient:
    def crash(prompt: str, target: str) -> RewriteResult:
        raise RuntimeError("boom")

    svc = RewriteService(
        pricing=Pricing({"gpt-4o": 2.5}),
        quality=crash if quality == "crash" else None,
        counter=lambda s, t: len(s.split()),
    )
    return TestClient(create_app(svc, ServiceMetrics(gpu=False)))


def test_rewrite_example_request() -> None:
    body = RewriteIn.model_json_schema()["examples"][0]
    r = _client().post("/rewrite", json=body)
    assert r.status_code == 200
    out = r.json()
    assert set(RewriteOut.model_fields) == set(out)
    assert out["tokens_after"] < out["tokens_before"] and out["path"] == "fast"
    assert not out["fallback_used"] and out["gate_score"] == 0.0


def test_validation_errors() -> None:
    c = _client()
    assert c.post("/rewrite", json={"prompt": ""}).status_code == 422
    assert c.post("/rewrite", json={"prompt": "hi", "mode": "turbo"}).status_code == 422
    r = c.post("/rewrite", json={"prompt": "hi", "target": "nope"})
    assert r.status_code == 422 and "nope" in r.json()["detail"]


def test_quality_crash_reports_fallback() -> None:
    body = {"prompt": "could you maybe sort this list", "mode": "quality"}
    r = _client("crash").post("/rewrite", json=body)
    out = r.json()
    assert out["fallback_used"] and out["fallback_reason"] == "error" and out["path"] == "fast"


def test_openapi_has_rewrite_with_examples() -> None:
    spec = _client().get("/openapi.json").json()
    assert "/rewrite" in spec["paths"] and "post" in spec["paths"]["/rewrite"]
    schemas = spec["components"]["schemas"]
    assert schemas["RewriteIn"]["examples"] and schemas["RewriteOut"]["examples"]
    assert _client().get("/docs").status_code == 200


def test_committed_openapi_is_current() -> None:
    """docs/serving/openapi.json must be regenerated when the API changes."""
    committed = json.loads((DOCS / "openapi.json").read_text(encoding="utf-8"))
    assert committed == _client().get("/openapi.json").json()


def test_metrics_endpoint() -> None:
    c = _client("crash")
    c.post("/rewrite", json={"prompt": "could you maybe sort this list", "mode": "quality"})
    c.post("/rewrite", json={"prompt": "could you maybe sort this list please"})
    c.post("/rewrite", json={"prompt": "could you maybe sort this list please"})
    r = c.get("/metrics")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/plain")
    text = r.text
    for name in (
        "terser_requests_total",
        "terser_request_latency_seconds_bucket",
        "terser_fallbacks_total",
        "terser_gate_rejections",
        "terser_tokens_saved_total",
        "terser_net_savings_usd_total",
        "terser_cache_hits_total 1.0",
    ):
        assert name in text, name
    assert 'terser_fallbacks_total{reason="error"} 1.0' in text
    assert 'terser_requests_total{mode="fast",path="fast"} 2.0' in text


def test_healthz_and_default_config() -> None:
    assert _client().get("/healthz").json() == {"status": "ok", "quality_path": False}
    cfg = load_config()
    assert cfg["quality_timeout_s"] > 0 and "gpt-4o" in cfg["pricing"]["target_input_per_mtok"]
