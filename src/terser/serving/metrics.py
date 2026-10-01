"""Prometheus metrics for the inference service (plan row **11.5.1**), served at ``/metrics``.

Each :class:`ServiceMetrics` owns its own :class:`prometheus_client.CollectorRegistry`, so tests
and multiple app instances never collide. The plan's quantities map to:

==========================  ============================================================
RPS                         ``rate(terser_requests_total[1m])`` (by ``path``)
p50 / p95 / p99 latency     ``histogram_quantile(q, terser_request_latency_seconds_bucket)``
fallback rate               ``terser_fallbacks_total / terser_requests_total``
gate rejections             ``terser_gate_rejections_total``
tokens saved                ``terser_tokens_saved_total`` (by ``target``)
net savings                 ``terser_net_savings_usd_total`` (by ``target``)
GPU util / VRAM             ``terser_gpu_utilization_ratio``, ``terser_gpu_memory_used_bytes``
                            (sampled at scrape time via NVML; absent on CPU-only hosts)
==========================  ============================================================
"""

from __future__ import annotations

from typing import Any

__all__ = ["LATENCY_BUCKETS", "ServiceMetrics"]

LATENCY_BUCKETS: tuple[float, ...] = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.075,
    0.1,
    0.15,
    0.2,
    0.3,
    0.4,
    0.6,
    1.0,
    2.0,
    5.0,
)


class _GpuCollector:
    """Scrape-time GPU utilisation and memory via NVML; yields nothing without a GPU."""

    def collect(self) -> Any:
        from prometheus_client.core import GaugeMetricFamily

        try:
            import pynvml

            pynvml.nvmlInit()
        except Exception:  # noqa: BLE001 - no NVML / no driver: CPU host
            return
        util = GaugeMetricFamily("terser_gpu_utilization_ratio", "GPU utilisation", labels=["gpu"])
        mem = GaugeMetricFamily("terser_gpu_memory_used_bytes", "GPU memory used", labels=["gpu"])
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            util.add_metric([str(i)], pynvml.nvmlDeviceGetUtilizationRates(h).gpu / 100.0)
            mem.add_metric([str(i)], float(pynvml.nvmlDeviceGetMemoryInfo(h).used))
        yield util
        yield mem


class ServiceMetrics:
    """Counters and histograms updated once per request."""

    def __init__(self, gpu: bool = True) -> None:
        from prometheus_client import CollectorRegistry, Counter, Histogram

        self.registry = CollectorRegistry()
        r = self.registry
        self.requests = Counter("terser_requests", "Rewrite requests", ["path", "mode"], registry=r)
        self.latency = Histogram(
            "terser_request_latency_seconds",
            "End-to-end request latency",
            ["path"],
            buckets=LATENCY_BUCKETS,
            registry=r,
        )
        self.fallbacks = Counter(
            "terser_fallbacks", "Quality-path fallbacks", ["reason"], registry=r
        )
        self.gate_rejections = Counter(
            "terser_gate_rejections", "Rewrites rejected by the gate", ["path"], registry=r
        )
        self.cache_hits = Counter("terser_cache_hits", "Cache hits", registry=r)
        self.tokens_saved = Counter(
            "terser_tokens_saved", "Target tokens saved", ["target"], registry=r
        )
        self.net_savings = Counter(
            "terser_net_savings_usd",
            "Net savings in USD (positive part)",
            ["target"],
            registry=r,
        )
        self.net_losses = Counter(
            "terser_net_losses_usd",
            "Net losses in USD (requests whose rewriter cost exceeded the savings)",
            ["target"],
            registry=r,
        )
        if gpu:
            r.register(_GpuCollector())

    def observe(
        self,
        *,
        path: str,
        mode: str,
        target: str,
        latency_s: float,
        tokens_saved: int,
        net_savings_usd: float,
        gate_rejected: bool,
        fallback_reason: str | None,
        cached: bool,
    ) -> None:
        """Record one finished request."""
        self.requests.labels(path=path, mode=mode).inc()
        self.latency.labels(path=path).observe(latency_s)
        if fallback_reason:
            self.fallbacks.labels(reason=fallback_reason).inc()
        if gate_rejected:
            self.gate_rejections.labels(path=path).inc()
        if cached:
            self.cache_hits.inc()
        if tokens_saved > 0:
            self.tokens_saved.labels(target=target).inc(tokens_saved)
        if net_savings_usd >= 0:
            self.net_savings.labels(target=target).inc(net_savings_usd)
        else:
            self.net_losses.labels(target=target).inc(-net_savings_usd)

    def render(self) -> tuple[bytes, str]:
        """Exposition-format body and its content type."""
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

        return generate_latest(self.registry), CONTENT_TYPE_LATEST
