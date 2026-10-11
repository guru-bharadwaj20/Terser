"""Rows 8.3.3 / 8.3.4: C8.3 val metrics for a rewriter rung."""

from __future__ import annotations

from terser.models.rewriter.val_report import rung_metrics


def _w(pid: str, cat: str, savings: float, fell_back: bool = False) -> dict[str, object]:
    return {
        "id": pid,
        "category": cat,
        "savings": savings,
        "oracle_savings": 0.5,
        "fell_back": fell_back,
        "latency_s": 0.1,
    }


def _l(pid: str, target: str, label: str, tag: str = "r1") -> dict[str, str]:
    return {"candidate_id": f"{pid}:{tag}", "target": target, "label": label}


def test_fallback_counts_preserved_and_gate() -> None:
    rewrites = [_w("a", "qa", 0.5), _w("b", "qa", 0.3), _w("c", "code", 0.0, fell_back=True)]
    labels = [
        _l("a", "q", "preserved"),
        _l("a", "p", "preserved"),
        _l("b", "q", "degraded"),
        _l("b", "p", "preserved"),
        _l("c", "q", "degraded"),
        _l("c", "p", "degraded"),
        _l("a", "q", "degraded", tag="r2"),
    ]
    m = rung_metrics(rewrites, labels, "r1")
    assert m["targets"] == ["p", "q"]
    assert m["median_savings"] == 0.3
    assert m["preserved"] == 5 / 6
    assert m["preserved_by_category"] == {"code": 1.0, "qa": 0.75}
    assert m["c8_3"] is False
