"""Canary metrics contract (rows 11.5.4, 11.5.5): pooled rolling value and the pushed gauges."""

from __future__ import annotations

from typing import Any

import pytest

from terser.monitoring import canary_metrics as cm


def test_rolling_pools_counts_over_window() -> None:
    runs = [(100, 100)] + [(185, 200)] * 6 + [(176, 200)]
    ratio, used = cm.rolling_preservation(runs)
    assert used == cm.ROLLING_WINDOW == 7  # the oldest run fell out of the window
    assert ratio == pytest.approx((6 * 185 + 176) / 1400)
    assert ratio < cm.ALERT_THRESHOLD  # the row 11.5.5 drill value fires the alert
    # pooling, not a mean of ratios: a 10-pair run cannot outweigh a 200-pair run
    assert cm.rolling_preservation([(200, 200), (0, 10)])[0] == pytest.approx(200 / 210)
    assert cm.rolling_preservation([(5, 10), (0, 0)]) == (0.5, 1)  # empty runs skipped


@pytest.mark.parametrize("runs", [[], [(0, 0)], [(3, 2)], [(-1, 2)]])
def test_rolling_rejects_bad_runs(runs: list[tuple[int, int]]) -> None:
    with pytest.raises(ValueError):
        cm.rolling_preservation(runs)


def test_registry_has_contract_gauges() -> None:
    reg = cm.canary_registry(190, 200, [(185, 200)] * 6, timestamp=1.5e9)
    values = {s.name: s.value for m in reg.collect() for s in m.samples}
    assert set(values) == set(cm.METRIC_NAMES)
    assert values[cm.PRESERVATION] == pytest.approx(0.95)
    assert values[cm.ROLLING_PRESERVATION] == pytest.approx(1300 / 1400)
    assert values[cm.PROMPTS] == 200
    assert values[cm.ROLLING_RUNS] == 7
    assert values[cm.LAST_RUN] == 1.5e9


def test_push_and_delete_use_canary_job() -> None:
    calls: list[dict[str, Any]] = []

    def handler(url: str, method: str, timeout: Any, headers: Any, data: bytes) -> Any:
        calls.append({"url": url, "method": method, "data": data})
        return lambda: None

    rolling = cm.push_canary("gw:9091", 176, 200, [(185, 200)] * 6, handler=handler)
    assert rolling == pytest.approx(1286 / 1400)
    cm.delete_canary("gw:9091", handler=handler)
    assert [c["method"] for c in calls] == ["PUT", "DELETE"]
    assert all(c["url"].endswith(f"/metrics/job/{cm.CANARY_JOB}") for c in calls)
    assert cm.ROLLING_PRESERVATION.encode() in calls[0]["data"]
