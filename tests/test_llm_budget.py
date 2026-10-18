"""Tests for the API budget guard (terser.llm): caps, ledger, disk cache. No real API calls."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

import terser.llm.budget as budget_mod
from terser.llm import (
    CACHE_ONLY_ENV,
    Budget,
    BudgetExceededError,
    Ledger,
    LLMCacheMissError,
    LLMClient,
    TransportResult,
    Usage,
    cache_key,
)

MSGS = [{"role": "user", "content": "make this prompt shorter please"}]


class FakeTransport:
    def __init__(self, input_tokens: int = 100, output_tokens: int = 50) -> None:
        self.calls = 0
        self.usage = (input_tokens, output_tokens)

    def __call__(self, provider, model, messages, max_tokens, system, params) -> TransportResult:
        self.calls += 1
        return TransportResult("short prompt", *self.usage)


def write_cfg(tmp_path: Path, phases: dict[str, Any], prices: dict[str, Any] | None = None) -> Path:
    cfg = {
        "ledger_path": str(tmp_path / "ledger.json"),
        "cache_dir": str(tmp_path / "responses"),
        "phases": phases,
        "prices": (
            prices
            if prices is not None
            else {"priced": {"input_per_mtok": 1.0, "output_per_mtok": 2.0}}
        ),
    }
    path = tmp_path / "budget.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def client(tmp_path: Path, transport: FakeTransport, **phase: Any) -> LLMClient:
    spec = {"max_usd": 1.0, "max_tokens": 10_000, "allow_unpriced": False, **phase}
    return LLMClient("p", budget_path=write_cfg(tmp_path, {"p": spec}), transport=transport)


def ledger_json(tmp_path: Path) -> dict[str, Any]:
    return json.loads((tmp_path / "ledger.json").read_text(encoding="utf-8"))


def test_repo_budget_yaml_loads_all_phases(tmp_path: Path) -> None:
    path = Path(__file__).resolve().parents[1] / "configs" / "budget.yaml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert {f"phase{i}" for i in range(14)} <= set(cfg["phases"])
    assert all(v["input_per_mtok"] is None for v in cfg["prices"].values())
    b = Budget.load(path, "phase3", ledger_path=tmp_path / "l.json")
    assert b.max_usd == 0 and b.max_tokens == 0


def test_cap_reached_refuses_without_calling_transport(tmp_path: Path) -> None:
    fake = FakeTransport(input_tokens=400, output_tokens=500)
    c = client(tmp_path, fake, max_tokens=1_000)
    c.complete("anthropic", "priced", MSGS, max_tokens=500)
    assert fake.calls == 1
    with pytest.raises(BudgetExceededError, match="token cap"):
        c.complete("anthropic", "priced", [{"role": "user", "content": "other"}], max_tokens=500)
    assert fake.calls == 1


def test_usd_cap_refuses(tmp_path: Path) -> None:
    fake = FakeTransport()
    c = client(tmp_path, fake, max_usd=0.001, max_tokens=10**9)
    # estimate: ~10 input tokens + 1000 output at $2/Mtok = $0.002 > $0.001
    with pytest.raises(BudgetExceededError, match="USD cap"):
        c.complete("anthropic", "priced", MSGS, max_tokens=1000)
    assert fake.calls == 0


def test_cached_call_costs_zero(tmp_path: Path) -> None:
    fake = FakeTransport()
    c = client(tmp_path, fake)
    first = c.complete("anthropic", "priced", MSGS, max_tokens=64, temperature=0)
    before = ledger_json(tmp_path)
    second = c.complete("anthropic", "priced", MSGS, max_tokens=64, temperature=0)
    assert fake.calls == 1
    assert not first.cached and first.usd == pytest.approx((100 * 1 + 50 * 2) / 1e6)
    assert second.cached and second.usd == 0 and second.text == first.text
    assert second.cache_key == first.cache_key
    assert ledger_json(tmp_path) == before
    assert len(before["phases"]["p"]["calls"]) == 1


def test_cache_only_serves_hits_and_refuses_misses(tmp_path: Path) -> None:
    fake = FakeTransport()
    first = client(tmp_path, fake).complete("local", "m", MSGS, max_tokens=64, temperature=0)
    before = ledger_json(tmp_path)
    offline = LLMClient("p", budget_path=tmp_path / "budget.yaml", transport=fake, cache_only=True)
    hit = offline.complete("local", "m", MSGS, max_tokens=64, temperature=0)
    assert hit.cached and hit.text == first.text
    with pytest.raises(LLMCacheMissError, match="cache miss"):
        offline.complete("local", "m", MSGS, max_tokens=64, temperature=1)
    assert fake.calls == 1  # the miss never reached the transport
    assert ledger_json(tmp_path) == before  # nor the ledger


@pytest.mark.parametrize(("value", "expected"), [("1", True), ("TRUE", True), ("0", False)])
def test_cache_only_from_env(tmp_path: Path, monkeypatch, value: str, expected: bool) -> None:
    monkeypatch.setenv(CACHE_ONLY_ENV, value)
    fake = FakeTransport()
    c = client(tmp_path, fake)
    assert c.cache_only is expected
    if expected:
        with pytest.raises(LLMCacheMissError):
            c.complete("local", "m", MSGS, max_tokens=8)
        assert fake.calls == 0
    monkeypatch.delenv(CACHE_ONLY_ENV)
    assert client(tmp_path, fake).cache_only is False


def test_zero_cap_phase_refuses(tmp_path: Path) -> None:
    fake = FakeTransport()
    c = client(tmp_path, fake, max_usd=0, max_tokens=0, allow_unpriced=True)
    with pytest.raises(BudgetExceededError, match="not allowed"):
        c.complete("anthropic", "priced", MSGS, max_tokens=10)
    assert fake.calls == 0


def test_unpriced_model_requires_allow_unpriced(tmp_path: Path) -> None:
    fake = FakeTransport()
    with pytest.raises(BudgetExceededError, match="no price"):
        client(tmp_path, fake).complete("openai", "mystery", MSGS, max_tokens=10)
    assert fake.calls == 0
    resp = client(tmp_path, fake, allow_unpriced=True).complete(
        "openai", "mystery", MSGS, max_tokens=10
    )
    assert fake.calls == 1 and resp.usd is None
    rec = ledger_json(tmp_path)["phases"]["p"]
    assert rec["tokens"] == 150 and rec["usd"] == 0


def test_unpriced_still_enforces_token_cap(tmp_path: Path) -> None:
    fake = FakeTransport()
    c = client(tmp_path, fake, max_tokens=50, allow_unpriced=True)
    with pytest.raises(BudgetExceededError, match="token cap"):
        c.complete("openai", "mystery", MSGS, max_tokens=100)
    assert fake.calls == 0


def test_ledger_survives_reload(tmp_path: Path) -> None:
    fake = FakeTransport()
    client(tmp_path, fake).complete("anthropic", "priced", MSGS, max_tokens=10)
    reloaded = Budget.load(tmp_path / "budget.yaml", "p")
    spent = reloaded.ledger.spent("p")
    assert spent["tokens"] == 150 and spent["usd"] == pytest.approx(200 / 1e6)
    call = spent["calls"][0]
    assert {"timestamp", "provider", "model", "input_tokens", "output_tokens", "usd"} <= set(call)
    assert call["cache_key"]
    # a new client sees prior spend
    c2 = client(tmp_path, FakeTransport(), max_tokens=150 + 20)
    with pytest.raises(BudgetExceededError):
        c2.complete("anthropic", "priced", [{"role": "user", "content": "x"}], max_tokens=30)


def test_cache_key_params_and_order() -> None:
    a = cache_key("anthropic", "m", MSGS, {"temperature": 0, "top_p": 1, "max_tokens": 5})
    b = cache_key("anthropic", "m", MSGS, {"max_tokens": 5, "top_p": 1, "temperature": 0})
    c = cache_key("anthropic", "m", MSGS, {"temperature": 1, "top_p": 1, "max_tokens": 5})
    assert a == b != c
    assert a != cache_key("anthropic", "m", MSGS, {"temperature": 0, "top_p": 1}, system="s")
    assert a != cache_key("openai", "m", MSGS, {"temperature": 0, "top_p": 1, "max_tokens": 5})


def test_different_params_miss_cache(tmp_path: Path) -> None:
    fake = FakeTransport()
    c = client(tmp_path, fake)
    r1 = c.complete("anthropic", "priced", MSGS, max_tokens=10, temperature=0)
    r2 = c.complete("anthropic", "priced", MSGS, max_tokens=10, temperature=1)
    assert fake.calls == 2 and r1.cache_key != r2.cache_key


def test_atomic_write_leaves_no_partial_file(tmp_path: Path, monkeypatch) -> None:
    ledger = Ledger(tmp_path / "ledger.json")
    ledger.add("p", Usage("anthropic", "m", 1, 1), 0.0)
    good = (tmp_path / "ledger.json").read_text(encoding="utf-8")

    def boom(*args: Any, **kwargs: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(budget_mod.os, "replace", boom)
    with pytest.raises(OSError):
        ledger.add("p", Usage("anthropic", "m", 5, 5), 0.0)
    assert (tmp_path / "ledger.json").read_text(encoding="utf-8") == good
    assert [p.name for p in tmp_path.iterdir()] == ["ledger.json"]
    monkeypatch.undo()
    ledger.reload()  # the failed add must not survive in memory
    assert ledger.spent("p")["tokens"] == 2


def test_two_ledgers_on_one_file_see_each_other(tmp_path: Path) -> None:
    a, b = Ledger(tmp_path / "ledger.json"), Ledger(tmp_path / "ledger.json")
    a.add("p", Usage("local", "m", 3, 4), 0.0)
    b.reload()
    assert b.spent("p")["tokens"] == 7
    b.add("p", Usage("local", "m", 1, 1), 0.0)
    a.add("p", Usage("local", "m", 1, 1), 0.0)  # merges with b's write on disk
    b.reload()
    assert b.spent("p")["tokens"] == 11 and len(b.spent("p")["calls"]) == 3
    assert "\n" not in (tmp_path / "ledger.json").read_text(encoding="utf-8")  # compact


def test_transport_error_records_nothing(tmp_path: Path) -> None:
    def failing(*args: Any) -> TransportResult:
        raise RuntimeError("network")

    c = LLMClient(
        "p",
        budget_path=write_cfg(tmp_path, {"p": {"max_usd": 1, "max_tokens": 1000}}),
        transport=failing,
    )
    with pytest.raises(RuntimeError):
        c.complete("anthropic", "priced", MSGS, max_tokens=10)
    # the reservation is released: nothing spent, nothing left pending
    data = json.loads((tmp_path / "ledger.json").read_text(encoding="utf-8"))
    assert data["phases"] == {} and not any(data["reserved"].values())


def test_reservations_stop_concurrent_workers_exceeding_the_cap(tmp_path: Path) -> None:
    """Two budgets on one ledger (two worker processes) see each other's pending calls."""
    cfg = write_cfg(tmp_path, {"p": {"max_usd": 1, "max_tokens": 100}})
    a, b = Budget.load(cfg, "p"), Budget.load(cfg, "p")
    rid = a.reserve(Usage("local", "m", 40, 40))
    with pytest.raises(BudgetExceededError):
        b.reserve(Usage("local", "m", 20, 20))  # 80 pending + 40 > 100
    a.record(Usage("local", "m", 10, 10), rid)  # the actual call was smaller
    b.reserve(Usage("local", "m", 20, 20))  # 20 spent + 40 fits now
