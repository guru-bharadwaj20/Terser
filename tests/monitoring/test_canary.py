"""Quality canary (row 11.5.4): canary selection, one cycle through a fake service, fake target
and judge, history / rolling value / push, the report, and the dashboard read-back.

No server, GPU or LLM: the service and Prometheus/Grafana are ``httpx.MockTransport`` stubs and
the target and judge are a fake :class:`terser.llm.LLMClient` transport.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

from terser.llm import LLMClient, TransportResult
from terser.monitoring import canary as cn
from terser.monitoring import canary_metrics as cm

REPO = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------------------------
# Selection


def _val_records() -> list[dict[str, Any]]:
    rows = []
    for cat, n_src in (("code", 30), ("fact", 50), ("explain", 20)):
        for i in range(n_src):
            for style in ("texting", "hedging"):  # two casual variants per source item
                rows.append(
                    {
                        "id": f"{cat}:{i}:{style}",
                        "source_id": f"{cat}:{i}",
                        "source": cat,
                        "category": cat,
                        "style": style,
                        "prompt": f"{cat} {i} {style}",
                        "reference": "r",
                        "rejected": "False",
                        "stand_in": "False",
                        "clean": "not kept",
                    }
                )
    rows.append({**rows[0], "id": "code:x:rej", "source_id": "code:x", "rejected": "True"})
    return rows


def test_allocate_is_proportional_largest_remainder() -> None:
    assert cn.allocate({"a": 60, "b": 100, "c": 40}, 20) == {"a": 6, "b": 10, "c": 4}
    # the real val split (1,133 records)
    val = {"code": 220, "creative": 169, "explain": 122, "fact": 235, "reason": 175}
    alloc = cn.allocate({**val, "transform": 212}, 200)
    want = {"code": 39, "creative": 30, "explain": 22, "fact": 41, "reason": 31}
    assert alloc == {**want, "transform": 37}
    assert sum(cn.allocate({"a": 1, "b": 1, "c": 1}, 2).values()) == 2
    assert cn.allocate({"a": 1, "b": 1, "c": 1}, 2) == {"a": 1, "b": 1, "c": 0}  # ties by name
    with pytest.raises(ValueError):
        cn.allocate({"a": 2}, 3)


def test_select_canary_stratified_deterministic_distinct_sources() -> None:
    rows = _val_records()
    out = cn.select_canary(rows, n=50, seed=0)
    assert out == cn.select_canary(list(reversed(rows)), n=50, seed=0)  # order-independent
    assert out != cn.select_canary(rows, n=50, seed=1)
    assert Counter(r["category"] for r in out) == {"code": 15, "fact": 25, "explain": 10}
    assert len({r["source_id"] for r in out}) == 50  # one variant per source item first
    assert all(r["id"] != "code:x:rej" for r in out)
    assert [r["id"] for r in out] == sorted(r["id"] for r in out)
    assert set(out[0]) == {*cn.CANARY_FIELDS, "split"} and out[0]["split"] == "val"
    # more prompts than source items: second variants fill the remainder
    big = cn.select_canary(rows, n=180, seed=0)
    assert len(big) == 180 and len({r["source_id"] for r in big}) == 100


def test_write_canary_manifest_and_refuses_test_split(tmp_path: Path) -> None:
    src = tmp_path / "val.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in _val_records()), encoding="utf-8")
    out = tmp_path / "canary" / "canary_v9.jsonl"
    manifest = cn.write_canary(src, out, n=20, seed=0)
    assert manifest["rows"] == 20 and manifest["version"] == "canary_v9"
    assert json.loads(out.with_suffix(".json").read_text())["sha256"] == manifest["sha256"]
    test_split = tmp_path / "test.jsonl"
    test_split.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ValueError, match="test split"):
        cn.write_canary(test_split, tmp_path / "c.jsonl", n=5)


def test_committed_canary_matches_rule(require_dvc: Any) -> None:
    manifest = json.loads((REPO / "data/canary/canary_v1.json").read_text(encoding="utf-8"))
    assert manifest["rows"] == cn.CANARY_SIZE and manifest["seed"] == cn.CANARY_SEED
    assert manifest["source"] == "data/processed/splits/val.jsonl"
    assert sum(manifest["by_category"].values()) == 200
    data = REPO / "data/canary/canary_v1.jsonl"
    require_dvc(data, "dvc pull data/canary/canary_v1.jsonl.dvc")
    rows = [json.loads(x) for x in data.read_text(encoding="utf-8").splitlines()]
    assert Counter(r["category"] for r in rows) == manifest["by_category"]
    assert {r["split"] for r in rows} == {"val"}


# --------------------------------------------------------------------------------------------
# Fakes: the service, the target and the judge

PROMPTS = [
    {"id": "e:1:t", "source_id": "e:1", "category": "explain", "prompt": "explain x pls GOOD KEEP"},
    {"id": "e:2:t", "source_id": "e:2", "category": "explain", "prompt": "explain y pls GOOD DROP"},
    {"id": "c:1:t", "source_id": "c:1", "category": "creative", "prompt": "poem pls GOOD KEEP"},
    {"id": "t:1:t", "source_id": "t:1", "category": "transform", "prompt": "fix this GOOD"},
    {"id": "t:2:t", "source_id": "t:2", "category": "transform", "prompt": "fix that GOOD"},
    {"id": "c:2:t", "source_id": "c:2", "category": "creative", "prompt": "story GOOD DROP"},
]
# Per target: KEEP rewrites keep "GOOD" (judged tie -> preserved), DROP rewrites lose it
# (judge prefers the original -> degraded), the rest are gate-rejected (unchanged -> preserved).
PER_TARGET = (4, 6)


def _service_rewrite(prompt: str) -> str:
    if "KEEP" in prompt:
        return prompt.replace(" pls", "")
    if "DROP" in prompt:
        return prompt.replace(" GOOD", "")
    return prompt


class FakeService:
    """``POST /rewrite`` and ``GET /healthz`` of the deployed service."""

    def __init__(self, fail_first: int = 0, always_fail: str | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.fail_first = fail_first
        self.always_fail = always_fail

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/healthz":
            return httpx.Response(200, json={"status": "ok", "quality_path": True})
        body = json.loads(request.content)
        self.calls.append(body)
        if len(self.calls) <= self.fail_first or body["prompt"] == self.always_fail:
            return httpx.Response(503, text="busy")
        rewrite = _service_rewrite(body["prompt"])
        unchanged = rewrite == body["prompt"]
        return httpx.Response(
            200,
            json={
                "rewrite": rewrite,
                "tokens_before": 10,
                "tokens_after": 10 if unchanged else 8,
                "gate_score": 0.9 if unchanged else 0.1,
                "fallback_used": False,
                "path": "quality",
                "mode": body["mode"],
                "target": body["target"],
                "gate_rejected": unchanged,
                "fallback_reason": None,
                "cached": False,
                "net_savings_usd": 0.0,
                "request_id": "x",
                "latency_ms": 1.0,
            },
        )


def _section(text: str, name: str) -> str:
    m = re.search(rf"\[{name}\]\n(.*?)\n\[End of {name}\]", text, re.DOTALL)
    assert m, name
    return m.group(1)


class FakeLLM:
    """Targets echo the text; the judge prefers the response that contains 'GOOD'."""

    def __init__(self) -> None:
        self.models: Counter[str] = Counter()

    def __call__(self, provider, model, messages, max_tokens, system, params):  # type: ignore[no-untyped-def]
        self.models[model] += 1
        content = messages[0]["content"]
        if model in {t.model for t in cn.TARGETS.values()}:
            seed = params.get("seed")
            return TransportResult(f"answer to {content}" + (f" s{seed}" if seed else ""), 5, 7)
        a, b = _section(content, "Response A"), _section(content, "Response B")
        pa = 0.8 if "GOOD" in a and "GOOD" not in b else 0.1 if "GOOD" in b else 0.3
        pb = 0.8 if "GOOD" in b and "GOOD" not in a else 0.1 if "GOOD" in a else 0.3
        pt = max(1e-6, 1.0 - pa - pb)
        top = (("A", math.log(pa)), ("B", math.log(pb)), ("tie", math.log(pt)))
        best = max(top, key=lambda x: x[1])[0]
        toks = (('{"verdict": "', (("x", 0.0),)), (best, top))
        return TransportResult(json.dumps({"verdict": best, "reason": "r"}), 50, 9, True, toks)


def _fake_client(tmp_path: Path, transport: Any, ledger: Path) -> LLMClient:
    budget = tmp_path / "budget.yaml"
    budget.write_text(
        yaml.safe_dump(
            {
                "phases": {"phase11": {"max_usd": 0, "max_tokens": 1_000_000}},
                "providers": {"local": {"base_url": "http://127.0.0.1:9/v1", "priced": False}},
            }
        ),
        encoding="utf-8",
    )
    return LLMClient(
        "phase11",
        budget_path=budget,
        transport=transport,
        ledger_path=ledger,
        cache_dir=tmp_path / "llm_cache",
    )


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """CLI wiring: the fake service over HTTP, the fake LLM, a recording Pushgateway handler."""
    canary = tmp_path / "canary_v1.jsonl"
    canary.write_text("".join(json.dumps(p) + "\n" for p in PROMPTS), encoding="utf-8")
    service = FakeService()
    llm = FakeLLM()
    urls: list[str | None] = []
    pushes: list[dict[str, Any]] = []

    def http(timeout: float, auth: Any = None) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(service), timeout=timeout)

    def client(url: str | None, *, phase: str, ledger_path: Path, cache_dir: Any) -> LLMClient:
        urls.append(url)
        return _fake_client(tmp_path, llm, ledger_path)

    def handler(url: str, method: str, timeout: Any, headers: Any, data: bytes) -> Any:
        pushes.append({"url": url, "method": method, "data": data.decode()})
        return lambda: None

    real_push = cm.push_canary
    monkeypatch.setattr(cn, "_http", http)
    monkeypatch.setattr(cn, "make_llm_client", client)
    monkeypatch.setattr(
        cn, "push_canary", lambda *a, **k: real_push(*a, **{**k, "handler": handler})
    )
    return {
        "dir": tmp_path,
        "canary": canary,
        "service": service,
        "llm": llm,
        "urls": urls,
        "pushes": pushes,
        "history": tmp_path / "history.jsonl",
        "report": tmp_path / "report.md",
    }


def _common(env: dict[str, Any]) -> list[str]:
    return [
        "--history",
        str(env["history"]),
        "--report",
        str(env["report"]),
    ]


def _llm_args(env: dict[str, Any]) -> list[str]:
    return ["--metric-cache", str(env["dir"] / "metrics"), "--raw-dir", str(env["dir"] / "raw")]


# --------------------------------------------------------------------------------------------
# One cycle


def test_run_cycle_end_to_end(env: dict[str, Any]) -> None:
    run_dir = env["dir"] / "runs" / "r1"
    rc = cn.main(
        [
            "run",
            "--canary",
            str(env["canary"]),
            "--run-dir",
            str(run_dir),
            "--base-url",
            "http://svc:8080",
            "--mode",
            "quality",
            "--llm-url",
            "http://all:8000/v1",
            "--target-url",
            "phi3.5-mini=http://phi:8000/v1",
            "--judge-url",
            "http://judge:8000/v1",
            "--gateway",
            "gw:9091",
            "--warmup",
            "1",
            *_common(env),
            *_llm_args(env),
        ]
    )
    assert rc == 0
    # service: one warm-up, then every prompt once per target, with the service target names
    calls = env["service"].calls
    assert calls[0]["prompt"] == cn.WARMUP_PROMPT
    assert Counter(c["target"] for c in calls[1:]) == {"qwen2.5": 6, "phi3.5": 6}
    assert {c["mode"] for c in calls} == {"quality"}
    # endpoints: per-target override, the shared URL, then the judge
    assert env["urls"] == ["http://all:8000/v1", "http://phi:8000/v1", "http://judge:8000/v1"]
    # only changed rewrites (and their originals) reach the targets and the judge
    cands = [json.loads(x) for x in (run_dir / "candidates_qwen2.5-7b.jsonl").open()]
    assert sorted(c["prompt_id"] for c in cands) == ["c:1:t", "c:2:t", "e:1:t", "e:2:t"]
    assert all(c["candidate_id"].endswith(":canary") for c in cands)
    responses = [json.loads(x) for x in (run_dir / "responses_qwen2.5-7b.jsonl").open()]
    assert {r["prompt_id"] for r in responses} == {"c:1:t", "c:2:t", "e:1:t", "e:2:t"}
    assert env["llm"].models[cn.DEFAULT_JUDGE] > 0

    result = json.loads((run_dir / "result.json").read_text())
    p, t = PER_TARGET
    assert (result["preserved"], result["total"]) == (2 * p, 2 * t)
    assert result["unchanged"] == 4 and result["judged"] == 8 and result["degraded"] == 4
    assert result["by_target"]["qwen2.5-7b"]["preservation"] == pytest.approx(p / t)
    assert result["by_category"]["transform"]["preservation"] == 1.0  # unchanged = preserved
    assert result["by_category"]["explain"]["preservation"] == 0.5
    tc = result["by_target_category"]["phi3.5-mini"]["creative"]
    assert (tc["preserved"], tc["total"]) == (1, 2)
    assert result["rolling"] == pytest.approx(p / t) and result["rolling_runs"] == 1
    assert result["alert"] and not result["meets_overall_bar"]
    assert result["pushed"]["gateway"] == "gw:9091"
    assert result["gate_rejected"] == 4 and result["paths"] == {"quality": 12}

    # pushed gauges: the pooled run value over both targets
    push = env["pushes"][-1]
    assert push["method"] == "PUT" and push["url"].endswith(f"/metrics/job/{cm.CANARY_JOB}")
    assert f"{cm.PROMPTS} 12.0" in push["data"]
    assert f"{cm.ROLLING_PRESERVATION} {8 / 12}" in push["data"]

    history = cn.read_history(env["history"])
    assert [h["run_id"] for h in history] == ["r1"]
    report = env["report"].read_text(encoding="utf-8")
    assert cn.REPORT_BEGIN in report and "Latest run: `r1`" in report
    assert "| qwen2.5-7b |" in report and "FIRING" in report
    meta = json.loads((run_dir / "run.json").read_text())
    assert meta["health"] == {"status": "ok", "quality_path": True}
    assert set(meta["stages"]) == {
        "rewrite",
        "respond:qwen2.5-7b",
        "respond:phi3.5-mini",
        "judge",
        "finalize",
    }


def test_staged_cli_rolling_history_and_rerun(env: dict[str, Any]) -> None:
    # six earlier runs of the same canary version, one of another version (not pooled)
    def hist(run_id: str, day: int, version: str, preserved: int) -> dict[str, Any]:
        time_ = f"2026-10-{day:02d}T02:00:00+00:00"
        counts = {"preserved": preserved, "total": 200}
        return {"run_id": run_id, "time": time_, "canary_version": version, **counts}

    earlier = [hist(f"old{i}", i + 1, "canary_v1", 185) for i in range(6)]
    other = hist("v0", 8, "canary_v0", 0)
    env["history"].write_text("".join(json.dumps(r) + "\n" for r in [*earlier, other]))
    run_dir = env["dir"] / "runs" / "r2"
    base = ["--run-dir", str(run_dir), *_common(env)]
    assert cn.main(["rewrite", "--canary", str(env["canary"]), "--mode", "fast", *base]) == 0
    with pytest.raises(SystemExit):  # an existing run is not overwritten without --force
        cn.main(["rewrite", "--canary", str(env["canary"]), *base])
    for tag in ("qwen2.5-7b", "phi3.5-mini"):
        assert cn.main(["respond", "--target", tag, *base, *_llm_args(env)]) == 0
    assert cn.main(["judge", *base, *_llm_args(env)]) == 0
    assert cn.main(["finalize", *base]) == 0  # no gateway: computed and recorded, not pushed
    assert env["pushes"] == []
    rec = json.loads((run_dir / "result.json").read_text())
    assert rec["mode"] == "fast" and rec["pushed"] is None
    assert rec["rolling_runs"] == 7
    assert rec["rolling"] == pytest.approx((6 * 185 + 8) / (6 * 200 + 12))
    # finalize is rerunnable: the run is replaced in the history, not duplicated
    assert cn.main(["finalize", *base, "--gateway", "gw:9091"]) == 0
    ids = [h["run_id"] for h in cn.read_history(env["history"])]
    assert ids.count("r2") == 1 and len(ids) == 8
    assert len(env["pushes"]) == 1
    # --force redoes the rewrite stage and clears the downstream files
    assert cn.main(["rewrite", "--canary", str(env["canary"]), "--force", *base]) == 0
    assert not (run_dir / "labels.jsonl").exists() and not (run_dir / "result.json").exists()
    with pytest.raises(cn.CanaryError, match="respond stage"):
        cn.main(["judge", *base, *_llm_args(env)])


def test_rewrite_stage_retries_and_error_budget(tmp_path: Path) -> None:
    targets = [cn.TARGETS["qwen2.5-7b"]]
    service = FakeService(fail_first=2)  # the first prompt fails twice, then succeeds
    with httpx.Client(transport=httpx.MockTransport(service)) as http:
        stats = cn.rewrite_stage(
            PROMPTS, targets, tmp_path / "a", http=http, warmup=0, sleep=lambda s: None
        )
    assert stats["errors"] == 0 and stats["requests"] == 6 and len(service.calls) == 8
    assert stats["changed"] == 4 and stats["unchanged"] == 2 and stats["gate_rejected"] == 2

    failing = FakeService(always_fail=PROMPTS[3]["prompt"])
    with httpx.Client(transport=httpx.MockTransport(failing)) as http:
        with pytest.raises(cn.CanaryError, match="1/6 service requests failed"):
            cn.rewrite_stage(
                PROMPTS, targets, tmp_path / "b", http=http, warmup=0, sleep=lambda s: None
            )
        stats = cn.rewrite_stage(
            PROMPTS,
            targets,
            tmp_path / "c",
            http=http,
            warmup=0,
            max_error_rate=0.5,
            sleep=lambda s: None,
        )
    assert stats["errors"] == 1
    rows = [json.loads(x) for x in (tmp_path / "c" / "rewrites.jsonl").open()]
    failed = [r for r in rows if r["error"]]
    assert len(failed) == 1 and "HTTP 503" in failed[0]["error"] and "rewrite" not in failed[0]
    # unchanged and failed rows need no label: the error is excluded, the unchanged is preserved
    summary = cn.summarize([r for r in rows if r["error"] or r["unchanged"]], [])
    assert (summary["errors"], summary["preserved"], summary["total"]) == (1, 1, 1)


def test_summarize_counting_rule() -> None:
    def rw(pid: str, cat: str, unchanged: bool, error: str | None = None) -> dict[str, Any]:
        rec = {"prompt_id": pid, "category": cat, "tag": "q", "target": "Q", "error": error}
        return {**rec, "unchanged": unchanged, "path": "fast"}

    def lab(pid: str, label: str | None) -> dict[str, Any]:
        return {"candidate_id": f"{pid}:canary", "target": "Q", "label": label}

    rows = [
        rw("a", "fact", True),
        rw("b", "fact", False),
        rw("c", "code", False),
        rw("d", "code", False),
        rw("e", "code", False, error="HTTP 500"),
    ]
    s = cn.summarize(rows, [lab("b", "preserved"), lab("c", "degraded"), lab("d", None)])
    assert (s["preserved"], s["total"], s["errors"]) == (2, 4, 1)
    assert (s["unchanged"], s["judged"], s["degraded"], s["unlabelled"]) == (1, 3, 1, 1)
    assert s["by_category"]["fact"]["preservation"] == 1.0
    assert s["by_category"]["code"]["preservation"] == 0.0  # unlabelled is not preserved
    with pytest.raises(cn.CanaryError, match="no label"):
        cn.summarize(rows, [lab("b", "preserved")])


def test_make_llm_client_url_beats_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TERSER_LOCAL_LLM_URL", "http://env:1/v1")
    client = cn.make_llm_client(
        "http://judge:8000/v1/", ledger_path=tmp_path / "l.json", cache_dir=tmp_path / "c"
    )
    providers = client.transport.providers  # type: ignore[attr-defined]
    assert providers["local"].resolved_base_url() == "http://judge:8000/v1"
    assert client.budget.is_free("local") and client.budget.phase == "phase11"
    assert client.budget.ledger.path == (tmp_path / "l.json").resolve()
    default = cn.make_llm_client(None, ledger_path=tmp_path / "l.json")
    assert default.transport.providers["local"].resolved_base_url() == "http://env:1/v1"  # type: ignore[attr-defined]


# --------------------------------------------------------------------------------------------
# Dashboard read-back


def _monitoring(rolling: float, run: float, pairs: float) -> Any:
    dashboard = json.loads(
        (REPO / "docker/monitoring/grafana/dashboards/terser-serving.json").read_text("utf-8")
    )
    values = {
        cm.ROLLING_PRESERVATION: rolling,
        cm.PRESERVATION: run,
        cm.PROMPTS: pairs,
    }
    queried: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/v1/query":
            q = request.url.params["query"]
            res = [{"metric": {}, "value": [0, str(values[q])]}] if q in values else []
            return httpx.Response(200, json={"status": "success", "data": {"result": res}})
        if path == "/api/v1/targets":
            t = [{"labels": {"job": "pushgateway"}, "health": "up"}]
            return httpx.Response(200, json={"data": {"activeTargets": t}})
        if path == "/api/dashboards/uid/terser-serving":
            return httpx.Response(200, json={"dashboard": dashboard})
        if path == "/api/ds/query":
            q = json.loads(request.content)["queries"][0]
            queried.append(q["expr"])
            m = re.search(r"terser_canary_\w+", q["expr"])
            v = values.get(m.group(0)) if m else None
            frames = [{"data": {"values": [[0], [v]]}}] if v is not None else []
            return httpx.Response(200, json={"results": {"A": {"frames": frames}}})
        return httpx.Response(404)

    return handler, queried


def test_verify_dashboard_reads_metric_back() -> None:
    handler, queried = _monitoring(0.93, 0.95, 400)
    expected = {cm.ROLLING_PRESERVATION: 0.93, cm.PRESERVATION: 0.95, cm.PROMPTS: 400.0}
    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        check = cn.verify_dashboard(http, expected=expected, wait_s=0)
        assert check["ok"]
        assert check["pushgateway_target_health"] == ["up"]
        assert f"min({cm.ROLLING_PRESERVATION})" in queried
        titles = {p["panel"] for p in check["panels"]}
        assert "Rolling preservation (canary)" in titles and len(check["panels"]) == 6
        rolling = [p for p in check["panels"] if p["panel"] == "Rolling preservation (canary)"]
        assert rolling[0]["value"] == 0.93
        stale = cn.verify_dashboard(
            http, expected={cm.ROLLING_PRESERVATION: 0.5}, wait_s=0, sleep=lambda s: None
        )
        assert not stale["ok"]
    block = cn._dashboard_block(check)
    assert "metric on dashboard" in block and "Rolling preservation (canary)" in block


def test_verify_cli_writes_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = tmp_path / "r"
    run_dir.mkdir()
    (run_dir / "result.json").write_text(
        json.dumps({"rolling": 0.93, "preservation": 0.95, "total": 400})
    )
    handler, _ = _monitoring(0.93, 0.95, 400)
    monkeypatch.setattr(
        cn, "_http", lambda timeout, auth=None: httpx.Client(transport=httpx.MockTransport(handler))
    )
    report = tmp_path / "report.md"
    report.write_text(f"# x\n\n{cn.DASH_BEGIN}\nold\n{cn.DASH_END}\n\ntail\n")
    rc = cn.main(["verify", "--run-dir", str(run_dir), "--report", str(report), "--wait", "0"])
    assert rc == 0
    assert json.loads((run_dir / "dashboard.json").read_text())["ok"]
    text = report.read_text()
    assert "old" not in text and text.endswith("tail\n") and "metric on dashboard" in text


def test_report_stub_has_markers() -> None:
    text = (REPO / "reports/p11_canary.md").read_text(encoding="utf-8")
    for marker in (cn.REPORT_BEGIN, cn.REPORT_END, cn.DASH_BEGIN, cn.DASH_END):
        assert marker in text
