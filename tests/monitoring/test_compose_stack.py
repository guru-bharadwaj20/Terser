"""Static checks of the container stack and its dashboard (rows 11.3.7, 11.5.5).

They need no Docker: they parse compose.yaml, the serving and Prometheus configs and the Grafana
provisioning, and cross-check names that must agree between files (service names and ports,
the served rewriter name, datasource uids, metric names, the 0.92 alert threshold).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from terser.monitoring import canary_metrics as cm

ROOT = Path(__file__).resolve().parents[2]
MON = ROOT / "docker" / "monitoring"
GRAFANA = MON / "grafana"
DASHBOARD = GRAFANA / "dashboards" / "terser-serving.json"
ALERTS = GRAFANA / "provisioning" / "alerting" / "terser-alerts.yaml"


def _yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def compose() -> dict[str, Any]:
    return dict(_yaml(ROOT / "compose.yaml"))


def _dockerfile_env(name: str) -> dict[str, str]:
    text = (ROOT / "docker" / name).read_text(encoding="utf-8").replace("\\\n", " ")
    env: dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith("ENV "):
            env.update(re.findall(r"(\w+)=(\S*)", line))
    return env


def test_plain_up_starts_both_paths_and_dashboard(compose: dict[str, Any]) -> None:
    services = compose["services"]
    assert {"terser-cpu", "terser-gpu", "prometheus", "pushgateway", "grafana"} <= set(services)
    # `docker compose up` starts every service without a profile (row 11.3.7 acceptance)
    assert not any("profiles" in s for s in services.values())
    gpu = services["terser-gpu"]
    devices = gpu["deploy"]["resources"]["reservations"]["devices"]
    assert devices[0]["driver"] == "nvidia" and "gpu" in devices[0]["capabilities"]
    assert services["terser-cpu"]["build"]["dockerfile"] == "docker/Dockerfile.cpu"
    assert gpu["build"]["dockerfile"] == "docker/Dockerfile.gpu"
    for svc in services.values():
        if "build" in svc:
            assert (ROOT / svc["build"]["dockerfile"]).is_file()
        else:
            assert re.search(r":v?\d+\.\d+\.\d+$", svc["image"]), "pin monitoring images"


def test_bind_mounts_exist(compose: dict[str, Any]) -> None:
    for svc in compose["services"].values():
        for vol in svc.get("volumes", []):
            src = vol.split(":")[0]
            if src.startswith("./"):
                assert (ROOT / src).exists(), src


def test_cpu_serving_config_matches_fast_onnx(compose: dict[str, Any]) -> None:
    cpu = compose["services"]["terser-cpu"]
    target = cpu["environment"]["TERSER_SERVING_CONFIG"]
    mounted = {v.split(":")[1]: v.split(":")[0] for v in cpu["volumes"]}
    cfg = _yaml(ROOT / mounted[target])
    fast = _yaml(ROOT / "configs" / "serving" / "fast_onnx.yaml")
    assert {k: v for k, v in cfg.items() if k != "quality"} == {
        k: v for k, v in fast.items() if k != "quality"
    }
    for key in ("gate_onnx", "tagger_onnx"):  # relative to the repo root (/app in the image)
        assert not Path(cfg[key]).is_absolute()
    # the quality path points at the terser-gpu service under the name vLLM serves the LoRA as
    url = cfg["quality"]["base_url"]
    host, port = re.match(r"http://([\w-]+):(\d+)/v1$", url).groups()  # type: ignore[union-attr]
    ports = compose["services"][host]["ports"]
    assert host == "terser-gpu" and f"127.0.0.1:{port}:{port}" in ports  # loopback only
    assert cfg["quality"]["model"] == _dockerfile_env("Dockerfile.gpu")["TERSER_SERVED_NAME"]


def test_cpu_image_bakes_onnx_models() -> None:
    text = (ROOT / "docker" / "Dockerfile.cpu").read_text(encoding="utf-8")
    assert "COPY models/fast ./models/fast" in text
    assert 'pip install -e ".[serve]"' in text  # repo-root-relative model paths resolve to /app
    assert _dockerfile_env("Dockerfile.cpu")["TERSER_SERVING_CONFIG"].endswith("fast_onnx.yaml")
    ignore = (ROOT / ".dockerignore").read_text(encoding="utf-8").split()
    assert "models" in ignore and "!models/fast" in ignore


def test_gpu_default_lora_matches_adapter_base(compose: dict[str, Any], require_dvc: Any) -> None:
    env = _dockerfile_env("Dockerfile.gpu")
    assert env["TERSER_LORA_PATH"].startswith("/models/rewriter/")
    gpu_env = compose["services"]["terser-gpu"]["environment"]
    assert env["TERSER_BASE_MODEL"] in gpu_env["TERSER_BASE_MODEL"]
    assert env["TERSER_LORA_PATH"] in gpu_env["TERSER_LORA_PATH"]
    adapter = ROOT / "models" / env["TERSER_LORA_PATH"].removeprefix("/models/")
    cfg = adapter / "adapter_config.json"
    require_dvc(cfg, "dvc pull models/rewriter")
    base = json.loads(cfg.read_text(encoding="utf-8"))["base_model_name_or_path"]
    assert base == env["TERSER_BASE_MODEL"]


def test_prometheus_scrapes_compose_services(compose: dict[str, Any]) -> None:
    prom = _yaml(MON / "prometheus.yml")
    targets = {
        job["job_name"]: t
        for job in prom["scrape_configs"]
        for sc in job["static_configs"]
        for t in sc["targets"]
    }
    assert set(targets) == {"terser-cpu", "terser-gpu", "pushgateway"}
    for job, target in targets.items():
        assert target.split(":")[0] in compose["services"], job
    assert next(j for j in prom["scrape_configs"] if j["job_name"] == "pushgateway")["honor_labels"]


def _dashboard() -> dict[str, Any]:
    return dict(json.loads(DASHBOARD.read_text(encoding="utf-8")))


def test_dashboard_panels_and_datasource() -> None:
    dash = _dashboard()
    ds_uid = _yaml(GRAFANA / "provisioning" / "datasources" / "prometheus.yaml")["datasources"][0][
        "uid"
    ]
    panels = [p for p in dash["panels"] if p["type"] != "row"]
    assert dash["uid"] == "terser-serving" and len(panels) >= 10
    ids = [p["id"] for p in dash["panels"]]
    assert len(ids) == len(set(ids))
    for p in panels:
        assert p["datasource"]["uid"] == ds_uid
        assert p["targets"] and all(t["datasource"]["uid"] == ds_uid for t in p["targets"])
    titles = " ".join(p["title"].lower() for p in panels)
    for needed in ("requests", "latency", "gate rejection", "fallback", "rolling preservation"):
        assert needed in titles


def _exported_names() -> set[str]:
    from terser.serving.metrics import ServiceMetrics

    m = ServiceMetrics()
    for net in (1e-5, -1e-5):
        m.observe(
            path="fast",
            mode="quality",
            target="gpt-4o",
            latency_s=0.01,
            tokens_saved=3,
            net_savings_usd=net,
            gate_rejected=True,
            fallback_reason="timeout",
            cached=True,
        )
    names = {s.name for fam in m.registry.collect() for s in fam.samples}
    gpu = {"terser_gpu_utilization_ratio", "terser_gpu_memory_used_bytes"}  # scrape-time, NVML
    return names | gpu | set(cm.METRIC_NAMES)


def test_dashboard_metrics_are_exported() -> None:
    exprs = [t["expr"] for p in _dashboard()["panels"] for t in p.get("targets", [])]
    used = {n for e in exprs for n in re.findall(r"\bterser_[a-z0-9_]+", e)}
    assert used and used <= _exported_names(), used - _exported_names()
    assert any(cm.ROLLING_PRESERVATION in e for e in exprs)


def test_alert_rule_fires_below_092() -> None:
    groups = _yaml(ALERTS)["groups"]
    rules = [r for g in groups for r in g["rules"]]
    (rule,) = [r for r in rules if r["uid"] == "terser-rolling-preservation"]
    data = {d["refId"]: d for d in rule["data"]}
    assert cm.ROLLING_PRESERVATION in data["A"]["model"]["expr"]
    assert data["A"]["datasourceUid"] == "prometheus"
    (cond,) = data[rule["condition"]]["model"]["conditions"]
    assert cond["evaluator"] == {"type": "lt", "params": [cm.ALERT_THRESHOLD]}
    assert cm.ALERT_THRESHOLD == 0.92
    # Grafana expands $VAR in provisioning files, so templates must not use $values / $labels
    assert "$" not in ALERTS.read_text(encoding="utf-8")
