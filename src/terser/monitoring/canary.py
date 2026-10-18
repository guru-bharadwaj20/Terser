"""Nightly quality canary (plan row **11.5.4**): a fixed prompt set through the deployed service
and the judge, with rolling preservation pushed to the dashboard (rows 11.5.5).

One cycle, stage by stage (each stage is also a CLI subcommand, so a single-GPU host can serve
the rewriter, the two targets and the judge one at a time between stages):

1. ``rewrite``: every canary prompt goes through ``POST {base_url}/rewrite`` of the deployed
   service once per target (``target`` = the service's tokenizer/price name, ``mode`` =
   ``quality`` by default). The responses go to ``<run_dir>/rewrites.jsonl``; rewrites that
   changed the prompt become the ``candidates_<tag>.jsonl`` of the respond stage.
2. ``respond``: the target model answers the original and the rewrite
   (:func:`terser.phase2.respond.run_respond`, same sampling plan as the dataset labels).
3. ``judge``: :func:`terser.phase2.label_stage.run_labels` (``pairwise_avg`` with the category
   task metric, the row 2.3.4 label rule) on every changed (prompt, target).
4. ``finalize``: this run's preservation, overall, per target and per category; appended to the
   run history (``logs/canary_history.jsonl``); pushed with
   :func:`terser.monitoring.canary_metrics.push_canary`; and the report
   (``reports/p11_canary.md``) is regenerated.
5. ``verify`` (optional): reads the pushed value back from Prometheus and from the Grafana
   dashboard panels.

Preservation counting, consistent with ``reports/p8_rewriter_val.md`` and the other val reports:

* a rewrite the service returned **unchanged** (gate rejection, a fallback that kept the original,
  or no change) is **preserved** by construction: the target receives the original prompt;
* a changed rewrite is preserved iff its label is ``preserved``; an unlabelled pair (unparseable
  judge verdict) counts as **not** preserved;
* a request the service failed (HTTP error after retries) is excluded from the ratio and reported
  as an error; more than ``max_error_rate`` of errors fails the stage.

The pushed run value pools both targets: ``preserved / judged (prompt, target) pairs`` (at most
2 x 200 = 400 pairs), and the rolling value pools the last 7 runs of the same canary version.

The canary set (``data/canary/canary_v1.jsonl``, DVC-tracked) is built once by ``select``: 200
val-split prompts stratified by category (largest-remainder proportional allocation), seed 0,
one prompt per source item before any second variant of the same item. The test split is never
read.

CLI::

    python -m terser.monitoring.canary select            # once: data/canary/canary_v1.jsonl
    python -m terser.monitoring.canary run --base-url http://localhost:8080 --mode quality \\
        --llm-url http://localhost:8000/v1 --judge-url http://localhost:8000/v1 \\
        --gateway localhost:9091
    # or stage by stage (one vLLM model on the GPU at a time):
    python -m terser.monitoring.canary rewrite  --run-dir logs/canary/R --mode quality
    python -m terser.monitoring.canary respond  --run-dir logs/canary/R --target qwen2.5-7b
    python -m terser.monitoring.canary respond  --run-dir logs/canary/R --target phi3.5-mini
    python -m terser.monitoring.canary judge    --run-dir logs/canary/R
    python -m terser.monitoring.canary finalize --run-dir logs/canary/R --gateway localhost:9091
    python -m terser.monitoring.canary verify   --run-dir logs/canary/R
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from terser.monitoring.canary_metrics import (
    ALERT_THRESHOLD,
    PRESERVATION,
    PROMPTS,
    ROLLING_PRESERVATION,
    ROLLING_WINDOW,
    push_canary,
    rolling_preservation,
)

if TYPE_CHECKING:
    import httpx

    from terser.judge.pairwise import PairwiseJudge
    from terser.llm import LLMClient
    from terser.phase2.label_stage import LabelStats, TaskReferences
    from terser.phase2.respond import RespondStats

__all__ = [
    "CANARY_PATH",
    "CANARY_SEED",
    "CANARY_SIZE",
    "CANARY_VERSION",
    "DEFAULT_JUDGE",
    "HISTORY_PATH",
    "REPORT_PATH",
    "TARGETS",
    "CanaryError",
    "CanaryTarget",
    "allocate",
    "append_history",
    "finalize",
    "judge_stage",
    "main",
    "make_llm_client",
    "read_history",
    "render_report",
    "respond_stage",
    "rewrite_stage",
    "select_canary",
    "summarize",
    "verify_dashboard",
    "write_canary",
]

CANARY_VERSION = "canary_v1"
CANARY_PATH = Path("data/canary/canary_v1.jsonl")
CANARY_SOURCE = Path("data/processed/splits/val.jsonl")
CANARY_SIZE = 200
CANARY_SEED = 0
CANARY_FIELDS: tuple[str, ...] = (
    "id",
    "source_id",
    "source",
    "category",
    "style",
    "prompt",
    "reference",
)
HISTORY_PATH = Path("logs/canary_history.jsonl")
RUNS_DIR = Path("logs/canary")
REPORT_PATH = Path("reports/p11_canary.md")
DEFAULT_SERVICE_URL = "http://localhost:8080"
DEFAULT_PROMETHEUS = "http://localhost:9090"
DEFAULT_GRAFANA = "http://localhost:3000"
DASHBOARD_UID = "terser-serving"
DEFAULT_JUDGE = "Qwen/Qwen2.5-32B-Instruct-AWQ"
DEFAULT_BUDGET_PHASE = "phase11"
CANDIDATE_SUFFIX = "canary"
MIN_CATEGORY = 0.85  # C8.3 per-category bar, reported next to the 0.92 overall alert bar
WARMUP_PROMPT = (
    "hey so um, could you maybe write a Python function that checks if a string is a "
    "palindrome? It should ignore case. Thanks so much!!"
)
IST = timezone(timedelta(hours=5, minutes=30), "IST")
REPORT_BEGIN = "<!-- canary:begin (generated by python -m terser.monitoring.canary) -->"
REPORT_END = "<!-- canary:end -->"
DASH_BEGIN = "<!-- canary-dashboard:begin (generated by python -m terser.monitoring.canary) -->"
DASH_END = "<!-- canary-dashboard:end -->"


class CanaryError(RuntimeError):
    """A canary stage could not produce a valid result."""


@dataclass(frozen=True)
class CanaryTarget:
    """One target LLM: run tag, served model id, and the service's ``target`` name."""

    tag: str
    model: str
    service_target: str


TARGETS: dict[str, CanaryTarget] = {
    "qwen2.5-7b": CanaryTarget("qwen2.5-7b", "Qwen/Qwen2.5-7B-Instruct", "qwen2.5"),
    "phi3.5-mini": CanaryTarget("phi3.5-mini", "microsoft/Phi-3.5-mini-instruct", "phi3.5"),
}


# --------------------------------------------------------------------------------------------
# I/O helpers


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _ist(iso: str) -> str:
    return datetime.fromisoformat(iso).astimezone(IST).strftime("%Y-%m-%d %H:%M IST")


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _refuse_test_split(path: Path) -> None:
    if "test" in path.name.lower():
        raise ValueError(f"the canary never reads the test split: {path}")


# --------------------------------------------------------------------------------------------
# 1. The fixed canary set


def allocate(counts: Mapping[str, int], n: int) -> dict[str, int]:
    """Largest-remainder proportional allocation of ``n`` over categories.

    Ties on the remainder go to the category that sorts first by name.
    """
    total = sum(counts.values())
    if n > total:
        raise ValueError(f"cannot draw {n} prompts from {total}")
    quotas = {c: n * k / total for c, k in counts.items()}
    alloc = {c: int(q) for c, q in quotas.items()}
    rest = n - sum(alloc.values())
    for c in sorted(counts, key=lambda c: (-(quotas[c] - alloc[c]), c))[:rest]:
        alloc[c] += 1
    return dict(sorted(alloc.items()))


def select_canary(
    records: Iterable[Mapping[str, Any]], n: int = CANARY_SIZE, seed: int = CANARY_SEED
) -> list[dict[str, Any]]:
    """The fixed canary: ``n`` prompts stratified by category, seed ``seed``.

    Rule: drop rejected and stand-in records; sort each category's records by ``id``; allocate
    ``n`` over the categories in proportion to their size (:func:`allocate`); then, with one
    ``random.Random(seed)`` walking the categories in name order, shuffle each category and take
    the first record of every distinct ``source_id`` before any second casual variant of the same
    source item. Output is sorted by ``id`` and keeps :data:`CANARY_FIELDS` plus ``split``.
    """
    by_cat: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in sorted(records, key=lambda r: str(r["id"])):
        if _truthy(r.get("rejected")) or _truthy(r.get("stand_in")):
            continue
        by_cat[str(r["category"])].append(r)
    alloc = allocate({c: len(v) for c, v in by_cat.items()}, n)
    rng = random.Random(seed)
    chosen: list[Mapping[str, Any]] = []
    for cat in sorted(by_cat):
        pool = list(by_cat[cat])
        rng.shuffle(pool)
        seen: set[str] = set()
        first: list[Mapping[str, Any]] = []
        rest: list[Mapping[str, Any]] = []
        for r in pool:
            sid = str(r.get("source_id", r["id"]))
            (rest if sid in seen else first).append(r)
            seen.add(sid)
        chosen += (first + rest)[: alloc[cat]]
    out = [{k: r.get(k) for k in CANARY_FIELDS} | {"split": "val"} for r in chosen]
    return sorted(out, key=lambda r: str(r["id"]))


def write_canary(
    source: Path = CANARY_SOURCE,
    out: Path = CANARY_PATH,
    n: int = CANARY_SIZE,
    seed: int = CANARY_SEED,
) -> dict[str, Any]:
    """Select the canary from ``source`` (a val file) and write it with a ``.json`` manifest."""
    _refuse_test_split(source)
    rows = select_canary(_read_jsonl(source), n, seed)
    _write_jsonl(out, rows)
    cats = Counter(str(r["category"]) for r in rows)
    manifest = {
        "version": out.stem,
        "rows": len(rows),
        "seed": seed,
        "source": source.as_posix(),
        "source_sha256": _sha256(source),
        "sha256": _sha256(out),
        "by_category": dict(sorted(cats.items())),
        "distinct_source_ids": len({r["source_id"] for r in rows}),
        "rule": (
            "val split only; rejected and stand-in records dropped; per category sorted by id; "
            f"{n} allocated in proportion to category size (largest remainder, ties by name); "
            f"one random.Random({seed}) shuffles each category in name order; first record of "
            "each distinct source_id before any second variant; output sorted by id"
        ),
    }
    _write_json(out.with_suffix(".json"), manifest)
    return manifest


# --------------------------------------------------------------------------------------------
# Run metadata


def _run_meta_path(run_dir: Path) -> Path:
    return run_dir / "run.json"


def _load_meta(run_dir: Path) -> dict[str, Any]:
    path = _run_meta_path(run_dir)
    if not path.is_file():
        raise CanaryError(f"{path} missing: run the rewrite stage first")
    data = json.loads(path.read_text(encoding="utf-8"))
    return dict(data)


def _update_meta(run_dir: Path, **fields: Any) -> dict[str, Any]:
    meta = _load_meta(run_dir) if _run_meta_path(run_dir).is_file() else {}
    for k, v in fields.items():
        if k == "stages":
            meta.setdefault("stages", {}).update(v)
        else:
            meta[k] = v
    _write_json(_run_meta_path(run_dir), meta)
    return meta


def _meta_targets(meta: Mapping[str, Any]) -> list[CanaryTarget]:
    return [CanaryTarget(**t) for t in meta["targets"]]


def _candidate_id(prompt_id: str) -> str:
    return f"{prompt_id}:{CANDIDATE_SUFFIX}"


# --------------------------------------------------------------------------------------------
# 2. The deployed service


def _post_rewrite(
    http: httpx.Client,
    base_url: str,
    prompt: str,
    target: str,
    mode: str,
    retries: int,
    sleep: Callable[[float], None],
) -> tuple[dict[str, Any] | None, str | None]:
    url = base_url.rstrip("/") + "/rewrite"
    error = None
    for attempt in range(retries + 1):
        try:
            resp = http.post(url, json={"prompt": prompt, "target": target, "mode": mode})
            if resp.status_code == 200:
                return dict(resp.json()), None
            error = f"HTTP {resp.status_code}: {resp.text[:200]}"
        except Exception as exc:  # any transport failure is retried
            error = f"{type(exc).__name__}: {exc}"
        if attempt < retries:
            sleep(2.0**attempt)
    return None, error


def rewrite_stage(
    prompts: Sequence[Mapping[str, Any]],
    targets: Sequence[CanaryTarget],
    run_dir: Path,
    *,
    http: httpx.Client,
    base_url: str = DEFAULT_SERVICE_URL,
    mode: str = "quality",
    workers: int = 1,
    retries: int = 2,
    warmup: int = 2,
    max_error_rate: float = 0.05,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Send every (prompt, target) through ``POST /rewrite``; write rewrites and candidates.

    ``workers=1`` (default) sends requests one at a time, like light production traffic: the
    quality path has a 0.4 s latency budget, and a burst would turn rewrites into timeouts.
    ``warmup`` requests with a fixed non-canary prompt go first, so a cold rewriter does not
    show up as canary fallbacks.
    """
    health: dict[str, Any] | str
    try:
        h = http.get(base_url.rstrip("/") + "/healthz")
        health = dict(h.json()) if h.status_code == 200 else f"HTTP {h.status_code}"
    except Exception as exc:
        health = f"{type(exc).__name__}: {exc}"
    for t in targets[:1]:
        for _ in range(warmup):
            _post_rewrite(http, base_url, WARMUP_PROMPT, t.service_target, mode, 0, sleep)

    jobs = [(p, t) for t in targets for p in prompts]

    def one(job: tuple[Mapping[str, Any], CanaryTarget]) -> dict[str, Any]:
        p, t = job
        body, error = _post_rewrite(
            http, base_url, str(p["prompt"]), t.service_target, mode, retries, sleep
        )
        rec: dict[str, Any] = {
            "prompt_id": str(p["id"]),
            "category": str(p["category"]),
            "tag": t.tag,
            "target": t.model,
            "service_target": t.service_target,
            "mode": mode,
            "error": error,
        }
        if body is not None:
            rewrite = str(body["rewrite"])
            rec |= {
                "rewrite": rewrite,
                "unchanged": rewrite == str(p["prompt"]),
                **{
                    k: body.get(k)
                    for k in (
                        "path",
                        "gate_rejected",
                        "gate_score",
                        "fallback_used",
                        "fallback_reason",
                        "cached",
                        "tokens_before",
                        "tokens_after",
                        "latency_ms",
                    )
                },
            }
        return rec

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        rows = list(pool.map(one, jobs))
    wall = time.perf_counter() - start
    _write_jsonl(run_dir / "rewrites.jsonl", rows)
    for t in targets:
        cands = [
            {
                "prompt_id": r["prompt_id"],
                "candidate_id": _candidate_id(r["prompt_id"]),
                "method": CANDIDATE_SUFFIX,
                "text": r["rewrite"],
                "rejected": False,
            }
            for r in rows
            if r["tag"] == t.tag and r["error"] is None and not r["unchanged"]
        ]
        _write_jsonl(run_dir / f"candidates_{t.tag}.jsonl", cands)
    ok = [r for r in rows if r["error"] is None]
    errors = len(rows) - len(ok)
    stats = {
        "requests": len(rows),
        "errors": errors,
        "changed": sum(not r["unchanged"] for r in ok),
        "unchanged": sum(bool(r["unchanged"]) for r in ok),
        "paths": dict(Counter(str(r.get("path")) for r in ok)),
        "gate_rejected": sum(bool(r.get("gate_rejected")) for r in ok),
        "fallbacks": dict(Counter(str(r["fallback_reason"]) for r in ok if r["fallback_reason"])),
        "wall_seconds": round(wall, 2),
        "finished": _now(),
    }
    _update_meta(run_dir, health=health, stages={"rewrite": stats})
    if rows and errors / len(rows) > max_error_rate:
        raise CanaryError(
            f"{errors}/{len(rows)} service requests failed (> {max_error_rate:.0%}); "
            f"first error: {next(r['error'] for r in rows if r['error'])}"
        )
    return stats


# --------------------------------------------------------------------------------------------
# 3. Targets and judge (the Phase 2 machinery)


def make_llm_client(
    url: str | None,
    *,
    phase: str = DEFAULT_BUDGET_PHASE,
    ledger_path: Path | None = None,
    cache_dir: Path | None = None,
    timeout_s: float = 300.0,
) -> LLMClient:
    """An :class:`terser.llm.LLMClient` whose ``local`` provider points at ``url``.

    The provider keeps the name ``local`` (so the response cache and the budget's free-provider
    rule are shared with the dataset stages: the cache key does not include the URL); ``url``
    overrides ``TERSER_LOCAL_LLM_URL``. ``ledger_path`` is the run's own budget ledger, so the
    phase's token cap bounds one canary run rather than every night's runs together.
    """
    from terser.llm import LLMClient
    from terser.llm.client import ProviderTransport
    from terser.llm.providers import LOCAL_PROVIDER, ProviderConfig

    client = LLMClient(
        phase,
        ledger_path=None if ledger_path is None else ledger_path.resolve(),
        cache_dir=None if cache_dir is None else cache_dir.resolve(),
    )
    if url:
        providers = dict(client.budget.providers)
        # A config name other than "local" is not overridden by TERSER_LOCAL_LLM_URL.
        providers[LOCAL_PROVIDER] = ProviderConfig("canary", url, timeout_s=timeout_s)
        client.transport = ProviderTransport(providers)
    return client


def _prompts_with_candidates(
    prompts: Sequence[Mapping[str, Any]], candidates: Sequence[Mapping[str, Any]]
) -> list[Mapping[str, Any]]:
    keep = {str(c["prompt_id"]) for c in candidates}
    return [p for p in prompts if str(p["id"]) in keep]


def respond_stage(
    prompts: Sequence[Mapping[str, Any]],
    target: CanaryTarget,
    run_dir: Path,
    client: LLMClient,
    *,
    workers: int = 64,
    max_tokens: int | None = None,
) -> RespondStats:
    """Target answers for the original and the rewrite of every changed (prompt, target).

    Unchanged rewrites are preserved by construction, so their originals are not answered.
    """
    from terser.phase2.respond import DEFAULT_MAX_TOKENS, build_items, run_respond

    cands = _read_jsonl(run_dir / f"candidates_{target.tag}.jsonl")
    items = build_items(_prompts_with_candidates(prompts, cands), cands)
    stats = run_respond(
        items,
        client,
        target=target.model,
        out_path=run_dir / f"responses_{target.tag}.jsonl",
        max_tokens=max_tokens or DEFAULT_MAX_TOKENS,
        workers=workers,
    )
    _update_meta(run_dir, stages={f"respond:{target.tag}": {**stats.to_dict(), "finished": _now()}})
    return stats


def judge_stage(
    prompts: Sequence[Mapping[str, Any]],
    run_dir: Path,
    judge: PairwiseJudge,
    *,
    references: TaskReferences | None = None,
    metric_cache: Path | None = Path(".cache/task_metrics"),
    workers: int = 32,
) -> LabelStats:
    """Judge and label every changed (prompt, target) of the run (``labels.jsonl``)."""
    from terser.phase2.label_stage import run_labels

    meta = _load_meta(run_dir)
    responses: list[dict[str, Any]] = []
    for t in _meta_targets(meta):
        path = run_dir / f"responses_{t.tag}.jsonl"
        cands = run_dir / f"candidates_{t.tag}.jsonl"
        if not path.is_file():
            if cands.is_file() and _read_jsonl(cands):
                raise CanaryError(f"{path} missing: run the respond stage for {t.tag}")
            continue
        responses += _read_jsonl(path)
    stats = run_labels(
        prompts,
        responses,
        judge,
        out_path=run_dir / "labels.jsonl",
        references=references,
        metric_cache=metric_cache,
        workers=workers,
        splits={str(p["id"]): "val" for p in prompts},
    )
    _update_meta(
        run_dir,
        judge_model=judge.model,
        stages={"judge": {**stats.to_dict(), "finished": _now()}},
    )
    return stats


# --------------------------------------------------------------------------------------------
# 4. Preservation, history, push, report


def _ratio(counts: Mapping[str, int]) -> dict[str, Any]:
    total = counts["total"]
    return {**counts, "preservation": counts["preserved"] / total if total else None}


def summarize(
    rewrites: Sequence[Mapping[str, Any]], labels: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """This run's preservation, pooled and per target / category / (target, category).

    Unchanged rewrites are preserved; a changed one is preserved iff its label is
    ``preserved`` (unlabelled = not preserved); service errors are excluded and counted.
    """
    label_of = {(str(r["candidate_id"]), str(r["target"])): r.get("label") for r in labels}
    keys = ("preserved", "total", "unchanged", "judged", "degraded", "unlabelled")
    pooled = dict.fromkeys(keys, 0)
    by_target: dict[str, dict[str, int]] = defaultdict(lambda: dict.fromkeys(keys, 0))
    by_cat: dict[str, dict[str, int]] = defaultdict(lambda: dict.fromkeys(keys, 0))
    by_tc: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: dict.fromkeys(keys, 0))
    )
    errors = 0
    missing = []
    for r in rewrites:
        if r.get("error") is not None:
            errors += 1
            continue
        if r["unchanged"]:
            inc = {"preserved": 1, "unchanged": 1}
        else:
            key = (_candidate_id(str(r["prompt_id"])), str(r["target"]))
            if key not in label_of:
                missing.append(key)
                continue
            label = label_of[key]
            inc = {"judged": 1}
            if label == "preserved":
                inc["preserved"] = 1
            elif label == "degraded":
                inc["degraded"] = 1
            else:
                inc["unlabelled"] = 1
        inc["total"] = 1
        for bucket in (
            pooled,
            by_target[str(r["tag"])],
            by_cat[str(r["category"])],
            by_tc[str(r["tag"])][str(r["category"])],
        ):
            for k, v in inc.items():
                bucket[k] += v
    if missing:
        raise CanaryError(f"{len(missing)} changed rewrites have no label (e.g. {missing[0]})")
    per_cat = {c: _ratio(v) for c, v in sorted(by_cat.items())}
    overall = _ratio(pooled)
    ok = [r for r in rewrites if r.get("error") is None]
    return {
        **overall,
        "errors": errors,
        "by_target": {t: _ratio(v) for t, v in sorted(by_target.items())},
        "by_category": per_cat,
        "by_target_category": {
            t: {c: _ratio(v) for c, v in sorted(cs.items())} for t, cs in sorted(by_tc.items())
        },
        "paths": dict(Counter(str(r.get("path")) for r in ok)),
        "gate_rejected": sum(bool(r.get("gate_rejected")) for r in ok),
        "fallbacks": dict(
            Counter(str(r["fallback_reason"]) for r in ok if r.get("fallback_reason"))
        ),
        "meets_overall_bar": overall["preservation"] is not None
        and overall["preservation"] >= ALERT_THRESHOLD,
        "meets_category_bar": all(
            v["preservation"] is not None and v["preservation"] >= MIN_CATEGORY
            for v in per_cat.values()
        ),
    }


def read_history(path: Path = HISTORY_PATH) -> list[dict[str, Any]]:
    """Every recorded canary run, oldest first (``[]`` if none)."""
    return _read_jsonl(path) if path.is_file() else []


def append_history(path: Path, record: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Append ``record``; a record with the same ``run_id`` is replaced (finalize is rerunnable)."""
    rows = [r for r in read_history(path) if r.get("run_id") != record["run_id"]]
    rows.append(dict(record))
    rows.sort(key=lambda r: str(r["time"]))
    _write_jsonl(path, rows)
    return rows


def _rolling_inputs(
    history: Sequence[Mapping[str, Any]], run_id: str, version: str
) -> list[tuple[int, int]]:
    """Earlier runs of the same canary version as ``(preserved, total)``, oldest first."""
    return [
        (int(r["preserved"]), int(r["total"]))
        for r in history
        if r.get("run_id") != run_id and r.get("canary_version") == version and int(r["total"])
    ]


def finalize(
    run_dir: Path,
    *,
    history_path: Path = HISTORY_PATH,
    gateway: str | None = None,
    report_path: Path | None = REPORT_PATH,
    handler: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Compute the run's preservation, record it in the history, push it, write the report."""
    meta = _load_meta(run_dir)
    rewrites = _read_jsonl(run_dir / "rewrites.jsonl")
    labels_path = run_dir / "labels.jsonl"
    labels = _read_jsonl(labels_path) if labels_path.is_file() else []
    summary = summarize(rewrites, labels)
    if not summary["total"]:
        raise CanaryError("no (prompt, target) pair was served: nothing to record")
    run_id = str(meta["run_id"])
    version = str(meta["canary_version"])
    earlier = _rolling_inputs(
        [r for r in read_history(history_path) if str(r.get("time", "")) <= meta["started"]],
        run_id,
        version,
    )
    rolling, used = rolling_preservation([*earlier, (summary["preserved"], summary["total"])])
    record = {
        "run_id": run_id,
        "time": meta["started"],
        "canary_version": version,
        "canary_sha256": meta["canary_sha256"],
        "base_url": meta["base_url"],
        "mode": meta["mode"],
        "targets": {t.tag: t.model for t in _meta_targets(meta)},
        "judge_model": meta.get("judge_model"),
        **summary,
        "rolling": rolling,
        "rolling_runs": used,
        "rolling_window": ROLLING_WINDOW,
        "alert": rolling < ALERT_THRESHOLD,
        "pushed": None,
    }
    if gateway:
        push_canary(gateway, summary["preserved"], summary["total"], earlier, handler=handler)
        record["pushed"] = {"gateway": gateway, "time": _now()}
    history = append_history(history_path, record)
    _write_json(run_dir / "result.json", record)
    _update_meta(run_dir, stages={"finalize": {"finished": _now(), "pushed": bool(gateway)}})
    if report_path is not None:
        render_report(history, report_path)
    return record


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.1%}"


def _report_block(history: Sequence[Mapping[str, Any]]) -> str:
    last = history[-1]
    tags = list(last["by_target"])
    cats = list(last["by_category"])
    lines = [
        f"### Latest run: `{last['run_id']}` ({_ist(str(last['time']))})",
        "",
        f"- Service: `{last['base_url']}`, mode `{last['mode']}`; canary "
        f"`{last['canary_version']}` (sha256 `{str(last['canary_sha256'])[:12]}`)",
        "- Targets: " + ", ".join(f"`{m}`" for m in last["targets"].values()),
        f"- Judge: `{last.get('judge_model')}` (`pairwise_avg` + task metric, label_v1 rule)",
        f"- **Run preservation {_pct(last['preservation'])}** ({last['preserved']}/"
        f"{last['total']} pairs); **rolling {_pct(last['rolling'])}** over "
        f"{last['rolling_runs']} run(s); alert below {ALERT_THRESHOLD:.0%}: "
        f"**{'FIRING' if last['alert'] else 'ok'}**",
        f"- Unchanged (preserved by construction) {last['unchanged']}, judged {last['judged']} "
        f"(degraded {last['degraded']}, unlabelled {last['unlabelled']}), service errors "
        f"{last['errors']}",
        f"- Service paths {json.dumps(last['paths'], sort_keys=True)}; gate rejected "
        f"{last['gate_rejected']}; fallbacks {json.dumps(last['fallbacks'], sort_keys=True)}",
        "- Pushed: "
        + (
            f"`{last['pushed']['gateway']}` at {_ist(last['pushed']['time'])}"
            if last.get("pushed")
            else "no (no gateway given)"
        ),
        "",
        "| Target | " + " | ".join(cats) + " | Overall |",
        "|---" * (len(cats) + 2) + "|",
    ]
    for t in tags:
        tc = last["by_target_category"][t]
        cells = [
            (
                f"{_pct(tc[c]['preservation'])} ({tc[c]['preserved']}/{tc[c]['total']})"
                if c in tc
                else "n/a"
            )
            for c in cats
        ]
        bt = last["by_target"][t]
        lines.append(
            f"| {t} | " + " | ".join(cells) + f" | {_pct(bt['preservation'])} "
            f"({bt['preserved']}/{bt['total']}) |"
        )
    bc = last["by_category"]
    lines.append(
        "| **both** | "
        + " | ".join(f"{_pct(bc[c]['preservation'])}" for c in cats)
        + f" | **{_pct(last['preservation'])}** |"
    )
    lines += [
        "",
        f"Per-category bar {MIN_CATEGORY:.0%} (C8.3): "
        f"{'met' if last['meets_category_bar'] else 'not met'}.",
        "",
        "### Run history",
        "",
        "| Run | Time | Mode | Pairs | Run preservation | Rolling (runs) | Alert |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in reversed(history):
        lines.append(
            f"| `{r['run_id']}` | {_ist(str(r['time']))} | {r.get('mode', '?')} | {r['total']} "
            f"| {_pct(r.get('preservation', r['preserved'] / r['total']))} "
            f"| {_pct(r.get('rolling'))} ({r.get('rolling_runs', '?')}) "
            f"| {'firing' if r.get('alert') else 'ok'} |"
        )
    return "\n".join(lines)


def _replace_block(text: str, begin: str, end: str, body: str) -> str:
    block = f"{begin}\n{body}\n{end}"
    if begin in text and end in text:
        head, _, rest = text.partition(begin)
        _, _, tail = rest.partition(end)
        return head + block + tail
    return text.rstrip("\n") + "\n\n" + block + "\n"


def render_report(history: Sequence[Mapping[str, Any]], path: Path = REPORT_PATH) -> str:
    """Regenerate the generated block of the report from the history (text outside is kept)."""
    text = path.read_text(encoding="utf-8") if path.is_file() else "# Quality canary\n"
    text = _replace_block(text, REPORT_BEGIN, REPORT_END, _report_block(history))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return text


# --------------------------------------------------------------------------------------------
# 5. Dashboard check


def _prom_value(http: httpx.Client, prometheus: str, query: str) -> float | None:
    r = http.get(prometheus.rstrip("/") + "/api/v1/query", params={"query": query})
    r.raise_for_status()
    result = r.json().get("data", {}).get("result", [])
    return float(result[0]["value"][1]) if result else None


def _walk_panels(panels: Sequence[Mapping[str, Any]]) -> Iterable[Mapping[str, Any]]:
    for p in panels:
        yield p
        yield from _walk_panels(p.get("panels", []))


def _grafana_frames_last(body: Mapping[str, Any]) -> float | None:
    last = None
    for res in body.get("results", {}).values():
        for frame in res.get("frames", []):
            values = frame.get("data", {}).get("values", [])
            if len(values) >= 2 and values[-1]:
                last = float(values[-1][-1])
    return last


def verify_dashboard(
    http: httpx.Client,
    *,
    prometheus: str = DEFAULT_PROMETHEUS,
    grafana: str = DEFAULT_GRAFANA,
    expected: Mapping[str, float] | None = None,
    wait_s: float = 120.0,
    sleep: Callable[[float], None] = time.sleep,
    poll_s: float = 5.0,
) -> dict[str, Any]:
    """Read the canary gauges back from Prometheus and the canary panels from Grafana.

    Polls Prometheus (scrape interval 15 s) until the rolling value matches ``expected`` or
    ``wait_s`` passes, then queries every dashboard panel whose expression uses a
    ``terser_canary_*`` metric through Grafana's ``/api/ds/query``. ``ok`` requires the
    Prometheus values to match ``expected`` and the rolling-preservation panel to return it.
    """
    exp = dict(expected or {})
    deadline = time.monotonic() + wait_s
    prom: dict[str, float | None] = {}
    while True:
        prom = {
            m: _prom_value(http, prometheus, m)
            for m in (ROLLING_PRESERVATION, PRESERVATION, PROMPTS)
        }
        if prom[ROLLING_PRESERVATION] is not None and all(
            prom.get(k) is not None and abs(float(prom[k] or 0) - v) < 1e-9 for k, v in exp.items()
        ):
            break
        if time.monotonic() >= deadline:
            break
        sleep(poll_s)
    targets = http.get(prometheus.rstrip("/") + "/api/v1/targets").json()
    push_health = [
        t.get("health")
        for t in targets.get("data", {}).get("activeTargets", [])
        if t.get("labels", {}).get("job") == "pushgateway"
    ]
    dash = http.get(grafana.rstrip("/") + f"/api/dashboards/uid/{DASHBOARD_UID}")
    dash.raise_for_status()
    panels = []
    for p in _walk_panels(dash.json()["dashboard"].get("panels", [])):
        for t in p.get("targets", []):
            expr = str(t.get("expr", ""))
            if "terser_canary" not in expr:
                continue
            body = {
                "from": "now-15m",
                "to": "now",
                "queries": [
                    {
                        "refId": t.get("refId", "A"),
                        "datasource": p.get("datasource") or {"uid": "prometheus"},
                        "expr": expr,
                        "instant": True,
                        "range": False,
                    }
                ],
            }
            r = http.post(grafana.rstrip("/") + "/api/ds/query", json=body)
            value = _grafana_frames_last(r.json()) if r.status_code == 200 else None
            panels.append(
                {"panel": p.get("title"), "expr": expr, "status": r.status_code, "value": value}
            )
    rolling_values = [x["value"] for x in panels if x["expr"] == f"min({ROLLING_PRESERVATION})"]
    want = exp.get(ROLLING_PRESERVATION)
    ok = (
        prom[ROLLING_PRESERVATION] is not None
        and all(
            prom.get(k) is not None and abs(float(prom[k] or 0) - v) < 1e-9 for k, v in exp.items()
        )
        and bool(rolling_values)
        and all(
            v is not None and (want is None or abs(float(v) - want) < 1e-9) for v in rolling_values
        )
    )
    return {
        "time": _now(),
        "ok": ok,
        "prometheus": prom,
        "expected": exp,
        "pushgateway_target_health": push_health,
        "dashboard": DASHBOARD_UID,
        "panels": panels,
    }


def _dashboard_block(check: Mapping[str, Any]) -> str:
    lines = [
        f"### Dashboard check ({_ist(str(check['time']))}): "
        f"**{'metric on dashboard' if check['ok'] else 'FAILED'}**",
        "",
        "Prometheus instant queries: "
        + ", ".join(f"`{k}` = {v}" for k, v in check["prometheus"].items())
        + f"; Pushgateway scrape target health {check['pushgateway_target_health']}.",
        "",
        f"Grafana dashboard `{check['dashboard']}`, canary panels through `/api/ds/query`:",
        "",
        "| Panel | Expression | HTTP | Value |",
        "|---|---|---|---|",
    ]
    for p in check["panels"]:
        lines.append(f"| {p['panel']} | `{p['expr']}` | {p['status']} | {p['value']} |")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------
# CLI


def _http(timeout: float, auth: tuple[str, str] | None = None) -> httpx.Client:
    import httpx

    return httpx.Client(timeout=timeout, auth=auth)


def _new_run_dir(run_id: str | None) -> tuple[str, Path]:
    rid = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return rid, RUNS_DIR / rid


def _targets_from_args(names: Sequence[str], models: Sequence[str]) -> list[CanaryTarget]:
    override = dict(m.split("=", 1) for m in models)
    out = []
    for n in names:
        if n not in TARGETS:
            raise SystemExit(f"unknown target {n!r}; choose from {list(TARGETS)}")
        t = TARGETS[n]
        out.append(CanaryTarget(t.tag, override.get(n, t.model), t.service_target))
    return out


def _start_run(args: argparse.Namespace) -> Path:
    canary = Path(args.canary)
    _refuse_test_split(canary)
    run_id, default_dir = _new_run_dir(args.run_id)
    run_dir = Path(args.run_dir) if args.run_dir else default_dir
    if _run_meta_path(run_dir).is_file() and not args.force:
        raise SystemExit(f"{run_dir} already has a run; pass --force to redo the rewrite stage")
    run_dir.mkdir(parents=True, exist_ok=True)
    # A redone rewrite stage invalidates everything downstream of it.
    for pattern in ("candidates_*", "responses_*", "labels.jsonl", "result.json", "dashboard.json"):
        for stale in run_dir.glob(pattern):
            stale.unlink()
    targets = _targets_from_args(args.targets, args.target_model)
    meta = {
        "run_id": run_dir.name if args.run_dir else run_id,
        "started": _now(),
        "canary": canary.as_posix(),
        "canary_version": canary.stem,
        "canary_sha256": _sha256(canary),
        "base_url": args.base_url,
        "mode": args.mode,
        "targets": [asdict(t) for t in targets],
        "judge_model": args.judge_model,
        "stages": {},
    }
    _write_json(_run_meta_path(run_dir), meta)
    return run_dir


def _cmd_rewrite(args: argparse.Namespace) -> Path:
    run_dir = _start_run(args)
    meta = _load_meta(run_dir)
    prompts = _read_jsonl(Path(meta["canary"]))
    with _http(args.http_timeout) as http:
        stats = rewrite_stage(
            prompts,
            _meta_targets(meta),
            run_dir,
            http=http,
            base_url=args.base_url,
            mode=args.mode,
            workers=args.workers,
            warmup=args.warmup,
            max_error_rate=args.max_error_rate,
        )
    print(json.dumps({"run_dir": run_dir.as_posix(), "rewrite": stats}))
    return run_dir


def _client(args: argparse.Namespace, run_dir: Path, url: str | None) -> LLMClient:
    return make_llm_client(
        url,
        phase=args.budget_phase,
        ledger_path=run_dir / "llm_ledger.json",
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
    )


def _cmd_respond(args: argparse.Namespace, run_dir: Path, tag: str, url: str | None) -> None:
    meta = _load_meta(run_dir)
    targets = {t.tag: t for t in _meta_targets(meta)}
    if tag not in targets:
        raise SystemExit(f"target {tag!r} is not part of run {meta['run_id']}: {list(targets)}")
    stats = respond_stage(
        _read_jsonl(Path(meta["canary"])),
        targets[tag],
        run_dir,
        _client(args, run_dir, url),
        workers=args.respond_workers,
    )
    print(json.dumps({"respond": tag, **stats.to_dict()}))


def _cmd_judge(args: argparse.Namespace, run_dir: Path) -> None:
    from terser.judge.protocols import averaged_judge
    from terser.phase2.label_stage import load_task_references

    meta = _load_meta(run_dir)
    prompts = _read_jsonl(Path(meta["canary"]))
    model = args.judge_model or meta.get("judge_model") or DEFAULT_JUDGE
    judge = averaged_judge(_client(args, run_dir, args.judge_url), provider="local", model=model)
    stats = judge_stage(
        prompts,
        run_dir,
        judge,
        references=load_task_references(prompts, Path(args.raw_dir)),
        metric_cache=Path(args.metric_cache),
        workers=args.judge_workers,
    )
    print(json.dumps({"judge": model, **stats.to_dict()}))


def _cmd_finalize(args: argparse.Namespace, run_dir: Path) -> dict[str, Any]:
    record = finalize(
        run_dir,
        history_path=Path(args.history),
        gateway=args.gateway,
        report_path=None if args.no_report else Path(args.report),
    )
    keys = ("run_id", "preserved", "total", "preservation", "rolling", "rolling_runs", "alert")
    print(json.dumps({k: record[k] for k in keys}))
    return record


def _cmd_verify(args: argparse.Namespace, run_dir: Path | None) -> int:
    expected: dict[str, float] = {}
    if run_dir is not None:
        rec = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        expected = {
            ROLLING_PRESERVATION: float(rec["rolling"]),
            PRESERVATION: float(rec["preservation"]),
            PROMPTS: float(rec["total"]),
        }
    auth = (args.grafana_user, args.grafana_password) if args.grafana_user else None
    with _http(30.0, auth) as http:
        check = verify_dashboard(
            http,
            prometheus=args.prometheus,
            grafana=args.grafana,
            expected=expected,
            wait_s=args.wait,
        )
    if run_dir is not None:
        _write_json(run_dir / "dashboard.json", check)
    if not args.no_report:
        path = Path(args.report)
        text = path.read_text(encoding="utf-8") if path.is_file() else "# Quality canary\n"
        path.write_text(
            _replace_block(text, DASH_BEGIN, DASH_END, _dashboard_block(check)),
            encoding="utf-8",
            newline="\n",
        )
    print(json.dumps(check))
    return 0 if check["ok"] else 1


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--run-dir", default=None, help="run directory (default logs/canary/<id>)")
    p.add_argument("--budget-phase", default=DEFAULT_BUDGET_PHASE)
    p.add_argument("--cache-dir", default=None, help="LLM response cache (default: budget.yaml)")
    p.add_argument("--history", default=str(HISTORY_PATH))
    p.add_argument("--report", default=str(REPORT_PATH))
    p.add_argument("--no-report", action="store_true")


def _add_rewrite(p: argparse.ArgumentParser) -> None:
    p.add_argument("--canary", default=str(CANARY_PATH))
    p.add_argument("--run-id", default=None)
    p.add_argument("--base-url", default=DEFAULT_SERVICE_URL, help="the deployed service")
    p.add_argument("--mode", choices=("fast", "quality"), default="quality")
    p.add_argument("--targets", nargs="+", default=list(TARGETS), choices=list(TARGETS))
    p.add_argument("--target-model", action="append", default=[], help="TAG=served model id")
    p.add_argument("--judge-model", default=None, help=f"default {DEFAULT_JUDGE}")
    p.add_argument("--workers", type=int, default=1, help="concurrent service requests")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--http-timeout", type=float, default=30.0)
    p.add_argument("--max-error-rate", type=float, default=0.05)
    p.add_argument("--force", action="store_true", help="redo the rewrite stage of a run dir")


def _add_llm(p: argparse.ArgumentParser) -> None:
    p.add_argument("--respond-workers", type=int, default=64)
    p.add_argument("--judge-workers", type=int, default=32)
    p.add_argument("--raw-dir", default="data/raw", help="HumanEval / MBPP rows for code tests")
    p.add_argument("--metric-cache", default=".cache/task_metrics")


def _add_push(p: argparse.ArgumentParser) -> None:
    p.add_argument("--gateway", default=None, help="Pushgateway, e.g. localhost:9091 (no push)")


def _add_verify(p: argparse.ArgumentParser) -> None:
    p.add_argument("--prometheus", default=DEFAULT_PROMETHEUS)
    p.add_argument("--grafana", default=DEFAULT_GRAFANA)
    p.add_argument("--grafana-user", default=None)
    p.add_argument("--grafana-password", default=None)
    p.add_argument("--wait", type=float, default=120.0, help="seconds to wait for the scrape")


def _required_run_dir(args: argparse.Namespace) -> Path:
    if not args.run_dir:
        raise SystemExit("--run-dir is required")
    return Path(args.run_dir)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point (see the module docstring)."""
    ap = argparse.ArgumentParser(prog="python -m terser.monitoring.canary", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("select", help="build the fixed canary set from the val split (once)")
    p.add_argument("--source", default=str(CANARY_SOURCE))
    p.add_argument("--out", default=str(CANARY_PATH))
    p.add_argument("--n", type=int, default=CANARY_SIZE)
    p.add_argument("--seed", type=int, default=CANARY_SEED)
    p = sub.add_parser("rewrite", help="stage 1: canary prompts through the deployed service")
    _add_common(p)
    _add_rewrite(p)
    p = sub.add_parser("respond", help="stage 2: one target answers originals and rewrites")
    _add_common(p)
    _add_llm(p)
    p.add_argument("--target", required=True, choices=list(TARGETS))
    p.add_argument("--llm-url", default=None, help="OpenAI-compatible URL of the target")
    p = sub.add_parser("judge", help="stage 3: judge and label the changed pairs")
    _add_common(p)
    _add_llm(p)
    p.add_argument("--judge-model", default=None)
    p.add_argument("--judge-url", default=None)
    p = sub.add_parser("finalize", help="stage 4: preservation, history, push, report")
    _add_common(p)
    _add_push(p)
    p = sub.add_parser("verify", help="read the metric back from Prometheus and Grafana")
    _add_common(p)
    _add_verify(p)
    p = sub.add_parser("run", help="one full cycle (every endpoint up at once)")
    _add_common(p)
    _add_rewrite(p)
    _add_llm(p)
    _add_push(p)
    _add_verify(p)
    p.add_argument("--llm-url", default=None, help="URL of every target (see --target-url)")
    p.add_argument("--target-url", action="append", default=[], help="TAG=URL per target")
    p.add_argument("--judge-url", default=None)
    p.add_argument("--verify", action="store_true", help="also run the dashboard check")
    args = ap.parse_args(argv)

    if args.command == "select":
        manifest = write_canary(Path(args.source), Path(args.out), args.n, args.seed)
        print(json.dumps(manifest, indent=2))
        return 0
    if args.command == "rewrite":
        _cmd_rewrite(args)
        return 0
    if args.command == "respond":
        _cmd_respond(args, _required_run_dir(args), args.target, args.llm_url)
        return 0
    if args.command == "judge":
        _cmd_judge(args, _required_run_dir(args))
        return 0
    if args.command == "finalize":
        _cmd_finalize(args, _required_run_dir(args))
        return 0
    if args.command == "verify":
        return _cmd_verify(args, Path(args.run_dir) if args.run_dir else None)
    # run: every stage in order
    run_dir = _cmd_rewrite(args)
    urls = dict(u.split("=", 1) for u in args.target_url)
    for t in _meta_targets(_load_meta(run_dir)):
        _cmd_respond(args, run_dir, t.tag, urls.get(t.tag, args.llm_url))
    _cmd_judge(args, run_dir)
    _cmd_finalize(args, run_dir)
    if args.verify:
        return _cmd_verify(args, run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
