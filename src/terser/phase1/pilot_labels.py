"""Pilot labelling pipeline (plan row **1.3.6**).

For every pilot prompt and every operator subset in :data:`OPERATOR_SUBSETS` (the 7 singletons
plus the cumulative stacks ``FHC``, ``FHCI``, ``FHCID``, ``FHCIDS``, ``FHCIDSL``):

1. rewrite the prompt with :func:`terser.operators.apply_ops`; subsets that leave the prompt
   unchanged are skipped (their label would be trivially ``preserved``);
2. get the target model's output for the original prompt and for the rewrite (temperature 0,
   through :class:`terser.llm.LLMClient`, so outputs are disk-cached and budget-checked; the
   original prompt's output is generated once and reused by all subsets);
3. judge (original output, rewrite output) in both A/B orders with
   :class:`terser.judge.pairwise.PairwiseJudge`;
4. label: with ``protocol: pairwise`` (``pairwise_v1``), ``preserved`` iff tie-or-rewrite-better
   in both orders (row 2.3.4, judge part, via :func:`terser.judge.labels.label_pair_from_result`
   with no task metric); with ``protocol: pairwise_avg`` (ADR-0005, which produced the current
   ``data/pilot/labels.jsonl``), the A/B/tie probabilities of both orders are averaged first and
   the same rule is applied to the averaged side;
5. append a record to ``labels.jsonl`` and write ``labels_review.csv`` for human review of every
   label (the CSV lists all rows; ``needs_human_review`` marks the ones to look at first).

Compute routing. ``stand_in: true`` (the default) means the models are CPU smoke-test stand-ins
(Qwen2.5-0.5B/1.5B via Ollama). Stand-in output is written **only** to ``data/pilot/smoke/`` and
every record carries ``stand_in: true``; it must never be used as a label. Real labels
(``stand_in: false``) come from GPU-served models on vLLM and go to ``data/pilot/``.

On the GPU the target and the judge cannot share the 24 GB card, so the pipeline runs in stages::

    # 1) vLLM serving the target model
    python -m terser.phase1.pilot_labels experiment=p1_pilot_labels hardware=rtx4500ada \\
        experiment.params.stand_in=false experiment.params.stage=respond \\
        experiment.params.target_model=Qwen/Qwen2.5-7B-Instruct
    # 2) restart vLLM with the judge; target outputs now come from the cache only
    python -m terser.phase1.pilot_labels experiment=p1_pilot_labels hardware=rtx4500ada \\
        experiment.params.stand_in=false experiment.params.stage=judge \\
        experiment.params.target_model=Qwen/Qwen2.5-7B-Instruct \\
        experiment.params.judge_model=Qwen/Qwen2.5-32B-Instruct-AWQ

CLI: ``python -m terser.phase1.pilot_labels experiment=p1_pilot_labels``.
"""

from __future__ import annotations

import csv
import json
import re
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from terser.data.casualise import is_stand_in
from terser.data.categories import parse_category
from terser.judge.labels import label_pair, label_pair_from_result
from terser.judge.pairwise import PairwiseJudge, PairwiseResult
from terser.judge.protocols import DEFAULT_TOP_LOGPROBS, LOW_MARGIN, order_averaged_side
from terser.llm import BudgetExceededError, LLMClient, LocalLLMUnavailableError, cache_key
from terser.phase1.token_study import OP_SETS

if TYPE_CHECKING:
    from omegaconf import DictConfig

__all__ = [
    "OPERATOR_SUBSETS",
    "PilotItem",
    "PilotServerUnavailableError",
    "PilotSettings",
    "PilotSummary",
    "TargetOutputMissingError",
    "iter_items",
    "load_prompts",
    "main",
    "output_dir_for",
    "run_from_config",
    "run_pilot",
    "settings_from_config",
]

OPERATOR_SUBSETS: tuple[str, ...] = OP_SETS
"""The 12 operator subsets of row 1.3.6: 7 singletons + cumulative stacks up to all operators."""

Stage = Literal["all", "respond", "judge"]
JudgeProtocolName = Literal["pairwise", "pairwise_avg"]
ApplyOps = Callable[[str, str, str | None], str]

REVIEW_COLUMNS: tuple[str, ...] = (
    "id",
    "category",
    "ops",
    "label",
    "verdict_original_first",
    "verdict_rewrite_first",
    "protocol",
    "averaged_side",
    "averaged_margin",
    "reason_original_first",
    "reason_rewrite_first",
    "needs_human_review",
    "review_reasons",
    "stand_in",
    "original",
    "rewrite",
    "human_label",
    "reviewer",
    "notes",
)


class PilotServerUnavailableError(RuntimeError):
    """The local OpenAI-compatible server (Ollama / vLLM) could not be reached."""


class TargetOutputMissingError(RuntimeError):
    """``stage=judge`` needs a target output that is not in the cache (run ``stage=respond``)."""


@dataclass(frozen=True)
class PilotSettings:
    """Resolved settings of one pilot run (from ``configs/experiment/p1_pilot_labels.yaml``)."""

    prompts_path: Path
    output_dir: Path
    smoke_dir: Path
    provider: str = "local"
    judge_provider: str | None = None
    target_model: str = "qwen2.5:1.5b"
    judge_model: str = "qwen2.5:0.5b"
    stand_in: bool = True
    stage: Stage = "all"
    operator_subsets: tuple[str, ...] = OPERATOR_SUBSETS
    operator_target: str | None = None
    categories: tuple[str, ...] | None = None
    limit: int | None = None
    max_target_tokens: int = 512
    max_judge_tokens: int = 256
    judge_retries: int = 2
    temperature: float = 0.0
    prompt_version: str = "pairwise_v1"
    budget_phase: str = "phase1"
    workers: int = 1
    protocol: JudgeProtocolName = "pairwise"

    def __post_init__(self) -> None:
        if self.protocol not in ("pairwise", "pairwise_avg"):
            raise ValueError(f"protocol must be pairwise|pairwise_avg, got {self.protocol!r}")
        if self.stage not in ("all", "respond", "judge"):
            raise ValueError(f"stage must be all|respond|judge, got {self.stage!r}")
        if self.workers < 1:
            raise ValueError(f"workers must be >= 1, got {self.workers}")
        if not self.stand_in:
            tags = [m for m in (self.target_model, self.judge_model) if is_stand_in(m)]
            if tags:
                raise ValueError(
                    f"stand_in=false but {tags} look like Ollama stand-in tags; real pilot labels "
                    "must come from GPU-served models (e.g. Qwen/Qwen2.5-7B-Instruct on vLLM). "
                    "Keep stand_in=true for CPU smoke runs."
                )

    @property
    def resolved_judge_provider(self) -> str:
        """Provider used for the judge (defaults to the target provider)."""
        return self.judge_provider or self.provider


def output_dir_for(settings: PilotSettings) -> Path:
    """Where outputs go: ``smoke_dir`` for stand-in runs, ``output_dir`` otherwise.

    Raises:
        ValueError: If a real (non-stand-in) run would write inside the smoke directory.
    """
    if settings.stand_in:
        return settings.smoke_dir
    out, smoke = settings.output_dir.resolve(), settings.smoke_dir.resolve()
    if out == smoke or smoke in out.parents:
        raise ValueError(f"non-stand-in output_dir {out} must not be inside smoke_dir {smoke}")
    return settings.output_dir


@dataclass(frozen=True)
class PilotItem:
    """One (prompt, operator subset) pair whose rewrite differs from the prompt."""

    id: str
    prompt_id: str
    category: str
    ops: str
    original: str
    rewrite: str


@dataclass
class PilotSummary:
    """Counts and accounting of one run."""

    stage: str
    stand_in: bool
    output_dir: Path
    n_prompts: int = 0
    n_pairs_considered: int = 0
    n_skipped_unchanged: int = 0
    n_items: int = 0
    labels: Counter[str] = field(default_factory=Counter)
    n_needs_review: int = 0
    n_position_inconsistent: int = 0
    generated_tokens: int = 0
    llm_calls: int = 0
    cached_calls: int = 0
    wall_seconds: float = 0.0
    stopped_reason: str | None = None

    def metrics(self) -> dict[str, float]:
        """Flat numeric metrics for MLflow."""
        return {
            "n_prompts": self.n_prompts,
            "n_pairs_considered": self.n_pairs_considered,
            "n_skipped_unchanged": self.n_skipped_unchanged,
            "n_items": self.n_items,
            "n_preserved": self.labels.get("preserved", 0),
            "n_degraded": self.labels.get("degraded", 0),
            "n_unlabelled": self.labels.get("none", 0),
            "n_needs_review": self.n_needs_review,
            "n_position_inconsistent": self.n_position_inconsistent,
            "generated_tokens": self.generated_tokens,
            "llm_calls": self.llm_calls,
            "cached_calls": self.cached_calls,
            "wall_seconds": self.wall_seconds,
        }


def load_prompts(
    path: Path, *, categories: Iterable[str] | None = None, limit: int | None = None
) -> list[dict[str, Any]]:
    """Read pilot prompts (JSONL with at least ``id``, ``category``, ``prompt``).

    Args:
        path: JSONL file.
        categories: Keep only these categories (None keeps all).
        limit: Keep at most this many prompts **per category** (None keeps all).

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If a line lacks a required field or an id repeats.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"pilot prompts not found: {path} (expected JSONL with id, category, prompt)"
        )
    wanted = set(categories) if categories is not None else None
    per_cat: Counter[str] = Counter()
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = [k for k in ("id", "category", "prompt") if not row.get(k)]
            if missing:
                raise ValueError(f"{path}:{lineno}: missing field(s) {missing}")
            if row["id"] in seen:
                raise ValueError(f"{path}:{lineno}: duplicate id {row['id']!r}")
            seen.add(row["id"])
            try:  # same category normalisation as terser.phase1.prompts
                row = {**row, "category": parse_category(row["category"]).value}
            except ValueError as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from None
            cat = str(row["category"])
            if wanted is not None and cat not in wanted:
                continue
            if limit is not None and per_cat[cat] >= limit:
                continue
            per_cat[cat] += 1
            out.append(row)
    return out


def _default_apply_ops(text: str, codes: str, target: str | None) -> str:
    """Lazy wrapper around :func:`terser.operators.apply_ops`."""
    import importlib

    apply = importlib.import_module("terser.operators").apply_ops
    return str(apply(text, codes, target=target))


def iter_items(
    prompts: Iterable[Mapping[str, Any]],
    subsets: Iterable[str],
    *,
    apply_ops: ApplyOps | None = None,
    target: str | None = None,
    on_skip: Callable[[str, str], None] | None = None,
) -> Iterator[PilotItem]:
    """Yield a :class:`PilotItem` per (prompt, subset) whose rewrite changes the prompt.

    ``on_skip(prompt_id, ops)`` is called for each unchanged (skipped) pair.
    """
    apply = apply_ops or _default_apply_ops
    subsets = tuple(subsets)
    for row in prompts:
        original = str(row["prompt"])
        for ops in subsets:
            rewrite = apply(original, ops, target)
            if rewrite.strip() == original.strip():
                if on_skip is not None:
                    on_skip(str(row["id"]), ops)
                continue
            yield PilotItem(
                id=f"{row['id']}:{ops}",
                prompt_id=str(row["id"]),
                category=str(row["category"]),
                ops=ops,
                original=original,
                rewrite=rewrite,
            )


class _Counting:
    """Tracks calls/tokens over a client's responses (thread-safe)."""

    def __init__(self, summary: PilotSummary) -> None:
        self.summary = summary
        self._lock = threading.Lock()

    def add(self, cached: bool, output_tokens: int) -> None:
        with self._lock:
            self.summary.llm_calls += 1
            if cached:
                self.summary.cached_calls += 1
            else:
                self.summary.generated_tokens += output_tokens


class _NoCounting(_Counting):
    """Counter for cache re-reads already counted by the parallel prefetch."""

    def __init__(self) -> None:
        pass

    def add(self, cached: bool, output_tokens: int) -> None:
        return None


def _target_output(
    client: LLMClient, settings: PilotSettings, text: str, counter: _Counting
) -> str:
    """Target output for ``text`` (temperature 0); in ``stage=judge`` only from the cache."""
    messages = [{"role": "user", "content": text}]
    if settings.stage == "judge":
        key = cache_key(
            settings.provider,
            settings.target_model,
            messages,
            {"temperature": settings.temperature, "max_tokens": settings.max_target_tokens},
            None,
        )
        hit = client.cache.get(key)
        if hit is None:
            raise TargetOutputMissingError(
                f"stage=judge: no cached output of {settings.target_model!r} for a prompt; "
                "run stage=respond with the target model served first"
            )
        counter.add(True, 0)
        return str(hit["text"])
    resp = client.complete(
        settings.provider,
        settings.target_model,
        messages,
        settings.max_target_tokens,
        temperature=settings.temperature,
    )
    counter.add(resp.cached, resp.output_tokens)
    return resp.text


def _review_reasons(
    result: PairwiseResult, settings: PilotSettings, margin: float | None = None
) -> list[str]:
    reasons = []
    if result.parse_failed:
        reasons.append("unparseable_verdict")
    elif not result.position_consistent:
        reasons.append("position_inconsistent")
    if settings.protocol == "pairwise_avg" and not result.parse_failed:
        if margin is None or margin < LOW_MARGIN:
            reasons.append("low_margin")
    if settings.stand_in:
        reasons.append("stand_in")
    return reasons


def _record(item: PilotItem, result: PairwiseResult, settings: PilotSettings) -> dict[str, Any]:
    aggregate: dict[str, Any] | None = None
    if settings.protocol == "pairwise_avg":
        side, probs = order_averaged_side(result.original_first, result.rewrite_first)
        margin = None
        if probs is not None:
            top, second = sorted(probs.values(), reverse=True)[:2]
            margin = top - second
        label = label_pair(None, side, side).label
        aggregate = {"protocol": "pairwise_avg", "side": side, "probs": probs, "margin": margin}
        reasons = _review_reasons(result, settings, margin)
    else:
        label = label_pair_from_result(None, result).label
        reasons = _review_reasons(result, settings)
    rec = {
        "id": item.id,
        "prompt_id": item.prompt_id,
        "category": item.category,
        "ops": item.ops,
        "original": item.original,
        "rewrite": item.rewrite,
        "target_model": settings.target_model,
        "judge_model": settings.judge_model,
        "judge_prompt_version": settings.prompt_version,
        "verdicts": result.verdicts(),
        "label": label,
        "stand_in": settings.stand_in,
        "needs_human_review": bool(reasons),
        "review_reasons": reasons,
    }
    if aggregate is not None:
        rec["judge_protocol"] = "pairwise_avg"
        rec["aggregate"] = aggregate
    return rec


def write_review_csv(records: Iterable[Mapping[str, Any]], path: Path) -> None:
    """Write the human-review sheet (one row per label; empty ``human_label`` to fill in)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(REVIEW_COLUMNS), lineterminator="\n")
        writer.writeheader()
        for rec in records:
            v = rec["verdicts"]
            writer.writerow(
                {
                    "id": rec["id"],
                    "category": rec["category"],
                    "ops": rec["ops"],
                    "label": rec["label"] or "",
                    "verdict_original_first": v["original_first"]["side"] or "",
                    "verdict_rewrite_first": v["rewrite_first"]["side"] or "",
                    "protocol": rec.get("judge_protocol", "pairwise"),
                    "averaged_side": (rec.get("aggregate") or {}).get("side") or "",
                    "averaged_margin": _fmt_margin((rec.get("aggregate") or {}).get("margin")),
                    "reason_original_first": v["original_first"]["reason"],
                    "reason_rewrite_first": v["rewrite_first"]["reason"],
                    "needs_human_review": rec["needs_human_review"],
                    "review_reasons": ";".join(rec["review_reasons"]),
                    "stand_in": rec["stand_in"],
                    "original": rec["original"],
                    "rewrite": rec["rewrite"],
                    "human_label": "",
                    "reviewer": "",
                    "notes": "",
                }
            )


def _fmt_margin(value: float | None) -> str:
    return "" if value is None else f"{value:.3f}"


def run_pilot(
    settings: PilotSettings,
    client: LLMClient,
    *,
    apply_ops: ApplyOps | None = None,
) -> PilotSummary:
    """Run the pilot pipeline and write outputs; return a :class:`PilotSummary`.

    ``stage=respond`` only fills the target-output cache and writes no label files;
    ``stage=judge`` reads target outputs from the cache only; ``stage=all`` does both.

    Raises:
        PilotServerUnavailableError: If the OpenAI-compatible server cannot be reached.
        BudgetExceededError: If the phase token cap would be exceeded. Records completed before
            the cap are kept in ``labels.jsonl`` and the review CSV.
        TargetOutputMissingError: ``stage=judge`` without a prior ``stage=respond``.
    """
    start = time.perf_counter()
    out_dir = output_dir_for(settings)
    summary = PilotSummary(stage=settings.stage, stand_in=settings.stand_in, output_dir=out_dir)
    counter = _Counting(summary)
    prompts = load_prompts(
        settings.prompts_path, categories=settings.categories, limit=settings.limit
    )
    summary.n_prompts = len(prompts)

    def on_skip(_pid: str, _ops: str) -> None:
        summary.n_skipped_unchanged += 1

    items = iter_items(
        prompts,
        settings.operator_subsets,
        apply_ops=apply_ops,
        target=settings.operator_target,
        on_skip=on_skip,
    )
    judge = PairwiseJudge(
        client,
        provider=settings.resolved_judge_provider,
        model=settings.judge_model,
        max_tokens=settings.max_judge_tokens,
        retries=settings.judge_retries,
        temperature=settings.temperature,
        prompt_version=settings.prompt_version,
        top_logprobs=DEFAULT_TOP_LOGPROBS if settings.protocol == "pairwise_avg" else None,
    )
    records: list[dict[str, Any]] = []
    labels_path = out_dir / "labels.jsonl"
    fh = None
    pool: ThreadPoolExecutor | None = None

    def judged(item: PilotItem, count: _Counting) -> PairwiseResult | None:
        original_out = _target_output(client, settings, item.original, count)
        rewrite_out = _target_output(client, settings, item.rewrite, count)
        if settings.stage == "respond":
            return None
        return judge.judge_pair(item.original, original_out, rewrite_out, category=item.category)

    try:
        if settings.stage != "respond":
            out_dir.mkdir(parents=True, exist_ok=True)
            fh = labels_path.open("w", encoding="utf-8", newline="\n")
        if settings.workers == 1:
            results: Iterator[tuple[PilotItem, PairwiseResult | None]] = (
                (item, judged(item, counter)) for item in _counted(items, summary)
            )
        else:
            # Concurrent requests let vLLM batch. Unique target texts are generated once; then
            # the first item of each distinct judge request is judged concurrently and repeats
            # are resolved in order from the cache, so records match a sequential run exactly.
            pool = ThreadPoolExecutor(max_workers=settings.workers)
            item_list = list(items)
            summary.n_items = len(item_list)
            texts = dict.fromkeys(t for it in item_list for t in (it.original, it.rewrite))
            # Target reads are counted here, once per unique text, in every stage (in the judge
            # stage they are cache reads); the per-item reads below go through ``no_count``.
            list(pool.map(lambda t: _target_output(client, settings, t, counter), texts))
            no_count = _NoCounting()
            seen: set[tuple[str, str, str]] = set()
            futures: list[Future[PairwiseResult | None] | None] = []
            for it in item_list:
                key = (it.original, it.rewrite, it.category)
                futures.append(None if key in seen else pool.submit(judged, it, no_count))
                seen.add(key)
            results = (
                (it, fut.result() if fut is not None else judged(it, no_count))
                for it, fut in zip(item_list, futures, strict=True)
            )
        for item, result in results:
            if result is None:
                continue
            for call in (result.original_first, result.rewrite_first):
                summary.llm_calls += call.attempts
                summary.cached_calls += call.attempts if call.cached else 0
                summary.generated_tokens += call.output_tokens
            rec = _record(item, result, settings)
            records.append(rec)
            summary.labels[rec["label"] or "none"] += 1
            summary.n_needs_review += int(rec["needs_human_review"])
            summary.n_position_inconsistent += int(
                not result.parse_failed and not result.position_consistent
            )
            assert fh is not None
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
    except LocalLLMUnavailableError as exc:
        summary.stopped_reason = "server_unavailable"
        raise PilotServerUnavailableError(
            f"pilot labelling needs a running OpenAI-compatible server for provider "
            f"{settings.provider!r} (target {settings.target_model!r}, judge "
            f"{settings.judge_model!r}). {exc}"
        ) from exc
    except BudgetExceededError:
        summary.stopped_reason = "budget_exceeded"
        raise
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        summary.n_pairs_considered = summary.n_items + summary.n_skipped_unchanged
        if fh is not None:
            fh.close()
            write_review_csv(records, out_dir / "labels_review.csv")
        summary.wall_seconds = time.perf_counter() - start
    return summary


def _counted(items: Iterable[PilotItem], summary: PilotSummary) -> Iterator[PilotItem]:
    """Count each item before it is processed (so a mid-item failure still counts it)."""
    for item in items:
        summary.n_items += 1
        yield item


def _opt_tuple(value: Any) -> tuple[str, ...] | None:
    return None if value is None else tuple(str(v) for v in value)


def settings_from_config(cfg: DictConfig) -> PilotSettings:
    """Build :class:`PilotSettings` from a composed root config (``experiment=p1_pilot_labels``)."""
    from omegaconf import OmegaConf

    params = OmegaConf.to_container(cfg.experiment.params, resolve=True)
    assert isinstance(params, dict)
    p: dict[str, Any] = {str(k): v for k, v in params.items()}
    return PilotSettings(
        prompts_path=Path(p["prompts_path"]),
        output_dir=Path(p["output_dir"]),
        smoke_dir=Path(p["smoke_dir"]),
        provider=str(p.get("provider", "local")),
        judge_provider=p.get("judge_provider"),
        target_model=str(p["target_model"]),
        judge_model=str(p["judge_model"]),
        stand_in=bool(p.get("stand_in", True)),
        stage=p.get("stage", "all"),
        operator_subsets=_opt_tuple(p.get("operator_subsets")) or OPERATOR_SUBSETS,
        operator_target=p.get("operator_target"),
        categories=_opt_tuple(p.get("categories")),
        limit=p.get("limit"),
        max_target_tokens=int(p.get("max_target_tokens", 512)),
        max_judge_tokens=int(p.get("max_judge_tokens", 256)),
        judge_retries=int(p.get("judge_retries", 2)),
        temperature=float(p.get("temperature", 0.0)),
        prompt_version=str(p.get("prompt_version", "pairwise_v1")),
        budget_phase=str(p.get("budget_phase", "phase1")),
        workers=int(p.get("workers", 1)),
        protocol=p.get("protocol", "pairwise"),
    )


def _run_name(settings: PilotSettings) -> str:
    model = re.sub(r"[^A-Za-z0-9_.]", "_", settings.target_model.split("/")[-1])
    variant: str = "standin" if settings.stand_in else settings.stage
    if settings.protocol == "pairwise_avg":
        variant += "avg"
    return f"1-pilotlabels-{model}-{variant}-0"


def run_from_config(
    cfg: DictConfig,
    *,
    client: LLMClient | None = None,
    apply_ops: ApplyOps | None = None,
    track: bool = True,
) -> PilotSummary:
    """Run the pilot from a composed config, optionally inside an MLflow run (phase 1, 1.3.6)."""
    settings = settings_from_config(cfg)
    client = client or LLMClient(settings.budget_phase)
    if not track:
        return run_pilot(settings, client, apply_ops=apply_ops)

    import mlflow

    from terser.tracking import start_run

    hardware = str(cfg.get("hardware", {}).get("name", "cpu"))
    params = {
        "provider": settings.provider,
        "target_model": settings.target_model,
        "judge_model": settings.judge_model,
        "stand_in": settings.stand_in,
        "stage": settings.stage,
        "operator_subsets": ",".join(settings.operator_subsets),
        "judge_prompt_version": settings.prompt_version,
        "judge_retries": settings.judge_retries,
        "temperature": settings.temperature,
        "limit": settings.limit,
        "prompts_path": settings.prompts_path.as_posix(),
        "workers": settings.workers,
        "protocol": settings.protocol,
    }
    with start_run(
        phase=1,
        row="1.3.6",
        task="pilotlabels",
        run_name=_run_name(settings),
        hardware=hardware,
        params=params,
    ):
        mlflow.set_tag("stand_in", str(settings.stand_in).lower())
        try:
            summary = run_pilot(settings, client, apply_ops=apply_ops)
        except BaseException as exc:
            mlflow.set_tag("stopped_reason", exc.__class__.__name__)
            raise
        mlflow.log_metrics(summary.metrics())
        gpu_hours = summary.wall_seconds / 3600 if hardware != "cpu" else 0.0
        mlflow.log_metric("gpu_hours", gpu_hours)
        for name in ("labels.jsonl", "labels_review.csv"):
            path = summary.output_dir / name
            if path.is_file():
                mlflow.log_artifact(str(path))
    return summary


def _hydra_entry() -> None:
    from terser.config import hydra_main

    @hydra_main()
    def _main(cfg: DictConfig) -> None:
        if cfg.experiment.get("name") != "p1_pilot_labels":
            print("usage: python -m terser.phase1.pilot_labels experiment=p1_pilot_labels")
            sys.exit(2)
        try:
            summary = run_from_config(cfg)
        except PilotServerUnavailableError as exc:
            print(f"error: {exc}", file=sys.stderr)
            sys.exit(2)
        except (BudgetExceededError, TargetOutputMissingError, FileNotFoundError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            sys.exit(1)
        print(json.dumps({**summary.metrics(), "output_dir": str(summary.output_dir)}, indent=2))

    _main()


def main() -> None:
    """CLI entry point (Hydra)."""
    _hydra_entry()


if __name__ == "__main__":
    main()
