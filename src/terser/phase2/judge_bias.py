"""Judge position-bias study on the pilot pairs (supports rows 2.3.3 / 2.3.4, checkpoint C2.6).

Re-judges the labelled pilot pairs of row 1.3.6 (``data/pilot/labels.jsonl``, target outputs
read from the LLM cache only) with a judge protocol from :mod:`terser.judge.protocols` and
compares protocols on the position-disagreement rate the C2.6 check measures. Results feed
ADR-0005 and ``reports/p2_judge_bias.md``.

``run`` judges every pair once per protocol run and writes ``<out_dir>/<name>.jsonl``:

* ``--protocol pairwise``: :class:`~terser.judge.pairwise.PairwiseJudge` in both orders; with
  ``--top-logprobs`` each call also carries A/B/tie probabilities, so the same run yields both
  the greedy ``pairwise`` sides and the order-averaged ``pairwise_avg`` side;
* ``--protocol pointwise``: :class:`~terser.judge.pointwise.PointwiseJudge` scores each output.

``report`` reads the runs plus the original pilot labels and writes a CSV with one row per
(run, derived protocol): position disagreement, first-position share, tie / degraded shares,
agreement with the reference set (pairs on which the original 1.3.6 judge was
position-consistent), identical-output handling, calls and generated tokens.

Like the pilot, judge stages need the judge model served by vLLM on the ``local`` provider::

    python -m terser.phase2.judge_bias run --protocol pairwise --name pairwise_32b \\
        --judge-model Qwen/Qwen2.5-32B-Instruct-AWQ --top-logprobs 20 --workers 16
    python -m terser.phase2.judge_bias report --runs pairwise_32b pointwise_32b \\
        --out reports/p2_judge_bias.csv
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from terser.eval.agreement import cohen_kappa
from terser.judge.labels import label_pair
from terser.judge.pairwise import PairwiseJudge
from terser.judge.pointwise import PointwiseJudge
from terser.judge.protocols import order_averaged_side
from terser.llm import LLMClient, cache_key

__all__ = [
    "PilotPair",
    "RunStats",
    "derive_sides",
    "load_pairs",
    "main",
    "protocol_rows",
    "run_protocol",
    "target_output",
]

# The original pilot verdicts; data/pilot/labels.jsonl now holds the pairwise_avg relabel.
DEFAULT_LABELS = Path("data/pilot/labels_pairwise_v1.jsonl")
DEFAULT_OUT_DIR = Path("data/pilot/judge_bias")
TARGET_MAX_TOKENS = 512


@dataclass(frozen=True)
class PilotPair:
    """One labelled pilot pair with both target outputs."""

    id: str
    prompt_id: str
    category: str
    ops: str
    original: str
    rewrite: str
    target_model: str
    original_output: str
    rewrite_output: str


def target_output(
    client: LLMClient, provider: str, model: str, text: str, temperature: float = 0.0
) -> str:
    """Cached target output for ``text`` (the request the pilot ``respond`` stage made)."""
    key = cache_key(
        provider,
        model,
        [{"role": "user", "content": text}],
        {"temperature": temperature, "max_tokens": TARGET_MAX_TOKENS},
        None,
    )
    hit = client.cache.get(key)
    if hit is None:
        raise KeyError(f"no cached {model} output for a pilot prompt; rerun the respond stage")
    return str(hit["text"])


def load_pairs(labels_path: Path, client: LLMClient, *, provider: str = "local") -> list[PilotPair]:
    """Pilot pairs from ``labels.jsonl`` with their target outputs from the cache."""
    pairs = []
    with labels_path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            model = rec["target_model"]
            pairs.append(
                PilotPair(
                    id=rec["id"],
                    prompt_id=rec["prompt_id"],
                    category=rec["category"],
                    ops=rec["ops"],
                    original=rec["original"],
                    rewrite=rec["rewrite"],
                    target_model=model,
                    original_output=target_output(client, provider, model, rec["original"]),
                    rewrite_output=target_output(client, provider, model, rec["rewrite"]),
                )
            )
    return pairs


@dataclass
class RunStats:
    """Accounting of one protocol run."""

    calls: int = 0
    cached_calls: int = 0
    generated_tokens: int = 0
    wall_seconds: float = 0.0


def _judge_record(
    pair: PilotPair,
    protocol: str,
    pairwise: PairwiseJudge | None,
    pointwise: PointwiseJudge | None,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": pair.id,
        "prompt_id": pair.prompt_id,
        "category": pair.category,
        "ops": pair.ops,
        "protocol": protocol,
        "identical_outputs": pair.original_output == pair.rewrite_output,
    }
    if protocol == "pairwise":
        assert pairwise is not None
        result = pairwise.judge_pair(
            pair.original, pair.original_output, pair.rewrite_output, category=pair.category
        )
        base["judge_model"] = pairwise.model
        base["calls"] = [result.original_first.to_dict(), result.rewrite_first.to_dict()]
    else:
        assert pointwise is not None
        orig, rew, side = pointwise.compare(
            pair.original, pair.original_output, pair.rewrite_output, pair.category
        )
        base["judge_model"] = pointwise.model
        base["scores"] = {"original": orig.to_dict(), "rewrite": rew.to_dict()}
        base["pointwise_side"] = side
    return base


def _account(rec: Mapping[str, Any], stats: RunStats, lock: threading.Lock) -> None:
    calls = rec["calls"] if "calls" in rec else list(rec["scores"].values())
    with lock:
        for call in calls:
            stats.calls += call["attempts"]
            stats.cached_calls += call["attempts"] if call["cached"] else 0
            stats.generated_tokens += call["output_tokens"]


def run_protocol(
    pairs: Sequence[PilotPair],
    *,
    protocol: str,
    client: LLMClient,
    judge_model: str,
    out_path: Path,
    provider: str = "local",
    top_logprobs: int | None = None,
    workers: int = 16,
    max_tokens: int = 256,
) -> RunStats:
    """Judge every pair with ``protocol`` and write one JSON line per pair (input order).

    Distinct requests are judged concurrently so vLLM can batch; repeats hit the disk cache.
    """
    if protocol not in ("pairwise", "pointwise"):
        raise ValueError(f"protocol must be pairwise or pointwise, got {protocol!r}")
    pairwise = (
        PairwiseJudge(
            client,
            provider=provider,
            model=judge_model,
            max_tokens=max_tokens,
            top_logprobs=top_logprobs,
        )
        if protocol == "pairwise"
        else None
    )
    pointwise = (
        PointwiseJudge(client, provider=provider, model=judge_model, max_tokens=max_tokens)
        if protocol == "pointwise"
        else None
    )
    stats = RunStats()
    lock = threading.Lock()
    start = time.perf_counter()
    # One representative per distinct request first (concurrently), then everything in order.
    seen: dict[tuple[str, str, str, str], PilotPair] = {}
    for p in pairs:
        seen.setdefault((p.original, p.original_output, p.rewrite_output, p.category), p)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        first = list(
            pool.map(lambda p: _judge_record(p, protocol, pairwise, pointwise), seen.values())
        )
    # Account from the distinct requests only (the in-order pass below is all cache hits).
    for rec in first:
        _account(rec, stats, lock)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as fh:
        for pair in pairs:
            rec = _judge_record(pair, protocol, pairwise, pointwise)
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    stats.wall_seconds = time.perf_counter() - start
    return stats


# --------------------------------------------------------------------------------------------
# analysis


def derive_sides(rec: Mapping[str, Any], derived: str) -> tuple[str | None, str | None]:
    """``(original_first side, rewrite_first side)`` of a run record under a derived protocol.

    ``derived`` is ``pairwise`` (greedy per order), ``pairwise_avg`` (order-averaged
    probabilities, same side for both orders) or ``pointwise`` (score comparison).
    """
    if derived == "pointwise":
        side = rec.get("pointwise_side")
        return side, side
    from terser.judge.pairwise import JudgeCall

    first, second = (JudgeCall(**c) for c in rec["calls"])
    if derived == "pairwise":
        return first.side, second.side
    if derived == "pairwise_avg":
        side, _ = order_averaged_side(first, second)
        return side, side
    raise ValueError(f"unknown derived protocol {derived!r}")


def _pilot_sides(rec: Mapping[str, Any]) -> tuple[str | None, str | None]:
    v = rec["verdicts"]
    return v["original_first"]["side"], v["rewrite_first"]["side"]


def _metrics(
    sides: Mapping[str, tuple[str | None, str | None]],
    categories: Mapping[str, str],
    identical: Mapping[str, bool],
    reference: Mapping[str, str | None],
) -> dict[str, Any]:
    parsed = {k: s for k, s in sides.items() if s[0] is not None and s[1] is not None}
    n = len(parsed)
    disagree = [k for k, (a, b) in parsed.items() if a != b]
    first_pos = sum(1 for k in disagree if parsed[k] == ("original", "rewrite"))
    labels = {k: label_pair(None, a, b).label for k, (a, b) in parsed.items()}  # type: ignore[arg-type]
    ties = sum(1 for a, b in parsed.values() if a == b == "tie")
    degraded = sum(1 for lab in labels.values() if lab == "degraded")
    ident = [k for k in parsed if identical.get(k)]
    ref_keys = [k for k in reference if k in labels]
    agree = sum(1 for k in ref_keys if labels[k] == reference[k])
    kappa = (
        cohen_kappa([reference[k] for k in ref_keys], [labels[k] for k in ref_keys])
        if ref_keys
        else float("nan")
    )
    row: dict[str, Any] = {
        "n_pairs": len(sides),
        "n_parsed": n,
        "position_disagreement": len(disagree) / n if n else float("nan"),
        "n_disagree": len(disagree),
        "first_position_share": first_pos / len(disagree) if disagree else float("nan"),
        "consistent_tie_share": ties / n if n else float("nan"),
        "degraded_share": degraded / n if n else float("nan"),
        "identical_pairs": len(ident),
        "identical_preserved": sum(1 for k in ident if labels[k] == "preserved"),
        "ref_n": len(ref_keys),
        "ref_agreement": agree / len(ref_keys) if ref_keys else float("nan"),
        "ref_kappa": kappa,
    }
    for cat in sorted(set(categories.values())):
        keys = [k for k in parsed if categories[k] == cat]
        d = sum(1 for k in keys if parsed[k][0] != parsed[k][1])
        row[f"disagree_{cat}"] = d / len(keys) if keys else float("nan")
    return row


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def protocol_rows(
    pilot: Iterable[Mapping[str, Any]],
    runs: Mapping[str, Sequence[Mapping[str, Any]]],
    run_stats: Mapping[str, Mapping[str, Any]] | None = None,
) -> pd.DataFrame:
    """One row per (run, derived protocol) plus the original pilot judge (``pilot_v1``).

    The reference set is the pilot pairs whose original judge verdicts were position-consistent;
    its label is the 1.3.6 label of those pairs.
    """
    pilot = list(pilot)
    categories = {r["id"]: r["category"] for r in pilot}
    pilot_sides = {r["id"]: _pilot_sides(r) for r in pilot}
    reference = {
        k: label_pair(None, a, b).label  # type: ignore[arg-type]
        for k, (a, b) in pilot_sides.items()
        if a is not None and a == b
    }
    identical: dict[str, bool] = {}
    for recs in runs.values():
        identical.update({r["id"]: bool(r["identical_outputs"]) for r in recs})
    rows = [
        {
            "run": "pilot_v1",
            "protocol": "pairwise",
            "judge_model": pilot[0]["judge_model"] if pilot else "",
            **_metrics(pilot_sides, categories, identical, reference),
        }
    ]
    for name, recs in runs.items():
        proto = recs[0]["protocol"] if recs else "pairwise"
        has_probs = proto == "pairwise" and any(
            c.get("verdict_probs") is not None for r in recs for c in r["calls"]
        )
        derived = ["pointwise"] if proto == "pointwise" else ["pairwise"]
        if has_probs:
            derived.append("pairwise_avg")
        for d in derived:
            sides = {r["id"]: derive_sides(r, d) for r in recs}
            rows.append(
                {
                    "run": name,
                    "protocol": d,
                    "judge_model": recs[0]["judge_model"] if recs else "",
                    **_metrics(sides, categories, identical, reference),
                    **dict((run_stats or {}).get(name, {})),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------
# CLI


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m terser.phase2.judge_bias", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="judge every pilot pair with one protocol")
    run.add_argument("--protocol", choices=("pairwise", "pointwise"), required=True)
    run.add_argument("--name", required=True, help="run name; output <out-dir>/<name>.jsonl")
    run.add_argument("--judge-model", required=True)
    run.add_argument("--top-logprobs", type=int, default=None)
    run.add_argument("--workers", type=int, default=16)
    run.add_argument("--max-tokens", type=int, default=256)
    run.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    run.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    run.add_argument("--budget-phase", default="phase2")
    run.add_argument("--hardware", default="rtx4500ada")
    run.add_argument("--no-track", action="store_true")
    rep = sub.add_parser("report", help="compare runs against the original pilot judge")
    rep.add_argument("--runs", nargs="+", required=True)
    rep.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    rep.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    rep.add_argument("--out", type=Path, default=Path("reports/p2_judge_bias.csv"))
    return ap


def _cmd_run(args: argparse.Namespace) -> int:
    client = LLMClient(args.budget_phase)
    pairs = load_pairs(args.labels, client)
    out_path = args.out_dir / f"{args.name}.jsonl"
    kwargs: dict[str, Any] = {
        "protocol": args.protocol,
        "client": client,
        "judge_model": args.judge_model,
        "out_path": out_path,
        "top_logprobs": args.top_logprobs,
        "workers": args.workers,
        "max_tokens": args.max_tokens,
    }
    if args.no_track:
        stats = run_protocol(pairs, **kwargs)
    else:
        import mlflow

        from terser.tracking import start_run

        model = args.judge_model.split("/")[-1].replace("-", "_")
        with start_run(
            phase=2,
            row="2.3.3",
            task="judgebias",
            run_name=f"2-judgebias-{model}-{args.protocol}-0",
            hardware=args.hardware,
            params={
                "protocol": args.protocol,
                "judge_model": args.judge_model,
                "top_logprobs": args.top_logprobs,
                "workers": args.workers,
                "n_pairs": len(pairs),
                "labels": args.labels.as_posix(),
            },
        ):
            stats = run_protocol(pairs, **kwargs)
            mlflow.log_metrics(
                {
                    "calls": stats.calls,
                    "cached_calls": stats.cached_calls,
                    "generated_tokens": stats.generated_tokens,
                    "wall_seconds": stats.wall_seconds,
                    "gpu_hours": stats.wall_seconds / 3600 if args.hardware != "cpu" else 0.0,
                }
            )
    stats_path = out_path.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats.__dict__, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out_path), "pairs": len(pairs), **stats.__dict__}, indent=2))
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    pilot = _read_jsonl(args.labels)
    runs = {name: _read_jsonl(args.out_dir / f"{name}.jsonl") for name in args.runs}
    stats = {}
    for name in args.runs:
        path = args.out_dir / f"{name}.stats.json"
        if path.is_file():
            stats[name] = json.loads(path.read_text(encoding="utf-8"))
    df = protocol_rows(pilot, runs, stats)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False, float_format="%.4f")
    cols = [
        "run",
        "protocol",
        "n_parsed",
        "position_disagreement",
        "first_position_share",
        "degraded_share",
        "ref_agreement",
        "ref_kappa",
        "identical_pairs",
        "identical_preserved",
    ]
    print(df[cols].to_markdown(index=False, floatfmt=".3f"))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    args = _parser().parse_args(argv)
    return _cmd_run(args) if args.cmd == "run" else _cmd_report(args)


if __name__ == "__main__":
    sys.exit(main())
