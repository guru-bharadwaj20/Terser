"""``casualise`` stage: clean instructions rewritten as casual prompts (plan row **2.1.4**).

Each clean instruction from the ``collect`` stage (:mod:`terser.data.collect`) is rewritten by a
local open model in five styles (:data:`STYLES`): ``rambling``, ``hedging``, ``texting``,
``over_polite`` and ``repetitive``, with the instruction
``src/terser/data/prompts/casualise_v1.txt``. Only the *instruction* is casualised; its input
(passage, code, data) is appended unchanged after a blank line, so payload is preserved by
construction. The clean prompt is kept on every record as the reference.

Validation (:func:`check_casual` then an LLM check):

* **Automatic checks**: non-empty; no assistant-style preamble ("Sure", "Here is"); every number,
  quoted span and inline-code span of the instruction still present; not longer than
  ``max_ratio`` x the instruction plus 80 words (the model must not answer the task); and, for
  every style except ``texting``, longer than the instruction (casualisation adds verbosity).
* **Meaning check**: the same model answers ``casualise_check_v1`` (same task? requirements
  added? requirements missing?). A casual prompt is accepted only if it is the same task with
  nothing added and nothing missing.

Rejected records are kept in a separate file with ``rejection_reason`` so the rejection rate is
auditable. Every record carries ``generator_model`` and ``prompt_version``.

All calls go through :class:`terser.llm.LLMClient`. The generation call samples at
``temperature`` 0.8 with a per-request ``seed`` (:func:`sample_seed`) so the five styles do not
collapse onto stock phrases; the meaning check runs at temperature 0. Requests are fixed for a
given input, so the stage is deterministic and resumable: a rerun is served from the disk cache,
and ``dvc repro`` rebuilds the stage without a GPU once the cache is populated. Concurrent
requests (``--workers``) let vLLM batch. Models that look like Ollama tags are CPU stand-ins:
their output is tagged ``stand_in: true`` and may only be written under a ``smoke/`` directory.

CLI::

    python -m terser.data.casualise --in data/interim/clean.jsonl \\
        --out data/interim/casual.jsonl --rejected data/interim/casual_rejected.jsonl \\
        --model Qwen/Qwen2.5-32B-Instruct-AWQ --workers 32
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from terser.judge.pairwise import json_candidates

if TYPE_CHECKING:
    from terser.llm import LLMClient

__all__ = [
    "OPENINGS",
    "STYLES",
    "CasualiseStats",
    "casualise_record",
    "check_casual",
    "clean_casual",
    "is_stand_in",
    "main",
    "parse_meaning_check",
    "render_casualise_prompt",
    "render_check_prompt",
    "run_casualise",
    "sample_seed",
]

STYLES: tuple[str, ...] = ("rambling", "hedging", "texting", "over_polite", "repetitive")
"""The five casual styles of row 2.1.4."""

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
DEFAULT_PROMPT_VERSION = "casualise_v1"
DEFAULT_CHECK_VERSION = "casualise_check_v1"
DEFAULT_MAX_RATIO = 4.0
DEFAULT_TEMPERATURE = 0.8
"""Sampling temperature of the generation call; the per-request seed keeps it reproducible."""

OPENINGS: tuple[str, ...] = (
    "start with the subject of the request itself",
    "start with a short remark about the person's situation",
    "start with the request as a question",
    "start with a word of hesitation or apology",
    "start with a greeting",
    "start mid-thought, as if continuing a conversation",
    "start with why the person needs this",
    "start with a direct ask",
)
"""Opening moves, one picked per (source, style) by :func:`sample_seed` for variety."""

# An Ollama tag such as ``qwen2.5:1.5b`` (name:tag, no org/ prefix). vLLM serves HF ids instead.
_OLLAMA_TAG_RE = re.compile(r"^[A-Za-z0-9._-]+:[A-Za-z0-9._-]+$")
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
_QUOTED_RE = re.compile(r'"([^"\n]{1,200})"|`([^`\n]{1,200})`')
_PREAMBLE_RE = re.compile(
    r"^\s*(sure|certainly|of course|here(?:'s| is| are)|casual message|rewritten)\b[^\n]*[:!]",
    re.IGNORECASE,
)
_LABEL_RE = re.compile(r"^\s*\[?(casual message|message)\]?\s*:?\s*\n", re.IGNORECASE)


def sample_seed(source_id: str, style: str) -> int:
    """Per-(source, style) sampling seed: deterministic, different for every request."""
    digest = hashlib.sha256(f"terser-casualise:{source_id}:{style}".encode()).hexdigest()
    return int(digest[:8], 16)


def is_stand_in(model: str) -> bool:
    """True for Ollama-style tags (``qwen2.5:1.5b``), i.e. CPU stand-in models."""
    return bool(_OLLAMA_TAG_RE.match(model))


@cache
def _sections(version: str) -> dict[str, str]:
    """``=== name ===`` sections of ``prompts/<version>.txt``."""
    path = PROMPTS_DIR / f"{version}.txt"
    text = path.read_text(encoding="utf-8")
    parts = re.split(r"^=== (\w+) ===\n", text, flags=re.MULTILINE)
    return {parts[i]: parts[i + 1].strip("\n") for i in range(1, len(parts) - 1, 2)}


def render_casualise_prompt(
    instruction: str,
    style: str,
    version: str = DEFAULT_PROMPT_VERSION,
    opening: str = OPENINGS[0],
) -> str:
    """The generation prompt for one (instruction, style, opening move)."""
    if style not in STYLES:
        raise ValueError(f"unknown style {style!r}; expected one of {STYLES}")
    sections = _sections(version)
    values = {
        "style": style.replace("_", "-"),
        "style_guidance": sections[style],
        "opening": opening,
        "instruction": instruction,
    }
    return re.sub(
        r"\{(style|style_guidance|opening|instruction)\}",
        lambda m: values[m.group(1)],
        sections["template"],
    )


def render_check_prompt(instruction: str, casual: str, version: str = DEFAULT_CHECK_VERSION) -> str:
    """The meaning-check prompt for one (instruction, casual instruction)."""
    template = (PROMPTS_DIR / f"{version}.txt").read_text(encoding="utf-8")
    values = {"instruction": instruction, "casual": casual}
    return re.sub(r"\{(instruction|casual)\}", lambda m: values[m.group(1)], template)


def clean_casual(text: str) -> str:
    """Strip a label line and quotes wrapping the whole message."""
    out = _LABEL_RE.sub("", text.strip()).strip()
    # Only quotes that wrap the whole message: '"foo" is what i mean, not "bar"' keeps both.
    if len(out) >= 2 and out[0] == out[-1] and out[0] in "\"'" and out[0] not in out[1:-1]:
        out = out[1:-1].strip()
    return out


def check_casual(
    instruction: str, casual: str, style: str, max_ratio: float = DEFAULT_MAX_RATIO
) -> str | None:
    """Automatic validation; return a rejection reason or None if the casual text passes."""
    if not casual.strip():
        return "empty"
    if _PREAMBLE_RE.match(casual):
        return "preamble"
    numbers = set(_NUMBER_RE.findall(instruction))
    casual_numbers = set(_NUMBER_RE.findall(casual))
    if numbers - casual_numbers:
        return "number_changed"
    for m in _QUOTED_RE.finditer(instruction):
        span = m.group(1) or m.group(2)
        if span not in casual:
            return "quoted_span_changed"
    n_in, n_out = len(instruction.split()), len(casual.split())
    if n_out > max_ratio * n_in + 80:
        return "too_long"
    if style != "texting" and n_out <= n_in:
        return "not_casual"
    return None


def parse_meaning_check(text: str) -> tuple[bool, list[str], list[str]] | None:
    """Parse ``{"same_task", "added", "missing"}``; None if unparseable."""
    for candidate in json_candidates(text or ""):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or "same_task" not in obj:
            continue
        added, missing = obj.get("added") or [], obj.get("missing") or []
        if not isinstance(added, list) or not isinstance(missing, list):
            continue
        same = obj["same_task"]
        if not isinstance(same, bool):
            same = str(same).strip().lower() in ("true", "yes")
        return (
            same,
            [str(a) for a in added if str(a).strip()],
            [str(m) for m in missing if str(m).strip()],
        )
    return None


@dataclass
class CasualiseStats:
    """Counts and accounting of one run."""

    accepted: int = 0
    rejected: int = 0
    calls: int = 0
    cached_calls: int = 0
    generated_tokens: int = 0
    wall_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form."""
        return dict(self.__dict__)


def casualise_record(
    client: LLMClient,
    clean: Mapping[str, Any],
    style: str,
    *,
    model: str,
    provider: str = "local",
    prompt_version: str = DEFAULT_PROMPT_VERSION,
    check_version: str = DEFAULT_CHECK_VERSION,
    max_tokens: int = 512,
    max_ratio: float = DEFAULT_MAX_RATIO,
    temperature: float = DEFAULT_TEMPERATURE,
) -> tuple[dict[str, Any], list[tuple[bool, int]]]:
    """Casualise one clean record in one style; return ``(record, [(cached, out_tokens)])``."""
    instruction = str(clean["instruction"])
    seed = sample_seed(str(clean["source_id"]), style)
    opening = OPENINGS[seed % len(OPENINGS)]
    usage: list[tuple[bool, int]] = []
    resp = client.complete(
        provider,
        model,
        [
            {
                "role": "user",
                "content": render_casualise_prompt(instruction, style, prompt_version, opening),
            }
        ],
        max_tokens,
        temperature=temperature,
        seed=seed,
    )
    usage.append((resp.cached, resp.output_tokens))
    casual_instruction = clean_casual(resp.text)
    reason = check_casual(instruction, casual_instruction, style, max_ratio)
    check: dict[str, Any] | None = None
    if reason is None:
        cresp = client.complete(
            provider,
            model,
            [
                {
                    "role": "user",
                    "content": render_check_prompt(instruction, casual_instruction, check_version),
                }
            ],
            256,
            temperature=0.0,
        )
        usage.append((cresp.cached, cresp.output_tokens))
        parsed = parse_meaning_check(cresp.text)
        if parsed is None:
            reason = "check_unparseable"
        else:
            same, added, missing = parsed
            check = {"same_task": same, "added": added, "missing": missing}
            if not same:
                reason = "different_task"
            elif added:
                reason = "added_requirements"
            elif missing:
                reason = "missing_requirements"
    input_text = clean.get("input")
    prompt = f"{casual_instruction}\n\n{input_text}" if input_text else casual_instruction
    record = {
        "id": f"{clean['source_id']}:{style}",
        "source_id": clean["source_id"],
        "source": clean["source"],
        "category": clean["category"],
        "style": style,
        "prompt": prompt,
        "casual_instruction": casual_instruction,
        "clean": clean["clean"],
        "instruction": instruction,
        "input": input_text,
        "reference": clean.get("reference"),
        "generator_model": model,
        "generator_temperature": temperature,
        "generator_seed": seed,
        "opening": opening,
        "prompt_version": prompt_version,
        "check_version": check_version,
        "meaning_check": check,
        "rejected": reason is not None,
        "rejection_reason": reason,
        "stand_in": is_stand_in(model),
    }
    return record, usage


def run_casualise(
    clean_records: Sequence[Mapping[str, Any]],
    client: LLMClient,
    *,
    model: str,
    out_path: Path,
    rejected_path: Path,
    styles: Sequence[str] = STYLES,
    provider: str = "local",
    workers: int = 32,
    max_ratio: float = DEFAULT_MAX_RATIO,
    temperature: float = DEFAULT_TEMPERATURE,
) -> CasualiseStats:
    """Casualise every (record, style); write accepted and rejected records in input order."""
    if is_stand_in(model) and not ("smoke" in out_path.parts and "smoke" in rejected_path.parts):
        raise ValueError(
            f"{model!r} looks like a CPU stand-in; its accepted and rejected output may only go "
            "under a smoke/ dir"
        )
    jobs = [(rec, style) for rec in clean_records for style in styles]
    stats = CasualiseStats()
    start = time.perf_counter()

    def one(job: tuple[Mapping[str, Any], str]) -> tuple[dict[str, Any], list[tuple[bool, int]]]:
        return casualise_record(
            client,
            job[0],
            job[1],
            model=model,
            provider=provider,
            max_ratio=max_ratio,
            temperature=temperature,
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(one, jobs))
    for path in (out_path, rejected_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    with (
        out_path.open("w", encoding="utf-8", newline="\n") as ok_fh,
        rejected_path.open("w", encoding="utf-8", newline="\n") as rej_fh,
    ):
        for rec, usage in results:
            for cached, tokens in usage:
                stats.calls += 1
                stats.cached_calls += int(cached)
                stats.generated_tokens += 0 if cached else tokens
            fh = rej_fh if rec["rejected"] else ok_fh
            fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
            if rec["rejected"]:
                stats.rejected += 1
            else:
                stats.accepted += 1
    stats.wall_seconds = time.perf_counter() - start
    return stats


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def summarise(path: Path, rejected_path: Path) -> dict[str, Any]:
    """Per-category / per-style counts of accepted records and rejection reasons."""
    ok, rej = _read_jsonl(path), _read_jsonl(rejected_path)
    return {
        "accepted": len(ok),
        "rejected": len(rej),
        "by_category": dict(sorted(Counter(r["category"] for r in ok).items())),
        "by_style": dict(sorted(Counter(r["style"] for r in ok).items())),
        "rejection_reasons": dict(Counter(r["rejection_reason"] for r in rej).most_common()),
        "rejected_by_style": dict(sorted(Counter(r["style"] for r in rej).items())),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(prog="python -m terser.data.casualise", description=__doc__)
    ap.add_argument("--in", dest="inp", type=Path, default=Path("data/interim/clean.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/interim/casual.jsonl"))
    ap.add_argument("--rejected", type=Path, default=Path("data/interim/casual_rejected.jsonl"))
    ap.add_argument("--model", default="Qwen/Qwen2.5-32B-Instruct-AWQ")
    ap.add_argument("--provider", default="local")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None, help="first N clean records (smoke)")
    ap.add_argument("--max-ratio", type=float, default=DEFAULT_MAX_RATIO)
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    ap.add_argument("--budget-phase", default="phase2")
    ap.add_argument("--stats", type=Path, default=None)
    ap.add_argument("--hardware", default="rtx4500ada")
    ap.add_argument("--track", action="store_true", help="log an MLflow run")
    args = ap.parse_args(argv)

    from terser.llm import LLMClient

    records = _read_jsonl(args.inp)
    if args.limit is not None:
        records = records[: args.limit]
    client = LLMClient(args.budget_phase)
    kwargs: dict[str, Any] = {
        "model": args.model,
        "out_path": args.out,
        "rejected_path": args.rejected,
        "provider": args.provider,
        "workers": args.workers,
        "max_ratio": args.max_ratio,
        "temperature": args.temperature,
    }
    if args.track:
        import mlflow

        from terser.tracking import start_run

        name = args.model.split("/")[-1].replace("-", "_")
        with start_run(
            phase=2,
            row="2.1.4",
            task="casualise",
            run_name=f"2-casualise-{name}-full-0",
            hardware=args.hardware,
            params={"model": args.model, "n_clean": len(records), "styles": ",".join(STYLES)},
        ):
            stats = run_casualise(records, client, **kwargs)
            mlflow.log_metrics(
                {
                    **{k: float(v) for k, v in stats.to_dict().items()},
                    "gpu_hours": stats.wall_seconds / 3600 if args.hardware != "cpu" else 0.0,
                }
            )
    else:
        stats = run_casualise(records, client, **kwargs)
    stable = summarise(args.out, args.rejected)
    summary = {**stats.to_dict(), **stable}
    if args.stats is not None:
        args.stats.parent.mkdir(parents=True, exist_ok=True)
        # Accounting (wall time, cache hits, tokens) varies between runs and is logged to
        # MLflow; the stats file holds only dataset counts so `dvc repro` reproduces it.
        args.stats.write_text(json.dumps(stable, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
