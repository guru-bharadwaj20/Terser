"""Target outputs (plan row **2.2.3**): the ``respond`` stage.

For every casual prompt and every accepted candidate rewrite (row 2.2.1), each target model
answers the text. On the GPU the targets are ``meta-llama/Llama-3.1-8B-Instruct`` and
``Qwen/Qwen2.5-7B-Instruct`` on vLLM bf16 at temperature 0; ``creative`` prompts get
``creative_samples`` (default 3) samples at ``creative_temperature`` with per-sample seeds,
because a single greedy answer under-represents an open-ended task.

All calls go through :class:`terser.llm.LLMClient` (provider ``local``), so responses are
cached and budgeted. On the CPU laptop only stand-in targets (Ollama tags such as
``qwen2.5:0.5b``) may be used: their records carry ``stand_in: true`` and go to
``smoke_output_path`` only, never to the real ``respond`` output.

Output: one JSON object per line with :data:`RESPONSE_FIELDS`. ``response_id`` is
``<text_id>|<target>|<sample>`` where ``text_id`` is ``<prompt_id>:original`` or the
candidate's ``candidate_id``, so runs are deterministic and resumable.

CLI::

    python -m terser.data.respond experiment=p2_respond
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from terser.data.candidates import load_prompts, read_candidates
from terser.llm import BudgetExceededError, LLMClient, LocalLLMUnavailableError

if TYPE_CHECKING:
    from omegaconf import DictConfig

__all__ = [
    "DEFAULT_TARGETS",
    "RESPONSE_FIELDS",
    "RespondServerUnavailableError",
    "RespondSettings",
    "RespondSummary",
    "iter_texts",
    "main",
    "response_id",
    "run_respond",
    "settings_from_config",
]

DEFAULT_TARGETS: tuple[str, ...] = (
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
)

RESPONSE_FIELDS: tuple[str, ...] = (
    "response_id",
    "prompt_id",
    "text_id",
    "kind",
    "category",
    "target",
    "sample",
    "temperature",
    "seed",
    "output",
    "output_tokens",
    "stand_in",
)

_OLLAMA_TAG_RE = re.compile(r"^[a-z0-9.\-]+:[A-Za-z0-9.\-]+$")


class RespondServerUnavailableError(RuntimeError):
    """The OpenAI-compatible server for a target model is not reachable."""


def response_id(text_id: str, target: str, sample: int) -> str:
    """Deterministic id ``<text_id>|<target>|<sample>``."""
    return f"{text_id}|{target}|{sample}"


@dataclass(frozen=True)
class RespondSettings:
    """Resolved settings of a respond run (``configs/experiment/p2_respond.yaml``)."""

    prompts_path: Path
    candidates_path: Path
    output_path: Path
    smoke_output_path: Path
    targets: tuple[str, ...] = DEFAULT_TARGETS
    provider: str = "local"
    stand_in: bool = False
    temperature: float = 0.0
    creative_category: str = "creative"
    creative_samples: int = 3
    creative_temperature: float = 0.7
    max_output_tokens: int = 1024
    base_seed: int = 0
    include_original: bool = True
    limit: int | None = None
    budget_phase: str = "phase2"

    def __post_init__(self) -> None:
        if not self.targets or len(set(self.targets)) != len(self.targets):
            raise ValueError(f"targets must be non-empty and unique, got {self.targets}")
        if self.creative_samples < 1:
            raise ValueError("creative_samples must be >= 1")
        if self.output_path.resolve() == self.smoke_output_path.resolve():
            raise ValueError("smoke_output_path must differ from output_path")
        if not self.stand_in:
            for t in self.targets:
                if _OLLAMA_TAG_RE.match(t):
                    raise ValueError(
                        f"stand_in=false but target {t!r} looks like an Ollama stand-in tag; "
                        "real target outputs come from vLLM bf16. Keep stand_in=true on CPU."
                    )

    @property
    def destination(self) -> Path:
        """Where records go: stand-in outputs only ever reach the smoke file."""
        return self.smoke_output_path if self.stand_in else self.output_path


@dataclass
class RespondSummary:
    """Counts of one run."""

    n_texts: int = 0
    written: Counter[str] = field(default_factory=Counter)
    skipped_existing: int = 0
    llm_calls: int = 0
    cached: int = 0

    def metrics(self) -> dict[str, float]:
        out: dict[str, float] = {
            "n_texts": float(self.n_texts),
            "skipped_existing": float(self.skipped_existing),
            "llm_calls": float(self.llm_calls),
            "cached": float(self.cached),
        }
        for target, n in self.written.items():
            out[f"written_{re.sub(r'[^A-Za-z0-9_.]', '_', target)}"] = float(n)
        return out


def iter_texts(
    prompts: Sequence[dict[str, Any]],
    candidates: Sequence[dict[str, Any]],
    *,
    include_original: bool = True,
) -> Iterator[dict[str, str]]:
    """Yield every text to answer: each original prompt, then its accepted candidates.

    Rejected candidates (``rejected: true``) and stand-in candidates are skipped, and
    candidates whose prompt is not in ``prompts`` are ignored.
    """
    by_prompt: dict[str, list[dict[str, Any]]] = {}
    for c in candidates:
        if c.get("rejected") or c.get("stand_in"):
            continue
        by_prompt.setdefault(str(c["prompt_id"]), []).append(c)
    for p in prompts:
        pid, cat = str(p["id"]), str(p.get("category", ""))
        if include_original:
            yield {
                "prompt_id": pid,
                "text_id": f"{pid}:original",
                "kind": "original",
                "category": cat,
                "text": str(p["prompt"]),
            }
        for c in sorted(by_prompt.get(pid, ()), key=lambda r: str(r["candidate_id"])):
            yield {
                "prompt_id": pid,
                "text_id": str(c["candidate_id"]),
                "kind": str(c.get("method", "candidate")),
                "category": cat,
                "text": str(c["text"]),
            }


def _samples(settings: RespondSettings, category: str) -> list[tuple[int, float, int | None]]:
    """``(sample, temperature, seed)`` per call: greedy once, or n seeded samples if creative."""
    if category == settings.creative_category and settings.creative_samples > 1:
        return [
            (i, settings.creative_temperature, settings.base_seed + i)
            for i in range(settings.creative_samples)
        ]
    return [(0, settings.temperature, None)]


def run_respond(settings: RespondSettings, *, client: LLMClient | None = None) -> RespondSummary:
    """Answer every missing (text, target, sample) and append it to the destination file.

    Raises:
        RespondServerUnavailableError: If the target server is unreachable.
        BudgetExceededError: If the phase token cap would be exceeded.
    """
    prompts = load_prompts(settings.prompts_path, limit=settings.limit)
    candidates = read_candidates(settings.candidates_path)
    texts = list(iter_texts(prompts, candidates, include_original=settings.include_original))
    summary = RespondSummary(n_texts=len(texts))
    dest = settings.destination
    done = {r["response_id"] for r in read_candidates(dest)}

    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("a", encoding="utf-8", newline="\n") as fh:
        for target in settings.targets:
            for t in texts:
                for sample, temp, seed in _samples(settings, t["category"]):
                    rid = response_id(t["text_id"], target, sample)
                    if rid in done:
                        summary.skipped_existing += 1
                        continue
                    if client is None:
                        client = LLMClient(settings.budget_phase)
                    params: dict[str, Any] = {"temperature": temp}
                    if seed is not None:
                        params["seed"] = seed
                    try:
                        resp = client.complete(
                            settings.provider,
                            target,
                            [{"role": "user", "content": t["text"]}],
                            settings.max_output_tokens,
                            **params,
                        )
                    except LocalLLMUnavailableError as exc:
                        raise RespondServerUnavailableError(
                            f"target {target!r} needs a running OpenAI-compatible server "
                            f"for provider {settings.provider!r}. {exc}"
                        ) from exc
                    summary.llm_calls += 1
                    summary.cached += int(resp.cached)
                    rec = {
                        "response_id": rid,
                        "prompt_id": t["prompt_id"],
                        "text_id": t["text_id"],
                        "kind": t["kind"],
                        "category": t["category"],
                        "target": target,
                        "sample": sample,
                        "temperature": temp,
                        "seed": seed,
                        "output": resp.text,
                        "output_tokens": resp.output_tokens,
                        "stand_in": settings.stand_in,
                    }
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                    done.add(rid)
                    summary.written[target] += 1
    return summary


def settings_from_config(cfg: DictConfig) -> RespondSettings:
    """Build :class:`RespondSettings` from a composed config (``experiment=p2_respond``)."""
    from omegaconf import OmegaConf

    params = OmegaConf.to_container(cfg.experiment.params, resolve=True)
    assert isinstance(params, dict)
    p: dict[str, Any] = {str(k): v for k, v in params.items()}
    return RespondSettings(
        prompts_path=Path(p["prompts_path"]),
        candidates_path=Path(p["candidates_path"]),
        output_path=Path(p["output_path"]),
        smoke_output_path=Path(p["smoke_output_path"]),
        targets=tuple(str(t) for t in p.get("targets") or DEFAULT_TARGETS),
        provider=str(p.get("provider", "local")),
        stand_in=bool(p.get("stand_in", False)),
        temperature=float(p.get("temperature", 0.0)),
        creative_category=str(p.get("creative_category", "creative")),
        creative_samples=int(p.get("creative_samples", 3)),
        creative_temperature=float(p.get("creative_temperature", 0.7)),
        max_output_tokens=int(p.get("max_output_tokens", 1024)),
        base_seed=int(p.get("base_seed", 0)),
        include_original=bool(p.get("include_original", True)),
        limit=p.get("limit"),
        budget_phase=str(p.get("budget_phase", "phase2")),
    )


def main() -> None:
    """CLI entry point (Hydra)."""
    from terser.config import hydra_main

    @hydra_main()
    def _main(cfg: DictConfig) -> None:
        if cfg.experiment.get("name") != "p2_respond":
            print("usage: python -m terser.data.respond experiment=p2_respond")
            sys.exit(2)
        try:
            summary = run_respond(settings_from_config(cfg))
        except RespondServerUnavailableError as exc:
            print(f"error: {exc}", file=sys.stderr)
            sys.exit(2)
        except (BudgetExceededError, FileNotFoundError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            sys.exit(1)
        print(json.dumps(summary.metrics(), indent=2))

    _main()


if __name__ == "__main__":
    main()
