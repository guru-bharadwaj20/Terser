"""Candidate generation (plan row **2.2.1**).

Six compressed candidates per casual prompt:

* ``llm_low`` / ``llm_medium`` / ``llm_high`` -- rewrites by a local open model through
  :class:`terser.llm.LLMClient` (provider ``local``) with the instruction
  ``src/terser/data/prompts/rewrite_v1.txt``. On the GPU the model is
  ``Qwen/Qwen2.5-32B-Instruct-AWQ`` on vLLM; on the CPU laptop only a stand-in
  (``qwen2.5:1.5b`` via Ollama) may be used, its records carry ``stand_in: true`` and are
  written to a separate smoke file, never to the real candidates file.
* ``ops_<codes>`` -- deterministic operator stacks via :func:`terser.operators.apply_ops`
  (default ``FHC`` and ``CHFIDSL``).
* ``llmlingua2`` -- the LLMLingua-2 token-classification baseline (``llmlingua`` package,
  optional dependency).

LLM rewrites are validated (:func:`validate_rewrite`): wrappers and quotes are stripped, and a
rewrite is *rejected* (kept in the file with ``rejected: true`` and a ``rejection_reason``) if it
is empty, costs more tokens than the original on any configured tokenizer, or alters a payload
span (fenced/inline code or quoted text, as found by :func:`terser.operators.masking.mask`).

Output: one JSON object per line with :data:`CANDIDATE_FIELDS`. ``candidate_id`` is
``<prompt_id>:<method>``, so runs are deterministic and resumable: candidates already in the
output file are skipped. LLM responses are cached by :class:`terser.llm.LLMClient`; LLMLingua-2
outputs are cached on disk under ``.cache/llmlingua2``.

CLI::

    python -m terser.data.candidates experiment=p2_candidates
    # non-LLM candidates only:
    python -m terser.data.candidates experiment=p2_candidates \\
        'experiment.params.methods=[ops_FHC,ops_CHFIDSL,llmlingua2]'
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from terser.llm import BudgetExceededError, LLMClient, LocalLLMUnavailableError, ResponseCache

if TYPE_CHECKING:
    from omegaconf import DictConfig

__all__ = [
    "CANDIDATE_FIELDS",
    "DEFAULT_METHODS",
    "AUG_LLM_LEVELS",
    "LLM_LEVELS",
    "CandidateServerUnavailableError",
    "CandidateSettings",
    "CandidateSummary",
    "LLMLingua2Compressor",
    "ResumeMismatchError",
    "candidate_id",
    "clean_rewrite",
    "load_prompts",
    "main",
    "method_kind",
    "output_path_for",
    "payload_altered",
    "read_candidates",
    "render_rewrite_prompt",
    "run_candidates",
    "run_from_config",
    "settings_from_config",
    "validate_rewrite",
]

CANDIDATE_FIELDS: tuple[str, ...] = (
    "prompt_id",
    "candidate_id",
    "method",
    "text",
    "generator",
    "stand_in",
    "rejected",
    "rejection_reason",
)
"""Fields of every candidate record, in output order."""

LLM_LEVELS: tuple[str, ...] = ("low", "medium", "high")
"""Aggressiveness levels of the LLM rewrites (methods ``llm_<level>``)."""

AUG_LLM_LEVELS: tuple[str, ...] = ("max", "min")
"""Extra levels for the row 8.3.4 target augmentation; never in the default method set."""

DEFAULT_METHODS: tuple[str, ...] = (
    "ops_FHC",
    "ops_CHFIDSL",
    "llmlingua2",
    "llm_low",
    "llm_medium",
    "llm_high",
)
"""The six methods of row 2.2.1. Cheap deterministic methods run first."""

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"
DEFAULT_LLMLINGUA_MODEL = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"

CountFn = Callable[[str, str], int]
ApplyOps = Callable[[str, str, str | None], str]
Compressor = Callable[[str], str]

# An Ollama tag such as ``qwen2.5:1.5b`` (name:tag, no org/ prefix). vLLM serves HF ids instead.
_OLLAMA_TAG_RE = re.compile(r"^[A-Za-z0-9._-]+:[A-Za-z0-9._-]+$")


class CandidateServerUnavailableError(RuntimeError):
    """The local OpenAI-compatible server needed for the LLM rewrites could not be reached."""


class ResumeMismatchError(RuntimeError):
    """An existing candidate was produced by a different generator than the current settings."""


# --------------------------------------------------------------------------------------------
# Methods, ids and prompts
# --------------------------------------------------------------------------------------------


def method_kind(method: str) -> str:
    """Return ``"llm"``, ``"ops"`` or ``"llmlingua2"`` for a method name.

    Raises:
        ValueError: For an unknown method or an unknown LLM level.
    """
    if method == "llmlingua2":
        return "llmlingua2"
    if method.startswith("ops_") and len(method) > 4:
        return "ops"
    if method.startswith("llm_") and method[4:] in (*LLM_LEVELS, *AUG_LLM_LEVELS):
        return "llm"
    raise ValueError(
        f"unknown candidate method {method!r}; expected llm_{{{','.join(LLM_LEVELS)}}}, "
        "ops_<operator codes> or llmlingua2"
    )


def candidate_id(prompt_id: str, method: str) -> str:
    """Deterministic candidate id ``<prompt_id>:<method>``."""
    return f"{prompt_id}:{method}"


def load_prompts(
    path: Path, *, limit: int | None = None, exclude_splits: Sequence[str] = ()
) -> list[dict[str, Any]]:
    """Read prompts from JSONL (fields ``id`` and ``prompt`` required); keep the first ``limit``.

    Records whose ``split`` is in ``exclude_splits`` are skipped before ``limit`` applies.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If a record lacks ``id``/``prompt`` or an id repeats.
    """
    if not path.is_file():
        raise FileNotFoundError(f"prompts not found: {path} (expected JSONL with id, prompt)")
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("id") or not isinstance(row.get("prompt"), str):
                raise ValueError(f"{path}:{lineno}: record needs 'id' and 'prompt'")
            if row["id"] in seen:
                raise ValueError(f"{path}:{lineno}: duplicate id {row['id']!r}")
            seen.add(row["id"])
            if row.get("split") in exclude_splits:
                continue
            out.append(row)
            if limit is not None and len(out) >= limit:
                break
    return out


@cache
def _rewrite_sections(version: str) -> dict[str, str]:
    path = PROMPTS_DIR / f"{version}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"rewrite prompt not found: {path}")
    text = path.read_text(encoding="utf-8")
    parts = re.split(r"^=== (\w+) ===\n", text, flags=re.M)
    sections = {name: body.strip("\n") for name, body in zip(parts[1::2], parts[2::2], strict=True)}
    missing = [s for s in ("template", *LLM_LEVELS, *AUG_LLM_LEVELS) if s not in sections]
    if missing:
        raise ValueError(f"{path}: missing section(s) {missing}")
    return sections


def render_rewrite_prompt(prompt: str, level: str, version: str = "rewrite_v1") -> str:
    """Fill the rewrite instruction ``prompts/<version>.txt`` for ``level`` and ``prompt``."""
    if level not in (*LLM_LEVELS, *AUG_LLM_LEVELS):
        raise ValueError(f"level must be one of {(*LLM_LEVELS, *AUG_LLM_LEVELS)}, got {level!r}")
    sections = _rewrite_sections(version)
    # Substitute the prompt last so braces inside user text are never interpreted.
    body = sections["template"].replace("{level}", level)
    body = body.replace("{level_guidance}", sections[level])
    return body.replace("{prompt}", prompt)


# --------------------------------------------------------------------------------------------
# Validation of LLM rewrites
# --------------------------------------------------------------------------------------------

_PREAMBLE_RE = re.compile(
    r"^\s*(?:(?:sure|okay|ok|certainly|of course)[,.!]?\s*)?"
    r"(?:here(?:'s| is)\s+(?:the|a|your)\s+(?:\w+\s+){0,3}(?:prompt|version|rewrite)"
    r"|(?:the\s+)?(?:rewritten|shortened|compressed|concise|revised)\s+(?:prompt|version)"
    r"|rewrite|prompt)\s*(?:\([^)\n]*\))?\s*:\s*\n?",
    re.I,
)
_TAG_RE = re.compile(
    r"^\s*\[(?:rewritten )?prompt\]\s*\n(.*?)\n\s*\[end of (?:rewritten )?prompt\]\s*$",
    re.I | re.S,
)
_FENCE_RE = re.compile(r"^```[\w-]*\n(.*)\n```$", re.S)
_QUOTE_PAIRS = (('"', '"'), ("“", "”"), ("'", "'"), ("‘", "’"))


def clean_rewrite(text: str, original: str) -> str:
    """Strip wrappers a chat model puts around its rewrite.

    Removes a leading preamble ("Here is the rewritten prompt:", "Rewritten prompt:"), echoed
    ``[Prompt]`` section markers, a code fence around the whole output and quotes around the
    whole output -- but a fence or quote pair is only stripped when the original prompt was not
    itself wrapped that way.
    """
    out = text.strip()
    for _ in range(3):  # wrappers can nest, e.g. a preamble followed by a quoted prompt
        before = out
        m = _TAG_RE.match(out)
        if m:
            out = m.group(1).strip()
        out = _PREAMBLE_RE.sub("", out, count=1).strip()
        orig = original.strip()
        m = _FENCE_RE.match(out)
        if m and not _FENCE_RE.match(orig):
            out = m.group(1).strip()
        for left, right in _QUOTE_PAIRS:
            if (
                len(out) >= 2
                and out.startswith(left)
                and out.endswith(right)
                and not (orig.startswith(left) and orig.endswith(right))
                and left not in out[1:-1]
                and right not in out[1:-1]
            ):
                out = out[1:-1].strip()
        if out == before:
            break
    return out


def payload_altered(original: str, rewrite: str) -> bool:
    """True if some payload span of ``original`` is missing verbatim from ``rewrite``'s spans.

    Payload spans are those protected by :func:`terser.operators.masking.mask` (fenced code,
    inline code, double-quoted text). Spans are compared as multisets, so reordering is allowed
    but dropping, editing or re-quoting a span is not.
    """
    from terser.operators.masking import mask

    _, before = mask(original)
    if not before:
        return False
    _, after = mask(rewrite)
    return bool(Counter(before) - Counter(after))


def validate_rewrite(
    raw: str,
    original: str,
    *,
    targets: Sequence[str],
    count: CountFn,
) -> tuple[str, str | None]:
    """Clean an LLM rewrite and decide whether to reject it.

    Returns:
        ``(cleaned_text, rejection_reason)``; the reason is ``None`` for an accepted rewrite,
        else ``"empty"``, ``"payload_altered"`` or ``"longer_than_original:<target>"`` (the first
        target on which the rewrite costs more tokens).
    """
    text = clean_rewrite(raw, original)
    if not text:
        return text, "empty"
    if payload_altered(original, text):
        return text, "payload_altered"
    for target in targets:
        if count(text, target) > count(original, target):
            return text, f"longer_than_original:{target}"
    return text, None


# --------------------------------------------------------------------------------------------
# LLMLingua-2 adapter (optional dependency)
# --------------------------------------------------------------------------------------------


class LLMLingua2Compressor:
    """LLMLingua-2 baseline behind a lazy, optional import of the ``llmlingua`` package.

    Outputs are cached on disk keyed by model, rate, force tokens and text, so re-runs do not
    reload the model.
    """

    def __init__(
        self,
        model: str = DEFAULT_LLMLINGUA_MODEL,
        *,
        rate: float = 0.5,
        device: str = "cpu",
        force_tokens: Sequence[str] = ("\n", "?", "."),
        cache_dir: Path | None = None,
    ) -> None:
        self.model = model
        self.rate = rate
        self.device = device
        self.force_tokens = list(force_tokens)
        self.cache = ResponseCache(cache_dir) if cache_dir is not None else None
        self._compressor: Any = None
        self._revision: str | None = None

    def _load(self) -> Any:
        if self._compressor is None:
            try:
                from llmlingua import PromptCompressor
            except ImportError as exc:  # optional dependency
                raise RuntimeError(
                    "the llmlingua2 method needs the 'llmlingua' package (pip install llmlingua)"
                ) from exc
            self._compressor = PromptCompressor(
                model_name=self.model, use_llmlingua2=True, device_map=self.device
            )
            config = getattr(getattr(self._compressor, "model", None), "config", None)
            self._revision = getattr(config, "_commit_hash", None)
        return self._compressor

    @property
    def generator(self) -> str:
        """``<model>@<revision>`` (revision known once loaded) plus the compression rate."""
        rev = f"@{self._revision}" if self._revision else ""
        return f"{self.model}{rev}?rate={self.rate:g}"

    def _key(self, text: str) -> str:
        payload = json.dumps(
            [self.model, self.rate, self.force_tokens, text], ensure_ascii=False
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def __call__(self, text: str) -> str:
        """Compress ``text`` (empty text stays empty)."""
        if not text.strip():
            return text
        key = self._key(text)
        if self.cache is not None:
            hit = self.cache.get(key)
            if hit is not None:
                self._revision = self._revision or hit.get("revision")
                return str(hit["text"])
        result = self._load().compress_prompt(
            text, rate=self.rate, force_tokens=self.force_tokens, drop_consecutive=True
        )
        out = str(result["compressed_prompt"])
        if self.cache is not None:
            self.cache.put(key, {"text": out, "model": self.model, "revision": self._revision})
        return out


# --------------------------------------------------------------------------------------------
# Settings, running and resuming
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateSettings:
    """Resolved settings of a candidate run (``configs/experiment/p2_candidates.yaml``)."""

    prompts_path: Path
    output_path: Path
    smoke_output_path: Path
    methods: tuple[str, ...] = DEFAULT_METHODS
    provider: str = "local"
    rewrite_model: str = "qwen2.5:1.5b"
    stand_in: bool = True
    prompt_version: str = "rewrite_v1"
    temperature: float = 0.0
    max_rewrite_tokens: int = 1024
    length_targets: tuple[str, ...] = ("gpt-4o", "qwen2.5", "llama-3.1")
    operator_target: str | None = None
    llmlingua_model: str = DEFAULT_LLMLINGUA_MODEL
    llmlingua_rate: float = 0.5
    llmlingua_device: str = "cpu"
    llmlingua_cache_dir: Path | None = None
    limit: int | None = None
    exclude_splits: tuple[str, ...] = ()
    budget_phase: str = "phase2"
    workers: int = 1

    def __post_init__(self) -> None:
        if self.workers < 1:
            raise ValueError(f"workers must be >= 1, got {self.workers}")
        for method in self.methods:
            method_kind(method)
        if len(set(self.methods)) != len(self.methods):
            raise ValueError(f"duplicate methods in {self.methods}")
        if not self.stand_in and _OLLAMA_TAG_RE.match(self.rewrite_model):
            raise ValueError(
                f"stand_in=false but rewrite_model {self.rewrite_model!r} looks like an Ollama "
                "stand-in tag; real candidates come from Qwen/Qwen2.5-32B-Instruct-AWQ on vLLM. "
                "Keep stand_in=true for CPU smoke runs."
            )

    @property
    def llm_methods(self) -> tuple[str, ...]:
        """The configured ``llm_*`` methods."""
        return tuple(m for m in self.methods if method_kind(m) == "llm")


def output_path_for(settings: CandidateSettings, method: str) -> Path:
    """File a method's records go to: stand-in LLM rewrites go to the smoke file only.

    Raises:
        ValueError: If the smoke file and the real output file are the same path.
    """
    if settings.output_path.resolve() == settings.smoke_output_path.resolve():
        raise ValueError("smoke_output_path must differ from output_path")
    if method_kind(method) == "llm" and settings.stand_in:
        return settings.smoke_output_path
    return settings.output_path


def read_candidates(path: Path) -> list[dict[str, Any]]:
    """Read a candidates JSONL file (missing file -> empty list; blank lines ignored)."""
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


@dataclass
class CandidateSummary:
    """Counts of one run."""

    n_prompts: int = 0
    written: Counter[str] = field(default_factory=Counter)
    skipped_existing: Counter[str] = field(default_factory=Counter)
    rejected: Counter[str] = field(default_factory=Counter)
    rejection_reasons: Counter[str] = field(default_factory=Counter)
    identical_to_original: Counter[str] = field(default_factory=Counter)
    seconds: dict[str, float] = field(default_factory=dict)
    llm_calls: int = 0
    cached_calls: int = 0
    wall_seconds: float = 0.0

    def metrics(self) -> dict[str, float]:
        """Flat numeric metrics for MLflow / the CLI."""
        out: dict[str, float] = {
            "n_prompts": self.n_prompts,
            "llm_calls": self.llm_calls,
            "cached_calls": self.cached_calls,
            "wall_seconds": round(self.wall_seconds, 3),
        }
        for name, counter in (
            ("written", self.written),
            ("skipped", self.skipped_existing),
            ("rejected", self.rejected),
            ("identical", self.identical_to_original),
        ):
            for method, n in sorted(counter.items()):
                out[f"{name}_{method}"] = n
        for method, s in sorted(self.seconds.items()):
            out[f"seconds_{method}"] = round(s, 3)
        return out


def _default_apply_ops(text: str, codes: str, target: str | None) -> str:
    from terser.operators import apply_ops

    return apply_ops(text, codes, target=target)


def _default_count(text: str, target: str) -> int:
    from terser.tokenizers import count_tokens

    return count_tokens(text, target)


def _record(
    prompt_id: str,
    method: str,
    text: str,
    generator: str,
    stand_in: bool,
    reason: str | None = None,
) -> dict[str, Any]:
    return {
        "prompt_id": prompt_id,
        "candidate_id": candidate_id(prompt_id, method),
        "method": method,
        "text": text,
        "generator": generator,
        "stand_in": stand_in,
        "rejected": reason is not None,
        "rejection_reason": reason,
    }


def _generator_for(settings: CandidateSettings, method: str) -> str | None:
    """Generator string known before running (None for LLMLingua-2, whose revision loads late)."""
    kind = method_kind(method)
    if kind == "ops":
        return f"operators:{method[4:]}" + (
            f"@{settings.operator_target}" if settings.operator_target else ""
        )
    if kind == "llm":
        return f"{settings.rewrite_model}|{settings.prompt_version}"
    return None


_CUSTOM_COMPRESSOR = "llmlingua2:custom"
"""Generator recorded for an injected compressor that has no ``generator`` attribute."""


def _generator_matches(
    settings: CandidateSettings,
    method: str,
    old: Any,
    expected: str | None,
    *,
    custom: bool = False,
) -> bool:
    """Whether an existing record's generator is compatible with the current settings."""
    if expected is not None:
        return bool(old == expected)
    # LLMLingua-2: ``<model>[@<revision>]?rate=<rate>``; the revision is unknown before loading.
    old = str(old)
    if old == _CUSTOM_COMPRESSOR:
        return custom
    model, _, rate = old.partition("?rate=")
    return model.split("@", 1)[0] == settings.llmlingua_model and rate == (
        f"{settings.llmlingua_rate:g}"
    )


def run_candidates(
    settings: CandidateSettings,
    *,
    client: LLMClient | None = None,
    apply_ops: ApplyOps | None = None,
    count: CountFn | None = None,
    compressor: Compressor | None = None,
) -> CandidateSummary:
    """Generate every missing candidate and append it to its output file.

    Methods run one at a time over all prompts (cheap deterministic methods first by default),
    each record is flushed as soon as it is produced, and candidates whose ``candidate_id``
    already exists in the target file are skipped.

    Raises:
        ResumeMismatchError: If an existing candidate has a different generator.
        CandidateServerUnavailableError: If an LLM method runs and the server is unreachable.
        BudgetExceededError: If the phase token cap would be exceeded.
    """
    start = time.perf_counter()
    apply = apply_ops or _default_apply_ops
    count_fn = count or _default_count
    prompts = load_prompts(
        settings.prompts_path, limit=settings.limit, exclude_splits=settings.exclude_splits
    )
    summary = CandidateSummary(n_prompts=len(prompts))

    existing: dict[Path, dict[str, dict[str, Any]]] = {}
    for method in settings.methods:
        path = output_path_for(settings, method)
        if path not in existing:
            existing[path] = {r["candidate_id"]: r for r in read_candidates(path)}

    lingua: Compressor | None = compressor
    for method in settings.methods:
        kind = method_kind(method)
        path = output_path_for(settings, method)
        done = existing[path]
        expected_gen = _generator_for(settings, method)
        todo = []
        for row in prompts:
            cid = candidate_id(str(row["id"]), method)
            if cid in done:
                old = done[cid]
                custom = compressor is not None and not hasattr(compressor, "generator")
                if not _generator_matches(
                    settings, method, old.get("generator"), expected_gen, custom=custom
                ):
                    raise ResumeMismatchError(
                        f"{path}: {cid} was generated by {old.get('generator')!r}, current "
                        f"settings use {expected_gen!r}; write to a new output file"
                    )
                summary.skipped_existing[method] += 1
            else:
                todo.append(row)
        if not todo:
            continue
        if kind == "llm" and client is None:
            client = LLMClient(settings.budget_phase)
        if kind == "llmlingua2" and lingua is None:
            lingua = LLMLingua2Compressor(
                settings.llmlingua_model,
                rate=settings.llmlingua_rate,
                device=settings.llmlingua_device,
                cache_dir=settings.llmlingua_cache_dir,
            )

        path.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        if kind == "llm" and settings.workers > 1:
            # Warm the response cache concurrently so vLLM can batch; the ordered loop below
            # then reads every rewrite from the cache and writes records in prompt order.
            assert client is not None

            def warm(row: dict[str, Any], method: str = method, llm: LLMClient = client) -> None:
                _llm_candidate(llm, settings, str(row["id"]), method, str(row["prompt"]), count_fn)

            with ThreadPoolExecutor(max_workers=settings.workers) as pool:
                list(pool.map(warm, todo))
        with path.open("a", encoding="utf-8", newline="\n") as fh:
            for row in todo:
                pid, original = str(row["id"]), str(row["prompt"])
                if kind == "ops":
                    text = apply(original, method[4:], settings.operator_target)
                    assert expected_gen is not None
                    rec = _record(pid, method, text, expected_gen, False)
                elif kind == "llmlingua2":
                    assert lingua is not None
                    text = lingua(original)
                    gen = getattr(lingua, "generator", _CUSTOM_COMPRESSOR)
                    rec = _record(pid, method, text, str(gen), False)
                else:
                    assert client is not None and expected_gen is not None
                    rec = _llm_candidate(client, settings, pid, method, original, count_fn)
                    summary.llm_calls += 1
                if rec["rejected"]:
                    summary.rejected[method] += 1
                    summary.rejection_reasons[str(rec["rejection_reason"])] += 1
                if rec["text"].strip() == original.strip():
                    summary.identical_to_original[method] += 1
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                done[rec["candidate_id"]] = rec
                summary.written[method] += 1
        summary.seconds[method] = summary.seconds.get(method, 0.0) + time.perf_counter() - t0
    summary.wall_seconds = time.perf_counter() - start
    return summary


def _llm_candidate(
    client: LLMClient,
    settings: CandidateSettings,
    prompt_id: str,
    method: str,
    original: str,
    count: CountFn,
) -> dict[str, Any]:
    level = method[4:]
    instruction = render_rewrite_prompt(original, level, settings.prompt_version)
    try:
        resp = client.complete(
            settings.provider,
            settings.rewrite_model,
            [{"role": "user", "content": instruction}],
            settings.max_rewrite_tokens,
            temperature=settings.temperature,
        )
    except LocalLLMUnavailableError as exc:
        raise CandidateServerUnavailableError(
            f"LLM rewrites need a running OpenAI-compatible server for provider "
            f"{settings.provider!r} (model {settings.rewrite_model!r}). {exc}"
        ) from exc
    text, reason = validate_rewrite(
        resp.text, original, targets=settings.length_targets, count=count
    )
    gen = _generator_for(settings, method)
    assert gen is not None
    return _record(prompt_id, method, text, gen, settings.stand_in, reason)


# --------------------------------------------------------------------------------------------
# Config / CLI
# --------------------------------------------------------------------------------------------


def _opt_path(value: Any) -> Path | None:
    return None if value in (None, "") else Path(value)


def settings_from_config(cfg: DictConfig) -> CandidateSettings:
    """Build :class:`CandidateSettings` from a composed config (``experiment=p2_candidates``)."""
    from omegaconf import OmegaConf

    params = OmegaConf.to_container(cfg.experiment.params, resolve=True)
    assert isinstance(params, dict)
    p: dict[str, Any] = {str(k): v for k, v in params.items()}
    ops = [f"ops_{codes}" for codes in p.get("operator_stacks") or ()]
    methods = p.get("methods")
    if methods is None:
        methods = [*ops, "llmlingua2", *(f"llm_{lvl}" for lvl in LLM_LEVELS)]
    return CandidateSettings(
        prompts_path=Path(p["prompts_path"]),
        output_path=Path(p["output_path"]),
        smoke_output_path=Path(p["smoke_output_path"]),
        methods=tuple(str(m) for m in methods),
        provider=str(p.get("provider", "local")),
        rewrite_model=str(p.get("rewrite_model", "qwen2.5:1.5b")),
        stand_in=bool(p.get("stand_in", True)),
        prompt_version=str(p.get("prompt_version", "rewrite_v1")),
        temperature=float(p.get("temperature", 0.0)),
        max_rewrite_tokens=int(p.get("max_rewrite_tokens", 1024)),
        length_targets=tuple(str(t) for t in p.get("length_targets") or ()),
        operator_target=p.get("operator_target"),
        llmlingua_model=str(p.get("llmlingua_model", DEFAULT_LLMLINGUA_MODEL)),
        llmlingua_rate=float(p.get("llmlingua_rate", 0.5)),
        llmlingua_device=str(p.get("llmlingua_device", "cpu")),
        llmlingua_cache_dir=_opt_path(p.get("llmlingua_cache_dir")),
        limit=p.get("limit"),
        exclude_splits=tuple(str(s) for s in p.get("exclude_splits") or ()),
        budget_phase=str(p.get("budget_phase", "phase2")),
        workers=int(p.get("workers", 1)),
    )


def run_from_config(
    cfg: DictConfig,
    *,
    client: LLMClient | None = None,
    apply_ops: ApplyOps | None = None,
    count: CountFn | None = None,
    compressor: Compressor | None = None,
    track: bool = True,
) -> CandidateSummary:
    """Run candidate generation from a composed config, optionally inside an MLflow run."""
    settings = settings_from_config(cfg)
    kwargs: dict[str, Any] = {
        "client": client,
        "apply_ops": apply_ops,
        "count": count,
        "compressor": compressor,
    }
    if not track:
        return run_candidates(settings, **kwargs)

    import mlflow

    from terser.tracking import start_run

    hardware = str(cfg.get("hardware", {}).get("name", "cpu"))
    model = re.sub(r"[^A-Za-z0-9_.]", "_", settings.rewrite_model.split("/")[-1])
    variant = "standin" if settings.stand_in and settings.llm_methods else "full"
    params = {
        "methods": ",".join(settings.methods),
        "rewrite_model": settings.rewrite_model,
        "stand_in": settings.stand_in,
        "prompt_version": settings.prompt_version,
        "llmlingua_model": settings.llmlingua_model,
        "llmlingua_rate": settings.llmlingua_rate,
        "length_targets": ",".join(settings.length_targets),
        "limit": settings.limit,
        "exclude_splits": ",".join(settings.exclude_splits),
        "prompts_path": settings.prompts_path.as_posix(),
    }
    with start_run(
        phase=2,
        row="2.2.1",
        task="candidates",
        run_name=f"2-candidates-{model}-{variant}-0",
        hardware=hardware,
        params=params,
    ):
        mlflow.set_tag("stand_in", str(settings.stand_in).lower())
        summary = run_candidates(settings, **kwargs)
        mlflow.log_metrics(summary.metrics())
    return summary


def _hydra_entry() -> None:
    from terser.config import hydra_main

    @hydra_main()
    def _main(cfg: DictConfig) -> None:
        if cfg.experiment.get("name") != "p2_candidates":
            print("usage: python -m terser.data.candidates experiment=p2_candidates")
            sys.exit(2)
        try:
            summary = run_from_config(cfg)
        except CandidateServerUnavailableError as exc:
            print(f"error: {exc}", file=sys.stderr)
            sys.exit(2)
        except (BudgetExceededError, ResumeMismatchError, FileNotFoundError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            sys.exit(1)
        print(json.dumps(summary.metrics(), indent=2))

    _main()


def main() -> None:
    """CLI entry point (Hydra)."""
    _hydra_entry()


if __name__ == "__main__":
    main()
