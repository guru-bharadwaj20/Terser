"""Row 13.8: demo script, casual prompt -> rewrite -> side-by-side outputs + token counter.

For one casual prompt the demo shows

1. the **fast path** (CPU only, always live): the serving stack of
   ``configs/serving/fast_onnx.yaml`` (ONNX span tagger + ONNX gate, :mod:`terser.serving`), or
   the rule-based operators of ``default.yaml`` when the ONNX models are not pulled;
2. the **quality path**: the generative rewriter served by vLLM, and the target models' answers to
   the original and to the rewrite, side by side;
3. a **token counter**: prompt tokens on every target tokenizer (ground truth, never estimated)
   before and after each rewrite, plus the answers' output tokens.

Step 2 is **live** only when GPU endpoints are configured and answer ``GET /models``
(``--quality-url`` and ``--target-url NAME=URL``, or ``TERSER_DEMO_QUALITY_URL`` /
``TERSER_DEMO_TARGET_URLS``).
Otherwise the demo falls back to a **cache** of pre-computed validation examples
(``data/demo/cache.jsonl``): the rung-2 rewriter's (Qwen2.5-1.5B LoRA, v1) val rewrite, both
targets' greedy answers and the 32B judge's labels, all copied from the v1 files in
``data/interim/rewrites/``. A prompt that is not in the cache still gets the live fast path and
the token counter; ``--nearest`` then shows the most similar cached example, clearly labelled.
No endpoint is probed unless one is configured, so the demo never sends work to a GPU by surprise.

CLI::

    python -m terser.demo --list                         # cached example ids
    python -m terser.demo --example dolly:10026:texting  # one cached example
    python -m terser.demo "hey so um can you explain what a mutex is?? thx"
    python -m terser.demo --build-cache                  # rebuild data/demo/cache.jsonl
    python -m terser.demo "..." --quality-url http://worker-1:8000/v1 \\
        --target-url qwen2.5=http://worker-1:8001/v1 --target-url phi3.5=http://worker-1:8002/v1

The runbook and a rehearsed transcript are in ``docs/demo.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import textwrap
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

from terser.serving.cache import normalise

__all__ = [
    "DEFAULT_CACHE",
    "TARGET_MODELS",
    "CacheIndex",
    "LiveEndpoints",
    "build_cache",
    "probe",
    "run_demo",
    "side_by_side",
]

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = ROOT / "data" / "demo" / "cache.jsonl"
REWRITES = ROOT / "data" / "interim" / "rewrites"
# demo target name (terser.tokenizers.count) -> served model id
TARGET_MODELS: dict[str, str] = {
    "qwen2.5": "Qwen/Qwen2.5-7B-Instruct",
    "phi3.5": "microsoft/Phi-3.5-mini-instruct",
}
COUNT_TARGETS: tuple[str, ...] = ("gpt-4o", "qwen2.5", "phi3.5")
CACHE_METHOD = "sft15"  # rung 2, Qwen2.5-1.5B LoRA (v1)
CACHE_MODEL = "models/rewriter/sft-qwen1.5b-lora"

Counter = Callable[[str, str], int]
FastRewrite = Callable[[str, str], dict[str, Any]]
GateScore = Callable[[str, str, str], float]  # (prompt, rewrite, target) -> P(degraded)


# ---------------------------------------------------------------- cache


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _short(model: str) -> str:
    for name, mid in TARGET_MODELS.items():
        if mid == model:
            return name
    return model


def build_cache(
    rewrites_dir: Path = REWRITES,
    out: Path = DEFAULT_CACHE,
    per_category: int = 3,
    max_prompt_chars: int = 900,
    method: str = CACHE_METHOD,
) -> list[dict[str, Any]]:
    """Select ``per_category`` val examples per category and write the demo cache.

    Eligible: the rewriter changed the prompt (no fallback), both targets answered both texts
    (sample 0) and both labels exist. Per category the selection is deterministic (sha256 of
    the id): ``per_category - 1`` preserved-on-both examples and one degraded-on-any example, so
    the demo shows failures too.
    """
    rewrites = {r["id"]: r for r in _read_jsonl(rewrites_dir / f"{_rung_file(method)}")}
    responses: dict[tuple[str, str, str], dict[str, Any]] = {}
    for name in ("val_resp_qwen2.5-7b.jsonl", "val_resp_phi3.5.jsonl"):
        for r in _read_jsonl(rewrites_dir / name):
            if r.get("sample", 0) != 0:
                continue
            if r["candidate_id"] == "original":
                cand = "original"
            elif r["candidate_id"] == f"{r['prompt_id']}:{method}":
                cand = "rewrite"
            else:  # another rung's rewrite
                continue
            responses[(r["prompt_id"], _short(r["target"]), cand)] = r
    labels: dict[tuple[str, str], dict[str, Any]] = {}
    for r in _read_jsonl(rewrites_dir / "val_labels.jsonl"):
        if r["candidate_id"].endswith(f":{method}"):
            labels[(r["prompt_id"], _short(r["target"]))] = r

    by_cat: dict[str, list[dict[str, Any]]] = {}
    for pid, rw in rewrites.items():
        if rw.get("fell_back") or rw["rewrite"] == rw["prompt"]:
            continue
        if len(rw["prompt"]) > max_prompt_chars:
            continue
        rec: dict[str, Any] = {
            "id": pid,
            "category": rw["category"],
            "prompt": rw["prompt"],
            "quality_rewrite": rw["rewrite"],
            "quality_model": rw.get("model", CACHE_MODEL),
            "quality_latency_s": rw.get("latency_s"),
            "responses": {},
        }
        ok = True
        for tgt, model in TARGET_MODELS.items():
            o = responses.get((pid, tgt, "original"))
            w = responses.get((pid, tgt, "rewrite"))
            lab = labels.get((pid, tgt))
            if o is None or w is None or lab is None:
                ok = False
                break
            rec["responses"][tgt] = {
                "model": model,
                "original": o["output"],
                "original_output_tokens": o.get("output_tokens"),
                "rewrite": w["output"],
                "rewrite_output_tokens": w.get("output_tokens"),
                "label": lab["label"],
                "reason": lab.get("reason"),
                "judge_model": lab.get("judge_model"),
            }
        if ok:
            by_cat.setdefault(rw["category"], []).append(rec)

    chosen: list[dict[str, Any]] = []
    for cat in sorted(by_cat):
        recs = sorted(by_cat[cat], key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())
        good = [r for r in recs if all(v["label"] == "preserved" for v in r["responses"].values())]
        bad = [r for r in recs if r not in good]
        chosen += good[: max(per_category - 1, 0)] + bad[:1]
    source = {
        "rewrites": f"data/interim/rewrites/{_rung_file(method)}",
        "responses": "data/interim/rewrites/val_resp_{qwen2.5-7b,phi3.5}.jsonl (sample 0)",
        "labels": "data/interim/rewrites/val_labels.jsonl",
    }
    for r in chosen:
        r["source"] = source
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in chosen), encoding="utf-8"
    )
    return chosen


def _rung_file(method: str) -> str:
    return {
        "sft05": "sft-qwen0.5b-full.val.jsonl",
        "sft15": "sft-qwen1.5b-lora.val.jsonl",
        "sft3b": "sft-qwen3b-lora.val.jsonl",
    }[method]


_WORD = re.compile(r"[a-z0-9]+")


def _words(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


@dataclass
class CacheIndex:
    """Cached examples, looked up by id, by normalised prompt, or by word overlap."""

    records: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path = DEFAULT_CACHE) -> CacheIndex:
        return cls(_read_jsonl(path) if path.exists() else [])

    def by_id(self, rid: str) -> dict[str, Any] | None:
        return next((r for r in self.records if r["id"] == rid), None)

    def exact(self, prompt: str) -> dict[str, Any] | None:
        key = normalise(prompt)
        return next((r for r in self.records if normalise(r["prompt"]) == key), None)

    def nearest(self, prompt: str) -> tuple[dict[str, Any], float] | None:
        q = _words(prompt)
        best: tuple[dict[str, Any], float] | None = None
        for r in self.records:
            w = _words(r["prompt"])
            sim = len(q & w) / len(q | w) if q | w else 0.0
            if best is None or sim > best[1]:
                best = (r, sim)
        return best


# ---------------------------------------------------------------- live endpoints


def probe(base_url: str, timeout_s: float = 1.5) -> str | None:
    """First model id served at ``base_url`` (OpenAI-compatible ``GET /models``), else None."""
    import httpx

    try:
        resp = httpx.get(f"{base_url.rstrip('/')}/models", timeout=timeout_s)
        resp.raise_for_status()
        data = resp.json().get("data") or []
        return str(data[0]["id"]) if data else None
    except Exception:  # noqa: BLE001 - unreachable, refused, not OpenAI-compatible: no GPU path
        return None


@dataclass
class LiveEndpoints:
    """Reachable GPU endpoints: quality rewriter and per-target answer servers."""

    quality_url: str | None = None
    quality_model: str | None = None
    targets: dict[str, tuple[str, str]] = field(default_factory=dict)  # name -> (url, model)

    @classmethod
    def discover(
        cls, quality_url: str | None, target_urls: Mapping[str, str], timeout_s: float = 1.5
    ) -> LiveEndpoints:
        live = cls()
        if quality_url:
            model = probe(quality_url, timeout_s)
            if model:
                live.quality_url, live.quality_model = quality_url, model
        for name, url in target_urls.items():
            model = probe(url, timeout_s)
            if model:
                live.targets[name] = (url, model)
        return live

    @property
    def any(self) -> bool:
        return bool(self.quality_url or self.targets)

    def rewrite(self, prompt: str, target: str) -> str:
        from terser.serving.paths import OpenAICompatibleRewriter

        assert self.quality_url and self.quality_model
        rw = OpenAICompatibleRewriter(
            self.quality_url, self.quality_model, timeout_s=60.0, prompt_format="sft"
        )
        return rw(prompt, target).text

    def answer(self, target: str, prompt: str, max_tokens: int = 512) -> tuple[str, int | None]:
        import httpx

        url, model = self.targets[target]
        body = {
            "model": model,
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        resp = httpx.post(f"{url.rstrip('/')}/chat/completions", json=body, timeout=120.0)
        resp.raise_for_status()
        data = resp.json()
        usage = data.get("usage") or {}
        out = usage.get("completion_tokens")
        return str(data["choices"][0]["message"]["content"]), int(out) if out else None


# ---------------------------------------------------------------- display


def _cell(text: str, width: int, max_lines: int) -> list[str]:
    lines: list[str] = []
    for para in text.splitlines() or [""]:
        lines += textwrap.wrap(para, width, replace_whitespace=False, drop_whitespace=True) or [""]
    if max_lines and len(lines) > max_lines:
        hidden = len(lines) - max_lines + 1
        lines = lines[: max_lines - 1] + [f"... (+{hidden} more lines)"]
    return lines


def side_by_side(
    left_title: str, left: str, right_title: str, right: str, width: int = 100, max_lines: int = 14
) -> str:
    """Two wrapped columns with titles, ASCII only."""
    col = max(10, (width - 3) // 2)
    a = _cell(left, col, max_lines)
    b = _cell(right, col, max_lines)
    rows = [f"{left_title[:col]:<{col}} | {right_title[:col]}", f"{'-' * col}-+-{'-' * col}"]
    for i in range(max(len(a), len(b))):
        x = a[i] if i < len(a) else ""
        y = b[i] if i < len(b) else ""
        rows.append(f"{x:<{col}} | {y}".rstrip())
    return "\n".join(rows)


def _pct(before: int, after: int) -> str:
    return f"{(after - before) / before:+.1%}" if before else "n/a"


def token_table(texts: Sequence[tuple[str, str]], targets: Sequence[str], counter: Counter) -> str:
    """Prompt tokens per target tokenizer for each text, with savings vs the first text."""
    head = f"{'tokens':<22}" + "".join(f"{t:>12}" for t in targets)
    rows = [head, "-" * len(head)]
    base = [counter(texts[0][1], t) for t in targets]
    for i, (name, text) in enumerate(texts):
        counts = [counter(text, t) for t in targets]
        cells = "".join(
            f"{c:>12}" if i == 0 else f"{f'{c} ({_pct(b, c)})':>12}"
            for c, b in zip(counts, base, strict=True)
        )
        rows.append(f"{name:<22}{cells}")
    return "\n".join(rows)


# ---------------------------------------------------------------- the demo


def _default_counter() -> Counter:
    from terser.tokenizers.count import count_tokens

    return count_tokens


def _default_fast(config: Path | None) -> tuple[FastRewrite, GateScore]:
    """The serving stack's fast path and gate (no quality path; the demo drives that)."""
    from terser.serving.api import build_service, load_config

    cfg_path = config
    if cfg_path is None:
        onnx_cfg = ROOT / "configs" / "serving" / "fast_onnx.yaml"
        have_models = (ROOT / "models" / "fast" / "gate.onnx").exists() and (
            ROOT / "models" / "fast" / "tagger.onnx"
        ).exists()
        cfg_path = onnx_cfg if have_models else ROOT / "configs" / "serving" / "default.yaml"
    cfg = load_config(cfg_path)
    cfg["quality"] = {}  # the demo drives the quality path itself
    svc = build_service(cfg)
    label = cfg_path.name

    def gate(prompt: str, rewrite: str, target: str) -> float:
        if rewrite == prompt:
            return 0.0
        gt = svc.gate_feature_target or target
        return float(svc.gate.score(svc.features(prompt, gt), svc.features(rewrite, gt)))

    def fast(prompt: str, target: str) -> dict[str, Any]:
        candidate = svc.fast(prompt, target).text  # what the span rewriter proposed
        r = svc.rewrite(prompt, target, "fast")  # what the service serves after the gate
        return {
            **r.as_dict(),
            "candidate": candidate,
            "threshold": svc.gate_threshold,
            "config": label,
        }

    fast("warm up the tokenizers and ONNX sessions", "qwen2.5")
    return fast, gate


def run_demo(
    prompt: str,
    *,
    cache: CacheIndex,
    fast: FastRewrite,
    counter: Counter,
    live: LiveEndpoints | None = None,
    gate: GateScore | None = None,
    count_targets: Sequence[str] = COUNT_TARGETS,
    answer_targets: Sequence[str] = tuple(TARGET_MODELS),
    nearest: bool = False,
    width: int = 100,
    max_lines: int = 14,
    out: TextIO = sys.stdout,
) -> dict[str, Any]:
    """Run the demo for one prompt and print it; returns what was shown (for tests)."""
    live = live or LiveEndpoints()
    p = lambda s="": print(s, file=out)  # noqa: E731
    rule = "=" * width
    hit = cache.exact(prompt)
    mode = "live" if live.any else ("cached" if hit else "fast-only")
    shown: dict[str, Any] = {"mode": mode, "cache_id": hit["id"] if hit else None}

    p(rule)
    p(f"TERSER DEMO  mode={mode}" + (f"  cached example={hit['id']}" if hit else ""))
    p(rule)

    # 1. fast path, always live on CPU
    primary = "qwen2.5" if "qwen2.5" in count_targets else count_targets[0]
    fr = fast(prompt, primary)
    shown["fast"] = fr
    candidate = str(fr.get("candidate") or fr["rewrite"])
    thr = fr.get("threshold")
    verdict = "REJECTED, original served" if fr["gate_rejected"] else "passed, rewrite served"
    p(
        f"[1] FAST PATH (CPU, live; {fr.get('config', 'fast')}, {fr['latency_ms']:.0f} ms): "
        f"gate P(degraded) {fr['gate_score']:.3f}"
        + (f" vs threshold {thr:.3f}" if thr is not None else "")
        + f" -> {verdict}"
    )
    p(side_by_side("ORIGINAL (casual)", prompt, "FAST CANDIDATE", candidate, width, max_lines))
    p()

    # 2. quality path: live, cached, or unavailable
    quality: str | None = None
    qsrc = ""
    if live.quality_url:
        quality = live.rewrite(prompt, primary)
        qsrc = f"live, {live.quality_model} @ {live.quality_url}"
    elif hit:
        quality = hit["quality_rewrite"]
        qsrc = f"CACHED, {hit['quality_model']} (v1 val rewrite)"
    shown["quality"] = quality
    if quality is not None:
        qline = f"[2] QUALITY PATH ({qsrc})"
        if gate is not None:
            score = gate(prompt, quality, primary)
            shown["quality_gate"] = score
            qline += f": gate P(degraded) {score:.3f}"
            if thr is not None:
                qline += " -> " + ("would be REJECTED" if score > thr else "would pass")
        p(qline)
        p(side_by_side("ORIGINAL (casual)", prompt, "QUALITY REWRITE", quality, width, max_lines))
    else:
        p("[2] QUALITY PATH: unavailable (no GPU endpoint reachable and prompt not in the cache)")
        if nearest and cache.records:
            near = cache.nearest(prompt)
            if near:
                p(
                    f"    nearest cached example (a DIFFERENT prompt, word Jaccard {near[1]:.2f}): "
                    f"python -m terser.demo --example {near[0]['id']}"
                )
        elif cache.records:
            p(f"    {len(cache.records)} cached examples: python -m terser.demo --list")
    p()

    # 3. token counter
    texts = [("original", prompt), ("fast candidate", candidate)]
    if quality is not None:
        texts.append(("quality rewrite", quality))
    table = token_table(texts, count_targets, counter)
    shown["tokens"] = {name: {t: counter(text, t) for t in count_targets} for name, text in texts}
    p("[3] TOKEN COUNTER (prompt tokens per target tokenizer; change vs original)")
    p(table)
    p()

    # 4. target answers side by side
    shown["answers"] = {}
    if quality is not None:
        for tgt in answer_targets:
            if tgt in live.targets:
                o_text, o_tok = live.answer(tgt, prompt)
                r_text, r_tok = live.answer(tgt, quality)
                label, src = "not judged live", f"live, {live.targets[tgt][1]}"
            elif hit and tgt in hit["responses"] and not live.quality_url:
                c = hit["responses"][tgt]
                o_text, o_tok = c["original"], c["original_output_tokens"]
                r_text, r_tok = c["rewrite"], c["rewrite_output_tokens"]
                label = f"{c['label']} ({c['reason']}; judge {c['judge_model']})"
                src = f"CACHED, {c['model']} greedy"
            else:
                continue
            shown["answers"][tgt] = {"label": label, "original": o_text, "rewrite": r_text}
            p(f"[4] TARGET {tgt} ({src}): judge label = {label}")
            p(
                side_by_side(
                    f"ANSWER TO ORIGINAL ({o_tok} out tok)",
                    o_text,
                    f"ANSWER TO REWRITE ({r_tok} out tok)",
                    r_text,
                    width,
                    max_lines,
                )
            )
            p()
    p(rule)
    return shown


def _target_urls(values: Iterable[str]) -> dict[str, str]:
    urls: dict[str, str] = {}
    env = os.environ.get("TERSER_DEMO_TARGET_URLS", "")
    for item in [*filter(None, env.split(",")), *values]:
        name, _, url = item.partition("=")
        if not url:
            raise SystemExit(f"--target-url expects NAME=URL, got {item!r}")
        urls[name.strip()] = url.strip()
    return urls


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Terser demo (row 13.8)")
    ap.add_argument("prompt", nargs="?", help="casual prompt (or use --example / stdin '-')")
    ap.add_argument("--example", help="cached example id (see --list)")
    ap.add_argument("--list", action="store_true", help="list cached examples")
    ap.add_argument("--build-cache", action="store_true", help="rebuild the demo cache")
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument(
        "--mode",
        choices=("auto", "cached"),
        default="auto",
        help="cached = never contact a GPU endpoint",
    )
    ap.add_argument("--quality-url", default=os.environ.get("TERSER_DEMO_QUALITY_URL"))
    ap.add_argument("--target-url", action="append", default=[], metavar="NAME=URL")
    ap.add_argument("--config", type=Path, help="serving config for the fast path")
    ap.add_argument(
        "--targets", nargs="+", default=list(COUNT_TARGETS), help="tokenizers for the token counter"
    )
    ap.add_argument("--nearest", action="store_true", help="suggest the nearest cached example")
    ap.add_argument("--width", type=int, default=110)
    ap.add_argument("--max-lines", type=int, default=14)
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    if args.build_cache:
        recs = build_cache(out=args.cache)
        print(f"wrote {len(recs)} examples to {args.cache}")
        return 0
    cache = CacheIndex.load(args.cache)
    if args.list:
        for r in cache.records:
            labels = ",".join(f"{t}:{v['label']}" for t, v in r["responses"].items())
            first = r["prompt"].splitlines()[0][:60]
            print(f"{r['id']:<40} {r['category']:<9} {labels:<36} {first}")
        return 0
    if args.example:
        rec = cache.by_id(args.example)
        if rec is None:
            raise SystemExit(f"no cached example {args.example!r}; see --list")
        prompt = rec["prompt"]
    elif args.prompt == "-":
        prompt = sys.stdin.read()
    elif args.prompt:
        prompt = args.prompt
    else:
        ap.error("give a prompt, --example ID, --list or --build-cache")
    live = LiveEndpoints()
    if args.mode == "auto":
        live = LiveEndpoints.discover(args.quality_url, _target_urls(args.target_url))
    fast, gate = _default_fast(args.config)
    run_demo(
        prompt,
        cache=cache,
        fast=fast,
        gate=gate,
        counter=_default_counter(),
        live=live,
        count_targets=args.targets,
        nearest=args.nearest,
        width=args.width,
        max_lines=args.max_lines,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
