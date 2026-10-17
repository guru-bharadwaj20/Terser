"""Final results, stratified tables, statistics, ablations, length figure and gallery.

Plan rows **12.2** (master table), **12.3** (stratified), **12.4** (CIs, seeds, McNemar),
**12.6** (ablation matrix), **12.7** (net savings by prompt length), **12.9** (gallery);
headline criteria S1-S5.

Definitions (shared by every system, so the table is one metric applied many times):

* **Served text.** A system's candidate is served for target ``T`` unless the 8.3.7 safeguard
  falls back to the original: the candidate is missing, rejected (row 2.2.1 validation), empty,
  or not strictly shorter on ``T``'s tokenizer. A gated system then sends ``(original, served)``
  to its gate and serves the original if the gate rejects (:mod:`terser.phase12.gates`).
* **Savings** on ``T``: ``1 - n_T(served) / n_T(original)`` (ADR-0002), fallbacks 0.
* **Preservation**: a served original is preserved by construction; otherwise the ``label_v1``
  label of that exact text on ``T`` (labels are matched by candidate id, else by prompt id +
  sha256 of the text, so identical texts of different methods share a judgement). A served text
  without a label is *missing*: it is left out of the rate and reported; a system with missing
  labels is marked incomplete and kept out of the McNemar / top-system ranking.
* **Net savings** (S4, row 12.7): :func:`terser.serving.accounting.net_savings` per request
  (``price_T * (before - after) - rewriter_cost``, prices from ``configs/serving/default.yaml``),
  divided by ``price_T * before`` -- i.e. savings minus the rewriter's cost as a share of the
  prompt's target cost. Generative systems (SFT, zero-shot 32B) pay the rewriter's input and
  output tokens on every request, fallbacks and gate rejections included; CPU systems pay
  ``fast_path_per_request`` (0 in the config).
* **Latency**: mean rewrite seconds per prompt (CPU systems: measured by
  :mod:`terser.phase12.candidates`; SFT: generation time per prompt in batches of 16; zero-shot
  32B: run wall time per prompt from the generation logs) plus the gate's mean feature + scoring
  time for gated systems.
* **CIs**: 95% cluster bootstrap over source groups, 1,000 resamples, seed 0
  (:mod:`terser.phase12.stats`); the same resamples for every system.

Full command sequence after the row 12.1 unlock (CPU steps under ``CUDA_VISIBLE_DEVICES=``)::

    # 0. one-time unlock (commit the code first; logs to docs/checkpoints/C12/)
    python -m terser.phase12.unlock unlock --reason "Phase 12 final evaluation"
    # 1. CPU candidates (test split + real slice), seeds for TextCNN and Q-learning
    python -m terser.phase12.candidates build --split test --out-dir data/interim/final \\
        --textcnn-seeds 0,1,2 --rl-seeds 0,1,2
    # 2. GPU: SFT rewrites -> raw-rewrite candidates
    python -m terser.models.rewriter.generate --base Qwen/Qwen2.5-3B-Instruct \\
        --model models/rewriter/sft-qwen3b-lora-v2 \\
        --data data/interim/final/test_sft_input.jsonl --out data/interim/final/test_sft3bv2.jsonl
    python -m terser.phase12.candidates import-sft --tag sft3bv2 \\
        --generate data/interim/final/test_sft3bv2.jsonl \\
        --out data/interim/final/test_candidates_gpu.jsonl
    # 3. GPU: zero-shot 32B (vLLM Qwen2.5-32B-Instruct-AWQ), same rewrite_v1 levels as Phase 2
    python -m terser.data.candidates experiment=p2_candidates hardware=rtx4500ada \\
        experiment.params.stand_in=false \\
        experiment.params.rewrite_model=Qwen/Qwen2.5-32B-Instruct-AWQ \\
        experiment.params.workers=32 \\
        experiment.params.prompts_path=data/interim/final/test_prompts.jsonl \\
        experiment.params.output_path=data/interim/final/test_candidates_llm.jsonl \\
        experiment.params.smoke_output_path=data/interim/smoke/test_candidates_llm.jsonl \\
        'experiment.params.methods=[llm_low,llm_medium,llm_high]' > logs/p12_llm.log
    # 4. GPU: respond per target (vLLM serving each target), then judge (32B AWQ)
    cat data/interim/final/test_candidates_{cpu,gpu,llm}.jsonl \\
        > data/interim/final/test_candidates_all.jsonl
    python -m terser.phase2.respond --prompts data/interim/final/test_prompts.jsonl \\
        --candidates data/interim/final/test_candidates_all.jsonl \\
        --target Qwen/Qwen2.5-7B-Instruct --out data/interim/final/test_resp_qwen2.5-7b.jsonl
    python -m terser.phase2.respond --prompts data/interim/final/test_prompts.jsonl \\
        --candidates data/interim/final/test_candidates_all.jsonl \\
        --target microsoft/Phi-3.5-mini-instruct --out data/interim/final/test_resp_phi3.5.jsonl
    python -m terser.phase2.label_stage --prompts data/interim/final/test_prompts.jsonl \\
        --responses data/interim/final/test_resp_qwen2.5-7b.jsonl \\
        data/interim/final/test_resp_phi3.5.jsonl --out data/interim/final/test_labels.jsonl
    # 5. CPU: results
    python -m terser.phase12.results --split test \\
        --prompts data/interim/final/test_prompts.jsonl \\
        --candidates data/interim/final/test_candidates_all.jsonl \\
        --labels data/interim/final/test_labels.jsonl \\
        --responses data/interim/final/test_resp_qwen2.5-7b.jsonl \\
        data/interim/final/test_resp_phi3.5.jsonl \\
        --meta data/interim/final/test_cpu_meta.json \\
        --sft sft3bv2=data/interim/final/test_sft3bv2.jsonl --full-system sft3bv2 \\
        --gen-logs logs/p12_llm.log --out-dir reports
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from terser.phase12.common import (
    MAIN_SLICE,
    REAL_SLICE,
    TARGETS,
    TOKEN_TARGET,
    group_of,
    read_jsonl,
    short_target,
    text_sha,
)
from terser.phase12.stats import (
    N_RESAMPLES,
    SEED,
    ci,
    group_resamples,
    mcnemar_paired,
    nanmean,
    nanmedian,
)

__all__ = [
    "GateBank",
    "LabelIndex",
    "System",
    "SystemResult",
    "build_report",
    "default_systems",
    "evaluate",
    "main",
]

S1_MEDIAN, S1_CI_LOW, S2_PRES, S3_PRES = 0.30, 0.25, 0.95, 0.90
N_LENGTH_BINS: Final = 10
Y_FLOOR, Y_CEIL = -150.0, 75.0
GALLERY_N: Final = 5
SEED_RE = re.compile(r"^(?P<base>.+)_s(?P<seed>\d+)$")
FloatArray = npt.NDArray[np.float64]


# --------------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------------


class LabelIndex:
    """``label_v1`` records keyed by (candidate id, target) and by (prompt, text sha, target)."""

    def __init__(
        self, labels: Sequence[Mapping[str, Any]], cand_text: Mapping[str, tuple[str, str]]
    ) -> None:
        self.by_cid: dict[tuple[str, str], Mapping[str, Any]] = {}
        self.by_text: dict[tuple[str, str, str], Mapping[str, Any]] = {}
        self.conflicts = 0
        self.unmatched = 0
        for r in labels:
            if r.get("stand_in"):
                continue
            cid, tgt = str(r["candidate_id"]), str(r.get("target") or r["target_model"])
            self.by_cid[(cid, tgt)] = r
            hit = cand_text.get(cid)
            if hit is None:
                self.unmatched += 1
                continue
            key = (hit[0], text_sha(hit[1]), tgt)
            old = self.by_text.get(key)
            if old is not None and old["label"] != r["label"]:
                self.conflicts += 1
            self.by_text.setdefault(key, r)
        self.cand_text = cand_text

    def get(self, pid: str, cid: str, text: str, target: str) -> Mapping[str, Any] | None:
        known = self.cand_text.get(cid)
        if known is not None and known[1] == text and (cid, target) in self.by_cid:
            return self.by_cid[(cid, target)]
        return self.by_text.get((pid, text_sha(text), target))


@dataclass(frozen=True)
class System:
    """One row of the master or ablation table."""

    name: str
    method: str  # candidate method; "" = the original prompt
    row: str = ""
    gate: str | None = None  # "tuned" | "untuned" | "onnx"
    fallback: bool = True
    path: str = "fast"  # accounting path: "fast" (CPU) or "quality" (generative rewriter)
    kind: str = "main"  # main | ablation | seed


FEATURES: Final = "_pair_features"
"""Timing key of the pair-feature extraction shared by the LightGBM pair gates."""


class GateBank:
    """Gate scorers with a JSON score cache (``{gate|sha(orig)|sha(text)|category|target}``)."""

    def __init__(
        self,
        gates: Mapping[str, Any],
        featurizer: Any | None = None,
        cache_path: Path | None = None,
    ) -> None:
        self.gates = dict(gates)
        self.featurizer = featurizer
        self.cache_path = cache_path
        self.header = {k: _jsonable(g.describe()) for k, g in self.gates.items()}
        self.cache: dict[str, float] = {}
        # measured scoring time per gate; kept in the cache so a cached rerun reports the same
        # gate latency as the run that computed the scores
        self.seconds: dict[str, float] = defaultdict(float)
        self.calls: dict[str, int] = defaultdict(int)
        if cache_path is not None and cache_path.is_file():
            blob = json.loads(cache_path.read_text(encoding="utf-8"))
            if blob.get("header") == self.header:
                self.cache = {str(k): float(v) for k, v in blob["scores"].items()}
                for g, (sec, n) in blob.get("timing", {}).items():
                    self.seconds[str(g)] = float(sec)
                    self.calls[str(g)] = int(n)

    def score(self, gate: str, original: str, text: str, category: str, target: str) -> float:
        key = f"{gate}|{text_sha(original)}|{text_sha(text)}|{category}|{target}"
        if key in self.cache:
            return self.cache[key]
        g = self.gates[gate]
        if hasattr(g, "score_rows"):
            assert self.featurizer is not None, "pair gates need a featurizer"
            # features are shared by the pair gates (one cache): time them once, separately
            t0 = time.perf_counter()
            calls0 = self.featurizer.calls
            row = self.featurizer.features(original, text, category, target)
            if self.featurizer.calls > calls0:
                self.seconds[FEATURES] += time.perf_counter() - t0
                self.calls[FEATURES] += 1
            t0 = time.perf_counter()
            s = g.score_rows([row])[0]
        else:
            t0 = time.perf_counter()
            s = float(g.score(original, text))
        self.seconds[gate] += time.perf_counter() - t0
        self.calls[gate] += 1
        self.cache[key] = float(s)
        return float(s)

    def rejects(self, gate: str, score: float) -> bool:
        return bool(self.gates[gate].rejects(score))

    def mean_seconds(self, gate: str) -> float | None:
        """Mean per-request gate time; pair gates add the mean shared feature time."""
        n = self.calls.get(gate, 0)
        if not n:
            return None
        own = self.seconds[gate] / n
        if hasattr(self.gates[gate], "score_rows") and self.calls.get(FEATURES, 0):
            own += self.seconds[FEATURES] / self.calls[FEATURES]
        return own

    def save(self) -> None:
        if self.cache_path is not None:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            timing = {g: [self.seconds[g], self.calls[g]] for g in self.calls}
            blob = {"header": self.header, "scores": self.cache, "timing": timing}
            self.cache_path.write_text(json.dumps(blob), encoding="utf-8")


def _jsonable(x: Any) -> Any:
    return json.loads(json.dumps(x, default=str))


# --------------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------------


@dataclass
class SystemResult:
    """Per-(prompt, target) outcomes of one system; arrays are in prompt order."""

    system: System
    savings: dict[str, FloatArray]
    net: dict[str, FloatArray]
    preserved: dict[str, FloatArray]  # 1 / 0 / nan (missing label)
    fell_back: dict[str, npt.NDArray[np.bool_]]
    gate_rejected: dict[str, npt.NDArray[np.bool_]]
    served: dict[str, list[str]]
    label: dict[str, list[Mapping[str, Any] | None]]
    missing_candidates: int = 0
    latency_s: float | None = None
    latency_note: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def missing_labels(self) -> int:
        return int(sum(np.isnan(v).sum() for v in self.preserved.values()))

    @property
    def complete(self) -> bool:
        return self.missing_labels == 0 and self.missing_candidates == 0


def _rewriter_tokens(
    method: str,
    rec: Mapping[str, Any],
    original: str,
    cands: Mapping[str, Mapping[str, Mapping[str, Any]]],
    count: Callable[[str, str], int],
) -> tuple[int, int]:
    """Rewriter (input, output) tokens for one rewrite.

    Recorded counts win. Otherwise the two families are counted differently, by design: a
    zero-shot ``llm_*`` rewrite counts the full rendered instruction prompt in and the rewrite
    out (chat-template and EOS tokens not counted), while an SFT rung counts prompt + separator
    (+1) in and rewrite + EOS (+1) out, matching its training format.
    """
    if "rewriter_tokens_in" in rec:
        return int(rec["rewriter_tokens_in"]), int(rec["rewriter_tokens_out"])
    base = method[: -len("_raw")] if method.endswith("_raw") else method
    raw = cands.get(f"{base}_raw", {}).get(str(rec["prompt_id"]))
    if raw is not None and "rewriter_tokens_in" in raw:
        return int(raw["rewriter_tokens_in"]), int(raw["rewriter_tokens_out"])
    if base.startswith("llm_"):
        from terser.data.candidates import render_rewrite_prompt

        prompt = render_rewrite_prompt(original, base[4:])
        return count(prompt, TOKEN_TARGET), count(str(rec["text"]), TOKEN_TARGET)
    return count(original, TOKEN_TARGET) + 1, count(str(rec["text"]), TOKEN_TARGET) + 1


def evaluate(
    system: System,
    prompts: Sequence[Mapping[str, Any]],
    cands: Mapping[str, Mapping[str, Mapping[str, Any]]],
    labels: LabelIndex,
    count: Callable[[str, str], int],
    pricing: Any,
    gates: GateBank | None,
    targets: Sequence[str] = tuple(TARGETS),
) -> SystemResult:
    """Outcomes of ``system`` on every (prompt, target); see the module docstring."""
    from terser.serving.accounting import net_savings

    by_pid = cands.get(system.method, {})
    if system.method and not by_pid and not system.method.endswith("_raw"):
        by_pid = cands.get(f"{system.method}_raw", {})  # SFT known only from its generate output
    n = len(prompts)
    out = SystemResult(
        system=system,
        savings={},
        net={},
        preserved={},
        fell_back={},
        gate_rejected={},
        served={},
        label={},
    )
    missing_c = 0
    for t in targets:
        ts = short_target(t)
        sav, net, pres = np.zeros(n), np.zeros(n), np.full(n, np.nan)
        fb, gr = np.zeros(n, bool), np.zeros(n, bool)
        served_l: list[str] = []
        lab_l: list[Mapping[str, Any] | None] = []
        price = pricing.target_price(ts)
        for i, p in enumerate(prompts):
            pid, orig = str(p["id"]), str(p["prompt"])
            n0 = count(orig, ts)
            rec = by_pid.get(pid) if system.method else None
            served, cost_in, cost_out = orig, 0, 0
            if system.method and rec is None:
                missing_c += 1
                fb[i] = True
            elif rec is not None:
                text = str(rec["text"])
                bad = bool(rec.get("rejected")) or not text.strip()
                if system.fallback:
                    fb[i] = bad or count(text, ts) >= n0
                    served = orig if fb[i] else text
                else:
                    served = text if text.strip() else orig
                    fb[i] = served == orig
                if system.path == "quality":
                    cost_in, cost_out = _rewriter_tokens(system.method, rec, orig, cands, count)
                if system.gate and served != orig:
                    assert gates is not None, f"{system.name} needs gate {system.gate}"
                    score = gates.score(system.gate, orig, served, str(p["category"]), ts)
                    if gates.rejects(system.gate, score):
                        gr[i] = True
                        served = orig
            n1 = count(served, ts) if served != orig else n0
            sav[i] = 1.0 - n1 / n0 if n0 else 0.0
            acct = net_savings(
                pricing,
                ts,
                n0,
                n1,
                path=system.path,
                rewriter_tokens_in=cost_in,
                rewriter_tokens_out=cost_out,
            )
            net[i] = acct.net_savings / (price * n0) if n0 and price else 0.0
            lab = None
            if served == orig:
                pres[i] = 1.0
            else:
                cid = str(rec["candidate_id"]) if rec is not None else ""
                lab = labels.get(pid, cid, served, t)
                if lab is not None:
                    pres[i] = float(lab["label"] == "preserved")
            served_l.append(served)
            lab_l.append(lab)
        out.savings[t], out.net[t], out.preserved[t] = sav, net, pres
        out.fell_back[t], out.gate_rejected[t] = fb, gr
        out.served[t], out.label[t] = served_l, lab_l
    out.missing_candidates = missing_c // max(1, len(targets))
    return out


# --------------------------------------------------------------------------------------------
# Systems
# --------------------------------------------------------------------------------------------


def default_systems(
    methods: Sequence[str], sft_tags: Sequence[str], full: str | None
) -> list[System]:
    """Master-table systems (row 12.2), seed variants and the row 12.6 ablations."""
    have = set(methods)
    s: list[System] = [System("Original", "", "original")]
    fixed = [
        ("Best operator (`ops_FHC`)", "ops_FHC", "best operator", None),
        ("Q-learning policy", "rl_greedy", "Q-learning policy", None),
        ("HMM compressor", "hmm", "HMM", None),
        ("GBM tagger + gate", "gbm_tagger", "GBM tagger+gate", "tuned"),
        ("TextCNN + gate", "textcnn_tagger", "TextCNN+gate", "tuned"),
    ]
    for name, m, row, gate in fixed:
        if m in have:
            s.append(System(name, m, row, gate))
    tag_set = set(sft_tags)
    for tag in sft_tags:
        mm = SEED_RE.match(tag)
        if mm and mm.group("base") in tag_set:
            continue  # a seed run of another rung: a seed variant below, not a main row
        if tag in have or f"{tag}_raw" in have:
            s.append(System(f"SFT rewriter `{tag}`", tag, "SFT rewriter", None, path="quality"))
    if full and (full in have or f"{full}_raw" in have):
        s.append(
            System(f"Terser full (`{full}` + gate)", full, "SFT rewriter", "tuned", True, "quality")
        )
    if "llmlingua2" in have:
        s.append(System("LLMLingua-2", "llmlingua2", "LLMLingua-2"))
    for m in sorted(x for x in have if x.startswith("llm_") and not SEED_RE.match(x)):
        s.append(System(f"Zero-shot 32B {m[4:]}", m, "zero-shot LLM", path="quality"))
    if "fast_path" in have:
        s.append(System("Fast path (deployed ONNX)", "fast_path", "fast path", "onnx"))
    # Keyed by method, holding every system on it: "Terser full" and the ungated SFT rung share
    # a method, and each gets its own seed group.
    by_method: dict[str, list[System]] = {}
    for x in s:
        by_method.setdefault(x.method, []).append(x)
    for m in sorted(have):
        mm = SEED_RE.match(m)
        if mm is None:
            continue
        for b in by_method.get(mm.group("base"), []):
            s.append(
                System(
                    f"{b.name} seed {mm.group('seed')}",
                    m,
                    b.row,
                    b.gate,
                    b.fallback,
                    b.path,
                    "seed",
                )
            )
    if full and (full in have or f"{full}_raw" in have):
        raw = f"{full}_raw" if f"{full}_raw" in have else full
        s += [
            System("Ablation: − gate", full, "12.6", None, True, "quality", "ablation"),
            System("Ablation: − fallback", raw, "12.6", "tuned", False, "quality", "ablation"),
            System(
                "Ablation: − HPO (untuned gate)",
                full,
                "12.6",
                "untuned",
                True,
                "quality",
                "ablation",
            ),
        ]
        if "ops_FHC" in have:
            s.append(
                System(
                    "Ablation: operator-only (`ops_FHC` + gate)",
                    "ops_FHC",
                    "12.6",
                    "tuned",
                    True,
                    "fast",
                    "ablation",
                )
            )
    return s


# --------------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------------


def _subset(res: SystemResult, idx: npt.NDArray[np.int64]) -> SystemResult:
    return SystemResult(
        system=res.system,
        savings={t: v[idx] for t, v in res.savings.items()},
        net={t: v[idx] for t, v in res.net.items()},
        preserved={t: v[idx] for t, v in res.preserved.items()},
        fell_back={t: v[idx] for t, v in res.fell_back.items()},
        gate_rejected={t: v[idx] for t, v in res.gate_rejected.items()},
        served={t: [v[i] for i in idx] for t, v in res.served.items()},
        label={t: [v[i] for i in idx] for t, v in res.label.items()},
        missing_candidates=res.missing_candidates,
        latency_s=res.latency_s,
        latency_note=res.latency_note,
    )


def _pooled(res: SystemResult, ix: npt.NDArray[np.int64]) -> FloatArray:
    return np.concatenate([v[ix] for v in res.preserved.values()])


def _worst_category(res: SystemResult, cats: npt.NDArray[Any], ix: npt.NDArray[np.int64]) -> float:
    vals = []
    c = cats[ix]
    for cat in np.unique(c):
        m = c == cat
        vals.append(nanmean(np.concatenate([v[ix][m] for v in res.preserved.values()])))
    vals = [v for v in vals if np.isfinite(v)]
    return float(min(vals)) if vals else float("nan")


def _reward(res: SystemResult, ix: npt.NDArray[np.int64]) -> float:
    vals = [100 * res.savings[t][ix] - 100 * (1 - res.preserved[t][ix]) for t in res.preserved]
    return nanmean(np.concatenate(vals))


Stat = Callable[[npt.NDArray[np.int64]], float]


def _median_of(x: FloatArray) -> Stat:
    return lambda ix: nanmedian(x[ix])


def _mean_of(x: FloatArray) -> Stat:
    return lambda ix: nanmean(x[ix])


def _bin_mean(x: FloatArray, bins: npt.NDArray[np.int64], b: int) -> Stat:
    def stat(ix: npt.NDArray[np.int64]) -> float:
        m = bins[ix] == b
        return nanmean(x[ix][m]) if m.any() else float("nan")

    return stat


def _pooled_rate(res: SystemResult, ix: npt.NDArray[np.int64]) -> float:
    return nanmean(_pooled(res, ix))


def length_bins(lengths: FloatArray, n_bins: int = N_LENGTH_BINS) -> npt.NDArray[np.int64]:
    """Percentile bin (0 .. n_bins-1) of each prompt's original token length."""
    if not len(lengths):
        return np.zeros(0, dtype=np.int64)
    edges = np.quantile(lengths, np.linspace(0, 1, n_bins + 1))[1:-1]
    return np.searchsorted(edges, lengths, side="right").astype(np.int64)


def summarise(
    res: SystemResult,
    cats: npt.NDArray[Any],
    lengths: Mapping[str, FloatArray],
    resamples: Sequence[npt.NDArray[np.int64]],
) -> dict[str, Any]:
    """Headline metrics of one system with bootstrap CIs (indices are local to ``res``)."""
    n = len(cats)
    per_t: dict[str, Any] = {}
    for t in res.savings:
        sav, net, pres = res.savings[t], res.net[t], res.preserved[t]
        bins = length_bins(lengths[t])
        p25_bin = min(N_LENGTH_BINS - 1, int(0.25 * N_LENGTH_BINS))  # the decile holding p25
        per_t[t] = {
            "n": n,
            "median_savings": ci(_median_of(sav), n, resamples).as_dict(),
            "mean_savings": float(np.mean(sav)) if n else None,
            "preserved": ci(_mean_of(pres), n, resamples).as_dict(),
            "median_net_savings": ci(_median_of(net), n, resamples).as_dict(),
            "mean_net_savings": ci(_mean_of(net), n, resamples).as_dict(),
            "net_savings_p25_length": ci(_bin_mean(net, bins, p25_bin), n, resamples).as_dict(),
            "fell_back": int(res.fell_back[t].sum()),
            "gate_rejected": int(res.gate_rejected[t].sum()),
            "missing_labels": int(np.isnan(pres).sum()),
            "worst_category_preserved": _worst_category(
                _one_target(res, t), cats, np.arange(n, dtype=np.int64)
            ),
        }
    return {
        "name": res.system.name,
        "method": res.system.method,
        "row": res.system.row,
        "gate": res.system.gate,
        "fallback": res.system.fallback,
        "path": res.system.path,
        "kind": res.system.kind,
        "complete": res.complete,
        "missing_labels": res.missing_labels,
        "missing_candidates": res.missing_candidates,
        "latency_s": res.latency_s,
        "latency_note": res.latency_note,
        "per_target": per_t,
        "preserved_pooled": ci(partial(_pooled_rate, res), n, resamples).as_dict(),
        "worst_category_preserved": ci(partial(_worst_category, res, cats), n, resamples).as_dict(),
        "reward": _reward(res, np.arange(n, dtype=np.int64)) if n else None,
    }


def _one_target(res: SystemResult, t: str) -> SystemResult:
    r = _subset(res, np.arange(len(res.savings[t]), dtype=np.int64))
    r.preserved = {t: res.preserved[t]}
    return r


def strata(
    res: SystemResult, keys: Sequence[Any]
) -> dict[str, dict[str, dict[str, float | int | None]]]:
    """``{target: {stratum: {n, median_savings, preserved, missing}}}`` (point estimates)."""
    k = np.asarray([str(x) for x in keys])
    out: dict[str, dict[str, dict[str, float | int | None]]] = {}
    for t in res.savings:
        out[t] = {}
        for s in sorted(set(k.tolist()), key=_natural):
            m = k == s
            pres = res.preserved[t][m]
            out[t][s] = {
                "n": int(m.sum()),
                "median_savings": nanmedian(res.savings[t][m]),
                "preserved": _none(nanmean(pres)),
                "mean_net_savings": nanmean(res.net[t][m]),
                "missing": int(np.isnan(pres).sum()),
            }
    return out


def _natural(s: str) -> tuple[int, Any]:
    return (0, int(s)) if s.isdigit() else (1, s)


def _none(x: float) -> float | None:
    return None if not np.isfinite(x) else float(x)


def criteria(summary: Mapping[str, Any]) -> dict[str, Any]:
    """S1-S5 for one system summary."""
    pt = summary["per_target"]
    s1 = {
        t: bool(
            (v["median_savings"]["estimate"] or 0) >= S1_MEDIAN
            and (v["median_savings"]["low"] or 0) >= S1_CI_LOW
        )
        for t, v in pt.items()
    }
    pooled = summary["preserved_pooled"]["estimate"]
    worst = summary["worst_category_preserved"]["estimate"]
    s4 = {t: bool((v["net_savings_p25_length"]["estimate"] or 0) > 0) for t, v in pt.items()}
    s5 = {
        t: bool(
            s1[t]
            and (v["preserved"]["estimate"] or 0) >= S2_PRES
            and (v["worst_category_preserved"] or 0) >= S3_PRES
        )
        for t, v in pt.items()
    }
    return {
        "S1": {"per_target": s1, "pass": all(s1.values())},
        "S2": {"value": pooled, "pass": bool(pooled is not None and pooled >= S2_PRES)},
        "S3": {"value": worst, "pass": bool(worst is not None and worst >= S3_PRES)},
        "S4": {"per_target": s4, "pass": all(s4.values())},
        "S5": {"per_target": s5, "pass": all(s5.values())},
        "complete": summary["complete"],
    }


# --------------------------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------------------------


def _pct(x: float | None, digits: int = 1) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{100 * x:.{digits}f}%"


def _cip(d: Mapping[str, float | None]) -> str:
    if d.get("estimate") is None:
        return "n/a"
    if d.get("low") is None:
        return _pct(d["estimate"])
    return f"{_pct(d['estimate'])} [{_pct(d['low'])}, {_pct(d['high'])}]"


def _sec(x: float | None) -> str:
    if x is None:
        return "n/r"
    if x < 0.01:
        return f"{1000 * x:.1f} ms"
    return f"{1000 * x:.0f} ms" if x < 1 else f"{x:.2f} s"


PARTIAL_MAX = 0.05
"""Above this share of unlabelled served rewrites a preservation cell shows n/a."""


def _pres(d: Mapping[str, float | None], missing: int, total: int) -> str:
    """A preservation cell: exact when fully labelled, ``†`` when partial, n/a when mostly not."""
    if not missing:
        return _cip(d)
    if total and missing / total > PARTIAL_MAX:
        return f"n/a ({missing} unlabelled)"
    return _cip(d) + " †"


def _short(t: str) -> str:
    return short_target(t)


def _md_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    return [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
        *("| " + " | ".join(r) + " |" for r in rows),
    ]


def _headline_rows(
    summaries: Sequence[Mapping[str, Any]], targets: Sequence[str]
) -> list[list[str]]:
    rows = []
    for s in summaries:
        pt = s["per_target"]
        status = "complete" if s["complete"] else f"**{s['missing_labels']} unlabelled**"
        n_pairs = sum(int(pt[t]["n"]) for t in targets)
        rows.append(
            [
                s["name"],
                *(_cip(pt[t]["median_savings"]) for t in targets),
                *(_pres(pt[t]["preserved"], pt[t]["missing_labels"], pt[t]["n"]) for t in targets),
                _pres(s["preserved_pooled"], s["missing_labels"], n_pairs),
                _pres(s["worst_category_preserved"], s["missing_labels"], n_pairs),
                *(_cip(pt[t]["median_net_savings"]) for t in targets),
                *(_cip(pt[t]["mean_net_savings"]) for t in targets),
                " / ".join(f"{pt[t]['fell_back']}·{pt[t]['gate_rejected']}" for t in targets),
                _sec(s["latency_s"]),
                status,
            ]
        )
    return rows


def _headline_header(targets: Sequence[str]) -> list[str]:
    ts = [_short(t) for t in targets]
    return [
        "System",
        *(f"Median savings {t}" for t in ts),
        *(f"Preserved {t}" for t in ts),
        "Preserved pooled",
        "Worst category",
        *(f"Median net {t}" for t in ts),
        *(f"Mean net {t}" for t in ts),
        "Fell back · gate-rejected (" + " / ".join(ts) + ")",
        "Latency / prompt",
        "Labels",
    ]


def build_report(res: Mapping[str, Any]) -> str:
    """``final_results.md`` from the results JSON."""
    targets = res["targets"]
    ts = [_short(t) for t in targets]
    main_s = [s for s in res["systems"] if s["kind"] == "main"]
    L: list[str] = [
        f"# Final results ({res['split']} split) — rows 12.2-12.4, 12.6, 12.7",
        "",
        f"Generated by `python -m terser.phase12.results --split {res['split']}`. "
        f"{res['n_prompts']} prompts ({res['n_groups']} source groups) in the main slice"
        + (f", {res['n_real']} in the real-prompt slice" if res["n_real"] else "")
        + f"; targets {', '.join(ts)}. Every cell: estimate [95% CI], cluster bootstrap over "
        f"source groups, {res['bootstrap']['n_resamples']} resamples, seed "
        f"{res['bootstrap']['seed']}. Savings on each target's tokenizer, 8.3.7 safeguard on"
        " (fallbacks 0%); preservation from `label_v1` labels (served original = preserved);"
        " net savings = savings minus rewriter cost as a share of the prompt's target cost"
        " (`terser.serving.accounting.net_savings`, prices in `configs/serving/default.yaml`)."
        " Definitions: module docstring of `terser.phase12.results`.",
    ]
    if res["split"] != "test":
        L += [
            "",
            f"> **Dry run on {res['split']}.** Not a Phase 12 result: the test split is locked"
            " until the row 12.1 unlock. Systems built for Phase 12 have no judge labels on"
            " this split yet, so their preservation is n/a and they are marked unlabelled.",
        ]
    L += ["", "## 12.2 Master table", ""]
    L += _md_table(_headline_header(targets), _headline_rows(main_s, targets))
    L += [
        "",
        "Preservation cells: † = some served rewrites have no judge label yet, so the rate is"
        f" over the labelled pairs only; n/a = more than {PARTIAL_MAX:.0%} unlabelled. Such"
        " systems are kept out of the McNemar ranking.",
        "",
        "Gates: "
        + "; ".join(
            f"`{k}` = {v.get('model')} (threshold {v.get('threshold'):.4g}, {v.get('rule')})"
            for k, v in res["gates"].items()
        )
        + ".",
        "",
        "Latency notes: "
        + "; ".join(f"{s['name']}: {s['latency_note']}" for s in main_s if s["latency_note"])
        + ".",
    ]

    full = res.get("full_system")
    if full and full in res["criteria"]:
        c = res["criteria"][full]
        fpt = res["by_name"][full]["per_target"]
        L += ["", f"## Headline criteria for {full}", ""]
        L += _md_table(
            ["Criterion", "Target", "Value", "Verdict"],
            [
                [
                    "S1 median savings per target",
                    ">= 30%, CI low >= 25%",
                    ", ".join(f"{_short(t)} {_cip(fpt[t]['median_savings'])}" for t in targets),
                    "pass" if c["S1"]["pass"] else "fail",
                ],
                ["S2 preservation overall", ">= 95%", _pct(c["S2"]["value"]), _pv(c["S2"])],
                ["S3 worst category", ">= 90%", _pct(c["S3"]["value"]), _pv(c["S3"])],
                [
                    "S4 net savings at p25 length",
                    "> 0 per target",
                    ", ".join(
                        f"{_short(t)} "
                        f"{_cip(res['by_name'][full]['per_target'][t]['net_savings_p25_length'])}"
                        for t in targets
                    ),
                    _pv(c["S4"]),
                ],
                [
                    "S5 S1-S3 per target",
                    "each target",
                    ", ".join(
                        f"{_short(t)} {'yes' if v else 'no'}"
                        for t, v in c["S5"]["per_target"].items()
                    ),
                    _pv(c["S5"]),
                ],
            ],
        )
        if not c["complete"]:
            L += ["", "Incomplete labels: the verdicts above are provisional."]

    # 12.3 strata
    L += ["", "## 12.3 Stratified: target x category", ""]
    cats = res["categories"]
    rows = []
    for s in main_s:
        st = res["strata"]["category"][s["name"]]
        for t in targets:
            rows.append([s["name"], _short(t)] + [_cell(st[t], c) for c in cats])
    L += _md_table(["System", "Target", *cats], rows)
    L += ["", "Cells: preserved / median savings."]

    cl = res.get("clusters")
    if cl:
        hr = cl["slice_clusters"]
        L += [
            "",
            "## 12.3 Stratified: target x Phase 6 cluster",
            "",
            f"K-means k = {cl['k']} refit on the train embeddings (Phase 6 recipe; agreement with"
            f" the stored Phase 6 cluster x category table {_pct(cl['refit_agreement'])}); each"
            " prompt goes to its nearest centroid (bge-small). "
            + (
                "High-risk clusters (C6.3: degradation >= 1.5x overall): " + ", ".join(map(str, hr))
                if cl["any_high_risk"]
                else "No cluster met the C6.3 high-risk bar (>= 1.5x), so the slice is the"
                f" highest-risk cluster(s) by train risk ratio: {', '.join(map(str, hr))}."
            ),
            "",
        ]
        ids = sorted(cl["risk"], key=_natural)
        rows = []
        for s in main_s:
            st = res["strata"]["cluster"][s["name"]]
            for t in targets:
                rows.append([s["name"], _short(t)] + [_cell(st[t], c) for c in ids])
        L += _md_table(
            ["System", "Target"]
            + [
                f"c{c} (risk {cl['risk'][c]:.2f}{', HR' if c in map(str, hr) else ''})" for c in ids
            ],
            rows,
        )
        L += ["", "### High-risk slice x category", ""]
        rows = []
        for s in main_s:
            st = res["strata"]["high_risk_category"][s["name"]]
            for t in targets:
                rows.append([s["name"], _short(t)] + [_cell(st[t], c, with_n=True) for c in cats])
        L += _md_table(["System", "Target", *cats], rows)

    L += ["", "## 12.3 Real-prompt slice (row 2.1.8)", ""]
    if res.get("real"):
        L += _md_table(_headline_header(targets), _headline_rows(res["real"], targets))
        L += ["", "S1-S3 on the real slice are reported, not gated."]
    else:
        L += ["Not in this run: the real-prompt slice is hash-locked with the test split (12.1)."]

    # 12.4
    L += ["", "## 12.4 McNemar between the top systems", ""]
    if res["mcnemar"]:
        L += [
            "Top systems: complete systems ranked by the Phase 9 reward (mean over (prompt, target)"
            " of `100 x savings - 100 x 1[degraded]`), plus the full system. Exact McNemar on"
            " preserved flags of the same (prompt, target) pairs; Δ = A - B with a paired"
            " bootstrap CI.",
            "",
        ]
        L += _md_table(
            [
                "A",
                "B",
                "pairs",
                "only A preserved",
                "only B preserved",
                "p",
                "Δ preserved",
                "Δ median savings",
            ],
            [
                [
                    m["a"],
                    m["b"],
                    str(m["n_pairs"]),
                    str(m["only_a"]),
                    str(m["only_b"]),
                    f"{m['p_value']:.3g}",
                    _cip_pp(m["diff_preserved"]),
                    _cip_pp(m["diff_median_savings"]),
                ]
                for m in res["mcnemar"]
            ],
        )
    else:
        L += ["No two complete systems besides the original; McNemar needs labelled systems."]
    L += ["", "### Seeds", ""]
    if res["seeds"]:
        L += _md_table(
            [
                "System",
                "seeds",
                f"median savings {ts[0]} (mean ± sd)",
                "preserved pooled (mean ± sd)",
            ],
            [
                [
                    k,
                    str(v["n_seeds"]),
                    f"{_pct(v['savings_mean'])} ± {_pct(v['savings_sd'])}",
                    f"{_pct(v['preserved_mean'])} ± {_pct(v['preserved_sd'])}",
                ]
                for k, v in res["seeds"].items()
            ],
        )
        L += [""]
    L += [res["seeds_note"]]

    # 12.6
    L += ["", "## 12.6 Ablation matrix", ""]
    abl = [res["by_name"][n] for n in res["ablation_order"] if n in res["by_name"]]
    if abl:
        ref = abl[0]
        rows = []
        for s in abl:
            pt, rp = s["per_target"], ref["per_target"]
            rows.append(
                [
                    s["name"],
                    *(_cip(pt[t]["median_savings"]) for t in targets),
                    _cip(s["preserved_pooled"]),
                    *(_cip(pt[t]["mean_net_savings"]) for t in targets),
                    *(
                        _dpp(
                            pt[t]["median_savings"]["estimate"], rp[t]["median_savings"]["estimate"]
                        )
                        for t in targets
                    ),
                    _dpp(s["preserved_pooled"]["estimate"], ref["preserved_pooled"]["estimate"]),
                    "complete" if s["complete"] else f"{s['missing_labels']} unlabelled",
                ]
            )
        L += _md_table(
            [
                "Variant",
                *(f"Median savings {t}" for t in ts),
                "Preserved pooled",
                *(f"Mean net {t}" for t in ts),
                *(f"Δ savings {t}" for t in ts),
                "Δ preserved",
                "Labels",
            ],
            rows,
        )
        L += [
            "",
            "Rows: the full system, then one change each. − gate: no quality gate. − fallback:"
            " the raw rewrite is served even when it is not shorter than the original (8.3.7"
            " safeguard off; an empty rewrite still sends the original), still gated. − HPO:"
            " the gate refit with the untuned LightGBM parameters of row 10.5 and its own"
            " row 3.2.3 threshold. Operator-only: the generative rewriter replaced by the best"
            " operator stack, same gate and fallback.",
        ]
    else:
        L += ["No full system given (`--full-system`)."]

    # 12.7
    L += ["", "## 12.7 Net savings by prompt length", ""]
    fig = res.get("figure")
    if fig:
        L += [f"![net savings by prompt length]({fig})", ""]
    lt = res["length_table"]
    if lt:
        L += _md_table(
            ["System", "Target", *(f"p{10 * b}-{10 * b + 10}" for b in range(N_LENGTH_BINS))],
            [
                [name, _short(t), *(_pct(v, 0) for v in vals)]
                for name, by_t in lt.items()
                for t, vals in by_t.items()
            ],
        )
        L += [
            "",
            "Cells: mean net savings per prompt-length decile (original tokens on the target);"
            " S4 reads the p20-30 decile (the 25th percentile).",
        ]
    L += [""]
    return "\n".join(L)


def _cell(st: Mapping[str, Mapping[str, Any]], c: str, with_n: bool = False) -> str:
    if c not in st:
        return "—"
    v = st[c]
    out = f"{_pct(v['preserved'], 0)} / {_pct(v['median_savings'], 0)}"
    return out + (f" (n {v['n']})" if with_n else "")


def _pv(c: Mapping[str, Any]) -> str:
    return "pass" if c["pass"] else "fail"


def _dpp(a: float | None, b: float | None) -> str:
    if a is None or b is None:
        return "n/a"
    return f"{100 * (a - b):+.1f} pp"


def _cip_pp(d: Mapping[str, float | None]) -> str:
    if d.get("estimate") is None:
        return "n/a"
    lo, hi = d.get("low"), d.get("high")
    tail = "" if lo is None or hi is None else f" [{100 * lo:+.1f}, {100 * hi:+.1f}]"
    est = d.get("estimate")
    return "n/a" if est is None else f"{100 * est:+.1f} pp{tail}"


def _gallery(
    res: SystemResult,
    prompts: Sequence[Mapping[str, Any]],
    responses: Mapping[tuple[str, str, str], str],
    split: str,
) -> str:
    def trunc(s: str | None, n: int) -> str:
        if s is None:
            return "_(no response file entry)_"
        s = s.strip()
        return s if len(s) <= n else s[:n] + " …"

    def block(i: int, t: str) -> list[str]:
        p = prompts[i]
        served = res.served[t][i]
        lab = res.label[t][i] or {}
        orig_out = responses.get((str(p["id"]), text_sha(str(p["prompt"])), t))
        rew_out = responses.get((str(p["id"]), text_sha(served), t))
        agg = lab.get("aggregate") or {}
        return [
            f"### `{p['id']}` ({p['category']}, {_short(t)}): savings "
            f"{_pct(float(res.savings[t][i]))}, {lab.get('label', 'n/a')}"
            + (
                f", judge margin {agg['margin']:.2f}"
                if isinstance(agg.get("margin"), float)
                else ""
            )
            + (f", reason `{lab['reason']}`" if lab.get("reason") else ""),
            "",
            "**Casual prompt**",
            "",
            "```text",
            trunc(str(p["prompt"]), 700),
            "```",
            "",
            "**Rewrite**",
            "",
            "```text",
            trunc(served, 700),
            "```",
            "",
            "**Output on the original** / **output on the rewrite** (sample 0)",
            "",
            "```text",
            trunc(orig_out, 500),
            "```",
            "",
            "```text",
            trunc(rew_out, 500),
            "```",
            "",
        ]

    best, worst = [], []
    for t in res.savings:
        for i in range(len(prompts)):
            if res.served[t][i] == prompts[i]["prompt"]:
                continue
            pr = res.preserved[t][i]
            lab = res.label[t][i] or {}
            margin = (lab.get("aggregate") or {}).get("margin") or 0.0
            if pr == 1:
                best.append((-float(res.savings[t][i]), str(prompts[i]["id"]), i, t))
            elif pr == 0:
                worst.append(
                    (-float(margin), -float(res.savings[t][i]), str(prompts[i]["id"]), i, t)
                )
    best.sort()
    worst.sort()

    def pick(rows: Sequence[tuple[Any, ...]]) -> list[tuple[int, str]]:
        seen: set[str] = set()
        out = []
        for r in rows:
            pid, i, t = r[-3], r[-2], r[-1]
            if pid not in seen:
                seen.add(pid)
                out.append((i, t))
            if len(out) == GALLERY_N:
                break
        return out

    L = [
        f"# Qualitative gallery ({split} split) — row 12.9",
        "",
        f"System: **{res.system.name}**. Best = preserved rewrites with the largest savings;"
        " worst = degraded rewrites with the largest judge margin for the original. One entry"
        " per prompt; outputs are the target's sample 0 from the respond files.",
    ]
    if split != "test":
        L += ["", f"> Dry run on {split}; the Phase 12 gallery is built from the test split."]
    L += ["", "## Best", ""]
    for i, t in pick(best):
        L += block(i, t)
    if not best:
        L += ["No preserved rewrite.", ""]
    L += ["## Worst", ""]
    for i, t in pick(worst):
        L += block(i, t)
    if not worst:
        L += ["No degraded rewrite.", ""]
    return "\n".join(L)


def _figure(
    lt: Mapping[str, Mapping[str, Sequence[float | None]]],
    lt_ci: Mapping[str, Mapping[str, Sequence[tuple[float | None, float | None]]]],
    path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
    targets = sorted({t for v in lt.values() for t in v})
    fig, axes = plt.subplots(1, len(targets), figsize=(5.6 * len(targets), 4.0), sharey=True)
    axes = np.atleast_1d(axes)
    x = np.arange(N_LENGTH_BINS) * 10 + 5
    for ax, t in zip(axes, targets, strict=True):
        for k, (name, by_t) in enumerate(lt.items()):
            if t not in by_t:
                continue
            y = np.array([np.nan if v is None else 100 * v for v in by_t[t]])
            lo = np.array([np.nan if a is None else 100 * a for a, _ in lt_ci[name][t]])
            hi = np.array([np.nan if b is None else 100 * b for _, b in lt_ci[name][t]])
            c = palette[k % len(palette)]
            ax.fill_between(x, lo, hi, color=c, alpha=0.12, linewidth=0)
            ax.plot(x, y, color=c, linewidth=2, marker="o", markersize=4, label=name)
        ax.set_ylim(Y_FLOOR, Y_CEIL)  # a few very negative short-prompt means would hide the rest
        ax.axhline(0, color="#52514e", linewidth=1)
        ax.axvline(25, color="#8a8984", linewidth=1, linestyle=":")
        ax.set_title(_short(t), fontsize=11, color="#0b0b0b")
        ax.set_xlabel("prompt length percentile (original tokens)", color="#52514e")
        ax.grid(axis="y", color="#e6e5e1", linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("mean net savings (% of prompt cost)", color="#52514e")
    axes[-1].legend(fontsize=8, frameon=False, loc="best")
    fig.suptitle(
        "Net savings by prompt length (row 12.7); dotted line = p25 (S4); "
        f"y clipped to [{Y_FLOOR:g}%, {Y_CEIL:g}%], values in the table",
        fontsize=11,
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor="#fcfcfb")
    plt.close(fig)


# --------------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------------


def run(
    *,
    split: str,
    prompts: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
    count: Callable[[str, str], int],
    pricing: Any,
    gates: GateBank | None,
    full_system: str | None = None,
    sft: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    meta: Mapping[str, Mapping[str, Any]] | None = None,
    gen_runs: Mapping[str, Mapping[str, Any]] | None = None,
    responses: Mapping[tuple[str, str, str], str] | None = None,
    clusters: Mapping[str, Any] | None = None,
    systems: Sequence[System] | None = None,
    n_resamples: int = N_RESAMPLES,
    seed: int = SEED,
    top_k: int = 3,
    out_dir: Path | None = None,
    figure_name: str = "final_net_savings_by_length.png",
) -> dict[str, Any]:
    """Evaluate every system and (with ``out_dir``) write the reports; returns the JSON."""
    targets = list(TARGETS)
    sft = dict(sft or {})
    meta = dict(meta or {})
    gen_runs = dict(gen_runs or {})
    pids = {str(p["id"]) for p in prompts}
    cands: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    text_conflicts = 0  # same candidate id, different text in two input files (later wins)
    for c in candidates:
        if str(c["prompt_id"]) in pids:
            old = cands[str(c["method"])].get(str(c["prompt_id"]))
            text_conflicts += old is not None and str(old["text"]) != str(c["text"])
            cands[str(c["method"])][str(c["prompt_id"])] = dict(c)
    sft_latency: dict[str, float] = {}
    for tag, gen_rows in sft.items():
        lats: list[float] = []
        for row in gen_rows:
            pid = str(row["id"])
            if pid not in pids:
                continue
            raw = str(row.get("raw_rewrite", row.get("rewrite", "")))
            cands[f"{tag}_raw"][pid] = {
                "prompt_id": pid,
                "candidate_id": f"{pid}:{tag}_raw",
                "method": f"{tag}_raw",
                "text": raw if raw.strip() else str(row["prompt"]),
                "rewriter_tokens_in": count(str(row["prompt"]), TOKEN_TARGET) + 1,
                "rewriter_tokens_out": count(raw, TOKEN_TARGET) + 1,
                "latency_s": row.get("latency_s"),
            }
            if row.get("latency_s") is not None:
                lats.append(float(row["latency_s"]))
        if lats:
            sft_latency[tag] = float(np.mean(lats))
    cand_text = {
        str(c["candidate_id"]): (str(c["prompt_id"]), str(c["text"]))
        for by in cands.values()
        for c in by.values()
    }
    index = LabelIndex([r for r in labels if str(r["prompt_id"]) in pids], cand_text)
    systems = list(systems or default_systems(sorted(cands), sorted(sft), full_system))

    main_idx = np.asarray(
        [i for i, p in enumerate(prompts) if p.get("slice", MAIN_SLICE) != REAL_SLICE], np.int64
    )
    real_idx = np.asarray(
        [i for i, p in enumerate(prompts) if p.get("slice") == REAL_SLICE], np.int64
    )
    results: list[SystemResult] = []
    for sys_ in systems:
        ev = evaluate(sys_, prompts, cands, index, count, pricing, gates, targets)
        ev.latency_s, ev.latency_note = _latency(sys_, meta, sft_latency, gen_runs, cands)
        results.append(ev)
    # Gate time is added only after every system is scored, so its mean does not depend on the
    # evaluation order; it is weighted by the share of requests the gate actually saw (the
    # rewrites that did not already fall back).
    for ev in results:
        sys_ = ev.system
        if not (sys_.gate and gates is not None and ev.latency_s is not None):
            continue
        g = gates.mean_seconds(sys_.gate)
        if g is None:
            continue
        seen = [float(np.mean(~fb)) for fb in ev.fell_back.values() if len(fb)]
        share = float(np.mean(seen)) if seen else 1.0
        ev.latency_s += g * share
        ev.latency_note = (ev.latency_note + "; " if ev.latency_note else "") + (
            f"+ gate {_sec(g)} × {share:.0%} of requests"
        )
    if gates is not None:
        gates.save()

    def block(idx: npt.NDArray[np.int64]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        sub = [prompts[i] for i in idx]
        cats = np.asarray([str(p["category"]) for p in sub])
        groups = [group_of(p) for p in sub]
        rs = group_resamples(groups, n_resamples, seed)
        lengths = {
            t: np.asarray([count(str(p["prompt"]), short_target(t)) for p in sub], float)
            for t in targets
        }
        sums = [summarise(_subset(r, idx), cats, lengths, rs) for r in results]
        return sums, {"cats": cats, "groups": groups, "rs": rs, "lengths": lengths, "sub": sub}

    summaries, ctx = block(main_idx)
    by_name = {s["name"]: s for s in summaries}
    sub_results = {r.system.name: _subset(r, main_idx) for r in results}
    out: dict[str, Any] = {
        "split": split,
        "targets": targets,
        "n_prompts": int(len(main_idx)),
        "n_groups": len(set(ctx["groups"])),
        "n_real": int(len(real_idx)),
        "bootstrap": {"n_resamples": n_resamples, "seed": seed, "unit": "source group"},
        "gates": dict(gates.header) if gates is not None else {},
        "label_index": {
            "conflicts": index.conflicts,
            "unmatched": index.unmatched,
            "candidate_text_conflicts": int(text_conflicts),
        },
        "categories": sorted(set(ctx["cats"].tolist())),
        "systems": summaries,
        "by_name": by_name,
    }
    full_name = next((s.name for s in systems if s.name.startswith("Terser full")), None)
    out["full_system"] = full_name
    out["criteria"] = {s["name"]: criteria(s) for s in summaries if s["kind"] != "seed"}

    # strata
    out["strata"] = {"category": {}, "cluster": {}, "high_risk_category": {}}
    for name, sr in sub_results.items():
        if sr.system.kind != "main":
            continue
        out["strata"]["category"][name] = strata(sr, ctx["cats"].tolist())
    if clusters:
        assign = clusters["assign"]
        keys = [str(assign.get(str(p["id"]), "?")) for p in ctx["sub"]]
        hr = {str(c) for c in clusters["slice_clusters"]}
        mask = np.asarray([k in hr for k in keys])
        hr_idx = np.flatnonzero(mask).astype(np.int64)
        for name, sr in sub_results.items():
            if sr.system.kind != "main":
                continue
            out["strata"]["cluster"][name] = strata(sr, keys)
            out["strata"]["high_risk_category"][name] = strata(
                _subset(sr, hr_idx), ctx["cats"][hr_idx].tolist()
            )
        out["clusters"] = {k: v for k, v in clusters.items() if k != "assign"}
        out["clusters"]["n_in_slice"] = int(mask.sum())

    if len(real_idx):
        real_sums, _ = block(real_idx)
        out["real"] = [s for s in real_sums if s["kind"] == "main"]
        out["real_criteria"] = {s["name"]: criteria(s) for s in out["real"]}

    # 12.4 McNemar among the top systems
    ranked = sorted(
        (s for s in summaries if s["kind"] == "main" and s["method"] and s["complete"]),
        key=lambda s: -(s["reward"] if s["reward"] is not None else -1e9),
    )
    top = [s["name"] for s in ranked[:top_k]]
    if full_name and by_name[full_name]["complete"] and full_name not in top:
        top.append(full_name)
    out["top_systems"] = top
    mc = []
    rs = ctx["rs"]
    for i, a in enumerate(top):
        for b in top[i + 1 :]:
            ra, rb = sub_results[a], sub_results[b]
            pa = np.concatenate([ra.preserved[t] for t in targets])
            pb = np.concatenate([rb.preserved[t] for t in targets])
            n = len(ctx["sub"])
            m = mcnemar_paired(pa, pb)

            def dpres(
                ix: npt.NDArray[np.int64], ra: SystemResult = ra, rb: SystemResult = rb
            ) -> float:
                return nanmean(_pooled(ra, ix)) - nanmean(_pooled(rb, ix))

            def dsav(
                ix: npt.NDArray[np.int64], ra: SystemResult = ra, rb: SystemResult = rb
            ) -> float:
                sa = np.concatenate([ra.savings[t][ix] for t in targets])
                sb = np.concatenate([rb.savings[t][ix] for t in targets])
                return nanmedian(sa) - nanmedian(sb)

            mc.append(
                {
                    "a": a,
                    "b": b,
                    **m,
                    "diff_preserved": ci(dpres, n, rs).as_dict(),
                    "diff_median_savings": ci(dsav, n, rs).as_dict(),
                }
            )
    out["mcnemar"] = mc

    # seeds
    seeds: dict[str, Any] = {}
    seed_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for summ in summaries:
        mm = SEED_RE.match(summ["method"]) if summ["kind"] == "seed" else None
        base = mm.group("base") if mm else summ["method"]
        if summ["kind"] in ("main", "seed") and summ["method"]:
            # one group = one system up to its seed (same gate, fallback and accounting path)
            seed_groups[f"{base}|{summ['gate']}|{summ['fallback']}|{summ['path']}"].append(summ)
    t0 = targets[0]
    for ss in seed_groups.values():
        if len(ss) < 2:
            continue
        sv = [x["per_target"][t0]["median_savings"]["estimate"] for x in ss]
        pv = [x["preserved_pooled"]["estimate"] for x in ss]
        sv_f = [v for v in sv if v is not None]
        pv_f = [v for v in pv if v is not None]
        seeds[ss[0]["name"]] = {
            "n_seeds": len(ss),
            "methods": [x["method"] for x in ss],
            "savings_mean": float(np.mean(sv_f)) if sv_f else None,
            "savings_sd": float(np.std(sv_f, ddof=1)) if len(sv_f) > 1 else None,
            "preserved_mean": float(np.mean(pv_f)) if pv_f else None,
            "preserved_sd": float(np.std(pv_f, ddof=1)) if len(pv_f) > 1 else None,
        }
    out["seeds"] = seeds
    out["seeds_note"] = SEEDS_NOTE

    # ablations
    if full_name:
        out["ablation_order"] = [full_name] + [s.name for s in systems if s.kind == "ablation"]
    else:
        out["ablation_order"] = []

    # 12.7 length table + figure
    key = [
        s["name"]
        for s in summaries
        if s["kind"] == "main"
        and s["method"]
        and s["name"] in _key_systems(summaries, full_name, top)
    ]
    lt: dict[str, dict[str, list[float | None]]] = {}
    lt_ci: dict[str, dict[str, list[tuple[float | None, float | None]]]] = {}
    for name in key:
        sr = sub_results[name]
        lt[name], lt_ci[name] = {}, {}
        for t in targets:
            bins = length_bins(ctx["lengths"][t])
            vals: list[float | None] = []
            cis: list[tuple[float | None, float | None]] = []
            for b in range(N_LENGTH_BINS):
                band = ci(_bin_mean(sr.net[t], bins, b), len(ctx["sub"]), rs)
                vals.append(band.estimate)
                cis.append((band.low, band.high))
            lt[name][t], lt_ci[name][t] = vals, cis
    out["length_table"] = lt
    out["length_table_ci"] = lt_ci

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        if lt:
            fig_path = out_dir / "figures" / figure_name
            _figure(lt, lt_ci, fig_path)
            out["figure"] = f"figures/{figure_name}"
        (out_dir / "final_results.md").write_text(build_report(out), encoding="utf-8")
        (out_dir / "final_results.json").write_text(
            json.dumps(_jsonable({k: v for k, v in out.items() if k != "by_name"}), indent=2)
            + "\n",
            encoding="utf-8",
        )
        gal_name = full_name or (top[0] if top else None)
        if gal_name:
            (out_dir / "gallery.md").write_text(
                _gallery(sub_results[gal_name], ctx["sub"], responses or {}, split) + "\n",
                encoding="utf-8",
            )
    return out


SEEDS_NOTE = (
    "Seeds (row 12.4, >= 3 seeds): deterministic systems need none (original, operators, HMM,"
    " LLMLingua-2, deployed fast path, LightGBM tagger and both gates at a fixed random_state;"
    " greedy decoding of the SFT and zero-shot rewriters). Seeded CPU systems get seed variants"
    " from `terser.phase12.candidates build --textcnn-seeds 0,1,2 --rl-seeds 0,1,2`"
    " (methods `<method>_s<k>`, judged like any candidate, summarised above). The SFT rewriter's"
    " seed is its training seed: the main session retrains the chosen rung with"
    " `python -m terser.models.rewriter.train experiment=<rung> seed=<k>` on the GPU for k = 1, 2,"
    " generates with `terser.models.rewriter.generate`, imports each as tag `<rung>_s<k>`"
    " (`terser.phase12.candidates import-sft`) and passes `--sft <rung>_s<k>=<file>`. Judge"
    " sampling is greedy; creative prompts use the three fixed respond seeds (common random"
    " numbers). Bootstrap CIs cover prompt sampling, the seed table covers training variance."
)


def _key_systems(
    summaries: Sequence[Mapping[str, Any]], full: str | None, top: Sequence[str]
) -> list[str]:
    """Figure lines: full system, its ungated SFT rung, top systems, then the CPU baselines."""
    full_method = next((s["method"] for s in summaries if s["name"] == full), None)
    names = [full] if full else []
    names += [
        s["name"]
        for s in summaries
        if s["kind"] == "main" and s["method"] == full_method and s["name"] != full
    ]
    names += list(top)
    names += [
        s["name"]
        for s in summaries
        if s["kind"] == "main" and s["method"] in ("ops_FHC", "fast_path", "llmlingua2")
    ]
    out: list[str] = []
    for n in names:
        if n and n not in out:
            out.append(n)
    return out[:6]


def _latency(
    s: System,
    meta: Mapping[str, Mapping[str, Any]],
    sft_latency: Mapping[str, float],
    gen_runs: Mapping[str, Mapping[str, Any]],
    cands: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> tuple[float | None, str]:
    if not s.method:
        return 0.0, ""
    base = s.method[: -len("_raw")] if s.method.endswith("_raw") else s.method
    if base in meta:
        m = meta[base]
        return m.get("latency_mean_s"), str(m.get("latency_note", ""))
    if base in sft_latency:
        return sft_latency[base], "HF generate, batch 16, per prompt"
    lat = [
        float(c["latency_s"])
        for c in cands.get(base, {}).values()
        if c.get("latency_s") is not None
    ]
    if lat:
        return float(np.mean(lat)), "HF generate, batch 16, per prompt"
    if base in gen_runs:
        return float(gen_runs[base]["s_per_prompt"]), "vLLM run wall time / prompts written"
    return None, ""


# --------------------------------------------------------------------------------------------
# Clusters (Phase 6)
# --------------------------------------------------------------------------------------------


def phase6_clusters(
    prompts: Sequence[Mapping[str, Any]],
    embed: Callable[[Sequence[str]], Any],
    *,
    embeddings_dir: Path = Path("data/processed/embeddings"),
    name: str = "train_bge_small",
    train: Path = Path("data/processed/splits/train.jsonl"),
    p6_results: Path = Path("reports/p6_results.json"),
    seed: int = 0,
) -> dict[str, Any]:
    """Refit the Phase 6 K-means, assign each prompt to its nearest centroid, attach risk."""
    from terser.features.embeddings import EmbeddingCache
    from terser.models.clustering.scratch_kmeans import KMeans

    saved = json.loads(p6_results.read_text(encoding="utf-8"))
    k = int(saved["chosen_k"])
    cache = EmbeddingCache.load(embeddings_dir, name)
    train_rows = {str(r["id"]): r for r in read_jsonl(train)}
    ids = [i for i in cache.ids if i in train_rows]
    X = cache.get(ids).astype(np.float64)
    km = KMeans(k, n_init=10, seed=seed).fit(X)
    stored = saved["categories"]["contingency"]  # {category: {cluster: n}}
    refit: Counter[tuple[str, str]] = Counter(
        (str(train_rows[i]["category"]), str(int(c))) for i, c in zip(ids, km.labels_, strict=True)
    )
    agree = sum(
        min(int(n), refit[(cat, str(c))]) for cat, d in stored.items() for c, n in d.items()
    )
    total = sum(int(n) for d in stored.values() for n in d.values())
    vecs = np.asarray(embed([str(p["prompt"]) for p in prompts]), dtype=np.float64)
    lab = km.predict(vecs) if len(prompts) else np.zeros(0, dtype=np.int64)
    risk = {str(r["cluster"]): float(r["risk_ratio"]) for r in saved["risk"]}
    high = [str(r["cluster"]) for r in saved["risk"] if r.get("high_risk")]
    top = max(risk.values()) if risk else 0.0
    slice_ = high or [c for c, v in risk.items() if v == top]
    return {
        "k": k,
        "assign": {str(p["id"]): int(c) for p, c in zip(prompts, lab, strict=True)},
        "risk": risk,
        "any_high_risk": bool(high),
        "slice_clusters": slice_,
        "refit_agreement": agree / total if total else None,
    }


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def _responses(paths: Sequence[Path], pids: set[str]) -> dict[tuple[str, str, str], str]:
    out: dict[tuple[str, str, str], str] = {}
    for path in paths:
        for r in read_jsonl(path, keep=lambda r: str(r.get("prompt_id")) in pids):
            if int(r.get("sample", 0) or 0) == 0:
                out.setdefault(
                    (str(r["prompt_id"]), str(r["text_sha256"]), str(r["target"])), str(r["output"])
                )
    return out


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(prog="python -m terser.phase12.results", description=__doc__)
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument(
        "--candidates",
        type=Path,
        nargs="+",
        required=True,
        help="candidate files; on a repeated candidate id the later file wins, so pass the "
        "files that were judged last (conflicts are counted in the JSON)",
    )
    ap.add_argument("--labels", type=Path, nargs="+", required=True)
    ap.add_argument("--responses", type=Path, nargs="*", default=[])
    ap.add_argument("--meta", type=Path, nargs="*", default=[])
    ap.add_argument("--sft", action="append", default=[], help="tag=generate output (jsonl)")
    ap.add_argument("--full-system", default=None, help="SFT tag of the full Terser system")
    ap.add_argument("--gen-logs", type=Path, nargs="*", default=[])
    ap.add_argument("--pairs", type=Path, default=Path("data/processed/p4_pairs.parquet"))
    ap.add_argument("--pricing", type=Path, default=Path("configs/serving/default.yaml"))
    ap.add_argument("--no-clusters", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=N_RESAMPLES)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--out-dir", type=Path, default=Path("reports"))
    ap.add_argument("--gate-cache", type=Path, default=None)
    args = ap.parse_args(argv)

    import pandas as pd
    import yaml

    from terser.data.splits import assert_no_test_access
    from terser.features.embeddings import bge_small_encoder
    from terser.features.span import gpt2_surprisal
    from terser.models.rewriter.baselines import parse_generation_logs
    from terser.phase10.hpo_study import UNTUNED_GBM
    from terser.phase12.gates import TUNED_GBM, OnnxFastGate, PairFeaturizer, PairGate
    from terser.serving.accounting import Pricing
    from terser.tokenizers.count import count_tokens

    for p in [args.prompts, *args.candidates, *args.labels, *args.responses]:
        assert_no_test_access(p)
    if args.split == "test":
        from terser.phase12.unlock import unlock_entry

        if unlock_entry() is None:
            raise SystemExit("test results need the row 12.1 unlock first")
    prompts = read_jsonl(args.prompts)
    if args.split == "test":
        # The --prompts file is an unlocked copy written by the candidate build; read the locked
        # split through the logged reader as well, so this run appears in the access log, and
        # refuse a copy that does not match it.
        from terser.phase12.candidates import load_split_prompts
        from terser.phase12.unlock import read_locked

        locked = load_split_prompts(
            "test", reader=lambda name, _p: read_locked(name, "row 12.2-12.7 evaluation/results")
        )
        key = lambda p: (str(p["id"]), str(p["prompt"]))  # noqa: E731
        if {key(p) for p in prompts} != {key(p) for p in locked}:
            raise SystemExit(f"{args.prompts} does not match the locked test split")
    pids = {str(p["id"]) for p in prompts}
    keep = lambda r: str(r.get("prompt_id")) in pids  # noqa: E731
    candidates = [c for path in args.candidates for c in read_jsonl(path, keep)]
    labels = [r for path in args.labels for r in read_jsonl(path, keep)]
    meta: dict[str, Any] = {}
    for path in args.meta:
        meta.update(json.loads(path.read_text(encoding="utf-8")).get("methods", {}))
    sft = {}
    for spec in args.sft:
        tag, _, path = spec.partition("=")
        sft[tag] = read_jsonl(Path(path), lambda r: str(r.get("id")) in pids)

    counts: dict[tuple[str, str], int] = {}

    def count(text: str, target: str) -> int:
        key = (text, target)
        if key not in counts:
            counts[key] = count_tokens(text, target)
        return counts[key]

    pricing = Pricing.from_config(
        yaml.safe_load(args.pricing.read_text(encoding="utf-8"))["pricing"]
    )
    table = pd.read_parquet(args.pairs)
    encode = bge_small_encoder()
    surprisal = gpt2_surprisal("gpt2")
    bank = GateBank(
        {
            "tuned": PairGate.fit(table, TUNED_GBM, "tuned"),
            "untuned": PairGate.fit(table, UNTUNED_GBM, "untuned"),
            "onnx": OnnxFastGate(cost=lambda t: count(t, TOKEN_TARGET)),
        },
        PairFeaturizer(surprisal, encode, lambda t: count(t, TOKEN_TARGET)),
        args.gate_cache or (args.out_dir / f".{args.split}_gate_cache.json"),
    )
    clusters = None if args.no_clusters else phase6_clusters(prompts, encode)
    res = run(
        split=args.split,
        prompts=prompts,
        candidates=candidates,
        labels=labels,
        count=count,
        pricing=pricing,
        gates=bank,
        full_system=args.full_system,
        sft=sft,
        meta=meta,
        gen_runs=parse_generation_logs(args.gen_logs),
        responses=_responses(args.responses, pids),
        clusters=clusters,
        n_resamples=args.bootstrap,
        seed=args.seed,
        top_k=args.top_k,
        out_dir=args.out_dir,
    )
    head = {
        s["name"]: {
            "median_savings": {
                _short(t): v["median_savings"]["estimate"] for t, v in s["per_target"].items()
            },
            "preserved": s["preserved_pooled"]["estimate"],
            "complete": s["complete"],
        }
        for s in res["systems"]
        if s["kind"] == "main"
    }
    print(json.dumps(head, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
