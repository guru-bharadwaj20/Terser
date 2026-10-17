"""Rewriter LoRA tuning (row **10.7**, checkpoint **C10.3**).

Tunes the SFT rewriter's LoRA (default base Qwen2.5-1.5B-Instruct on ``data/processed/sft_v2``,
the ``sft_qwen15b_lora_v2`` rung: 35.0% median savings at 57.0% preserved, C8.3 fail) for
**median savings subject to preservation >= 0.92** (the C8.3 bar, together with savings >= 25%).

Why two stages
--------------
A judged evaluation costs about 25 GPU minutes per model (swap the trainer out, serve both
targets for ``terser.phase2.respond``, serve the 32B judge for ``label_stage``), so it is too
expensive per trial. Stage 1 therefore optimises a **judge-free proxy**, and only the top-k
trials are judged in stage 2.

Search space (:data:`SPACE`)
----------------------------
``rank`` {8, 16, 32, 64}; ``alpha_mult`` {1, 2, 4} (``alpha = alpha_mult x rank``); ``lr``
log-uniform [5e-5, 5e-4]; ``warmup_frac`` [0, 0.1] of the steps, with linear warmup then linear
decay (``schedule: linear`` in :func:`terser.models.rewriter.train.train_sft`); ``epochs`` {1, 2,
3}, mapped to ``steps = ceil(epochs x n_train / batch_size)`` (1262 pairs, batch 4: 316 / 631 /
947 steps). Trial 0 is the untuned rung's settings inside the space (r 16, alpha 32, lr 2e-4,
no warmup, 3 epochs) so the study has its own reference point.

Stage 1 (``run``): Optuna TPE + MedianPruner
--------------------------------------------
:func:`terser.hpo.search.run_search` (row 10.1 conventions: ``TPESampler(seed, n_startup)``,
an MLflow parent run tagged phase 10 / row 10.7 with one nested child run per trial) with a
SQLite storage under ``experiments/p10_rewriter_hpo/`` so a crashed chain resumes. Each trial:

1. trains the LoRA in-process through :func:`terser.models.rewriter.train.run`; every
   ``eval_every`` steps the masked val loss on ``sft_v2/val.jsonl`` is reported to Optuna as
   ``-val_loss`` (the study maximises, and :class:`optuna.pruners.MedianPruner` prunes a trial
   whose intermediate value is below the median of earlier trials at the same step, so the
   sign must follow the study's direction). Pruning starts after ``--pruner-startup`` complete
   trials and ``--pruner-warmup`` steps;
2. generates greedy, safeguarded rewrites of the 285 val prompts
   (:func:`terser.models.rewriter.generate.generate_rows`, the CLI's code);
3. scores them with the preservation proxy (below) and returns

   ``J = s`` if ``p_hat >= 0.92`` and ``s >= 0.25`` (feasible), else
   ``J = -1 - [max(0, 0.92 - p_hat) + max(0, 0.25 - s)]``

   where ``s`` is the median savings and ``p_hat`` the proxy's preservation estimate. Every
   feasible trial ranks above every infeasible one, and infeasible trials rank by their distance
   to the C8.3 box. Savings alone would reward the 35%-savings / 57%-preserved behaviour of the
   untuned rung; preservation alone would reward a model that copies its input (a copied
   prompt falls back to the original and is preserved by construction, at 0% savings).

The preservation proxy
----------------------
Two judge-free candidates were measured against the 2,850 existing judge labels of the five
judged rungs (``val_labels_all_v2.jsonl``; ``python -m terser.phase10.rewriter_hpo proxy``):

* **constraint-keyword retention** (:class:`terser.serving.paths.ConstraintGate`): ROC-AUC
  about 0.48-0.50 against "degraded" on non-fallback rewrites, i.e. no signal. LLM rewrites keep
  the constraint keywords and lose content in other ways.
* **the trained quality gate** ``models/fast/gate.onnx`` (row 11.3.2; features from
  :mod:`terser.features.shared` on the Qwen2.5 tokenizer, as served): AUC about 0.58 (Qwen
  target) and 0.62 (Phi). This signal is weak, but it is the only one. Used as a hard gate at its
  serving operating point (0.264564) it rejects 93-96% of SFT rewrites and the gated median
  savings fall to 0, so it cannot be used as a filter.

The proxy is therefore the gate's score **calibrated to the judge**: ``p_hat`` is the share of
val prompts that fall back (preserved by construction) plus, for every other rewrite,
``sigmoid(a + b x gate_score)``. ``(a, b)`` is a logistic fit of the judge's "preserved" (per
target) on the gate score over the five rungs' non-fallback rewrites. A leave-one-rung-out check
(fit on four rungs, estimate the fifth rung's judged preservation) gives a rung-level mean
absolute error of about 2 pp (worst +5 pp, on the 1.5B v2 rung) and a Spearman rank correlation of
0.9 over the five rungs. The proxy can rank trials but cannot certify 0.92 per rewrite, so stage 2
judges the winners. ``proxy`` writes the fit and the agreement table to
``reports/p10_rewriter_hpo/proxy.json``; ``run`` refuses to start without it.

Measured fit (``proxy.json``, 1,840 labels): ``a = 1.006``, ``b = -2.100``, pooled gate AUC
0.598, constraint-retention AUC 0.488, LORO MAE 2.2 pp, max 5.2 pp, Spearman 0.90. Because the
signal is weak, the calibrated estimate shrinks towards the mean: a non-fallback rewrite scores
at most ``sigmoid(a) = 0.73``. So ``p_hat >= 0.92`` needs at least 70% fallbacks, which pins the
median savings at 0. Under the proxy no trial can be feasible, and stage 1 in practice minimises
the distance to the C8.3 box. Only the stage-2 judge can show a trial inside it.

Stage 2 (``candidates``, judge, ``finalize``)
---------------------------------------------
``candidates`` writes the top-k complete trials by ``J`` (default 3) as
``data/interim/rewrites/hpo<NN>.val.jsonl`` plus one candidate file
``val_candidates_hpo.jsonl`` in the ``val_candidates_v2.jsonl`` format (candidate ids
``<prompt id>:hpo<NN>``), which ``terser.phase2.respond`` + ``label_stage`` judge on both targets
(``/home/temp/hpo_chain.sh``). ``finalize`` computes each candidate's C8.3 metrics with
:func:`terser.models.rewriter.val_report.rung_metrics` and picks the trial with the most
savings among those with judged preservation >= 0.92, or, if none qualifies, the best-preserving
trial, labelled as such. It writes ``reports/p10_rewriter_hpo.md`` / ``.json`` with the C10.3
verdict, copies the chosen adapter to ``models/rewriter/sft-<base>-lora-hpo`` and registers it
as a new ``terser-rewriter`` version with lineage
(:func:`terser.phase11.register.register_one`). The version is promoted to Staging only if it
passes C8.3.

C10.3 ("still passes C8.3 with >= 1 pp more savings, or documented plateau"): **pass** if the
chosen trial passes C8.3 and, when the untuned rung on the same base passed C8.3 too, saves at
least 1 pp more than it. Otherwise the verdict is **plateau**, and the report documents the
judged savings / preservation of the top-k against the untuned rung.

Caveats: the proxy, the pruner's val loss and the stage-2 judgement all use the val split (as
C8.3 does), so the chosen trial's val numbers are optimistically biased by selection among k.
The MedianPruner compares trials at equal step counts, which slightly favours short schedules
(their learning rate has decayed further by a given step). Data access goes through
:class:`terser.hpo.audit.DataAccessLog` (``reports/p10_rewriter_hpo/data_access.jsonl``,
row 10.8); nothing reads the test split.

CLI::

    python -m terser.phase10.rewriter_hpo proxy
    python -m terser.phase10.rewriter_hpo run --trials 12 [--base Qwen/Qwen2.5-3B-Instruct]
    python -m terser.phase10.rewriter_hpo candidates --k 3
    # judge val_candidates_hpo.jsonl -> data/interim/rewrites/val_labels_hpo.jsonl
    python -m terser.phase10.rewriter_hpo finalize
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import shutil
import statistics
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from terser.hpo.audit import DataAccessLog, audit_access_log
from terser.hpo.search import Categorical, Float, SearchSpace, run_search
from terser.models.rewriter.val_report import MIN_PRESERVED, MIN_SAVINGS, rung_metrics

__all__ = [
    "DEFAULT_PARAMS",
    "SPACE",
    "Calibration",
    "GateScorer",
    "HpoSettings",
    "c10_3_verdict",
    "candidate_records",
    "choose_trial",
    "estimate_preservation",
    "fit_calibration",
    "main",
    "make_objective",
    "objective_value",
    "run_stage1",
    "select_topk",
    "steps_for",
    "trial_settings",
    "trial_tag",
]

DEFAULT_BASE = "Qwen/Qwen2.5-1.5B-Instruct"
BASE_EXPERIMENTS: Mapping[str, str] = {
    "Qwen/Qwen2.5-1.5B-Instruct": "sft_qwen15b_lora_v2",
    "Qwen/Qwen2.5-3B-Instruct": "sft_qwen3b_lora_v2",
    "Qwen/Qwen2.5-7B-Instruct": "sft_qwen7b_qlora_v2",
}
# The untuned rung on the same base and data (reports/p8_rewriter_val.json key), for C10.3.
BASELINE_RUNG: Mapping[str, str] = {
    "Qwen/Qwen2.5-1.5B-Instruct": "sft15v2",
    "Qwen/Qwen2.5-3B-Instruct": "sft3bv2",
    "Qwen/Qwen2.5-7B-Instruct": "sft7bv2",
}
REWRITES = Path("data/interim/rewrites")
JUDGED_RUNGS: Mapping[str, str] = {
    "sft05": "sft-qwen0.5b-full.val.jsonl",
    "sft15": "sft-qwen1.5b-lora.val.jsonl",
    "sft3b": "sft-qwen3b-lora.val.jsonl",
    "sft15v2": "sft15v2.val.jsonl",
    "sft3bv2": "sft3bv2.val.jsonl",
}
JUDGED_LABELS = REWRITES / "val_labels_all_v2.jsonl"
HPO_LABELS = REWRITES / "val_labels_hpo.jsonl"
HPO_CANDIDATES = REWRITES / "val_candidates_hpo.jsonl"
GATE_ONNX = Path("models/fast/gate.onnx")
GATE_TARGET = "qwen2.5"  # the gate was trained on Qwen2.5 token counts (configs/serving)
SERVING_CONFIG = Path("configs/serving/fast_onnx.yaml")
OUT = Path("reports/p10_rewriter_hpo")
WORK = Path("experiments/p10_rewriter_hpo")
REPORT = Path("reports/p10_rewriter_hpo.md")
STUDY_NAME = "p10-rewriter-lora"
TAG_PREFIX = "hpo"

SPACE = SearchSpace(
    {
        "rank": Categorical((8, 16, 32, 64)),
        "alpha_mult": Categorical((1, 2, 4)),
        "lr": Float(5e-5, 5e-4, log=True),
        "warmup_frac": Float(0.0, 0.1),
        "epochs": Categorical((1, 2, 3)),
    }
)
# The untuned rung's settings inside the space (r 16, alpha 32, lr 2e-4, 3 epochs, no warmup).
DEFAULT_PARAMS: dict[str, Any] = {
    "rank": 16,
    "alpha_mult": 2,
    "lr": 2e-4,
    "warmup_frac": 0.0,
    "epochs": 3,
}

Rows = list[dict[str, Any]]
Generator = Callable[[Path, Any, Rows], Rows]
Scorer = Callable[[Rows], list[float]]


def _jsonl(path: Path) -> Rows:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---- search-space mapping and objective ---------------------------------------------------------
def steps_for(epochs: int, n_train: int, batch_size: int) -> int:
    """Optimiser steps for ``epochs`` passes over ``n_train`` pairs at ``batch_size``."""
    if epochs < 1 or n_train < 1 or batch_size < 1:
        raise ValueError("epochs, n_train and batch_size must be >= 1")
    return math.ceil(epochs * n_train / batch_size)


def trial_settings(params: Mapping[str, Any], n_train: int, batch_size: int) -> dict[str, Any]:
    """Trainer settings of one point of :data:`SPACE`."""
    steps = steps_for(int(params["epochs"]), n_train, batch_size)
    rank = int(params["rank"])
    return {
        "rank": rank,
        "alpha": rank * int(params["alpha_mult"]),
        "lr": float(params["lr"]),
        "steps": steps,
        "warmup_steps": int(round(float(params["warmup_frac"]) * steps)),
        "epochs": int(params["epochs"]),
    }


def objective_value(
    savings: float,
    preserved: float,
    tau: float = MIN_PRESERVED,
    min_savings: float = MIN_SAVINGS,
) -> float:
    """``savings`` if inside the C8.3 box, else ``-1 -`` (distance to the box); see module doc."""
    gap = max(0.0, tau - preserved) + max(0.0, min_savings - savings)
    return float(savings) if gap == 0.0 else -1.0 - gap


def trial_tag(number: int) -> str:
    """Candidate tag of a trial (no ``:``, which separates prompt id and tag)."""
    return f"{TAG_PREFIX}{number:02d}"


# ---- preservation proxy -------------------------------------------------------------------------
@dataclass(frozen=True)
class Calibration:
    """``P(preserved | gate score) = sigmoid(a + b * score)``."""

    a: float
    b: float

    def __call__(self, scores: Sequence[float]) -> np.ndarray:
        z = self.a + self.b * np.asarray(scores, dtype=float)
        return np.asarray(1.0 / (1.0 + np.exp(-z)))


def fit_calibration(scores: Sequence[float], preserved: Sequence[bool]) -> Calibration:
    """Logistic fit of judged preservation on the gate score."""
    from sklearn.linear_model import LogisticRegression

    y = np.asarray(preserved, dtype=int)
    if len(set(y.tolist())) < 2:
        raise ValueError("calibration needs both preserved and degraded labels")
    m = LogisticRegression(C=1.0).fit(np.asarray(scores, dtype=float).reshape(-1, 1), y)
    return Calibration(float(m.intercept_[0]), float(m.coef_[0][0]))


def estimate_preservation(
    rewrites: Sequence[Mapping[str, Any]], scores: Sequence[float], cal: Calibration
) -> dict[str, Any]:
    """Proxy preservation: fallbacks count 1, other rewrites ``cal(gate score)``."""
    if len(rewrites) != len(scores):
        raise ValueError("one score per rewrite")
    probs = cal(scores)
    by_cat: dict[str, list[float]] = defaultdict(list)
    for w, p in zip(rewrites, probs, strict=True):
        by_cat[str(w["category"])].append(1.0 if w["fell_back"] else float(p))
    flat = [x for xs in by_cat.values() for x in xs]
    per_cat = {c: float(np.mean(xs)) for c, xs in sorted(by_cat.items())}
    return {
        "preserved": float(np.mean(flat)) if flat else 0.0,
        "by_category": per_cat,
        "min_category": min(per_cat.values()) if per_cat else 0.0,
    }


class GateScorer:
    """``P(degraded)`` of ``models/fast/gate.onnx`` per rewrite (0 for a fallback), as served."""

    def __init__(self, path: Path = GATE_ONNX, target: str = GATE_TARGET) -> None:
        from terser.serving.paths import OnnxGate

        self.gate = OnnxGate(path)
        self.target = target
        self._cache: dict[str, dict[str, float]] = {}

    def _features(self, text: str) -> dict[str, float]:
        from terser.features.shared import extract_features

        if text not in self._cache:
            self._cache[text] = extract_features(text, self.target)
        return self._cache[text]

    def __call__(self, rewrites: Rows) -> list[float]:
        from terser.features.shared import gate_vector

        out: list[float] = [0.0] * len(rewrites)
        idx, vecs = [], []
        for i, w in enumerate(rewrites):
            if w["fell_back"] or w["rewrite"] == w["prompt"]:
                continue
            idx.append(i)
            vecs.append(gate_vector(self._features(w["prompt"]), self._features(w["rewrite"])))
        for i, s in zip(idx, self.gate.score_vectors(vecs) if vecs else [], strict=True):
            out[i] = float(s)
        return out


def gate_threshold(path: Path = SERVING_CONFIG) -> float:
    """The served gate's row 3.2.3 operating point."""
    import yaml

    return float(yaml.safe_load(path.read_text(encoding="utf-8"))["gate_threshold"])


def _label_map(labels: Rows) -> dict[tuple[str, str, str], bool]:
    out: dict[tuple[str, str, str], bool] = {}
    for r in labels:
        pid, _, tag = r["candidate_id"].rpartition(":")
        out[(tag, pid, r["target"])] = r["label"] == "preserved"
    return out


def _pairs(
    rewrites: Rows, scores: Sequence[float], ok: Mapping[tuple[str, str, str], bool], tag: str
) -> tuple[list[float], list[bool]]:
    """(gate score, judged preserved) per non-fallback rewrite and target."""
    targets = sorted({t for (g, _, t) in ok if g == tag})
    xs: list[float] = []
    ys: list[bool] = []
    for w, s in zip(rewrites, scores, strict=True):
        if w["fell_back"]:
            continue
        for t in targets:
            if (tag, w["id"], t) in ok:
                xs.append(float(s))
                ys.append(ok[(tag, w["id"], t)])
    return xs, ys


def proxy_agreement(
    rungs: Mapping[str, Rows],
    labels: Rows,
    scores: Mapping[str, list[float]],
    constraint_scores: Mapping[str, list[float]] | None = None,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Calibration on all rungs plus the leave-one-rung-out agreement with the judge."""
    from scipy.stats import spearmanr

    from terser.eval.curves import roc_auc

    ok = _label_map(labels)
    pairs = {t: _pairs(rungs[t], scores[t], ok, t) for t in rungs}
    xs = [x for t in rungs for x in pairs[t][0]]
    ys = [y for t in rungs for y in pairs[t][1]]
    cal = fit_calibration(xs, ys)
    out: dict[str, Any] = {
        "calibration": {"a": cal.a, "b": cal.b, "n_labels": len(ys)},
        "gate_auc_degraded": roc_auc([0 if y else 1 for y in ys], xs),
        "rungs": {},
    }
    if constraint_scores is not None:
        cx = [x for t in rungs for x in _pairs(rungs[t], constraint_scores[t], ok, t)[0]]
        out["constraint_retention_auc_degraded"] = roc_auc([0 if y else 1 for y in ys], cx)
    for held in rungs:
        rest = [t for t in rungs if t != held]
        cal_h = fit_calibration(
            [x for t in rest for x in pairs[t][0]], [y for t in rest for y in pairs[t][1]]
        )
        judged = rung_metrics(rungs[held], labels, held)
        est = estimate_preservation(rungs[held], scores[held], cal_h)["preserved"]
        row = {
            "judged_preserved": judged["preserved"],
            "loro_estimate": est,
            "error": est - judged["preserved"],
            "median_savings": judged["median_savings"],
            "fell_back": judged["fell_back"],
        }
        if threshold is not None:  # the serving gate as a hard filter, judged
            row.update(_gated(rungs[held], scores[held], ok, held, threshold))
        out["rungs"][held] = row
    r = out["rungs"]
    out["loro_mae"] = float(np.mean([abs(v["error"]) for v in r.values()]))
    out["loro_max_abs_error"] = float(max(abs(v["error"]) for v in r.values()))
    rho = spearmanr(
        [v["loro_estimate"] for v in r.values()], [v["judged_preserved"] for v in r.values()]
    )
    out["loro_spearman"] = float(rho.statistic)
    out["gate_threshold"] = threshold
    return out


def _gated(
    rewrites: Rows,
    scores: Sequence[float],
    ok: Mapping[tuple[str, str, str], bool],
    tag: str,
    threshold: float,
) -> dict[str, float]:
    """Judged preservation / median savings with the gate rejecting P(degraded) > threshold."""
    targets = sorted({t for (g, _, t) in ok if g == tag})
    passed = [not w["fell_back"] and s <= threshold for w, s in zip(rewrites, scores, strict=True)]
    pres = [
        (not p) or ok.get((tag, w["id"], t), False)
        for w, p in zip(rewrites, passed, strict=True)
        for t in targets
    ]
    return {
        "gate_pass_share": float(np.mean(passed)),
        "gated_preserved": float(np.mean(pres)) if pres else 0.0,
        "gated_median_savings": float(
            statistics.median(
                w["savings"] if p else 0.0 for w, p in zip(rewrites, passed, strict=True)
            )
        ),
    }


# ---- stage 1 ------------------------------------------------------------------------------------
@dataclass
class HpoSettings:
    """Everything a stage-1 trial needs besides its parameters."""

    experiment: str
    base: str
    hardware: str
    train: Path
    val: Path
    n_train: int
    batch_size: int
    val_rows: Rows
    calibration: Calibration
    work: Path = WORK
    eval_every: int = 100
    seed: int = 0
    gen_batch_size: int = 16
    gate_threshold: float | None = None
    extra_overrides: list[str] = field(default_factory=list)


def _free_accelerator() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is in the train extra
        return
    if torch.cuda.is_available():  # pragma: no cover - GPU only
        torch.cuda.empty_cache()


def hf_generator(model_dir: Path, cfg: Any, rows: Rows, batch_size: int = 16) -> Rows:
    """Rewrites of ``rows`` by the adapter in ``model_dir``, exactly as the generate CLI makes them.

    The adapter is reloaded from disk (merged into a fresh bf16 base, or on the NF4 base for
    QLoRA), so the judged candidates are what the registered model produces.
    """
    p = cfg.experiment.params
    if p.model == "tiny":
        raise ValueError("the tiny model has no generator; pass one to make_objective")
    from terser.models.rewriter.generate import generate_rows, load_model  # pragma: no cover

    model, tok = load_model(str(p.model), str(model_dir), p.get("quant"))  # pragma: no cover
    try:  # pragma: no cover - GPU only
        return generate_rows(model, tok, rows, batch_size=batch_size, model_name=str(model_dir))
    finally:  # pragma: no cover
        del model
        _free_accelerator()


def trial_overrides(s: HpoSettings, st: Mapping[str, Any], out_dir: Path) -> list[str]:
    """Hydra overrides of one trial on top of ``experiment=<s.experiment>``."""
    pp = "++experiment.params"
    return [
        f"experiment={s.experiment}",
        f"hardware={s.hardware}",
        f"seed={s.seed}",
        f"{pp}.model={s.base}",
        f"{pp}.data={s.train.as_posix()}",
        f"{pp}.val_data={s.val.as_posix()}",
        f"{pp}.eval_every={s.eval_every}",
        f"{pp}.steps={st['steps']}",
        f"{pp}.lr={st['lr']!r}",
        f"{pp}.schedule=linear",
        f"{pp}.warmup_steps={st['warmup_steps']}",
        f"{pp}.lora.r={st['rank']}",
        f"{pp}.lora.alpha={st['alpha']}",
        f"{pp}.output_dir={out_dir.resolve().as_posix()}",
        f"{pp}.save_weights=true",
        *s.extra_overrides,
    ]


def make_objective(
    s: HpoSettings,
    generator: Generator | None = None,
    scorer: Scorer | None = None,
) -> Callable[[dict[str, Any], Any], float]:
    """Stage-1 objective ``(params, trial) -> J`` (see the module docstring)."""
    import optuna

    from terser.config import load_config
    from terser.models.rewriter.train import run as train_run

    def gen(model_dir: Path, cfg: Any, rows: Rows) -> Rows:
        if generator is not None:
            return generator(model_dir, cfg, rows)
        return hf_generator(model_dir, cfg, rows, s.gen_batch_size)  # pragma: no cover

    score: Scorer
    if scorer is not None:
        score = scorer
    else:  # pragma: no cover - needs models/fast/gate.onnx (dvc pull) and the Qwen tokenizer
        score = GateScorer()

    def objective(params: dict[str, Any], trial: Any) -> float:
        _free_accelerator()
        st = trial_settings(params, s.n_train, s.batch_size)
        tag = trial_tag(trial.number)
        tdir = s.work / f"trial-{trial.number:02d}"
        cfg = load_config(trial_overrides(s, st, tdir))
        trial.set_user_attr("tag", tag)
        for k in ("steps", "warmup_steps", "alpha"):
            trial.set_user_attr(k, st[k])
        curve: list[list[float]] = []

        def on_eval(step: int, loss: float) -> None:
            curve.append([step, loss])
            trial.report(-loss, step)  # maximised study: lower loss = higher value
            if trial.should_prune():
                trial.set_user_attr("val_loss_curve", curve)
                trial.set_user_attr("pruned_at_step", step)
                raise optuna.TrialPruned(f"val loss {loss:.4f} below median at step {step}")

        train_run(cfg, on_eval=on_eval)
        _free_accelerator()  # release the training copy before the generation copy loads
        rows = gen(tdir, cfg, s.val_rows)
        _write_jsonl(tdir / "val_rewrites.jsonl", rows)
        scores = score(rows)
        est = estimate_preservation(rows, scores, s.calibration)
        savings = float(statistics.median(w["savings"] for w in rows))
        value = objective_value(savings, est["preserved"])
        attrs: dict[str, Any] = {
            "val_loss_curve": curve,
            "final_val_loss": curve[-1][1] if curve else float("nan"),
            "median_savings": savings,
            "preserved_est": est["preserved"],
            "preserved_est_min_category": est["min_category"],
            "fell_back": int(sum(bool(w["fell_back"]) for w in rows)),
            "feasible": value >= 0.0,
            "model_dir": tdir.as_posix(),
        }
        if s.gate_threshold is not None:
            attrs["gate_pass_share"] = float(
                np.mean(
                    [
                        not w["fell_back"] and x <= s.gate_threshold
                        for w, x in zip(rows, scores, strict=True)
                    ]
                )
            )
        for k, v in attrs.items():
            trial.set_user_attr(k, v)
        summary = {"trial": trial.number, "params": params, **st, **attrs, "value": value}
        (tdir / "trial.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return value

    return objective


def summarise_study(study: Any) -> list[dict[str, Any]]:
    """One JSON-able record per trial (state, params, value of COMPLETE trials, user attrs)."""
    out = []
    for t in study.trials:
        complete = t.state.name == "COMPLETE"
        out.append(
            {
                "trial": t.number,
                "state": t.state.name,
                "params": dict(t.params),
                "value": float(t.value) if complete and t.value is not None else None,
                "intermediate": {str(k): v for k, v in sorted(t.intermediate_values.items())},
                **dict(t.user_attrs),
            }
        )
    return out


def run_stage1(
    s: HpoSettings,
    n_trials: int,
    *,
    n_startup: int = 5,
    pruner_startup: int = 3,
    pruner_warmup: int = 300,
    storage: str | None = None,
    mlflow: bool = False,
    enqueue_default: bool = True,
    enqueue: Sequence[Mapping[str, Any]] | None = None,
    generator: Generator | None = None,
    scorer: Scorer | None = None,
    data_log: DataAccessLog | None = None,
) -> dict[str, Any]:
    """TPE + MedianPruner study over :data:`SPACE`; returns the stage-1 summary.

    The first trial is :data:`DEFAULT_PARAMS` unless ``enqueue_default`` is false; ``enqueue``
    replaces that list of fixed first trials.
    """
    import optuna

    if enqueue is None:
        enqueue = [DEFAULT_PARAMS] if enqueue_default else []
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=pruner_startup, n_warmup_steps=pruner_warmup
    )
    res = run_search(
        make_objective(s, generator, scorer),
        SPACE,
        "tpe",
        n_trials,
        seed=s.seed,
        n_startup=n_startup,
        pruner=pruner,
        mlflow=mlflow,
        mlflow_tags={
            "phase": 10,
            "row": "10.7",
            "task": "hpo_rewriter",
            "run_name": f"10-hpo-rewriter-tpe-{s.seed}",
            "base": s.base,
            "experiment": s.experiment,
        },
        data_log=data_log,
        with_trial=True,
        storage=storage,
        study_name=STUDY_NAME if storage else None,
        enqueue=[dict(p) for p in enqueue],
        catch=(FloatingPointError, RuntimeError),
    )
    trials = summarise_study(res.extra["study"])
    audit = audit_access_log(data_log.records()) if data_log is not None else None
    return {
        "row": "10.7",
        "base": s.base,
        "experiment": s.experiment,
        "n_train": s.n_train,
        "batch_size": s.batch_size,
        "eval_every": s.eval_every,
        "seed": s.seed,
        "n_startup": n_startup,
        "pruner": {"type": "MedianPruner", "n_startup_trials": pruner_startup,
                   "n_warmup_steps": pruner_warmup},  # fmt: skip
        "calibration": {"a": s.calibration.a, "b": s.calibration.b},
        "best_value": res.best_value,
        "best_params": res.best_params,
        "mlflow_run_id": res.mlflow_run_id,
        "data_access_ok": None if audit is None else audit.ok,
        "trials": trials,
    }


# ---- stage 2 ------------------------------------------------------------------------------------
def select_topk(trials: Sequence[Mapping[str, Any]], k: int) -> list[Mapping[str, Any]]:
    """Top ``k`` COMPLETE trials by objective (ties: more median savings, then trial number)."""
    done = [t for t in trials if t["state"] == "COMPLETE" and t.get("value") is not None]
    done.sort(key=lambda t: (-float(t["value"]), -float(t.get("median_savings", 0)), t["trial"]))
    return done[:k]


def candidate_records(rewrites: Rows, tag: str) -> Rows:
    """Rows of ``val_candidates_v2.jsonl`` format for one tagged rewrite file."""
    return [
        {
            "prompt_id": r["id"],
            "candidate_id": f"{r['id']}:{tag}",
            "method": tag,
            "text": r["rewrite"],
        }
        for r in rewrites
    ]


def choose_trial(
    results: Mapping[str, Mapping[str, Any]], tau: float = MIN_PRESERVED
) -> tuple[str, bool]:
    """``(tag, qualifies)``: most savings with preserved >= ``tau``, else best-preserving."""
    if not results:
        raise ValueError("no judged trials")
    ok = [t for t, m in results.items() if m["preserved"] >= tau]
    if ok:
        return max(ok, key=lambda t: (results[t]["median_savings"], results[t]["preserved"])), True
    best = max(results, key=lambda t: (results[t]["preserved"], results[t]["median_savings"]))
    return best, False


def c10_3_verdict(chosen: Mapping[str, Any], baseline: Mapping[str, Any] | None) -> dict[str, Any]:
    """C10.3: passes C8.3, and >= 1 pp more savings than the untuned rung if that passed too."""
    base_sav = None if baseline is None else float(baseline["median_savings"])
    base_pass = bool(baseline and baseline.get("c8_3"))
    gain = None if base_sav is None else float(chosen["median_savings"]) - base_sav
    passed = bool(chosen["c8_3"]) and (not base_pass or (gain is not None and gain >= 0.01))
    return {
        "verdict": "pass" if passed else "plateau",
        "chosen_c8_3": bool(chosen["c8_3"]),
        "baseline_c8_3": base_pass,
        "savings_gain_vs_untuned": gain,
        "preserved_gain_vs_untuned": (
            None if baseline is None else float(chosen["preserved"]) - float(baseline["preserved"])
        ),
    }


def _short_base(base: str) -> str:
    m = re.search(r"Qwen2\.5-([\d.]+B)", base)
    return f"qwen{m.group(1).lower()}" if m else re.sub(r"[^a-z0-9.]+", "-", base.lower())


# ---- CLI ----------------------------------------------------------------------------------------
def cmd_proxy(args: argparse.Namespace) -> int:  # pragma: no cover - needs gate.onnx + labels
    from terser.features.shared import extract_features
    from terser.serving.paths import ConstraintGate

    log = DataAccessLog(args.out / "data_access.jsonl")
    labels = log.read_jsonl(args.labels)
    rungs = {t: log.read_jsonl(args.rewrites / f) for t, f in JUDGED_RUNGS.items()}
    scorer = GateScorer()
    scores = {t: scorer(r) for t, r in rungs.items()}
    cg = ConstraintGate()
    cons = {
        t: [
            cg.score(
                extract_features(w["prompt"], GATE_TARGET),
                extract_features(w["rewrite"], GATE_TARGET),
            )
            for w in r
        ]
        for t, r in rungs.items()
    }
    res = proxy_agreement(rungs, labels, scores, cons, gate_threshold())
    res["labels"] = args.labels.as_posix()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "proxy.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("calibration", "gate_auc_degraded", "loro_mae",
                                           "loro_spearman")}))  # fmt: skip
    return 0


def _settings(args: argparse.Namespace, log: DataAccessLog) -> HpoSettings:
    from terser.config import load_config

    proxy = json.loads((args.out / "proxy.json").read_text(encoding="utf-8"))
    experiment = args.experiment or BASE_EXPERIMENTS.get(args.base)
    if experiment is None:
        raise SystemExit(f"no default experiment for {args.base}; pass --experiment")
    cfg = load_config([f"experiment={experiment}"])
    n_train = len(log.read_jsonl(args.train))
    return HpoSettings(
        experiment=experiment,
        base=args.base,
        hardware=args.hardware,
        train=args.train,
        val=args.val,
        n_train=n_train,
        batch_size=int(cfg.experiment.params.batch_size),
        val_rows=log.read_jsonl(args.val),
        calibration=Calibration(**{k: proxy["calibration"][k] for k in ("a", "b")}),
        work=args.work,
        eval_every=args.eval_every,
        seed=args.seed,
        gen_batch_size=args.gen_batch_size,
        gate_threshold=proxy.get("gate_threshold"),
    )


def cmd_run(args: argparse.Namespace) -> int:  # pragma: no cover - GPU
    log = DataAccessLog(args.out / "data_access.jsonl")
    s = _settings(args, log)
    args.work.mkdir(parents=True, exist_ok=True)
    storage = args.storage or f"sqlite:///{(args.work / 'optuna.db').resolve().as_posix()}"
    summary = run_stage1(
        s,
        args.trials,
        n_startup=args.n_startup,
        pruner_startup=args.pruner_startup,
        pruner_warmup=args.pruner_warmup,
        storage=storage,
        mlflow=not args.no_mlflow,
        enqueue_default=not args.no_default_trial,
        data_log=log,
    )
    summary["storage"] = storage
    (args.out / "stage1.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"best_value": summary["best_value"], "best_params": summary["best_params"]}))
    return 0


def write_candidates(
    stage1: Mapping[str, Any], k: int, rewrites_dir: Path, out: Path
) -> dict[str, Any]:
    """Stage-2 inputs: per-trial rewrite files, one candidate file, ``topk.json``."""
    top = select_topk(stage1["trials"], k)
    if not top:
        raise SystemExit("no complete stage-1 trial")
    records: Rows = []
    picked = []
    for t in top:
        tag = str(t["tag"])
        rows = _jsonl(Path(t["model_dir"]) / "val_rewrites.jsonl")
        dest = rewrites_dir / f"{tag}.val.jsonl"
        _write_jsonl(dest, rows)
        records += candidate_records(rows, tag)
        picked.append(
            {
                "tag": tag,
                "trial": t["trial"],
                "params": t["params"],
                "value": t["value"],
                "median_savings": t.get("median_savings"),
                "preserved_est": t.get("preserved_est"),
                "steps": t.get("steps"),
                "warmup_steps": t.get("warmup_steps"),
                "alpha": t.get("alpha"),
                "model_dir": t["model_dir"],
                "rewrites": dest.as_posix(),
            }
        )
    cand = rewrites_dir / HPO_CANDIDATES.name
    _write_jsonl(cand, records)
    topk = {
        "base": stage1["base"],
        "experiment": stage1["experiment"],
        "baseline_rung": BASELINE_RUNG.get(stage1["base"]),
        "candidates": cand.as_posix(),
        "k": len(picked),
        "trials": picked,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "topk.json").write_text(json.dumps(topk, indent=2) + "\n", encoding="utf-8")
    return topk


def cmd_candidates(args: argparse.Namespace) -> int:
    stage1 = json.loads((args.out / "stage1.json").read_text(encoding="utf-8"))
    topk = write_candidates(stage1, args.k, args.rewrites, args.out)
    print(
        json.dumps({"tags": [t["tag"] for t in topk["trials"]], "candidates": topk["candidates"]})
    )
    return 0


def _fmt(v: Any, pct: bool = True) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.1%}" if pct else f"{v:.4g}"
    return str(v)


def finalize(
    out: Path,
    labels_path: Path,
    *,
    baseline_json: Path = Path("reports/p8_rewriter_val.json"),
    report: Path = REPORT,
    models_root: Path = Path("models/rewriter"),
    register: bool = True,
    stage: str | None = None,
    log: DataAccessLog | None = None,
) -> dict[str, Any]:
    """Judged metrics of the top-k, the chosen trial, C10.3, the report and the registration."""
    topk = json.loads((out / "topk.json").read_text(encoding="utf-8"))
    stage1 = json.loads((out / "stage1.json").read_text(encoding="utf-8"))
    proxy_path = out / "proxy.json"
    proxy = json.loads(proxy_path.read_text(encoding="utf-8")) if proxy_path.exists() else None

    def read(path: str | Path) -> Rows:
        return log.read_jsonl(path) if log is not None else _jsonl(Path(path))

    labels = read(labels_path)
    results: dict[str, dict[str, Any]] = {}
    for t in topk["trials"]:
        m = rung_metrics(read(t["rewrites"]), labels, t["tag"])
        if not m["targets"]:
            raise SystemExit(f"no judge labels for {t['tag']} in {labels_path}")
        m["preserved_est"] = t["preserved_est"]
        results[t["tag"]] = m
    tag, qualifies = choose_trial(results)
    chosen_t = next(t for t in topk["trials"] if t["tag"] == tag)
    chosen = results[tag]
    baseline = None
    if topk.get("baseline_rung") and baseline_json.exists():
        baseline = json.loads(baseline_json.read_text(encoding="utf-8")).get(topk["baseline_rung"])
    verdict = c10_3_verdict(chosen, baseline)
    checkpoints = (["C8.3"] if chosen["c8_3"] else []) + (
        ["C10.3"] if verdict["verdict"] == "pass" else []
    )
    dest = models_root / f"sft-{_short_base(topk['base'])}-lora-hpo"
    config = {
        "experiment": topk["experiment"],
        "row": "10.7",
        "base": topk["base"],
        "trial": chosen_t["trial"],
        **{k: chosen_t[k] for k in ("steps", "warmup_steps", "alpha")},
        **chosen_t["params"],
        "schedule": "linear",
    }
    registry: dict[str, Any] | None = None
    stages = [stage] if stage else (["Staging"] if chosen["c8_3"] else [])
    if register:  # pragma: no cover - writes models/ and the MLflow registry
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(chosen_t["model_dir"], dest)
        from terser.phase11.register import register_one

        registry = register_one(
            "rewriter",
            dest.as_posix(),
            config,
            checkpoints,
            stages,
            {"median_savings": chosen["median_savings"], "preserved": chosen["preserved"]},
            phase=10,
            row="10.7",
            run_name=f"10-register-rewriter-{tag}-{stage1.get('seed', 0)}",
        )
    result = {
        "row": "10.7",
        "checkpoint": "C10.3",
        "base": topk["base"],
        "chosen": tag,
        "chosen_meets_preservation": qualifies,
        "chosen_label": (
            "best with preserved >= 0.92"
            if qualifies
            else "best-preserving (no judged trial reaches 0.92)"
        ),
        "chosen_config": config,
        "c10_3": verdict,
        "baseline_rung": topk.get("baseline_rung"),
        "baseline": baseline,
        "judged": results,
        "registered": registry,
        "registered_path": dest.as_posix(),
        "checkpoints_passed": checkpoints,
        "stage1": stage1,
        "proxy": proxy,
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    report.write_text(_markdown(result), encoding="utf-8")
    return result


def _markdown(r: Mapping[str, Any]) -> str:
    s1, px, v = r["stage1"], r["proxy"], r["c10_3"]
    lines = [
        "# Rewriter LoRA tuning (row 10.7, checkpoint C10.3)",
        "",
        "Generated by `python -m terser.phase10.rewriter_hpo finalize` (design: module docstring "
        "of `terser.phase10.rewriter_hpo`). Base "
        f"`{r['base']}` on `data/processed/sft_v2`, val split only.",
        "",
        "Search space: rank {8, 16, 32, 64}; alpha = {1, 2, 4} x rank; lr log-uniform "
        "[5e-5, 5e-4]; warmup fraction [0, 0.1] (linear warmup then linear decay); epochs "
        f"{{1, 2, 3}} ({s1['n_train']} pairs, batch {s1['batch_size']}). Optuna TPE "
        f"(seed {s1['seed']}, {s1['n_startup']} random start-up trials) with a MedianPruner "
        f"({s1['pruner']['n_startup_trials']} start-up trials, warmup "
        f"{s1['pruner']['n_warmup_steps']} steps) on the val loss every {s1['eval_every']} steps. "
        "Trial 0 is the untuned rung's settings (r 16, alpha 32, lr 2e-4, 3 epochs).",
        "",
        "Stage-1 objective: J = median savings if (proxy preservation >= 0.92 and savings >= "
        "0.25), else -1 - (distance to that box). Proxy: the served quality gate's P(degraded) "
        "calibrated to the judge (logistic fit on the five judged rungs); fallbacks count as "
        "preserved.",
        "",
    ]
    if px:
        lines += [
            "## Proxy agreement with the judge (five judged rungs)",
            "",
            f"Gate AUC vs judged degraded (non-fallback rewrites, both targets): "
            f"**{px['gate_auc_degraded']:.3f}**; constraint-keyword retention: "
            f"{px.get('constraint_retention_auc_degraded', float('nan')):.3f}. "
            f"Leave-one-rung-out rung-level MAE **{px['loro_mae']:.1%}**, max "
            f"{px['loro_max_abs_error']:.1%}, Spearman {px['loro_spearman']:.2f}.",
            "",
            "| Rung | Median savings | Judged preserved | LORO proxy | Error | Gate pass "
            "| Gated preserved | Gated savings |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for t, m in px["rungs"].items():
            lines.append(
                f"| {t} | {_fmt(m['median_savings'])} | {_fmt(m['judged_preserved'])} "
                f"| {_fmt(m['loro_estimate'])} | {m['error']:+.1%} "
                f"| {_fmt(m.get('gate_pass_share'))} | {_fmt(m.get('gated_preserved'))} "
                f"| {_fmt(m.get('gated_median_savings'))} |"
            )
        lines.append("")
    lines += [
        "## Stage 1 trials",
        "",
        "| Trial | State | rank | alpha | lr | warmup | epochs (steps) | Final val loss "
        "| Median savings | Proxy preserved | J |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for t in s1["trials"]:
        p = t["params"]
        loss = t.get("final_val_loss")
        if loss is None and t.get("val_loss_curve"):
            loss = t["val_loss_curve"][-1][1]
        state = t["state"] + (f" @ {t['pruned_at_step']}" if t.get("pruned_at_step") else "")
        lines.append(
            f"| {t['trial']} | {state} | {p.get('rank', '-')} | {t.get('alpha', '-')} "
            f"| {_fmt(p.get('lr'), False)} | {t.get('warmup_steps', '-')} "
            f"| {p.get('epochs', '-')} ({t.get('steps', '-')}) | {_fmt(loss, False)} "
            f"| {_fmt(t.get('median_savings'))} | {_fmt(t.get('preserved_est'))} "
            f"| {_fmt(t.get('value'), False)} |"
        )
    lines += [
        "",
        "## Stage 2: judged top-k (Qwen2.5-32B-Instruct-AWQ, `pairwise_avg`, both targets)",
        "",
        "| Candidate | Median savings | Fell back | Proxy preserved | Judged preserved "
        "| Min category | C8.3 |",
        "|---|---|---|---|---|---|---|",
    ]
    rows = dict(r["judged"])
    if r.get("baseline"):
        rows[f"{r['baseline_rung']} (untuned)"] = r["baseline"]
    for t, m in rows.items():
        pc = m.get("preserved_by_category") or {}
        lines.append(
            f"| {t} | {_fmt(m['median_savings'])} | {m.get('fell_back', '-')} "
            f"| {_fmt(m.get('preserved_est'))} | {_fmt(m['preserved'])} "
            f"| {_fmt(min(pc.values()) if pc else None)} | {'pass' if m['c8_3'] else 'fail'} |"
        )
    gain = v["savings_gain_vs_untuned"]
    pgain = v["preserved_gain_vs_untuned"]
    reg = r.get("registered")
    lines += [
        "",
        f"Chosen: **{r['chosen']}**, {r['chosen_label']}. "
        f"Config: `{json.dumps(r['chosen_config'])}`.",
        "",
        f"**C10.3: {v['verdict']}.** Chosen trial C8.3 "
        f"{'pass' if v['chosen_c8_3'] else 'fail'}; untuned rung {r.get('baseline_rung')} C8.3 "
        f"{'pass' if v['baseline_c8_3'] else 'fail'}; savings vs untuned "
        f"{'-' if gain is None else f'{gain:+.1%}'}, preservation vs untuned "
        f"{'-' if pgain is None else f'{pgain:+.1%}'}.",
        "",
    ]
    if v["verdict"] != "pass":
        lines += [
            "Documented plateau: no judged trial reaches the C8.3 box (median savings >= 25%, "
            "preserved >= 92% overall and >= 85% per category). The table above is the "
            "evidence: LoRA rank, alpha, learning rate, warmup and epochs move the "
            "savings/preservation trade-off but do not lift preservation to 0.92.",
            "",
        ]
    if reg:
        lines.append(
            f"Registered: `{reg['registry name']}` v{reg['version']} from `{reg['source']}` "
            f"(stages {reg['stage history']}; checkpoints "
            f"{', '.join(r['checkpoints_passed']) or 'none'})."
        )
    else:
        lines.append(
            f"Not registered (`--no-register`); weights would be `{r['registered_path']}`."
        )
    return "\n".join(lines) + "\n"


def cmd_finalize(args: argparse.Namespace) -> int:  # pragma: no cover - registry + labels
    log = DataAccessLog(args.out / "data_access.jsonl")
    res = finalize(
        args.out,
        args.labels,
        report=args.report,
        register=not args.no_register,
        stage=args.stage,
        log=log,
    )
    print(json.dumps({"chosen": res["chosen"], "c10_3": res["c10_3"]["verdict"]}))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Row 10.7 rewriter LoRA tuning.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    px = sub.add_parser("proxy", help="fit the gate calibration, measure agreement with the judge")
    px.add_argument("--labels", type=Path, default=JUDGED_LABELS)
    px.add_argument("--rewrites", type=Path, default=REWRITES)
    r = sub.add_parser("run", help="stage 1: TPE + MedianPruner trials (GPU)")
    r.add_argument("--base", default=DEFAULT_BASE)
    r.add_argument("--experiment", default=None, help="default: by --base")
    r.add_argument("--hardware", default="rtx4500ada")
    r.add_argument("--train", type=Path, default=Path("data/processed/sft_v2/train.jsonl"))
    r.add_argument("--val", type=Path, default=Path("data/processed/sft_v2/val.jsonl"))
    r.add_argument("--trials", type=int, default=12)
    r.add_argument("--n-startup", type=int, default=5)
    r.add_argument("--pruner-startup", type=int, default=3)
    r.add_argument("--pruner-warmup", type=int, default=300)
    r.add_argument("--eval-every", type=int, default=100)
    r.add_argument("--gen-batch-size", type=int, default=16)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--work", type=Path, default=WORK)
    r.add_argument("--storage", default=None, help="default sqlite:///<work>/optuna.db")
    r.add_argument("--no-mlflow", action="store_true")
    r.add_argument("--no-default-trial", action="store_true")
    c = sub.add_parser("candidates", help="stage 2 inputs: top-k candidate files")
    c.add_argument("--k", type=int, default=3)
    c.add_argument("--rewrites", type=Path, default=REWRITES)
    f = sub.add_parser("finalize", help="judged choice, report, C10.3, registration")
    f.add_argument("--labels", type=Path, default=HPO_LABELS)
    f.add_argument("--report", type=Path, default=REPORT)
    f.add_argument("--no-register", action="store_true")
    f.add_argument("--stage", default=None, help="override the alias (default: Staging iff C8.3)")
    for p in (px, r, c, f):
        p.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    cmds: dict[str, Callable[[argparse.Namespace], int]] = {
        "proxy": cmd_proxy,
        "run": cmd_run,
        "candidates": cmd_candidates,
        "finalize": cmd_finalize,
    }
    return cmds[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
