"""Quality gates of the final evaluation (rows 12.2, 12.6).

* :class:`PairGate` -- the champion quality gate: LightGBM on the row 3.1.3 pair table
  (``data/processed/p4_pairs.parquet``, 27 features: Qwen2.5 token costs, GPT-2 surprisal of the
  removed words, bge-small cosine, category, ``target_phi``, prompt features). ``tuned`` uses the
  Phase 10 best hyperparameters (:data:`TUNED_GBM`, row 10.5), ``untuned`` the LightGBM defaults
  of row 10.5 (:data:`terser.phase10.hpo_study.UNTUNED_GBM`); both 150 trees as in the HPO
  objective. The threshold follows row 3.2.3: maximise specificity subject to sensitivity >= 0.95
  on grouped 5-fold out-of-fold train scores (:func:`terser.eval.threshold.choose_threshold`);
  a rewrite is rejected when ``P(degraded) >= threshold``. The gate is target-aware
  (``target_phi``), so the decision is made per (prompt, target).
* :class:`OnnxFastGate` -- the deployed fast-path gate (``models/fast/gate.onnx``, surface
  features of :mod:`terser.features.shared`, Qwen2.5 counts) with the serving rule
  ``P(degraded) > gate_threshold`` from ``configs/serving/fast_onnx.yaml``.

The gate-off / HPO-off ablations of row 12.6 swap these objects; nothing else changes.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np

__all__ = [
    "N_ESTIMATORS",
    "TUNED_GBM",
    "OnnxFastGate",
    "PairFeaturizer",
    "PairGate",
    "fast_gate_threshold",
]

TUNED_GBM: Final[dict[str, Any]] = {
    "learning_rate": 0.019083055421086518,
    "num_leaves": 27,
    "min_child_samples": 197,
}
"""Best Phase 10 parameters (GP search, seed 0, OOF AUC 0.7829; ``reports/p10_hpo.csv``)."""

N_ESTIMATORS: Final = 150
NON_FEATURES: Final = ("id", "group", "category", "label")


def _lgbm(params: Mapping[str, Any], seed: int) -> Any:
    import lightgbm as lgb

    return lgb.LGBMClassifier(
        n_estimators=N_ESTIMATORS,
        learning_rate=float(params["learning_rate"]),
        num_leaves=int(params["num_leaves"]),
        min_child_samples=int(params["min_child_samples"]),
        random_state=seed,
        verbose=-1,
        n_jobs=2,
    )


@dataclass
class PairGate:
    """LightGBM pair gate with its row 3.2.3 threshold (``reject if score >= threshold``)."""

    name: str
    params: dict[str, Any]
    columns: list[str]
    model: Any
    threshold: float
    oof_auc: float
    sensitivity: float
    specificity: float
    n_train: int

    @classmethod
    def fit(cls, table: Any, params: Mapping[str, Any], name: str, seed: int = 0) -> PairGate:
        """Grouped OOF threshold, then refit on the whole train pair table."""
        from terser.eval.curves import roc_auc
        from terser.eval.cv import out_of_fold_scores
        from terser.eval.threshold import choose_threshold

        cols = [c for c in table.columns if c not in NON_FEATURES]
        X = table[cols].to_numpy(np.float64)
        y = table.label.to_numpy().astype(int)

        def fp(Xt: np.ndarray, yt: np.ndarray, Xv: np.ndarray) -> np.ndarray:
            return np.asarray(_lgbm(params, seed).fit(Xt, yt).predict_proba(Xv))[:, 1]

        oof = out_of_fold_scores(
            fp,
            X,
            y,
            table.category.astype(str).tolist(),
            table.group.astype(str).tolist(),
            seed=seed,
        )
        choice = choose_threshold(y, oof.scores)
        return cls(
            name=name,
            params=dict(params),
            columns=cols,
            model=_lgbm(params, seed).fit(X, y),
            threshold=float(choice.threshold),
            oof_auc=float(roc_auc(y, oof.scores)),
            sensitivity=float(choice.sensitivity),
            specificity=float(choice.specificity),
            n_train=len(y),
        )

    def score_rows(self, rows: Sequence[Mapping[str, float]]) -> list[float]:
        """``P(degraded)`` for feature dicts (missing columns are 0, as in Phase 9)."""
        if not rows:
            return []
        X = np.asarray([[r.get(c, 0.0) for c in self.columns] for r in rows], dtype=np.float64)
        return [float(p) for p in np.asarray(self.model.predict_proba(X))[:, 1]]

    def rejects(self, score: float) -> bool:
        return score >= self.threshold

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model": f"LightGBM {N_ESTIMATORS} trees on the row 3.1.3 pair table",
            "params": self.params,
            "threshold": self.threshold,
            "rule": "reject if P(degraded) >= threshold; threshold = max specificity s.t. "
            "sensitivity >= 0.95 on grouped 5-fold OOF train scores (row 3.2.3)",
            "oof_auc": self.oof_auc,
            "oof_sensitivity": self.sensitivity,
            "oof_specificity": self.specificity,
            "n_train_pairs": self.n_train,
        }


@dataclass
class PairFeaturizer:
    """Row 3.1.3 pair features of (original, rewrite), the code the Phase 9 gate uses.

    ``surprisal`` maps a token list to GPT-2 bits, ``embed`` a list of texts to bge-small
    vectors, ``cost`` a text to Qwen2.5 tokens; all three are cached here.
    """

    surprisal: Callable[[Sequence[str]], Sequence[float]]
    embed: Callable[[Sequence[str]], Any]
    cost: Callable[[str], int]
    _bits: dict[tuple[str, ...], list[float]] = field(default_factory=dict, repr=False)
    _emb: dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    _base: dict[tuple[str, str, str], dict[str, float]] = field(default_factory=dict, repr=False)
    seconds: float = 0.0
    calls: int = 0

    def _b(self, toks: Sequence[str]) -> list[float]:
        key = tuple(toks)
        if key not in self._bits:
            self._bits[key] = list(self.surprisal(list(key))) if key else []
        return self._bits[key]

    def _e(self, texts: Sequence[str]) -> np.ndarray:
        missing = [t for t in dict.fromkeys(texts) if t not in self._emb]
        if missing:
            for t, v in zip(missing, np.asarray(self.embed(missing)), strict=True):
                self._emb[t] = np.asarray(v, dtype=np.float64)
        return np.stack([self._emb[t] for t in texts])

    def features(self, original: str, rewrite: str, category: str, target: str) -> dict[str, float]:
        """Feature dict; ``target`` is the tokenizer key (``phi3.5`` sets ``target_phi``)."""
        from terser.features.alignment import tokenize
        from terser.features.pair import pair_features
        from terser.phase3.build_features import prompt_features

        key = (original, rewrite, category)
        if key not in self._base:
            t0 = time.perf_counter()
            f = pair_features(original, rewrite, category, self.cost, self._b, self._e)
            f.update(prompt_features(original, self.cost(original), self._b(tokenize(original))))
            self._base[key] = f
            self.seconds += time.perf_counter() - t0
            self.calls += 1
        return {**self._base[key], "target_phi": float("phi" in target.lower())}

    @property
    def mean_seconds(self) -> float | None:
        return self.seconds / self.calls if self.calls else None


def fast_gate_threshold(config: Path = Path("configs/serving/fast_onnx.yaml")) -> float:
    """The deployed fast-path ``gate_threshold`` (row 3.2.3 operating point of ``gate.onnx``)."""
    import yaml

    return float(yaml.safe_load(config.read_text(encoding="utf-8"))["gate_threshold"])


class OnnxFastGate:
    """Deployed ``gate.onnx`` on the serving features (Qwen2.5 counts, as ``fast_onnx.yaml``)."""

    def __init__(
        self,
        path: Path = Path("models/fast/gate.onnx"),
        threshold: float | None = None,
        cost: Callable[[str], int] | None = None,
    ) -> None:
        from terser.serving.paths import OnnxGate

        self.gate = OnnxGate(path)
        self.threshold = fast_gate_threshold() if threshold is None else threshold
        self.cost = cost
        self.path = path

    def score(self, original: str, rewrite: str) -> float:
        from terser.features.shared import extract_features

        return float(
            self.gate.score(
                extract_features(original, "qwen2.5", self.cost),
                extract_features(rewrite, "qwen2.5", self.cost),
            )
        )

    def rejects(self, score: float) -> bool:
        return score > self.threshold

    def describe(self) -> dict[str, Any]:
        return {
            "name": "onnx",
            "model": f"{self.path.as_posix()} (sklearn GradientBoosting, serving features)",
            "threshold": self.threshold,
            "rule": "reject if P(degraded) > gate_threshold (configs/serving/fast_onnx.yaml)",
        }
