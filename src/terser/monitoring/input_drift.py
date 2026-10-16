"""Input drift report (plan row **11.5.2**).

Compares a *reference* window of prompts (the train split) with a *current* window (the val split
by default, or a serving request log / any prompts JSONL) on three views of the input:

* **embedding drift**: bge-small prompt embeddings (row 6.1) projected on a PCA fitted to the
  reference; population stability index (PSI) per component, plus a kernel two-sample test
  (unbiased MMD², RBF kernel with the median-heuristic bandwidth, permutation p-value) on a
  subsample of the projected points;
* **length drift**: PSI of the prompt length in whitespace words;
* **category drift**: PSI of the category mix (rows without a ``category`` field are tagged with
  the real-slice rules, :func:`terser.data.real_slice.tag_category`, else ``"unknown"``).

The same frame (PCA components, length, category) also goes through Evidently's
``DataDriftPreset`` with its PSI stat test; the Evidently HTML (or, if it is too large, only its
JSON) and a JSON summary land in ``reports/monitoring/``.

Alert rule (:class:`DriftThresholds`): PSI > 0.2 on any monitored feature, or MMD permutation
p < 0.01. An alert is a ``CRITICAL`` record on the ``terser.alerts`` logger and exit code 1 from the
CLI, so the scheduled ``drift_flow`` (:mod:`terser.pipelines.flows`) fails loudly; the summary
JSON carries ``"alert": true`` and the reasons::

    python -m terser.monitoring.input_drift                                   # train vs val
    python -m terser.monitoring.input_drift --current logs/requests.jsonl --name 2026-10-17

Embeddings come from the row 6.1 cache when a prompt id is in it; any other prompt is embedded
once with the same bge-small model on CPU and stored in a text-hash keyed cache
(``data/interim/drift_embeddings``) so daily runs only embed new prompts. Request-log records
without raw text (no consent, row 11.5.6) carry no prompt and are skipped and counted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from terser.features.embeddings import EmbeddingCache, Encoder

__all__ = [
    "ALERT_LOGGER",
    "DriftResult",
    "DriftThresholds",
    "PromptRecord",
    "categorical_psi",
    "detect_input_drift",
    "evidently_report",
    "load_prompts",
    "main",
    "mmd_permutation_test",
    "psi",
    "resolve_embeddings",
    "run",
]

ALERT_LOGGER = "terser.alerts"
_ROOT = Path(__file__).resolve().parents[3]
REFERENCE_PATH = _ROOT / "data" / "processed" / "splits" / "train.jsonl"
CURRENT_PATH = _ROOT / "data" / "processed" / "splits" / "val.jsonl"
EMBEDDINGS_DIR = _ROOT / "data" / "processed" / "embeddings"
EMBEDDINGS_NAME = "train_bge_small"
TEXT_CACHE_DIR = _ROOT / "data" / "interim" / "drift_embeddings"
TEXT_CACHE_NAME = "drift_bge_small"
OUT_DIR = _ROOT / "reports" / "monitoring"
MAX_HTML_BYTES = 5_000_000
_EPS = 1e-4

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class DriftThresholds:
    """Alert when any PSI exceeds ``psi`` or the MMD permutation p-value is below ``mmd_p``."""

    psi: float = 0.2
    mmd_p: float = 0.01


@dataclass(frozen=True)
class PromptRecord:
    """One prompt of a window: cache id, text and task category."""

    id: str
    text: str
    category: str

    @property
    def length(self) -> int:
        return len(self.text.split())


# --------------------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------------------


def psi(reference: npt.ArrayLike, current: npt.ArrayLike, bins: int = 10) -> float:
    """PSI of a numeric feature on reference-quantile bins (proportions floored at 1e-4).

    ``sum((c - r) * ln(c / r))`` over the bins; < 0.1 is usually read as stable, > 0.2 as a
    significant shift.
    """
    ref = np.asarray(reference, dtype=np.float64)
    cur = np.asarray(current, dtype=np.float64)
    edges = np.unique(np.quantile(ref, np.linspace(0.0, 1.0, bins + 1))[1:-1])
    r = np.bincount(np.searchsorted(edges, ref, side="right"), minlength=len(edges) + 1)
    c = np.bincount(np.searchsorted(edges, cur, side="right"), minlength=len(edges) + 1)
    return _psi_from_counts(r.astype(np.float64), c.astype(np.float64))


def categorical_psi(reference: Sequence[str], current: Sequence[str]) -> float:
    """PSI of a categorical feature over the union of the observed levels."""
    levels = sorted(set(reference) | set(current))
    rc, cc = Counter(reference), Counter(current)
    r = np.array([rc[k] for k in levels], dtype=np.float64)
    c = np.array([cc[k] for k in levels], dtype=np.float64)
    return _psi_from_counts(r, c)


def _psi_from_counts(r: FloatArray, c: FloatArray) -> float:
    rp = np.maximum(r / r.sum(), _EPS)
    cp = np.maximum(c / c.sum(), _EPS)
    return float(np.sum((cp - rp) * np.log(cp / rp)))


def mmd_permutation_test(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    n_permutations: int = 500,
    max_samples: int = 500,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Unbiased MMD² with an RBF kernel, and its permutation p-value.

    Each side is subsampled to at most ``max_samples`` rows. The bandwidth is the median heuristic
    (median pairwise Euclidean distance of the pooled sample). Returns ``(mmd2, p_value, sigma)``;
    the p-value is ``(1 + #{perm >= observed}) / (1 + n_permutations)``.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(x, dtype=np.float64)
    b = np.asarray(y, dtype=np.float64)
    if len(a) > max_samples:
        a = a[rng.choice(len(a), max_samples, replace=False)]
    if len(b) > max_samples:
        b = b[rng.choice(len(b), max_samples, replace=False)]
    z = np.vstack([a, b])
    sq = np.sum(z * z, axis=1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * z @ z.T, 0.0)
    off = d2[np.triu_indices(len(z), k=1)]
    sigma = float(np.sqrt(np.median(off))) or 1.0
    k = np.exp(-d2 / (2.0 * sigma * sigma))
    np.fill_diagonal(k, 0.0)

    m, n = len(a), len(b)
    labels = np.zeros((n_permutations + 1, m + n), dtype=np.float64)
    labels[0, :m] = 1.0
    for i in range(1, n_permutations + 1):
        labels[i, rng.permutation(m + n)[:m]] = 1.0
    other = 1.0 - labels
    kx = labels @ k
    sxx = np.sum(kx * labels, axis=1)
    sxy = np.sum(kx * other, axis=1)
    syy = np.sum((other @ k) * other, axis=1)
    stats = sxx / (m * (m - 1)) + syy / (n * (n - 1)) - 2.0 * sxy / (m * n)
    p = float((1 + np.sum(stats[1:] >= stats[0])) / (1 + n_permutations))
    return float(stats[0]), p, sigma


# --------------------------------------------------------------------------------------------------
# Drift result
# --------------------------------------------------------------------------------------------------


@dataclass
class DriftResult:
    """Every number behind one reference-vs-current comparison, plus the alert decision."""

    n_reference: int
    n_current: int
    n_components: int
    explained_variance: float
    psi: dict[str, float]
    mmd2: float
    mmd_p: float
    mmd_sigma: float
    mmd_samples: int
    length: dict[str, dict[str, float]]
    category: dict[str, dict[str, float]]
    thresholds: DriftThresholds = field(default_factory=DriftThresholds)
    n_skipped: int = 0
    evidently: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)

    @property
    def drifted(self) -> list[str]:
        """Features whose PSI exceeds the threshold (components first, then length, category)."""
        return [k for k, v in self.psi.items() if v > self.thresholds.psi]

    @property
    def reasons(self) -> list[str]:
        out = [f"PSI {k} = {self.psi[k]:.3f} > {self.thresholds.psi}" for k in self.drifted]
        if self.mmd_p < self.thresholds.mmd_p:
            out.append(f"MMD p = {self.mmd_p:.4f} < {self.thresholds.mmd_p}")
        return out

    @property
    def alert(self) -> bool:
        return bool(self.reasons)

    def summary(self) -> str:
        if not self.alert:
            worst = max(self.psi, key=lambda k: self.psi[k])
            return (
                f"no input drift (n_ref={self.n_reference}, n_cur={self.n_current}; max PSI "
                f"{worst} = {self.psi[worst]:.3f}; MMD p = {self.mmd_p:.3f})"
            )
        return "INPUT DRIFT " + "; ".join(self.reasons)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out.update(alert=self.alert, reasons=self.reasons, drifted=self.drifted)
        return out

    def markdown(self, title: str) -> str:
        """A report section: verdict, MMD, per-feature PSI, length and category tables."""
        ev_psi: Mapping[str, float] = self.evidently.get("psi", {})
        lines = [
            f"### {title}",
            "",
            f"- Verdict: **{'DRIFT' if self.alert else 'no drift'}**"
            + (f" ({'; '.join(self.reasons)})" if self.alert else ""),
            f"- Rows: reference {self.n_reference:,}, current {self.n_current:,}"
            + (
                f" ({self.n_skipped:,} current records without raw text skipped)"
                if self.n_skipped
                else ""
            ),
            f"- PCA: {self.n_components} components fitted on the reference, "
            f"{self.explained_variance:.1%} of embedding variance",
            f"- MMD² (RBF, σ = {self.mmd_sigma:.3f}, {self.mmd_samples} rows per side): "
            f"{self.mmd2:.5f}, permutation p = {self.mmd_p:.4f}",
        ]
        if self.evidently:
            lines.append(
                f"- Evidently DataDriftPreset (PSI, threshold {self.thresholds.psi}): "
                f"{self.evidently.get('drifted_count', 0):g} of {len(ev_psi)} columns drifted"
            )
        lines += ["", "| Feature | PSI | Evidently PSI | > threshold |", "|---|---:|---:|:---:|"]
        for name, value in self.psi.items():
            ev = ev_psi.get(name)
            ev_s = f"{ev:.3f}" if ev is not None else "n/a"
            flag = "yes" if value > self.thresholds.psi else ""
            lines.append(f"| {name} | {value:.3f} | {ev_s} | {flag} |")
        ref_l, cur_l = self.length["reference"], self.length["current"]
        lines += [
            "",
            "| Length (words) | median | p90 | mean |",
            "|---|---:|---:|---:|",
            f"| reference | {ref_l['median']:.0f} | {ref_l['p90']:.0f} | {ref_l['mean']:.1f} |",
            f"| current | {cur_l['median']:.0f} | {cur_l['p90']:.0f} | {cur_l['mean']:.1f} |",
            "",
            "| Category | reference share | current share |",
            "|---|---:|---:|",
        ]
        ref_c, cur_c = self.category["reference"], self.category["current"]
        for k in sorted(set(ref_c) | set(cur_c)):
            lines.append(f"| {k} | {ref_c.get(k, 0.0):.1%} | {cur_c.get(k, 0.0):.1%} |")
        if self.artifacts:
            lines += ["", "Artifacts: " + ", ".join(f"`{v}`" for v in self.artifacts.values())]
        return "\n".join(lines) + "\n"


def _length_stats(lengths: FloatArray) -> dict[str, float]:
    return {
        "median": float(np.median(lengths)),
        "p90": float(np.quantile(lengths, 0.9)),
        "mean": float(np.mean(lengths)),
    }


def _shares(values: Sequence[str]) -> dict[str, float]:
    counts = Counter(values)
    return {k: counts[k] / len(values) for k in sorted(counts)}


def _pca(reference: FloatArray, n_components: int) -> tuple[FloatArray, FloatArray, float]:
    """Mean and principal axes of ``reference`` (SVD), and the explained variance share."""
    mean = reference.mean(axis=0)
    _, s, vt = np.linalg.svd(reference - mean, full_matrices=False)
    k = min(n_components, vt.shape[0])
    var = s * s
    return mean, vt[:k], float(var[:k].sum() / var.sum())


def detect_input_drift(
    reference_embeddings: npt.ArrayLike,
    current_embeddings: npt.ArrayLike,
    reference_lengths: npt.ArrayLike,
    current_lengths: npt.ArrayLike,
    reference_categories: Sequence[str],
    current_categories: Sequence[str],
    *,
    n_components: int = 16,
    thresholds: DriftThresholds | None = None,
    n_permutations: int = 500,
    max_mmd_samples: int = 500,
    seed: int = 0,
) -> tuple[DriftResult, FloatArray, FloatArray]:
    """Compare the windows; returns the result and the reference/current PCA projections."""
    ref = np.asarray(reference_embeddings, dtype=np.float64)
    cur = np.asarray(current_embeddings, dtype=np.float64)
    if ref.ndim != 2 or cur.ndim != 2 or ref.shape[1] != cur.shape[1]:
        raise ValueError(f"embedding shapes differ: {ref.shape} vs {cur.shape}")
    if len(ref) < 20 or len(cur) < 20:
        raise ValueError(f"need >= 20 rows per window, got {len(ref)} and {len(cur)}")
    ref_len = np.asarray(reference_lengths, dtype=np.float64)
    cur_len = np.asarray(current_lengths, dtype=np.float64)
    if len(ref_len) != len(ref) or len(cur_len) != len(cur):
        raise ValueError("lengths and embeddings differ in row count")
    if len(reference_categories) != len(ref) or len(current_categories) != len(cur):
        raise ValueError("categories and embeddings differ in row count")

    mean, axes, explained = _pca(ref, n_components)
    ref_pc = (ref - mean) @ axes.T
    cur_pc = (cur - mean) @ axes.T
    scores = {f"pc_{i + 1}": psi(ref_pc[:, i], cur_pc[:, i]) for i in range(axes.shape[0])}
    scores["length_words"] = psi(ref_len, cur_len)
    scores["category"] = categorical_psi(reference_categories, current_categories)
    mmd2, mmd_p, sigma = mmd_permutation_test(
        ref_pc, cur_pc, n_permutations=n_permutations, max_samples=max_mmd_samples, seed=seed
    )
    result = DriftResult(
        n_reference=len(ref),
        n_current=len(cur),
        n_components=int(axes.shape[0]),
        explained_variance=explained,
        psi=scores,
        mmd2=mmd2,
        mmd_p=mmd_p,
        mmd_sigma=sigma,
        mmd_samples=min(max_mmd_samples, len(ref), len(cur)),
        length={"reference": _length_stats(ref_len), "current": _length_stats(cur_len)},
        category={
            "reference": _shares(reference_categories),
            "current": _shares(current_categories),
        },
        thresholds=thresholds or DriftThresholds(),
    )
    return result, ref_pc, cur_pc


# --------------------------------------------------------------------------------------------------
# Evidently
# --------------------------------------------------------------------------------------------------


def _frame(pcs: FloatArray, lengths: npt.ArrayLike, categories: Sequence[str]) -> Any:
    import pandas as pd

    df = pd.DataFrame(pcs, columns=[f"pc_{i + 1}" for i in range(pcs.shape[1])])
    df["length_words"] = np.asarray(lengths, dtype=np.float64)
    df["category"] = list(categories)
    return df


def evidently_report(
    reference: Any,
    current: Any,
    out_dir: Path | None,
    name: str,
    threshold: float = 0.2,
    max_html_bytes: int = MAX_HTML_BYTES,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Evidently ``DataDriftPreset`` (PSI) over two frames; saves HTML (if small enough) + JSON.

    Returns ``({"psi": {column: value}, "drifted_count", "drifted_share"}, {kind: path})``.
    """
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    numeric = [c for c in reference.columns if c != "category"]
    definition = DataDefinition(numerical_columns=numeric, categorical_columns=["category"])
    report = Report([DataDriftPreset(method="psi", threshold=threshold)])
    snapshot = report.run(
        Dataset.from_pandas(current, data_definition=definition),
        Dataset.from_pandas(reference, data_definition=definition),
    )
    values: dict[str, Any] = {"psi": {}}
    for metric in snapshot.dict()["metrics"]:
        cfg = metric.get("config", {})
        if cfg.get("type", "").endswith(":ValueDrift"):
            values["psi"][cfg["column"]] = float(metric["value"])
        elif cfg.get("type", "").endswith(":DriftedColumnsCount"):
            values["drifted_count"] = float(metric["value"]["count"])
            values["drifted_share"] = float(metric["value"]["share"])
    artifacts: dict[str, str] = {}
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / f"input_drift_{name}.evidently.json"
        snapshot.save_json(str(json_path))
        artifacts["evidently_json"] = _rel(json_path)
        html_path = out_dir / f"input_drift_{name}.html"
        snapshot.save_html(str(html_path))
        if html_path.stat().st_size > max_html_bytes:
            html_path.unlink()
            artifacts["evidently_html"] = f"omitted (> {max_html_bytes:,} bytes)"
        else:
            artifacts["evidently_html"] = _rel(html_path)
    return values, artifacts


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(_ROOT))
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------------------------------
# Data loading and embeddings
# --------------------------------------------------------------------------------------------------


def _text_key(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_prompts(path: Path, category: str | None = None) -> tuple[list[PromptRecord], int]:
    """Prompt records of a JSONL window, and the number of records without raw prompt text.

    Accepts split rows, real-slice rows and request-log records (row 11.5.6): ``prompt`` is the
    text, ``id`` the cache id (else a text hash), ``category`` the category (else the real-slice
    rule tag, else ``"unknown"``). ``category`` filters the window to one category.
    """
    from terser.data.real_slice import tag_category

    records: list[PromptRecord] = []
    skipped = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        row = json.loads(line)
        text = row.get("prompt")
        if not isinstance(text, str) or not text.strip():
            skipped += 1
            continue
        cat = row.get("category") or tag_category(text) or "unknown"
        rid = str(row.get("id") or row.get("request_id") or _text_key(text))
        records.append(PromptRecord(rid, text, str(cat)))
    if category is not None:
        records = [r for r in records if r.category == category]
    return records, skipped


def resolve_embeddings(
    records: Sequence[PromptRecord],
    caches: Sequence[EmbeddingCache],
    encoder_factory: Callable[[], Encoder] | None = None,
    text_cache_dir: Path | None = None,
    text_cache_name: str = TEXT_CACHE_NAME,
) -> npt.NDArray[np.float32]:
    """Embedding of each record: by id from ``caches``, else by text hash from the text cache,
    else encoded with ``encoder_factory()`` (called once, only if needed) and added to the text
    cache. ``KeyError`` when a prompt is uncached and no encoder is given."""
    text_cache: EmbeddingCache | None = None
    if text_cache_dir is not None and (text_cache_dir / f"{text_cache_name}.npy").exists():
        text_cache = EmbeddingCache.load(text_cache_dir, text_cache_name)
    lookups = [(c, set(c.ids)) for c in caches]
    if text_cache is not None:
        lookups.append((text_cache, set(text_cache.ids)))

    rows: list[npt.NDArray[np.float32] | None] = []
    for rec in records:
        row = None
        for cache, ids in lookups:
            key = rec.id if rec.id in ids else _text_key(rec.text)
            if key in ids:
                row = cache.get([key])[0]
                break
        rows.append(row)

    missing = sorted(
        {_text_key(r.text): r.text for r, v in zip(records, rows, strict=True) if v is None}.items()
    )
    if missing:
        if encoder_factory is None:
            raise KeyError(f"{len(missing)} prompt(s) not in any embedding cache")
        new = EmbeddingCache.build(
            [k for k, _ in missing], [t for _, t in missing], encoder_factory()
        )
        if text_cache_dir is not None:
            merged = new
            if text_cache is not None:
                merged = EmbeddingCache(
                    text_cache.ids + new.ids, np.vstack([text_cache.matrix, new.matrix])
                )
            merged.save(text_cache_dir, text_cache_name)
        index = {k: i for i, k in enumerate(new.ids)}
        rows = [
            v if v is not None else new.matrix[index[_text_key(r.text)]]
            for r, v in zip(records, rows, strict=True)
        ]
    return np.vstack([np.asarray(v, dtype=np.float32) for v in rows])


# --------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------


def run(
    reference: Sequence[PromptRecord],
    current: Sequence[PromptRecord],
    reference_embeddings: npt.ArrayLike,
    current_embeddings: npt.ArrayLike,
    *,
    name: str,
    out_dir: Path | None = OUT_DIR,
    thresholds: DriftThresholds | None = None,
    n_components: int = 16,
    use_evidently: bool = True,
    n_skipped: int = 0,
    logger: logging.Logger | None = None,
) -> DriftResult:
    """Drift statistics, Evidently report, JSON + markdown summary, and the alert log record."""
    thr = thresholds or DriftThresholds()
    ref_len = [r.length for r in reference]
    cur_len = [r.length for r in current]
    ref_cat = [r.category for r in reference]
    cur_cat = [r.category for r in current]
    result, ref_pc, cur_pc = detect_input_drift(
        reference_embeddings,
        current_embeddings,
        ref_len,
        cur_len,
        ref_cat,
        cur_cat,
        n_components=n_components,
        thresholds=thr,
    )
    result.n_skipped = n_skipped
    if use_evidently:
        result.evidently, result.artifacts = evidently_report(
            _frame(ref_pc, ref_len, ref_cat),
            _frame(cur_pc, cur_len, cur_cat),
            out_dir,
            name,
            threshold=thr.psi,
        )
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_path = out_dir / f"input_drift_{name}.json"
        md_path = out_dir / f"input_drift_{name}.md"
        result.artifacts["summary_json"] = _rel(summary_path)
        result.artifacts["summary_md"] = _rel(md_path)
        summary_path.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
        md_path.write_text(result.markdown(f"Input drift: {name}"), encoding="utf-8")
    log = logger or logging.getLogger(ALERT_LOGGER)
    if result.alert:
        log.critical(result.summary())
    else:
        log.info(result.summary())
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: exit 1 on a drift alert, 0 otherwise; prints the JSON verdict on the last line."""
    ap = argparse.ArgumentParser(description="Input drift report (row 11.5.2)")
    ap.add_argument("--reference", type=Path, default=REFERENCE_PATH)
    ap.add_argument("--current", type=Path, default=CURRENT_PATH)
    ap.add_argument("--current-category", default=None, help="keep only this category")
    ap.add_argument("--name", default=None, help="artifact name (default: current file stem)")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--embeddings-dir", type=Path, default=EMBEDDINGS_DIR)
    ap.add_argument("--embeddings-name", default=EMBEDDINGS_NAME)
    ap.add_argument("--text-cache-dir", type=Path, default=TEXT_CACHE_DIR)
    ap.add_argument("--no-encode", action="store_true", help="fail instead of embedding prompts")
    ap.add_argument("--no-evidently", action="store_true")
    ap.add_argument("--components", type=int, default=16)
    ap.add_argument("--psi-threshold", type=float, default=DriftThresholds.psi)
    ap.add_argument("--mmd-p-threshold", type=float, default=DriftThresholds.mmd_p)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    reference, _ = load_prompts(args.reference)
    current, skipped = load_prompts(args.current, args.current_category)
    name = args.name or args.current.stem + (
        f"_{args.current_category}" if args.current_category else ""
    )
    caches = []
    if (args.embeddings_dir / f"{args.embeddings_name}.npy").exists():
        caches.append(EmbeddingCache.load(args.embeddings_dir, args.embeddings_name))

    def factory() -> Encoder:
        from terser.features.embeddings import bge_small_encoder

        return bge_small_encoder(device="cpu")

    encoder = None if args.no_encode else factory
    ref_emb = resolve_embeddings(reference, caches, encoder, args.text_cache_dir)
    cur_emb = resolve_embeddings(current, caches, encoder, args.text_cache_dir)
    result = run(
        reference,
        current,
        ref_emb,
        cur_emb,
        name=name,
        out_dir=args.out_dir,
        thresholds=DriftThresholds(args.psi_threshold, args.mmd_p_threshold),
        n_components=args.components,
        use_evidently=not args.no_evidently,
        n_skipped=skipped,
    )
    print(result.summary())
    print(json.dumps({"alert": result.alert, "reasons": result.reasons, **result.artifacts}))
    return 1 if result.alert else 0


if __name__ == "__main__":
    raise SystemExit(main())
