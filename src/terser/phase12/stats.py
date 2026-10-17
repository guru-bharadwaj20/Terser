"""Bootstrap and McNemar helpers for the final tables (row **12.4**).

Every interval is a 95% percentile bootstrap over **source groups** (all casual variants of one
clean prompt resample together, the same grouping as the splits and the CV), 1,000 resamples,
seed 0. One set of resample indices is drawn per table and shared by every system, so intervals
of different systems come from the same resamples (paired).
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = [
    "CI",
    "N_RESAMPLES",
    "SEED",
    "ci",
    "group_resamples",
    "mcnemar_paired",
    "nanmean",
    "nanmedian",
]

N_RESAMPLES = 1000
SEED = 0
FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class CI:
    """Point estimate on the full data and a percentile interval (``None`` if undefined)."""

    estimate: float | None
    low: float | None
    high: float | None
    dropped: int = 0
    """Resamples whose statistic was not finite (e.g. an empty length bin); the interval is over
    the remaining ones, i.e. conditional on the statistic being defined."""

    def as_dict(self) -> dict[str, float | None]:
        return {"estimate": self.estimate, "low": self.low, "high": self.high}


def group_resamples(
    groups: Sequence[Hashable], n_resamples: int = N_RESAMPLES, seed: int = SEED
) -> list[npt.NDArray[np.int64]]:
    """Index arrays of a cluster bootstrap: draw groups with replacement, keep all members."""
    if not len(groups):
        return []
    _, gid = np.unique(np.asarray([str(g) for g in groups]), return_inverse=True)
    members = [np.flatnonzero(gid == k) for k in range(int(gid.max()) + 1)]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_resamples):
        pick = rng.integers(0, len(members), len(members))
        out.append(np.concatenate([members[k] for k in pick]).astype(np.int64))
    return out


def nanmedian(x: FloatArray) -> float:
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else float("nan")


def nanmean(x: FloatArray) -> float:
    x = x[np.isfinite(x)]
    return float(np.mean(x)) if x.size else float("nan")


def _f(x: float) -> float | None:
    return None if not np.isfinite(x) else float(x)


def ci(
    stat: Callable[[npt.NDArray[np.int64]], float],
    n: int,
    resamples: Sequence[npt.NDArray[np.int64]],
    alpha: float = 0.05,
) -> CI:
    """``stat(indices)`` on all ``n`` rows and its percentile interval over ``resamples``."""
    est = stat(np.arange(n, dtype=np.int64)) if n else float("nan")
    reps = np.asarray([stat(idx) for idx in resamples], dtype=np.float64)
    finite = np.isfinite(reps)
    dropped = int((~finite).sum())
    reps = reps[finite]
    if not reps.size:
        return CI(_f(est), None, None, dropped)
    lo, hi = np.quantile(reps, [alpha / 2, 1 - alpha / 2])
    return CI(_f(est), float(lo), float(hi), dropped)


def mcnemar_paired(a: npt.ArrayLike, b: npt.ArrayLike) -> dict[str, float | int]:
    """Exact McNemar on two systems' preserved flags over the same (prompt, target) pairs.

    ``nan`` entries (unlabelled) drop the pair. ``b`` = only A preserved, ``c`` = only B.
    """
    from terser.eval.significance import mcnemar

    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    xa, ya = x[ok].astype(int), y[ok].astype(int)
    if not len(xa):
        return {"n_pairs": 0, "only_a": 0, "only_b": 0, "statistic": 0.0, "p_value": 1.0}
    r = mcnemar(np.ones(len(xa), dtype=int), xa, ya)
    return {
        "n_pairs": int(ok.sum()),
        "only_a": r.b,
        "only_b": r.c,
        "statistic": r.statistic,
        "p_value": r.p_value,
    }
