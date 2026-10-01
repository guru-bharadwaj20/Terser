"""Loading the Phase 3 feature tables that Phase 4 and 5 experiments consume.

Contract (produced by rows 3.1.2 / 3.1.3, or by :mod:`terser.features.pair_embedding` for 4.1.2):
a CSV or Parquet table with one row per example and the columns

* ``id``: example id;
* ``group``: source-prompt id (the grouping key of the 3.2.4 protocol);
* ``category``: task category;
* ``label``: 1 = degraded / filler / positive class, 0 otherwise;
* every other numeric column is a feature.

Alternatively ``features`` may point at a ``.npy`` matrix (e.g. ``data/features/pair_emb.npy``)
whose rows align with a metadata table holding the four key columns.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

__all__ = ["KEY_COLUMNS", "FeatureTable", "from_frame", "load_table"]

KEY_COLUMNS: tuple[str, ...] = ("id", "group", "category", "label")


@dataclass(frozen=True)
class FeatureTable:
    X: npt.NDArray[np.float64]
    y: npt.NDArray[np.int64]
    groups: list[str]
    categories: list[str]
    ids: list[str]
    feature_names: list[str]


def _read(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def from_frame(df: pd.DataFrame, X: npt.ArrayLike | None = None) -> FeatureTable:
    """Build a :class:`FeatureTable` from a frame (and an optional aligned feature matrix)."""
    missing = [c for c in KEY_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"feature table is missing key columns {missing}")
    if X is None:
        feats = [c for c in df.columns if c not in KEY_COLUMNS]
        Xa = df[feats].to_numpy(dtype=np.float64)
    else:
        Xa = np.asarray(X, dtype=np.float64)
        if len(Xa) != len(df):
            raise ValueError(f"feature matrix has {len(Xa)} rows, metadata {len(df)}")
        feats = [f"f{i}" for i in range(Xa.shape[1])]
    return FeatureTable(
        X=Xa,
        y=df["label"].to_numpy(dtype=np.int64),
        groups=df["group"].astype(str).tolist(),
        categories=df["category"].astype(str).tolist(),
        ids=df["id"].astype(str).tolist(),
        feature_names=feats,
    )


def load_table(path: Path, features: Path | None = None) -> FeatureTable:
    """Load a feature table; ``features`` (``.npy``) overrides the table's feature columns."""
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} does not exist: build the Phase 3 feature tables (rows 3.1.2 / 3.1.3) first"
        )
    df = _read(path)
    return from_frame(df, None if features is None else np.load(features))
