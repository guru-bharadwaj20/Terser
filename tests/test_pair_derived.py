"""Derived pair features are computed from table columns (row 5.1.7), not stored by the builder."""

from __future__ import annotations

import pandas as pd
import pytest

from terser.features.pair import DERIVED_FEATURES, PAIR_FEATURES, add_derived_features


def test_si_removed_share_from_columns() -> None:
    df = pd.DataFrame(
        {
            "sum_si_removed": [10.0, 0.0, 5.0],
            "prompt_mean_si": [2.0, 3.0, 0.0],
            "prompt_words": [10, 4, 7],
        }
    )
    out = add_derived_features(df)
    assert out["si_removed_share"].tolist() == pytest.approx([0.5, 0.0, 0.0])
    assert "si_removed_share" not in df.columns  # input untouched


def test_derived_features_are_not_table_columns() -> None:
    assert not set(DERIVED_FEATURES) & set(PAIR_FEATURES)
