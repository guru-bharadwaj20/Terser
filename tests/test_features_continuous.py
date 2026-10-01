"""Row 3.1.4: standardisation and equal-frequency binning."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from terser.features.continuous import EqualFrequencyBinner, Standardizer


def test_standardizer_matches_sklearn() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(3, 2, size=(50, 4))
    X[:, 3] = 7.0  # constant column
    ours = Standardizer().fit_transform(X)
    ref = StandardScaler().fit_transform(X)
    np.testing.assert_allclose(ours, ref, atol=1e-12)


def test_standardizer_uses_training_statistics() -> None:
    s = Standardizer().fit([[0.0], [2.0]])
    assert s.transform([[4.0]])[0, 0] == pytest.approx(3.0)


def test_equal_frequency_bins_hand_example() -> None:
    x = np.arange(1, 9, dtype=float).reshape(-1, 1)  # 1..8
    b = EqualFrequencyBinner(n_bins=4).fit(x)
    assert b.edges_[0].tolist() == [2.0, 4.0, 6.0]
    assert b.transform(x)[:, 0].tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    assert b.transform([[-5.0], [100.0]])[:, 0].tolist() == [0, 3]


def test_binary_flag_keeps_two_bins() -> None:
    x = np.array([[0.0], [0.0], [0.0], [1.0], [1.0], [0.0]])
    out = EqualFrequencyBinner(n_bins=4).fit_transform(x)
    assert sorted(set(out[:, 0].tolist())) == [0, 1]
    assert (out[:, 0] == x[:, 0]).all()


def test_rejects_single_bin() -> None:
    with pytest.raises(ValueError):
        EqualFrequencyBinner(n_bins=1).fit([[1.0]])
