"""Cluster bootstrap and paired McNemar helpers."""

from __future__ import annotations

import numpy as np

from terser.phase12.stats import ci, group_resamples, mcnemar_paired, nanmean, nanmedian


def test_group_resamples_keep_groups_whole() -> None:
    groups = ["a", "a", "b", "c", "c", "c"]
    for idx in group_resamples(groups, n_resamples=50, seed=1):
        picked = [groups[i] for i in idx]
        for g, size in (("a", 2), ("c", 3)):
            assert picked.count(g) % size == 0
    assert [list(x) for x in group_resamples(groups, 3, 7)] == [
        list(x) for x in group_resamples(groups, 3, 7)
    ]


def test_ci_brackets_estimate() -> None:
    x = np.linspace(0, 1, 101)
    rs = group_resamples([str(i) for i in range(101)], 200, 0)
    c = ci(lambda ix: nanmedian(x[ix]), 101, rs)
    assert c.low is not None and c.high is not None
    assert c.low <= c.estimate <= c.high  # type: ignore[operator]
    const = ci(lambda ix: 0.5, 5, rs[:10])
    assert const.as_dict() == {"estimate": 0.5, "low": 0.5, "high": 0.5}
    assert np.isnan(nanmean(np.array([np.nan])))


def test_mcnemar_drops_unlabelled_pairs() -> None:
    a = [1, 1, 1, 0, np.nan, 1]
    b = [1, 0, 0, 0, 1, np.nan]
    r = mcnemar_paired(a, b)
    assert r["n_pairs"] == 4 and r["only_a"] == 2 and r["only_b"] == 0
    assert mcnemar_paired([np.nan], [1])["p_value"] == 1.0
