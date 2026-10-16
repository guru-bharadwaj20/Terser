"""Row 3.2.3 operating point of the fast-path gate under the service's ``score > t`` rule."""

from __future__ import annotations

import numpy as np

from terser.eval.threshold import choose_threshold
from terser.phase11.train_fast_path import fast_path_savings, operating_point, service_threshold


def test_service_threshold_flags_same_examples_as_choose_threshold() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    # ties on purpose: each score appears twice, as identical feature rows do in the gate data
    s = np.repeat(np.round(rng.random(250), 3), 2) + 0.3 * y
    t = choose_threshold(y, s).threshold
    v = service_threshold(y, s)
    assert v < t
    assert np.array_equal(s > v, s >= t)
    op = operating_point(y, s, v)
    assert op["sensitivity"] >= 0.95


def test_operating_point_rates_and_categories() -> None:
    y = np.array([1, 1, 0, 0, 0])
    s = np.array([0.9, 0.2, 0.6, 0.1, 0.3])
    op = operating_point(y, s, 0.5, ["a", "a", "b", "b", "b"])
    assert op["sensitivity"] == 0.5  # 0.9 rejected, 0.2 passed
    assert op["specificity"] == 2 / 3  # 0.6 rejected
    assert op["pass_rate"] == 3 / 5
    assert op["degraded_share_passed"] == 1 / 3
    assert set(op["per_category"]) == {"a", "b"}
    assert op["per_category"]["a"]["n"] == 2
    assert "auc" not in op["per_category"]["a"]  # one class only: no AUC


def test_fast_path_savings_rejected_rewrite_saves_nothing() -> None:
    prompts = ["a b c d", "e f g h", "keep"]
    res = fast_path_savings(
        prompts,
        rewrite=lambda t: t if t == "keep" else t[:3],
        cost=lambda t: len(t.split()),
        score=lambda i, t, out: 0.9 if i == 0 else 0.1,
        thresholds={"x": 0.5},
    )
    assert res["n_changed"] == 2
    assert res["x"]["rejected"] == 1
    assert res["x"]["mean_savings_after_gate"] == (0 + 0.5 + 0) / 3
