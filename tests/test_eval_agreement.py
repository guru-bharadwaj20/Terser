"""Tests for terser.eval.agreement on synthetic data with known kappa."""

from __future__ import annotations

import math
import random

import numpy as np
import pytest
from sklearn.metrics import cohen_kappa_score

from terser.eval.agreement import (
    THRESHOLDS,
    checkpoint_checks,
    cohen_kappa,
    confusion_table,
    detector_metrics,
    kappa_with_ci,
    per_category_kappa,
    position_disagreement_rate,
)


def known_kappa_pair(n_per_class: int, flips_per_class: int) -> tuple[list[str], list[str]]:
    """Balanced marginals for both raters, so p_e = 0.5 and kappa = 1 - 4 * flips / n."""
    r1 = ["preserved"] * n_per_class + ["degraded"] * n_per_class
    r2 = list(r1)
    for k in range(flips_per_class):
        r2[k] = "degraded"
        r2[n_per_class + k] = "preserved"
    return r1, r2


@pytest.mark.parametrize(("flips", "expected"), [(0, 1.0), (15, 0.70), (25, 0.50), (50, 0.0)])
def test_known_kappa(flips: int, expected: float) -> None:
    r1, r2 = known_kappa_pair(100, flips)
    assert cohen_kappa(r1, r2) == pytest.approx(expected)


def test_full_disagreement_is_minus_one() -> None:
    r1, r2 = known_kappa_pair(100, 100)
    assert cohen_kappa(r1, r2) == pytest.approx(-1.0)


@pytest.mark.parametrize("seed", range(20))
def test_matches_sklearn(seed: int) -> None:
    rng = random.Random(seed)
    cats = ["original", "rewrite", "tie"][: rng.choice([2, 3])]
    n = rng.randint(5, 400)
    y1 = [rng.choice(cats) for _ in range(n)]
    y2 = [a if rng.random() < rng.random() else rng.choice(cats) for a in y1]
    ours = cohen_kappa(y1, y2)
    theirs = cohen_kappa_score(y1, y2)
    if math.isnan(theirs):
        assert math.isnan(ours)
    else:
        assert ours == pytest.approx(theirs, abs=1e-12)


def test_matches_sklearn_with_ints_and_label_set() -> None:
    y1 = [0, 1, 1, 0, 1, 1, 0, 0, 1, 1]
    y2 = [0, 1, 0, 0, 1, 1, 1, 0, 1, 0]
    assert cohen_kappa(y1, y2, labels=[0, 1]) == pytest.approx(cohen_kappa_score(y1, y2))


def test_none_dropped_and_undefined_cases() -> None:
    assert cohen_kappa(["a", None, "b"], ["a", "b", "b"]) == pytest.approx(1.0)
    assert math.isnan(cohen_kappa(["a", "a"], ["a", "a"]))
    assert math.isnan(cohen_kappa([None], ["a"]))
    with pytest.raises(ValueError, match="length"):
        cohen_kappa(["a"], ["a", "b"])
    with pytest.raises(ValueError, match="label set"):
        cohen_kappa(["a", "x"], ["a", "a"], labels=["a", "b"])


def test_bootstrap_ci_covers_known_kappa_and_is_deterministic() -> None:
    r1, r2 = known_kappa_pair(150, 30)  # kappa = 1 - 120/300 = 0.60
    res = kappa_with_ci(r1 + [None], r2 + ["degraded"], n_boot=1000, seed=3)
    assert res.kappa == pytest.approx(0.60)
    assert res.n == 300 and res.n_dropped == 1
    assert res.observed_agreement == pytest.approx(0.80)
    assert res.ci_low < 0.60 < res.ci_high
    assert 0.05 < res.ci_high - res.ci_low < 0.30
    again = kappa_with_ci(r1, r2, n_boot=1000, seed=3)
    assert (again.ci_low, again.ci_high) == (res.ci_low, res.ci_high)
    assert "0.600" in res.format()


def test_bootstrap_ci_width_shrinks_with_n() -> None:
    small = kappa_with_ci(*known_kappa_pair(25, 5), n_boot=800, seed=1)
    large = kappa_with_ci(*known_kappa_pair(400, 80), n_boot=800, seed=1)
    assert small.kappa == pytest.approx(large.kappa)
    assert (large.ci_high - large.ci_low) < (small.ci_high - small.ci_low)


def test_bootstrap_argument_checks() -> None:
    with pytest.raises(ValueError):
        kappa_with_ci(["a"], ["a"], n_boot=0)
    with pytest.raises(ValueError):
        kappa_with_ci(["a"], ["a"], alpha=1.5)
    empty = kappa_with_ci([None], [None], n_boot=10)
    assert math.isnan(empty.kappa) and math.isnan(empty.ci_low) and empty.n == 0


def test_per_category_kappa() -> None:
    a1, a2 = known_kappa_pair(25, 0)  # kappa 1
    b1, b2 = known_kappa_pair(24, 12)  # kappa = 1 - 4 * 12 / 48 = 0
    cats = ["code"] * 50 + ["fact"] * 48
    res = per_category_kappa(cats, a1 + b1, a2 + b2, n_boot=200)
    assert list(res) == ["code", "fact"]
    assert res["code"].kappa == pytest.approx(1.0)
    assert res["fact"].kappa == pytest.approx(0.0)
    with pytest.raises(ValueError):
        per_category_kappa(["code"], ["a", "b"], ["a", "b"])


def test_confusion_table() -> None:
    r1, r2 = known_kappa_pair(10, 3)
    table = confusion_table(r1 + [None], r2 + ["preserved"], row_name="h", col_name="j")
    assert table.labels == ("preserved", "degraded")
    assert table.matrix == ((7, 3), (3, 7))
    md = table.markdown()
    assert "| h \\ j | preserved | degraded | total |" in md
    assert "| total | 10 | 10 | 20 |" in md


def test_detector_metrics_degraded_is_positive() -> None:
    human = ["degraded"] * 20 + ["preserved"] * 80 + [None]
    judge = ["degraded"] * 17 + ["preserved"] * 3 + ["preserved"] * 64 + ["degraded"] * 16
    judge.append("degraded")
    m = detector_metrics(human, judge)
    assert (m.tp, m.fn, m.tn, m.fp) == (17, 3, 64, 16)
    assert m.sensitivity == pytest.approx(0.85)
    assert m.specificity == pytest.approx(0.80)
    assert m.n_dropped == 1


def test_position_disagreement_rate() -> None:
    ab = ["rewrite", "tie", "original", None, "tie", "rewrite", "rewrite", "tie", "tie", "tie", "o"]
    ba = ["rewrite", "rewrite", "original", "tie", "tie", "rewrite", "rewrite", "tie", "tie", "tie"]
    ba.append("o")
    rate, dis, n = position_disagreement_rate(ab, ba)
    assert (dis, n) == (1, 10)
    assert rate == pytest.approx(0.10)
    assert math.isnan(position_disagreement_rate([None], [None])[0])


def _checks(**over: float) -> list[bool]:
    cat_kappa = over.pop("cat", 0.9)
    values = {
        "human_human_kappa": 0.60,
        "judge_human_kappa": 0.60,
        "sensitivity": 0.85,
        "specificity": 0.80,
        "position_disagreement": 0.10,
    }
    values.update(over)
    per_cat = {"code": 0.5, "fact": cat_kappa}
    results = checkpoint_checks(judge_human_kappa_per_category=per_cat, **values)
    return [c.passed for c in results]


def test_checkpoint_boundaries_pass() -> None:
    assert all(_checks())
    lines = [
        c.line()
        for c in checkpoint_checks(
            human_human_kappa=0.6,
            judge_human_kappa=0.7,
            judge_human_kappa_per_category={"code": 0.4},
            sensitivity=0.9,
            specificity=float("nan"),
            position_disagreement=0.2,
        )
    ]
    assert lines[0] == "C2.3 PASS - human-human kappa = 0.600 (>= 0.60)"
    assert lines[2].startswith("C2.4 FAIL - judge-human kappa (code)")
    assert lines[4] == "C2.5 FAIL - judge specificity = n/a (>= 0.80)"
    assert lines[5].startswith("C2.6 FAIL") and "(<= 0.10)" in lines[5]


@pytest.mark.parametrize(
    ("override", "failing_index"),
    [
        ({"human_human_kappa": 0.59}, 0),
        ({"judge_human_kappa": 0.599}, 1),
        ({"cat": 0.49}, 3),
        ({"sensitivity": 0.849}, 4),
        ({"specificity": 0.79}, 5),
        ({"position_disagreement": 0.1001}, 6),
        ({"human_human_kappa": float("nan")}, 0),
    ],
)
def test_checkpoint_failures(override: dict[str, float], failing_index: int) -> None:
    passed = _checks(**override)
    assert [i for i, p in enumerate(passed) if not p] == [failing_index]


def test_thresholds_match_plan() -> None:
    assert dict(THRESHOLDS) == {
        "human_human_kappa": 0.60,
        "judge_human_kappa": 0.60,
        "judge_human_kappa_per_category": 0.50,
        "sensitivity": 0.85,
        "specificity": 0.80,
        "position_disagreement": 0.10,
    }


def test_kappa_symmetric_random() -> None:
    rng = np.random.default_rng(0)
    y1 = rng.choice(["preserved", "degraded"], size=100).tolist()
    y2 = rng.choice(["preserved", "degraded"], size=100).tolist()
    assert cohen_kappa(y1, y2) == pytest.approx(cohen_kappa(y2, y1))
