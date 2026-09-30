"""Tests for terser.eval.metrics against scikit-learn (row 3.2.1: match to 1e-9)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn import metrics as skm

from terser.eval.labels import POSITIVE_LABEL
from terser.eval.metrics import classification_metrics, confusion_matrix, per_category_metrics


@pytest.mark.parametrize("seed", range(20))
def test_matches_sklearn(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(5, 300))
    y = rng.integers(0, 2, n)
    p = np.where(rng.random(n) < 0.7, y, 1 - y)
    m = classification_metrics(y, p)
    kw = {"pos_label": POSITIVE_LABEL, "zero_division": 0}
    assert m.accuracy == pytest.approx(skm.accuracy_score(y, p), abs=1e-9)
    assert m.precision == pytest.approx(skm.precision_score(y, p, **kw), abs=1e-9)
    assert m.recall == pytest.approx(skm.recall_score(y, p, **kw), abs=1e-9)
    assert m.sensitivity == m.recall
    assert m.specificity == pytest.approx(skm.recall_score(y, p, pos_label=0, zero_division=0))
    assert m.f1 == pytest.approx(skm.f1_score(y, p, **kw), abs=1e-9)
    np.testing.assert_array_equal(confusion_matrix(y, p), skm.confusion_matrix(y, p, labels=[0, 1]))


def test_string_labels_and_degenerate_cases() -> None:
    m = classification_metrics(["preserved", "preserved"], ["preserved", "preserved"])
    assert (m.accuracy, m.precision, m.recall, m.f1, m.specificity) == (1.0, 0.0, 0.0, 0.0, 1.0)
    assert m.as_dict()["n"] == 2
    with pytest.raises(ValueError):
        classification_metrics([], [])
    with pytest.raises(ValueError):
        classification_metrics([0, 1], [0])


def test_per_category() -> None:
    y = [1, 0, 1, 0, 1, 1]
    p = [1, 0, 0, 0, 1, 0]
    cats = ["code", "code", "fact", "fact", "code", "fact"]
    out = per_category_metrics(y, p, cats)
    assert list(out) == ["code", "fact", "overall"]
    assert out["code"].recall == 1.0
    assert out["fact"].recall == 0.0
    assert out["overall"].tp == 2 and out["overall"].n == 6
    with pytest.raises(ValueError):
        per_category_metrics(y, p, cats[:-1])
