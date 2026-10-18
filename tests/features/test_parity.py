"""Offline/online feature parity (row 11.1.3): 100/100 prompts must give identical vectors.

Offline = :func:`featurize_frame` over a DataFrame (the training path).
Online  = :meth:`RewriteService.features`, called per request by the serving path.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from terser.features.shared import (
    FEATURE_NAMES,
    GATE_FEATURE_NAMES,
    extract_features,
    featurize_frame,
    featurize_records,
    gate_vector,
)
from terser.serving.accounting import Pricing
from terser.serving.service import RewriteService

FIX = Path(__file__).resolve().parents[1] / "fixtures"


def _prompts(n: int = 100) -> list[str]:
    lines = (FIX / "tokenizer_strings.jsonl").read_text(encoding="utf-8").split("\n")
    texts = [json.loads(x)["text"] for x in lines if x]
    casual = [
        json.loads(x)["prompt"]
        for x in (FIX / "casual_prompts.jsonl").read_text(encoding="utf-8").split("\n")
        if x
    ]
    return (casual + texts)[:n]


@pytest.mark.network
def test_offline_online_parity_100_prompts() -> None:
    prompts = _prompts()
    assert len(prompts) == 100
    offline = featurize_frame(pd.DataFrame({"prompt": prompts}), target="gpt-4o")
    svc = RewriteService(pricing=Pricing({"gpt-4o": 2.5}))
    identical = sum(
        list(offline.iloc[i]) == list(svc.features(p, "gpt-4o").values())
        for i, p in enumerate(prompts)
    )
    assert identical == 100
    assert list(offline.columns) == list(FEATURE_NAMES)


def test_feature_values_and_gate_vector() -> None:
    def words(s: str) -> int:
        return len(s.split())

    f = extract_features("Hey, could you ONLY use Python?\n```x```", counter=words)
    assert list(f) == list(FEATURE_NAMES)
    assert f["n_questions"] == 1 and f["has_code"] == 1 and f["n_constraints"] == 1
    assert f["n_hedges"] == 1 and f["n_lines"] == 2
    assert extract_features("", counter=words)["n_tokens"] == 0
    recs = featurize_records([{"prompt": "a b"}, {"prompt": "c"}], counter=words)
    assert [r["n_words"] for r in recs] == [2, 1]
    v = gate_vector(f, extract_features("use Python", counter=words))
    assert len(v) == len(GATE_FEATURE_NAMES) and v[-1] == 0.0


def _words(s: str) -> int:
    return len(s.split())


def test_has_code_ignores_blank_lines() -> None:
    prose = "First paragraph.\n\n\n\nSecond paragraph after blank lines."
    assert extract_features(prose, counter=_words)["has_code"] == 0
    assert extract_features("look:\n    x = 1", counter=_words)["has_code"] == 1


# Golden values (word counter, no network): an independent record of the training-time feature
# definition, so a change on either the offline or the online side shows up here.
_GOLDEN = {
    "um so basically can you ONLY list 3 things?": {
        "n_words": 9.0,
        "filler_ratio": 3 / 9,
        "n_hedges": 1.0,
        "n_constraints": 1.0,
        "n_questions": 1.0,
        "has_code": 0.0,
    },
    "Do not use numpy.\n\tprint(x)": {
        "n_words": 6.0,
        "n_constraints": 1.0,
        "n_lines": 2.0,
        "has_code": 1.0,
    },
}


@pytest.mark.parametrize("prompt", list(_GOLDEN))
def test_features_match_golden_values(prompt: str) -> None:
    got = extract_features(prompt, counter=_words)
    for name, want in _GOLDEN[prompt].items():
        assert got[name] == pytest.approx(want), name


def test_span_lexicons_and_tokens() -> None:
    from terser.features.alignment import tokenize
    from terser.features.lexicons import is_constraint, is_negation, is_number

    assert tokenize("keep 2.5 kg and 40% now") == ["keep", "2.5", "kg", "and", "40%", "now"]
    assert is_number("2.5") and is_number("40%")
    assert is_negation("don’t") and is_negation("Can’t")
    assert not any(map(is_constraint, ["at", "least", "most"]))
