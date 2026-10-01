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
