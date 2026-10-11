"""Row 8.3.1: target-agnostic SFT pairs from judge labels."""

from __future__ import annotations

from terser.models.rewriter.sft_data import build_sft_rows

PROMPTS = [
    {"id": "a", "category": "qa", "prompt": "hey so um what is two plus two"},
    {"id": "b", "category": "qa", "prompt": "pls name a colour"},
]
CANDIDATES = [
    {"prompt_id": "a", "candidate_id": "a:short", "method": "llm_high", "text": "2+2?"},
    {"prompt_id": "a", "candidate_id": "a:mid", "method": "llm_low", "text": "What is 2+2?"},
    {"prompt_id": "b", "candidate_id": "b:short", "method": "llm_high", "text": "Colour?"},
]
TOKENS = {"a:original": 10, "a:short": 3, "a:mid": 5, "b:original": 4, "b:short": 2}


def _label(cid: str, target: str, label: str) -> dict[str, str]:
    return {"candidate_id": cid, "target": target, "label": label}


def test_shortest_preserved_on_every_target() -> None:
    labels = [
        _label("a:short", "q", "preserved"),
        _label("a:short", "p", "degraded"),
        _label("a:mid", "q", "preserved"),
        _label("a:mid", "p", "preserved"),
        _label("b:short", "q", "degraded"),
        _label("b:short", "p", "preserved"),
    ]
    a, b = build_sft_rows(PROMPTS, CANDIDATES, labels, TOKENS)
    assert a["rewrite"] == "What is 2+2?" and a["savings"] == 0.5
    assert b["method"] == "original" and b["rewrite"] == PROMPTS[1]["prompt"]
    assert b["savings"] == 0.0
