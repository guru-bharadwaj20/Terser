"""Tests for the Phase 2 respond and judge/label stages. Fake transport only."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from terser.judge.code_runner import CodeTests
from terser.judge.protocols import averaged_judge
from terser.llm import LLMClient, TransportResult
from terser.phase2.label_stage import TaskReferences, aggregate_sides, run_labels
from terser.phase2.respond import build_items, run_respond, sample_plan

TARGET = "t/target"
JUDGE = "j/judge"


def make_client(tmp_path: Path, transport: Any) -> LLMClient:
    budget = tmp_path / "budget.yaml"
    budget.write_text(
        yaml.safe_dump(
            {
                "phases": {"phase2": {"max_usd": 0, "max_tokens": 10_000_000}},
                "providers": {"local": {"base_url": "http://127.0.0.1:9/v1", "priced": False}},
            }
        ),
        encoding="utf-8",
    )
    return LLMClient(
        "phase2",
        budget_path=budget,
        transport=transport,
        ledger_path=tmp_path / "ledger.json",
        cache_dir=tmp_path / "cache",
    )


def section(text: str, name: str) -> str:
    m = re.search(rf"\[{name}\]\n(.*?)\n\[End of {name}\]", text, re.DOTALL)
    assert m, name
    return m.group(1)


class Fake:
    """Target echoes the prompt (seeded samples add the seed); judge prefers 'GOOD' outputs."""

    def __init__(self) -> None:
        self.target_params: list[dict[str, Any]] = []
        self.judge_calls = 0

    def __call__(self, provider, model, messages, max_tokens, system, params):  # type: ignore[no-untyped-def]
        content = messages[0]["content"]
        if model == TARGET:
            self.target_params.append(params)
            seed = params.get("seed")
            if "add(" in content:
                code = (
                    "def add(a, b):\n    return a + b"
                    if "GOOD" in content
                    else "def add(a, b):\n    return a - b"
                )
                return TransportResult(f"```python\n{code}\n```", 5, 9)
            return TransportResult(f"answer to {content}" + (f" s{seed}" if seed else ""), 5, 7)
        self.judge_calls += 1
        a, b = section(content, "Response A"), section(content, "Response B")
        pa = 0.8 if "GOOD" in a and "GOOD" not in b else 0.1 if "GOOD" in b else 0.3
        pb = 0.8 if "GOOD" in b and "GOOD" not in a else 0.1 if "GOOD" in a else 0.3
        pt = max(1e-6, 1.0 - pa - pb)
        top = (("A", math.log(pa)), ("B", math.log(pb)), ("tie", math.log(pt)))
        best = max(top, key=lambda x: x[1])[0]
        toks = (('{"verdict": "', (("x", 0.0),)), (best, top))
        return TransportResult(json.dumps({"verdict": best, "reason": "r"}), 50, 9, True, toks)


PROMPTS = [
    {
        "id": "f:1:texting",
        "source_id": "f:1",
        "category": "fact",
        "prompt": "q one pls GOOD",
        "reference": "answer",
    },
    {"id": "c:1:hedging", "source_id": "c:1", "category": "creative", "prompt": "poem pls GOOD"},
    {"id": "k:1:texting", "source_id": "k:1", "category": "code", "prompt": "write add( GOOD"},
]
CANDS = [
    {
        "prompt_id": "f:1:texting",
        "candidate_id": "f:1:texting:llm_high",
        "text": "q one",
        "rejected": False,
    },
    {
        "prompt_id": "f:1:texting",
        "candidate_id": "f:1:texting:ops_FHC",
        "text": "q one pls GOOD",
        "rejected": False,
    },
    {
        "prompt_id": "f:1:texting",
        "candidate_id": "f:1:texting:llm_low",
        "text": "x",
        "rejected": True,
    },
    {
        "prompt_id": "c:1:hedging",
        "candidate_id": "c:1:hedging:llm_high",
        "text": "poem GOOD",
        "rejected": False,
    },
    {
        "prompt_id": "k:1:texting",
        "candidate_id": "k:1:texting:llm_high",
        "text": "write add(",
        "rejected": False,
    },
]


def test_sample_plan() -> None:
    assert sample_plan("fact") == [(0, 0.0, None)]
    assert sample_plan("creative") == [(0, 0.7, 1000), (1, 0.7, 1001), (2, 0.7, 1002)]
    assert sample_plan("creative", creative_samples=1) == [(0, 0.0, None)]


def test_respond_then_label(tmp_path: Path) -> None:
    fake = Fake()
    client = make_client(tmp_path, fake)
    items = build_items(PROMPTS, CANDS)
    assert len(items) == 3 + 4  # rejected candidate dropped
    out = tmp_path / "responses.jsonl"
    stats = run_respond(items, client, target=TARGET, out_path=out, workers=4)
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    # fact: 3 items x 1; creative: 2 items x 3 samples; code: 2 x 1
    assert stats.records == len(rows) == 3 + 6 + 2
    # identical texts are requested once
    assert stats.requests == 2 + 6 + 2
    assert {p.get("seed") for p in fake.target_params} == {None, 1000, 1001, 1002}

    refs = TaskReferences(
        {
            "f:1": "answer",
            "k:1": CodeTests.mbpp(["assert add(1, 2) == 3", "assert add(0, 0) == 0"]),
        }
    )
    judge = averaged_judge(client, provider="local", model=JUDGE)
    labels_path = tmp_path / "labels.jsonl"
    lstats = run_labels(
        PROMPTS,
        rows,
        judge,
        out_path=labels_path,
        references=refs,
        metric_cache=tmp_path / "mcache",
        workers=4,
        splits={"f:1:texting": "train"},
    )
    recs = {r["candidate_id"]: r for r in map(json.loads, labels_path.read_text().splitlines())}
    assert lstats.records == 4 and set(recs) == {
        c["candidate_id"] for c in CANDS if not c["rejected"]
    }
    same = recs["f:1:texting:ops_FHC"]
    assert same["identical_outputs"] and same["label"] == "preserved"
    assert same["split"] == "train" and same["target"] == TARGET
    worse = recs["f:1:texting:llm_high"]
    assert worse["aggregate"]["side"] == "original" and worse["label"] == "degraded"
    poem = recs["c:1:hedging:llm_high"]
    assert len(poem["samples"]) == 3 and poem["label"] == "preserved"
    code = recs["k:1:texting:llm_high"]
    assert code["task_metric"] == {"original": 1.0, "rewrite": 0.5}
    assert code["task_not_worse"] is False and code["label"] == "degraded"
    assert lstats.identical_pairs == 1 and lstats.judged_pairs == 3
    calls_before = fake.judge_calls
    run_labels(
        PROMPTS,
        rows,
        judge,
        out_path=labels_path,
        references=refs,
        metric_cache=tmp_path / "mcache",
    )
    assert fake.judge_calls == calls_before  # fully cached rerun


def test_aggregate_sides() -> None:
    assert aggregate_sides([]) == (None, None)
    assert aggregate_sides([None]) == (None, None)
    side, avg = aggregate_sides(
        [
            {"original": 0.6, "rewrite": 0.3, "tie": 0.1},
            {"original": 0.2, "rewrite": 0.7, "tie": 0.1},
        ]
    )
    assert side == "rewrite" and avg == pytest.approx({"original": 0.4, "rewrite": 0.5, "tie": 0.1})


def test_stand_ins_never_reach_outputs_or_labels(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="stand-in candidate"):
        build_items(PROMPTS, [{**CANDS[0], "stand_in": True}])
    with pytest.raises(ValueError, match="stand-in prompt"):
        build_items([{**PROMPTS[0], "stand_in": "True"}], CANDS)
    resp = {"target": "t", "prompt_id": "p", "candidate_id": "p:x", "sample": 0, "output": "o"}
    with pytest.raises(ValueError, match="never be labelled"):
        run_labels(PROMPTS, [{**resp, "stand_in": True}], None, out_path=tmp_path / "l.jsonl")  # type: ignore[arg-type]
