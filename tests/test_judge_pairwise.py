"""Tests for terser.judge.pairwise: parsing, A/B swap, label rule, retries. Fake transport only."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from terser.judge.pairwise import (
    PairwiseJudge,
    label_from_sides,
    load_prompt_template,
    parse_verdict,
    render_prompt,
    to_side,
)
from terser.llm import LLMClient, TransportResult


def make_client(tmp_path: Path, transport: Any, max_tokens: int = 1_000_000) -> LLMClient:
    budget = tmp_path / "budget.yaml"
    budget.write_text(
        yaml.safe_dump(
            {
                "phases": {"phase1": {"max_usd": 0, "max_tokens": max_tokens}},
                "providers": {"local": {"base_url": "http://127.0.0.1:9/v1", "priced": False}},
            }
        ),
        encoding="utf-8",
    )
    return LLMClient(
        "phase1",
        budget_path=budget,
        transport=transport,
        ledger_path=tmp_path / "ledger.json",
        cache_dir=tmp_path / "cache",
    )


def section(text: str, name: str) -> str:
    m = re.search(rf"\[{name}\]\n(.*?)\n\[End of {name}\]", text, re.DOTALL | re.I)
    assert m, name
    return m.group(1)


class PrefersJudge:
    """Fake judge transport that always prefers the response containing ``favourite``."""

    def __init__(self, favourite: str | None, replies: list[str] | None = None) -> None:
        self.favourite = favourite
        self.replies = list(replies or [])
        self.calls: list[list[dict[str, Any]]] = []

    def __call__(
        self,
        provider: str,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        system: str | None,
        params: dict[str, Any],
    ) -> TransportResult:
        self.calls.append(messages)
        if self.replies:
            return TransportResult(self.replies.pop(0), 10, 5)
        body = messages[0]["content"]
        a, b = section(body, "Response A"), section(body, "Response B")
        if self.favourite is None:
            verdict = "tie"
        else:
            verdict = "A" if self.favourite in a else "B" if self.favourite in b else "tie"
        return TransportResult(f'{{"verdict": "{verdict}", "reason": "because"}}', 10, 5)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"verdict": "A", "reason": "more complete"}', ("A", "more complete")),
        ('{"verdict": "b", "reason": "x"}', ("B", "x")),
        ('{"verdict": "Tie", "reason": ""}', ("tie", "")),
        ('Sure!\n```json\n{"verdict": "B", "reason": "ok"}\n```\nHope this helps', ("B", "ok")),
        ('Here: {"winner": "Response A", "explanation": "e"} done', ("A", "e")),
        ('{"Verdict": "equal", "Reasoning": "same"}', ("tie", "same")),
        (
            '{"verdict": "A", "reason": "uses {braces} and \\"quotes\\""}',
            ("A", 'uses {braces} and "quotes"'),
        ),
        ("verdict: B", ("B", "")),
        ('{"verdict": "A"', ("A", "")),  # truncated after the verdict
    ],
)
def test_parse_verdict_accepts_common_formats(text: str, expected: tuple[str, str]) -> None:
    assert parse_verdict(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "I think A is better",
        '{"verdict": "A or B"}',
        '{"reason": "no verdict"}',
        "[1, 2, 3]",
        '{"verdict": 1}',
    ],
)
def test_parse_verdict_rejects_malformed(text: str) -> None:
    assert parse_verdict(text) is None


def test_to_side_maps_positions_per_order() -> None:
    assert to_side("A", "original_first") == "original"
    assert to_side("B", "original_first") == "rewrite"
    assert to_side("A", "rewrite_first") == "rewrite"
    assert to_side("B", "rewrite_first") == "original"
    assert to_side("tie", "rewrite_first") == "tie"
    assert to_side(None, "original_first") is None


@pytest.mark.parametrize(
    ("first", "second", "label"),
    [
        ("rewrite", "rewrite", "preserved"),
        ("rewrite", "tie", "preserved"),
        ("tie", "rewrite", "preserved"),
        ("tie", "tie", "preserved"),
        ("original", "original", "degraded"),
        ("original", "tie", "degraded"),
        ("tie", "original", "degraded"),
        ("original", "rewrite", "degraded"),
        ("rewrite", "original", "degraded"),
        ("original", None, "degraded"),
        (None, "original", "degraded"),
        (None, "tie", None),
        ("rewrite", None, None),
        (None, None, None),
    ],
)
def test_label_truth_table(first: Any, second: Any, label: str | None) -> None:
    assert label_from_sides(first, second) == label


def test_render_prompt_single_pass_and_template_braces_kept() -> None:
    template = load_prompt_template()
    out = render_prompt(template, "Q {response_b}", "ans {prompt}", "other")
    assert section(out, "User request") == "Q {response_b}"
    assert section(out, "Response A") == "ans {prompt}"
    assert section(out, "Response B") == "other"
    assert '{"verdict": "A" | "B" | "tie"' in out


def test_judge_pair_swaps_positions(tmp_path: Path) -> None:
    fake = PrefersJudge(favourite="ORIG")
    judge = PairwiseJudge(make_client(tmp_path, fake), provider="local", model="judge")
    result = judge.judge_pair("prompt", "ORIG output", "rewrite output")
    first, second = (section(c[0]["content"], "Response A") for c in fake.calls)
    assert first == "ORIG output" and second == "rewrite output"
    # Position-consistent preference for the original: A then B.
    assert (result.original_first.verdict, result.rewrite_first.verdict) == ("A", "B")
    assert result.original_first.side == result.rewrite_first.side == "original"
    assert result.position_consistent and result.label == "degraded"


def test_judge_pair_position_bias_is_degraded(tmp_path: Path) -> None:
    # A judge that always says "A" prefers the original in the first order only.
    fake = PrefersJudge(None, replies=['{"verdict": "A", "reason": "r"}'] * 2)
    judge = PairwiseJudge(make_client(tmp_path, fake), provider="local", model="judge")
    result = judge.judge_pair("prompt", "o", "r")
    assert result.original_first.side == "original"
    assert result.rewrite_first.side == "rewrite"
    assert not result.position_consistent
    assert result.label == "degraded"


def test_judge_retries_malformed_then_succeeds(tmp_path: Path) -> None:
    fake = PrefersJudge(None, replies=["garbage", '{"verdict": "tie", "reason": "same"}'])
    judge = PairwiseJudge(make_client(tmp_path, fake), provider="local", model="judge", retries=2)
    call = judge.judge_once("p", "a", "b", "original_first")
    assert call.verdict == "tie" and call.attempts == 2
    retry_msgs = fake.calls[1]
    assert retry_msgs[1] == {"role": "assistant", "content": "garbage"}
    assert "could not be parsed" in retry_msgs[2]["content"]


def test_judge_gives_up_after_retries(tmp_path: Path) -> None:
    fake = PrefersJudge(None, replies=["nope"] * 3 + ['{"verdict": "tie"}'] * 3)
    judge = PairwiseJudge(make_client(tmp_path, fake), provider="local", model="judge", retries=2)
    result = judge.judge_pair("p", "a", "b")
    assert result.original_first.verdict is None and result.original_first.attempts == 3
    assert result.parse_failed and result.label is None


def test_judge_uses_temperature_zero(tmp_path: Path) -> None:
    seen: list[dict[str, Any]] = []

    def transport(*args: Any) -> TransportResult:
        seen.append(args[5])
        return TransportResult('{"verdict": "tie", "reason": ""}', 1, 1)

    judge = PairwiseJudge(make_client(tmp_path, transport), provider="local", model="j")
    judge.judge_once("p", "a", "b", "original_first")
    assert seen == [{"temperature": 0.0}]
