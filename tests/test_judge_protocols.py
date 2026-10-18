"""Tests for the ADR-0005 judge protocols: logprob verdicts, order averaging, pointwise scoring."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from terser.judge.pairwise import JudgeCall, PairwiseJudge, verdict_probs
from terser.judge.pointwise import (
    PointScore,
    PointwiseJudge,
    parse_scores,
    side_from_scores,
    weighted_score,
)
from terser.judge.protocols import call_side_probs, order_averaged_side
from terser.judge.rubrics import load_rubric
from terser.llm import LLMClient, TransportResult


def make_client(tmp_path: Path, transport: Any) -> LLMClient:
    budget = tmp_path / "budget.yaml"
    budget.write_text(
        yaml.safe_dump(
            {
                "phases": {"phase2": {"max_usd": 0, "max_tokens": 1_000_000}},
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


def lp(tokens: list[tuple[str, dict[str, float]]]) -> tuple[Any, ...]:
    """Build TokenLogprobs from (token, {candidate: prob}) with probabilities, not logprobs."""
    return tuple((tok, tuple((c, math.log(p)) for c, p in top.items())) for tok, top in tokens)


# ---------------------------------------------------------------------------- verdict_probs


def test_verdict_probs_reads_the_verdict_token() -> None:
    logprobs = lp(
        [
            ('{"', {'{"': 1.0}),
            ("verdict", {"verdict": 1.0}),
            ('":', {'":': 1.0}),
            (' "', {' "': 1.0}),
            ("A", {"A": 0.6, "tie": 0.3, "B": 0.05, "The": 0.05}),
            ('",', {'",': 1.0}),
        ]
    )
    probs = verdict_probs('{"verdict": "A",', logprobs)
    assert probs is not None
    assert probs["A"] == pytest.approx(0.6 / 0.95)
    assert probs["tie"] == pytest.approx(0.3 / 0.95)
    assert probs["B"] == pytest.approx(0.05 / 0.95)


def test_verdict_probs_handles_merged_quote_tokens_and_missing_cases() -> None:
    merged = lp([('{"verdict": ', {"x": 1.0}), ('"B', {'"B': 0.7, '"A': 0.2, '"tie': 0.1})])
    probs = verdict_probs('{"verdict": "B', merged)
    assert probs == pytest.approx({"A": 0.2, "B": 0.7, "tie": 0.1})
    assert verdict_probs("x", None) is None
    assert verdict_probs("no key", lp([("no key", {"no key": 1.0})])) is None
    no_cand = lp([('{"verdict": "', {"x": 1.0}), ("maybe", {"maybe": 1.0})])
    assert verdict_probs('{"verdict": "maybe', no_cand) is None


# ---------------------------------------------------------------------------- order averaging


def call(order: str, side: str | None, probs: dict[str, float] | None) -> JudgeCall:
    verdict = None if side is None else "tie"
    return JudgeCall(order, verdict, side, "", "", 1, False, 1, probs)  # type: ignore[arg-type]


def test_call_side_probs_maps_positions_to_sides() -> None:
    p = {"A": 0.7, "B": 0.2, "tie": 0.1}
    assert call_side_probs(call("original_first", "original", p)) == {
        "original": 0.7,
        "rewrite": 0.2,
        "tie": 0.1,
    }
    assert call_side_probs(call("rewrite_first", "rewrite", p)) == {
        "original": 0.2,
        "rewrite": 0.7,
        "tie": 0.1,
    }
    assert call_side_probs(call("rewrite_first", "tie", None)) == {
        "original": 0.0,
        "rewrite": 0.0,
        "tie": 1.0,
    }
    assert call_side_probs(call("rewrite_first", None, p)) is None


def test_order_average_cancels_a_pure_first_position_bias() -> None:
    # Judge always picks A with 0.6, B 0.3, tie 0.1: averaged, the two sides balance exactly,
    # so neither wins on position alone and the conservative tie-break applies.
    a_bias = {"A": 0.6, "B": 0.3, "tie": 0.1}
    side, avg = order_averaged_side(
        call("original_first", "original", a_bias), call("rewrite_first", "rewrite", a_bias)
    )
    assert avg == pytest.approx({"original": 0.45, "rewrite": 0.45, "tie": 0.1})
    assert side == "original"


def test_order_average_keeps_a_real_preference_and_is_order_free() -> None:
    first = call("original_first", "original", {"A": 0.55, "B": 0.4, "tie": 0.05})
    second = call("rewrite_first", "original", {"A": 0.1, "B": 0.85, "tie": 0.05})
    side, _ = order_averaged_side(first, second)
    assert side == "original"
    assert order_averaged_side(first, call("rewrite_first", None, None)) == (None, None)


def test_pairwise_judge_requests_logprobs(tmp_path: Path) -> None:
    seen: list[dict[str, Any]] = []

    def transport(provider, model, messages, max_tokens, system, params):  # type: ignore[no-untyped-def]
        seen.append(params)
        text = '{"verdict": "tie", "reason": "same"}'
        toks = lp([('{"verdict": "', {"x": 1.0}), ("tie", {"tie": 0.8, "A": 0.2})])
        return TransportResult(text, 10, 5, True, toks)

    judge = PairwiseJudge(
        make_client(tmp_path, transport), provider="local", model="m", top_logprobs=5
    )
    result = judge.judge_pair("p", "x", "y")
    assert seen[0] == {"temperature": 0.0, "logprobs": True, "top_logprobs": 5}
    assert result.original_first.verdict_probs == pytest.approx({"A": 0.2, "B": 0.0, "tie": 0.8})
    # Served from the cache, logprobs survive the JSON round trip.
    again = judge.judge_pair("p", "x", "y")
    assert again.original_first.cached
    assert again.original_first.verdict_probs == result.original_first.verdict_probs
    with pytest.raises(ValueError):
        PairwiseJudge(make_client(tmp_path, transport), provider="local", model="m", top_logprobs=0)


# ---------------------------------------------------------------------------- pointwise

CREATIVE_DIMS = tuple(d.name for d in load_rubric("creative").dimensions)


def scores_json(value: int, hard_fail: bool = False, dims: tuple[str, ...] = CREATIVE_DIMS) -> str:
    return json.dumps(
        {"scores": dict.fromkeys(dims, value), "hard_fail": hard_fail, "reason": "ok"}
    )


def test_parse_scores_accepts_valid_and_rejects_invalid() -> None:
    parsed = parse_scores("Sure:\n```json\n" + scores_json(4) + "\n```", CREATIVE_DIMS)
    assert parsed == (dict.fromkeys(CREATIVE_DIMS, 4), False, "ok")
    loose = {d.replace("_", " ").title(): 3 for d in CREATIVE_DIMS}
    assert parse_scores(json.dumps({"scores": loose}), CREATIVE_DIMS) == (
        dict.fromkeys(CREATIVE_DIMS, 3),
        False,
        "",
    )
    missing = {"scores": {CREATIVE_DIMS[0]: 3}}
    assert parse_scores(json.dumps(missing), CREATIVE_DIMS) is None
    out_of_range = {"scores": dict.fromkeys(CREATIVE_DIMS, 7)}
    assert parse_scores(json.dumps(out_of_range), CREATIVE_DIMS) is None
    fractional = {"scores": dict.fromkeys(CREATIVE_DIMS, 3.5)}
    assert parse_scores(json.dumps(fractional), CREATIVE_DIMS) is None
    assert parse_scores("", CREATIVE_DIMS) is None


def test_weighted_score_uses_rubric_weights() -> None:
    weights = {d.name: d.weight for d in load_rubric("creative").dimensions}
    assert weighted_score(dict.fromkeys(weights, 5), weights) == pytest.approx(5.0)


def ps(score: float | None, hard_fail: bool = False) -> PointScore:
    return PointScore({} if score is not None else None, hard_fail, score, "", "", 1, False, 1)


@pytest.mark.parametrize(
    ("orig", "rew", "expected"),
    [
        (ps(4.0), ps(4.0), "tie"),
        (ps(4.0), ps(4.2), "rewrite"),
        (ps(4.2), ps(4.0), "original"),
        (ps(5.0, True), ps(2.0), "rewrite"),
        (ps(2.0), ps(5.0, True), "original"),
        (ps(3.0, True), ps(4.0, True), "rewrite"),
        (ps(None), ps(4.0), None),
    ],
)
def test_side_from_scores(orig: PointScore, rew: PointScore, expected: str | None) -> None:
    assert side_from_scores(orig, rew) == expected


def test_pointwise_judge_scores_each_output_alone(tmp_path: Path) -> None:
    prompts: list[str] = []

    def transport(provider, model, messages, max_tokens, system, params):  # type: ignore[no-untyped-def]
        body = messages[0]["content"]
        prompts.append(body)
        response = re.search(r"\[Response\]\n(.*?)\n\[End of Response\]", body, re.S)
        assert response is not None
        return TransportResult(scores_json(5 if "good" in response.group(1) else 3), 10, 5)

    judge = PointwiseJudge(make_client(tmp_path, transport), provider="local", model="m")
    orig, rew, side = judge.compare("write a poem", "a good poem", "a poem", "creative")
    assert side == "original"
    assert orig.score == pytest.approx(5.0) and rew.score == pytest.approx(3.0)
    assert all("Category: creative" in p for p in prompts)
    assert all("Response A" not in p for p in prompts)
    # Identical outputs share one cached request and always tie.
    _, _, same = judge.compare("write a poem", "a poem", "a poem", "creative")
    assert same == "tie"


def test_pointwise_judge_retries_then_gives_up(tmp_path: Path) -> None:
    replies = ["not json", scores_json(4)]

    def transport(provider, model, messages, max_tokens, system, params):  # type: ignore[no-untyped-def]
        return TransportResult(replies.pop(0) if replies else "nope", 10, 5)

    judge = PointwiseJudge(make_client(tmp_path, transport), provider="local", model="m", retries=1)
    first = judge.score("p", "x", "creative")
    assert first.parsed and first.attempts == 2
    failed = judge.score("p", "other", "creative")
    assert not failed.parsed and failed.attempts == 2
    with pytest.raises(ValueError):
        PointwiseJudge(make_client(tmp_path, transport), provider="local", model="m", retries=-1)


def test_judge_pair_averaged(tmp_path: Path) -> None:
    from terser.judge.protocols import averaged_judge, judge_pair_averaged

    def transport(provider, model, messages, max_tokens, system, params):  # type: ignore[no-untyped-def]
        # Position-biased judge: A at 0.55 in both orders, so the averaged result is near-even.
        toks = lp([('{"verdict": "', {"x": 1.0}), ("A", {"A": 0.55, "B": 0.35, "tie": 0.10})])
        return TransportResult('{"verdict": "A", "reason": "r"}', 10, 5, True, toks)

    client = make_client(tmp_path, transport)
    judge = averaged_judge(client, provider="local", model="m")
    res = judge_pair_averaged(judge, "p", "orig out", "rew out", "fact")
    assert res.pairwise.original_first.side == "original"
    assert res.pairwise.rewrite_first.side == "rewrite"
    assert res.side == "original" and res.sides == ("original", "original")
    assert res.margin == pytest.approx(0.0)
    assert res.low_confidence
    agg = res.aggregate()
    assert agg["protocol"] == "pairwise_avg" and agg["low_confidence"] is True
    plain = PairwiseJudge(client, provider="local", model="m")
    with pytest.raises(ValueError):
        judge_pair_averaged(plain, "p", "a", "b")


def test_parse_scores_rejects_non_finite_values() -> None:
    for bad in ("NaN", "Infinity", "-Infinity"):
        reply = '{"scores": {"correctness": ' + bad + '}, "hard_fail": false}'
        assert parse_scores(reply, ("correctness",)) is None
