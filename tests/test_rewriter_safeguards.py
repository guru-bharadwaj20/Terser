"""Rows 8.3.7 (decoding safeguard property test) and 8.3.8 (gate integration test)."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from terser.models.rewriter.safeguards import GatedRewriter, safeguard
from terser.operators.registry import apply_ops
from terser.tokenizers.count import count_tokens

FIXTURES = Path(__file__).resolve().parents[1] / "data" / "samples" / "casual_prompts.jsonl"
WORDS = "please could you maybe just write a python function that sorts the list fast ok".split()


def _o200k(text: str) -> int:
    return count_tokens(text, "gpt-4o")


def _random_text(rng: random.Random) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(rng.randint(0, 25)))


def _whitespace(text: str) -> int:
    return len(text.split())


@pytest.mark.parametrize("count", [_whitespace, _o200k], ids=["whitespace", "o200k"])
def test_property_never_longer_than_input(count: object) -> None:
    """For 500 random (original, rewrite) pairs: output never costs more than the input on T."""
    counter = count  # type: ignore[assignment]
    rng = random.Random(0)
    for _ in range(500):
        original = _random_text(rng)
        # Rewrites drawn to be shorter, equal, longer, empty or whitespace.
        rewrite = rng.choice(
            [
                _random_text(rng),
                original,
                original + " " + _random_text(rng),
                " ".join(original.split()[: rng.randint(0, 5)]),
                "",
                "   \n",
            ]
        )
        out, fell_back = safeguard(original, rewrite, counter)  # type: ignore[arg-type]
        assert out in (original, rewrite)
        assert counter(out) <= counter(original)  # type: ignore[operator]
        if fell_back:
            assert out == original
        else:
            assert counter(out) < counter(original) and out.strip()  # type: ignore[operator]


def test_safeguard_cases() -> None:
    assert safeguard("a b c", "a b", _whitespace) == ("a b", False)
    assert safeguard("a b c", "a b c d", _whitespace) == ("a b c", True)
    assert safeguard("a b c", "x y z", _whitespace) == ("a b c", True)  # equal is not cheaper
    assert safeguard("a b c", " ", _whitespace) == ("a b c", True)


def _constraint_gate(constraints: list[str]):  # type: ignore[no-untyped-def]
    """Stand-in for the champion gate: P(degraded) = share of constraints dropped."""

    def gate(original: str, rewrite: str) -> float:
        present = [c for c in constraints if c.lower() in original.lower()]
        if not present:
            return 0.0
        dropped = [c for c in present if c.lower() not in rewrite.lower()]
        return len(dropped) / len(present)

    return gate


def _ladder() -> list:  # type: ignore[type-arg]
    def drop_second_half(text: str) -> str:  # deliberately risky "model" rewrite
        words = text.split()
        return " ".join(words[: max(1, len(words) // 2)])

    return [
        drop_second_half,
        lambda t: apply_ops(t, "CHFIDS", "gpt-4o"),
        lambda t: apply_ops(t, "F", "gpt-4o"),
    ]


def test_gate_integration_on_fixture_prompts() -> None:
    """End to end on the 20 fixture prompts: real operators, real tokenizer, gate + fallback."""
    rows = [json.loads(line) for line in FIXTURES.read_text(encoding="utf-8").splitlines()]
    assert len(rows) >= 20
    rungs_used = set()
    for row in rows:
        gr = GatedRewriter(_ladder(), _constraint_gate(row["constraints"]), _o200k, 0.01)
        d = gr(row["prompt"])
        assert d.tokens_output <= d.tokens_original
        assert d.tokens_output == _o200k(d.text)
        # The accepted text keeps every constraint the gate checks.
        assert _constraint_gate(row["constraints"])(row["prompt"], d.text) < 0.01
        if d.rung is None:
            assert d.text == row["prompt"] and d.savings == 0.0
        else:
            assert d.savings > 0
            assert d.trace[-1][1] == "accepted"
            # Every heavier rung was tried and rejected first.
            assert [t[0] for t in d.trace] == list(range(d.rung + 1))
            assert all(t[1] != "accepted" for t in d.trace[:-1])
        rungs_used.add(d.rung)
    # The risky heaviest rung is rejected somewhere, and a lighter operator rung is used.
    assert rungs_used & {1, 2}


def test_gate_rejects_everything_returns_original() -> None:
    gr = GatedRewriter([lambda t: t[:5], lambda t: t[:10]], lambda o, r: 1.0, _whitespace)
    d = gr("one two three four five six")
    assert d.rung is None and d.text == "one two three four five six"
    assert [t[1] for t in d.trace] == ["gate_rejected", "gate_rejected"]


def test_lighter_rung_used_when_heavy_rejected() -> None:
    gate = lambda o, r: 0.9 if len(r.split()) < 3 else 0.1  # noqa: E731
    gr = GatedRewriter([lambda t: "one", lambda t: "one two three"], gate, _whitespace)
    d = gr("one two three four five")
    assert d.rung == 1 and d.text == "one two three"
    assert d.savings == pytest.approx(0.4)
