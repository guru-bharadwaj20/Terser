"""Operator registry, canonical order (row 1.2.8) and checkpoint C1.2 (operator safety)."""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import pytest

from terser.operators.base import Operator
from terser.operators.registry import CANONICAL_ORDER, OPERATORS, apply_ops
from terser.tokenizers import count_tokens

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _load(name: str) -> list[dict[str, Any]]:
    with open(FIXTURES / name, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


CASUAL = _load("casual_prompts.jsonl")
EFFICIENT = _load("efficient_prompts.jsonl")
CASUAL_IDS = [row["id"] for row in CASUAL]
EFFICIENT_IDS = [row["id"] for row in EFFICIENT]


def test_registry_contents() -> None:
    assert set(OPERATORS) == set(CANONICAL_ORDER) == set("FHDILCS")
    assert len(CANONICAL_ORDER) == 7
    for code, op in OPERATORS.items():
        assert isinstance(op, Operator)
        assert op.code == code


def test_efficient_fixture_shape() -> None:
    assert len(EFFICIENT) == 50
    assert len(set(EFFICIENT_IDS)) == 50
    for category in ("code", "fact", "reason", "explain", "creative", "transform"):
        assert sum(row["category"] == category for row in EFFICIENT) >= 7


@pytest.mark.parametrize("row", CASUAL, ids=CASUAL_IDS)
def test_two_operator_order_independent(row: dict[str, Any]) -> None:
    text = row["prompt"]
    assert apply_ops(text, "FH") == apply_ops(text, "HF")


@pytest.mark.parametrize("subset", ["CHF", "FID", "HISL", "CFDS"])
@pytest.mark.parametrize("row", CASUAL[:5], ids=CASUAL_IDS[:5])
def test_all_permutations_agree(row: dict[str, Any], subset: str) -> None:
    text = row["prompt"]
    expected = apply_ops(text, subset)
    for perm in itertools.permutations(subset):
        assert apply_ops(text, "".join(perm)) == expected
        assert apply_ops(text, list(perm)) == expected


@pytest.mark.parametrize("row", CASUAL[:5], ids=CASUAL_IDS[:5])
def test_repeats_ignored(row: dict[str, Any]) -> None:
    text = row["prompt"]
    assert apply_ops(text, "FFHHF") == apply_ops(text, "FH")
    assert apply_ops(text, CANONICAL_ORDER * 3) == apply_ops(text, CANONICAL_ORDER)


def test_empty_codes_is_identity() -> None:
    assert apply_ops("hey so um, explain DNS", "") == "hey so um, explain DNS"


def test_unknown_code_raises_helpful_key_error() -> None:
    with pytest.raises(KeyError, match=r"unknown operator code.*'X'.*valid codes"):
        apply_ops("Explain DNS.", "FX")


PILOT_PATH = Path(__file__).resolve().parents[2] / "data" / "pilot" / "prompts.jsonl"


@pytest.mark.parametrize("row", CASUAL, ids=CASUAL_IDS)
def test_full_stack_idempotent(row: dict[str, Any]) -> None:
    once = apply_ops(row["prompt"], CANONICAL_ORDER)
    assert apply_ops(once, CANONICAL_ORDER) == once


@pytest.mark.skipif(not PILOT_PATH.exists(), reason="pilot prompts not present")
def test_full_stack_idempotent_on_pilot_prompts() -> None:
    with open(PILOT_PATH, encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    assert len(rows) == 500
    failures = []
    for row in rows:
        once = apply_ops(row["prompt"], CANONICAL_ORDER, target=None)
        twice = apply_ops(once, CANONICAL_ORDER, target=None)
        if twice != once:
            failures.append((row["id"], once, twice))
    assert failures == []


@pytest.mark.network  # tiktoken downloads o200k_base on first use
@pytest.mark.parametrize("row", CASUAL, ids=CASUAL_IDS)
def test_full_stack_never_longer_on_gpt4o(row: dict[str, Any]) -> None:
    text = row["prompt"]
    out = apply_ops(text, CANONICAL_ORDER, target="gpt-4o")
    assert count_tokens(out, "gpt-4o") <= count_tokens(text, "gpt-4o")


# --- C1.2: every operator is an exact no-op on already-efficient prompts ----------------


@pytest.mark.parametrize("row", EFFICIENT, ids=EFFICIENT_IDS)
@pytest.mark.parametrize("code", list(CANONICAL_ORDER))
def test_c1_2_operator_noop_on_efficient(code: str, row: dict[str, Any]) -> None:
    text = row["prompt"]
    assert OPERATORS[code](text) == text


@pytest.mark.parametrize("row", EFFICIENT, ids=EFFICIENT_IDS)
def test_c1_2_full_stack_noop_on_efficient(row: dict[str, Any]) -> None:
    text = row["prompt"]
    assert apply_ops(text, CANONICAL_ORDER) == text


@pytest.mark.network
@pytest.mark.parametrize("row", EFFICIENT, ids=EFFICIENT_IDS)
@pytest.mark.parametrize("target", ["gpt-4o", "qwen2.5", "llama-3.1"])
def test_c1_2_substitute_and_stack_noop_per_target(target: str, row: dict[str, Any]) -> None:
    text = row["prompt"]
    assert OPERATORS["S"](text, target) == text
    assert apply_ops(text, CANONICAL_ORDER, target=target) == text
