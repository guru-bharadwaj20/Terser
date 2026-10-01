"""Tests for the collect and casualise stages (rows 2.1.4, 2.1.7). Fake transport only."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from terser.data.casualise import (
    STYLES,
    check_casual,
    clean_casual,
    is_stand_in,
    parse_meaning_check,
    render_casualise_prompt,
    render_check_prompt,
    run_casualise,
    summarise,
)
from terser.data.collect import clean_text, collect, select_balanced
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


# ---------------------------------------------------------------------------- collect


def src(
    i: int,
    cat: str | None,
    instruction: str = "Explain how tides work in detail.",
    input_text: str | None = None,
) -> dict[str, Any]:
    return {
        "source": "s",
        "source_id": f"s:{i}",
        "instruction": instruction,
        "input": input_text,
        "reference": "r",
        "category_hint": cat,
    }


def test_collect_filters_scrubs_and_balances() -> None:
    records = [src(i, "explain") for i in range(10)] + [src(100 + i, "fact") for i in range(3)]
    records += [
        src(200, None),
        src(201, "fact", "Hi"),
        src(202, "fact", "Email me at jane.doe@gmail.com about tides please."),
        src(203, "transform", "Summarise the passage below.", "Tides rise twice a day."),
    ]
    out, counts = collect(records, per_category=4, min_words=3, max_words=50)
    by_cat = {c: [r for r in out if r["category"] == c] for c in ("explain", "fact", "transform")}
    assert len(by_cat["explain"]) == 4 and len(by_cat["transform"]) == 1
    assert counts["no_category"] == 1 and counts["length"] == 1
    assert counts["kept:explain"] == 4
    assert all("@" not in r["clean"] for r in out)
    t = by_cat["transform"][0]
    assert t["clean"] == "Summarise the passage below.\n\nTides rise twice a day."
    # Deterministic for a seed, different for another seed.
    again, _ = collect(records, per_category=4, min_words=3, max_words=50)
    assert again == out
    other = select_balanced([r for r in out if r["category"] == "explain"], 2, seed=1)
    assert len(other) == 2


def test_clean_text() -> None:
    assert clean_text("Do x.", None) == "Do x."
    assert clean_text("Do x.", "data") == "Do x.\n\ndata"


# ---------------------------------------------------------------------------- checks


def test_render_prompts() -> None:
    p = render_casualise_prompt("Write a haiku about {rain}.", "over_polite")
    assert "Style: over-polite" in p and "Write a haiku about {rain}." in p
    assert "{style" not in p
    with pytest.raises(ValueError):
        render_casualise_prompt("x", "angry")
    c = render_check_prompt("Do A.", "pls do A")
    assert "Do A." in c and "pls do A" in c


@pytest.mark.parametrize(
    ("casual", "style", "reason"),
    [
        ("so um could you like list 3 fruits that are red, anyway thanks", "rambling", None),
        ("", "rambling", "empty"),
        ("Sure! Here is the casual message: list 3 red fruits pls", "texting", "preamble"),
        ("list some red fruits pls", "texting", "number_changed"),
        ("list 3 red fruits", "hedging", "not_casual"),
        ("3 red fruits " + "blah " * 200, "rambling", "too_long"),
        ("list 3 red fruits pls", "texting", None),
    ],
)
def test_check_casual(casual: str, style: str, reason: str | None) -> None:
    assert check_casual("List 3 red fruits.", casual, style) == reason


def test_check_casual_keeps_quoted_and_code_spans() -> None:
    inst = 'Translate "good morning" and call `foo()`.'
    assert (
        check_casual(inst, 'hey could u translate "good morning" n call `foo()` thx', "texting")
        is None
    )
    assert check_casual(inst, "hey could u translate good morning n call foo thx", "texting") == (
        "quoted_span_changed"
    )


def test_clean_and_parse_helpers() -> None:
    assert clean_casual('Casual message:\n"hey pls do x"') == "hey pls do x"
    assert parse_meaning_check('{"same_task": true, "added": [], "missing": []}') == (
        True,
        [],
        [],
    )
    assert parse_meaning_check('ok ```json\n{"same_task": "no", "added": ["x"]}```') == (
        False,
        ["x"],
        [],
    )
    assert parse_meaning_check("nope") is None
    assert is_stand_in("qwen2.5:1.5b") and not is_stand_in("Qwen/Qwen2.5-32B-Instruct-AWQ")


# ---------------------------------------------------------------------------- run


class FakeModel:
    """Casualises by prefixing a filler; the meaning check flags a marker word as added."""

    def __call__(self, provider, model, messages, max_tokens, system, params):  # type: ignore[no-untyped-def]
        body = messages[0]["content"]
        if body.startswith("You check"):
            casual = re.search(r"\[Casual message\]\n(.*?)\n\[End", body, re.S)
            assert casual is not None
            added = ["extra"] if "EXTRA" in casual.group(1) else []
            return TransportResult(
                json.dumps({"same_task": True, "added": added, "missing": []}), 20, 10
            )
        inst = re.search(r"\[Clean instruction\]\n(.*?)\n\[End", body, re.S)
        style = re.search(r"Style: (\S+)", body)
        assert inst is not None and style is not None
        text = inst.group(1)
        if style.group(1) == "repetitive" and "fruits" in text:
            text += " EXTRA"
        return TransportResult(f"so um basically {text} thanks a lot", 50, 12)


def clean_records() -> list[dict[str, Any]]:
    return [
        {
            "source_id": "s:1",
            "source": "s",
            "category": "fact",
            "instruction": "List 3 red fruits.",
            "input": None,
            "clean": "List 3 red fruits.",
            "reference": "x",
        },
        {
            "source_id": "s:2",
            "source": "s",
            "category": "transform",
            "instruction": "Summarise the passage below.",
            "input": "Tides rise.",
            "clean": "Summarise the passage below.\n\nTides rise.",
            "reference": None,
        },
    ]


def test_run_casualise_writes_accepted_and_rejected(tmp_path: Path) -> None:
    client = make_client(tmp_path, FakeModel())
    out, rej = tmp_path / "casual.jsonl", tmp_path / "rej.jsonl"
    stats = run_casualise(
        clean_records(), client, model="m/real", out_path=out, rejected_path=rej, workers=4
    )
    assert (stats.accepted, stats.rejected) == (9, 1)
    assert stats.calls == 20
    ok = [json.loads(x) for x in out.read_text().splitlines()]
    assert {r["style"] for r in ok} == set(STYLES)
    t = next(r for r in ok if r["source_id"] == "s:2")
    assert t["prompt"].endswith("\n\nTides rise.") and t["clean"].startswith("Summarise")
    assert t["generator_model"] == "m/real" and t["stand_in"] is False
    bad = json.loads(rej.read_text())
    assert bad["id"] == "s:1:repetitive" and bad["rejection_reason"] == "added_requirements"
    s = summarise(out, rej)
    assert s["by_category"] == {"fact": 4, "transform": 5}
    # Rerun is served entirely from the cache and writes identical files.
    before = out.read_bytes()
    again = run_casualise(
        clean_records(), client, model="m/real", out_path=out, rejected_path=rej, workers=4
    )
    assert again.cached_calls == again.calls and out.read_bytes() == before


def test_stand_in_only_writes_to_smoke(tmp_path: Path) -> None:
    client = make_client(tmp_path, FakeModel())
    with pytest.raises(ValueError):
        run_casualise(
            clean_records(),
            client,
            model="qwen2.5:1.5b",
            out_path=tmp_path / "c.jsonl",
            rejected_path=tmp_path / "r.jsonl",
        )
    stats = run_casualise(
        clean_records()[:1],
        client,
        model="qwen2.5:1.5b",
        out_path=tmp_path / "smoke" / "c.jsonl",
        rejected_path=tmp_path / "smoke" / "r.jsonl",
        styles=("texting",),
    )
    assert stats.accepted == 1
    rec = json.loads((tmp_path / "smoke" / "c.jsonl").read_text())
    assert rec["stand_in"] is True
