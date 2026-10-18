"""Tests for the 1.3.6 pilot labelling pipeline. No real LLM calls: fake transports only."""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml

from terser.llm import BudgetExceededError, LLMClient, TransportResult
from terser.phase1.pilot_labels import (
    OPERATOR_SUBSETS,
    REVIEW_COLUMNS,
    PilotServerUnavailableError,
    PilotSettings,
    TargetOutputMissingError,
    iter_items,
    load_prompts,
    output_dir_for,
    run_pilot,
    settings_from_config,
)

TARGET = "fake-target"
JUDGE = "fake-judge"

PROMPTS = [
    {"id": "code-01", "category": "code", "prompt": "um please write a sort function"},
    {"id": "fact-01", "category": "fact", "prompt": "please name the capital of France"},
    {"id": "reason-01", "category": "reason", "prompt": "what is 2 + 2"},  # nothing to drop
]


def fake_apply_ops(text: str, codes: str, target: str | None) -> str:
    """F drops 'um ', H drops 'please '; every other operator is a no-op."""
    if "F" in codes:
        text = text.replace("um ", "")
    if "H" in codes:
        text = text.replace("please ", "")
    return text


def section(text: str, name: str) -> str:
    m = re.search(rf"\[{name}\]\n(.*?)\n\[End of {name}\]", text, re.DOTALL | re.I)
    assert m
    return m.group(1)


class FakeLLM:
    """Target echoes its prompt; judge prefers the response whose prompt kept 'please'.

    ``judge_reply`` overrides the judge's raw text (e.g. malformed output).
    """

    def __init__(self, judge_reply: str | None = None) -> None:
        self.judge_reply = judge_reply
        self.target_calls: list[str] = []
        self.judge_calls: list[str] = []

    def __call__(
        self,
        provider: str,
        model: str,
        messages: list[dict[str, Any]],
        max_tokens: int,
        system: str | None,
        params: dict[str, Any],
    ) -> TransportResult:
        assert params.get("temperature") == 0.0
        content = messages[0]["content"]
        if model == TARGET:
            self.target_calls.append(content)
            return TransportResult(f"OUTPUT<{content}>", 5, 7)
        assert model == JUDGE
        self.judge_calls.append(content)
        if self.judge_reply is not None:
            return TransportResult(self.judge_reply, 50, 9)
        a, b = section(content, "Response A"), section(content, "Response B")
        a_polite, b_polite = "please" in a, "please" in b
        verdict = "tie" if a_polite == b_polite else ("A" if a_polite else "B")
        return TransportResult(json.dumps({"verdict": verdict, "reason": "r"}), 50, 9)


def write_budget(tmp_path: Path, max_tokens: int = 10_000_000) -> Path:
    path = tmp_path / "budget.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "phases": {"phase1": {"max_usd": 0, "max_tokens": max_tokens}},
                "providers": {"local": {"base_url": "http://127.0.0.1:9/v1", "priced": False}},
            }
        ),
        encoding="utf-8",
    )
    return path


def make_client(tmp_path: Path, transport: Any = None, **kwargs: Any) -> LLMClient:
    max_tokens = kwargs.pop("max_tokens", 10_000_000)
    return LLMClient(
        "phase1",
        budget_path=write_budget(tmp_path, max_tokens),
        transport=transport,
        ledger_path=tmp_path / "ledger.json",
        cache_dir=tmp_path / "cache",
        **kwargs,
    )


@pytest.fixture
def settings(tmp_path: Path) -> PilotSettings:
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_text("".join(json.dumps(p) + "\n" for p in PROMPTS), encoding="utf-8")
    return PilotSettings(
        prompts_path=prompts,
        output_dir=tmp_path / "data" / "pilot",
        smoke_dir=tmp_path / "data" / "pilot" / "smoke",
        target_model=TARGET,
        judge_model=JUDGE,
        stand_in=True,
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_operator_subsets_match_plan() -> None:
    assert OPERATOR_SUBSETS[:7] == ("F", "H", "D", "I", "L", "C", "S")
    assert OPERATOR_SUBSETS[7:] == ("FHC", "FHCI", "FHCID", "FHCIDS", "FHCIDSL")


def test_unchanged_rewrites_are_skipped() -> None:
    skipped: list[tuple[str, str]] = []
    items = list(
        iter_items(
            PROMPTS,
            OPERATOR_SUBSETS,
            apply_ops=fake_apply_ops,
            on_skip=lambda pid, ops: skipped.append((pid, ops)),
        )
    )
    # code-01 changes under every subset containing F or H: F, H + 5 stacks = 7.
    # fact-01 has no "um ", so F alone is a no-op: 6. reason-01 never changes.
    assert len(items) == 13
    assert all(it.rewrite != it.original for it in items)
    assert len(skipped) == 3 * 12 - 13
    assert all(
        pid == "reason-01" or ops in {"D", "I", "L", "C", "S"} or (pid, ops) == ("fact-01", "F")
        for pid, ops in skipped
    )
    assert {it.id for it in items} >= {"code-01:F", "fact-01:FHCIDSL"}


def test_stand_in_run_writes_only_to_smoke_dir(tmp_path: Path, settings: PilotSettings) -> None:
    fake = FakeLLM()
    summary = run_pilot(settings, make_client(tmp_path, fake), apply_ops=fake_apply_ops)
    assert summary.output_dir == settings.smoke_dir
    records = read_jsonl(settings.smoke_dir / "labels.jsonl")
    assert len(records) == summary.n_items == 13
    assert not (settings.output_dir / "labels.jsonl").exists()
    assert all(r["stand_in"] is True and r["needs_human_review"] for r in records)
    required = {
        "id", "category", "ops", "original", "rewrite", "target_model", "judge_model",
        "verdicts", "label", "stand_in", "needs_human_review",
    }  # fmt: skip
    assert required <= set(records[0])
    assert summary.n_skipped_unchanged == 23 and summary.n_pairs_considered == 36


def test_real_run_routing_and_guards(tmp_path: Path, settings: PilotSettings) -> None:
    real = replace(settings, stand_in=False)
    run_pilot(real, make_client(tmp_path, FakeLLM()), apply_ops=fake_apply_ops)
    assert (settings.output_dir / "labels.jsonl").is_file()
    assert not (settings.smoke_dir / "labels.jsonl").exists()
    records = read_jsonl(settings.output_dir / "labels.jsonl")
    assert all(r["stand_in"] is False for r in records)

    with pytest.raises(ValueError, match="stand-in tags"):
        replace(settings, stand_in=False, target_model="qwen2.5:1.5b")
    with pytest.raises(ValueError, match="smoke_dir"):
        output_dir_for(replace(real, output_dir=settings.smoke_dir / "x"))


def test_labels_follow_both_orders_rule(tmp_path: Path, settings: PilotSettings) -> None:
    fake = FakeLLM()
    run_pilot(settings, make_client(tmp_path, fake), apply_ops=fake_apply_ops)
    records = {r["id"]: r for r in read_jsonl(settings.smoke_dir / "labels.jsonl")}
    # F on code-01 keeps "please" -> tie in both orders -> preserved.
    f = records["code-01:F"]
    assert f["verdicts"]["original_first"]["side"] == "tie"
    assert f["verdicts"]["rewrite_first"]["side"] == "tie"
    assert f["label"] == "preserved"
    # H drops "please" -> judge prefers the original in both orders -> degraded.
    h = records["code-01:H"]
    assert h["verdicts"]["original_first"]["verdict"] == "A"
    assert h["verdicts"]["rewrite_first"]["verdict"] == "B"
    assert h["verdicts"]["position_consistent"] is True
    assert h["label"] == "degraded"
    assert h["review_reasons"] == ["stand_in"]


def test_judge_sees_swapped_positions(tmp_path: Path, settings: PilotSettings) -> None:
    fake = FakeLLM()
    one = replace(settings, operator_subsets=("H",), categories=("fact",))
    run_pilot(one, make_client(tmp_path, fake), apply_ops=fake_apply_ops)
    assert len(fake.judge_calls) == 2
    first, second = fake.judge_calls
    orig_out = "OUTPUT<please name the capital of France>"
    rew_out = "OUTPUT<name the capital of France>"
    assert section(first, "Response A") == orig_out and section(first, "Response B") == rew_out
    assert section(second, "Response A") == rew_out and section(second, "Response B") == orig_out
    assert section(first, "User request") == PROMPTS[1]["prompt"]


def test_malformed_judge_output_is_unlabelled(tmp_path: Path, settings: PilotSettings) -> None:
    fake = FakeLLM(judge_reply="I cannot decide, both are fine!")
    one = replace(settings, operator_subsets=("F",), categories=("code",), judge_retries=1)
    summary = run_pilot(one, make_client(tmp_path, fake), apply_ops=fake_apply_ops)
    (rec,) = read_jsonl(settings.smoke_dir / "labels.jsonl")
    assert rec["label"] is None
    assert rec["needs_human_review"] and "unparseable_verdict" in rec["review_reasons"]
    assert rec["verdicts"]["original_first"]["attempts"] == 2
    assert len(fake.judge_calls) == 4  # 2 orders x (1 try + 1 retry)
    assert summary.labels["none"] == 1


def test_review_csv_has_every_row(tmp_path: Path, settings: PilotSettings) -> None:
    run_pilot(settings, make_client(tmp_path, FakeLLM()), apply_ops=fake_apply_ops)
    with (settings.smoke_dir / "labels_review.csv").open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 13
    assert tuple(rows[0]) == REVIEW_COLUMNS


def test_cache_reuse(tmp_path: Path, settings: PilotSettings) -> None:
    fake = FakeLLM()
    client = make_client(tmp_path, fake)
    summary = run_pilot(settings, client, apply_ops=fake_apply_ops)
    # 2 changed prompts: 1 original output each + distinct rewrites (code-01: 3, fact-01: 1).
    assert len(fake.target_calls) == len(set(fake.target_calls)) == 2 + 3 + 1
    assert summary.generated_tokens > 0
    n_judge = len(fake.judge_calls)

    again = run_pilot(settings, make_client(tmp_path, fake), apply_ops=fake_apply_ops)
    assert len(fake.target_calls) == 6 and len(fake.judge_calls) == n_judge
    assert again.generated_tokens == 0 and again.cached_calls == again.llm_calls


def test_budget_cap_stops_before_calling(tmp_path: Path, settings: PilotSettings) -> None:
    fake = FakeLLM()
    client = make_client(tmp_path, fake, max_tokens=100)  # < max_target_tokens estimate
    with pytest.raises(BudgetExceededError):
        run_pilot(settings, client, apply_ops=fake_apply_ops)
    assert fake.target_calls == [] and fake.judge_calls == []
    assert read_jsonl(settings.smoke_dir / "labels.jsonl") == []


def test_budget_cap_keeps_completed_records(tmp_path: Path, settings: PilotSettings) -> None:
    fake = FakeLLM()
    # Enough for a few target and judge calls (judge prompts include the rubric), not all 13.
    small = replace(settings, max_target_tokens=20, max_judge_tokens=20)
    client = make_client(tmp_path, fake, max_tokens=1100)
    with pytest.raises(BudgetExceededError):
        run_pilot(small, client, apply_ops=fake_apply_ops)
    records = read_jsonl(settings.smoke_dir / "labels.jsonl")
    assert 0 < len(records) < 13
    assert (settings.smoke_dir / "labels_review.csv").is_file()


def test_unreachable_server_gives_clear_error(tmp_path: Path, settings: PilotSettings) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    http = httpx.Client(transport=httpx.MockTransport(refuse))
    client = make_client(tmp_path, http_client=http)
    with pytest.raises(PilotServerUnavailableError, match="running OpenAI-compatible server"):
        run_pilot(settings, client, apply_ops=fake_apply_ops)


def test_respond_then_judge_stages(tmp_path: Path, settings: PilotSettings) -> None:
    fake = FakeLLM()
    with pytest.raises(TargetOutputMissingError):
        run_pilot(
            replace(settings, stage="judge"), make_client(tmp_path, fake), apply_ops=fake_apply_ops
        )
    assert fake.target_calls == []
    (settings.smoke_dir / "labels.jsonl").unlink(missing_ok=True)

    respond = replace(settings, stage="respond")
    run_pilot(respond, make_client(tmp_path, fake), apply_ops=fake_apply_ops)
    assert len(fake.target_calls) == 6 and fake.judge_calls == []
    assert not (settings.smoke_dir / "labels.jsonl").exists()

    summary = run_pilot(
        replace(settings, stage="judge"), make_client(tmp_path, fake), apply_ops=fake_apply_ops
    )
    assert len(fake.target_calls) == 6  # judge stage never calls the target
    assert summary.n_items == 13 and len(read_jsonl(settings.smoke_dir / "labels.jsonl")) == 13


def test_load_prompts_limit_and_validation(tmp_path: Path, settings: PilotSettings) -> None:
    assert [p["id"] for p in load_prompts(settings.prompts_path, categories=["fact"])] == [
        "fact-01"
    ]
    assert len(load_prompts(settings.prompts_path, limit=1)) == 3
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": "x", "category": "code"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        load_prompts(bad)
    with pytest.raises(FileNotFoundError):
        load_prompts(tmp_path / "nope.jsonl")


def test_config_defaults_are_stand_in() -> None:
    from terser.config import load_config

    cfg = load_config(["experiment=p1_pilot_labels"])
    s = settings_from_config(cfg)
    assert s.stand_in is True and s.provider == "local"
    assert (s.target_model, s.judge_model) == ("qwen2.5:1.5b", "qwen2.5:0.5b")
    assert s.operator_subsets == OPERATOR_SUBSETS
    assert s.smoke_dir.as_posix().endswith("data/pilot/smoke")
    assert output_dir_for(s) == s.smoke_dir
    assert cfg.experiment.phase == 1 and cfg.experiment.row == "1.3.6"


def test_config_gpu_variant_overrides() -> None:
    from terser.config import load_config

    cfg = load_config(
        [
            "experiment=p1_pilot_labels",
            "experiment.params.stand_in=false",
            "experiment.params.stage=judge",
            "experiment.params.target_model=Qwen/Qwen2.5-7B-Instruct",
            "experiment.params.judge_model=Qwen/Qwen2.5-32B-Instruct-AWQ",
        ]
    )
    s = settings_from_config(cfg)
    assert not s.stand_in and s.stage == "judge"
    assert output_dir_for(s).as_posix().endswith("data/pilot")


def test_with_real_operators_if_available(tmp_path: Path, settings: PilotSettings) -> None:
    summary = run_pilot(
        replace(settings, operator_subsets=("F", "H", "FHCIDSL")), make_client(tmp_path, FakeLLM())
    )
    assert summary.n_pairs_considered == 9
    for rec in read_jsonl(settings.smoke_dir / "labels.jsonl"):
        assert rec["rewrite"] != rec["original"]


def test_parallel_workers_match_sequential(tmp_path: Path, settings: PilotSettings) -> None:
    seq_fake, par_fake = FakeLLM(), FakeLLM()
    (tmp_path / "seq").mkdir()
    (tmp_path / "par").mkdir()
    seq = run_pilot(settings, make_client(tmp_path / "seq", seq_fake), apply_ops=fake_apply_ops)
    expected = read_jsonl(settings.smoke_dir / "labels.jsonl")
    expected_csv = (settings.smoke_dir / "labels_review.csv").read_text(encoding="utf-8")

    par = run_pilot(
        replace(settings, workers=4),
        make_client(tmp_path / "par", par_fake),
        apply_ops=fake_apply_ops,
    )
    assert read_jsonl(settings.smoke_dir / "labels.jsonl") == expected
    assert (settings.smoke_dir / "labels_review.csv").read_text(encoding="utf-8") == expected_csv
    # Each unique target text is generated exactly once, as in the sequential run.
    assert sorted(par_fake.target_calls) == sorted(seq_fake.target_calls)
    assert len(par_fake.judge_calls) == len(seq_fake.judge_calls)
    assert par.n_items == seq.n_items and par.labels == seq.labels
    assert par.generated_tokens == seq.generated_tokens


def test_parallel_respond_then_judge(tmp_path: Path, settings: PilotSettings) -> None:
    fake = FakeLLM()
    fast = replace(settings, workers=3)
    run_pilot(replace(fast, stage="respond"), make_client(tmp_path, fake), apply_ops=fake_apply_ops)
    assert len(fake.target_calls) == 6 and fake.judge_calls == []
    summary = run_pilot(
        replace(fast, stage="judge"), make_client(tmp_path, fake), apply_ops=fake_apply_ops
    )
    assert len(fake.target_calls) == 6
    assert summary.n_items == 13 and len(read_jsonl(settings.smoke_dir / "labels.jsonl")) == 13


def test_workers_must_be_positive(settings: PilotSettings) -> None:
    with pytest.raises(ValueError, match="workers"):
        replace(settings, workers=0)


class BiasedLogprobLLM(FakeLLM):
    """Judge that always answers A; logprobs say A 0.5 / B 0.45 / tie 0.05 in every order."""

    def __call__(self, provider, model, messages, max_tokens, system, params):  # type: ignore[no-untyped-def]
        if model == TARGET:
            return super().__call__(provider, model, messages, max_tokens, system, params)
        assert params.get("logprobs") is True and params.get("top_logprobs") == 20
        top = (("A", math.log(0.5)), ("B", math.log(0.45)), ("tie", math.log(0.05)))
        toks = (('{"verdict": "', (("x", 0.0),)), ("A", top))
        return TransportResult('{"verdict": "A", "reason": "r"}', 50, 9, True, toks)


def test_pairwise_avg_protocol(tmp_path: Path, settings: PilotSettings) -> None:
    avg = replace(settings, protocol="pairwise_avg")
    run_pilot(avg, make_client(tmp_path, BiasedLogprobLLM()), apply_ops=fake_apply_ops)
    recs = read_jsonl(settings.smoke_dir / "labels.jsonl")
    rec = recs[0]
    # Raw orders disagree (always A); averaged they are an exact original/rewrite tie, broken
    # conservatively towards original, so the label is degraded and the margin is 0.
    assert rec["verdicts"]["position_consistent"] is False
    assert rec["judge_protocol"] == "pairwise_avg"
    assert rec["aggregate"]["side"] == "original" and rec["label"] == "degraded"
    assert rec["aggregate"]["margin"] == pytest.approx(0.0)
    assert {"position_inconsistent", "low_margin"} <= set(rec["review_reasons"])
    with (settings.smoke_dir / "labels_review.csv").open(encoding="utf-8", newline="") as fh:
        row = next(csv.DictReader(fh))
    assert row["protocol"] == "pairwise_avg" and row["averaged_side"] == "original"
    with pytest.raises(ValueError):
        replace(settings, protocol="bogus")
