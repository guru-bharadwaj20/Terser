"""Final results on synthetic prompts, candidates and labels (no models, no tokenizers)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from terser.phase12 import results as R
from terser.serving.accounting import Pricing

QWEN, PHI = "Qwen/Qwen2.5-7B-Instruct", "microsoft/Phi-3.5-mini-instruct"
CATS = ("fact", "code")


def count(text: str, target: str) -> int:
    return len(text.split())


PRICING = Pricing(
    target_input_per_mtok={"qwen2.5": 0.2, "phi3.5": 0.2},
    rewriter_input_per_mtok=0.05,
    rewriter_output_per_mtok=0.1,
)


class FakeGate:
    """Rejects any rewrite that drops the word ``keep``."""

    def __init__(self, name: str) -> None:
        self.name = name

    def score(self, original: str, text: str) -> float:
        return 1.0 if "keep" in original and "keep" not in text else 0.0

    def rejects(self, s: float) -> bool:
        return s >= 0.5

    def describe(self) -> dict[str, Any]:
        return {"model": f"fake {self.name}", "threshold": 0.5, "rule": "drop keep"}


def _data(n: int = 24) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    prompts, cands, labels = [], [], []
    for i in range(n):
        pid = f"p{i}"
        words = ["w"] * (8 + i) + (["keep"] if i % 4 == 0 else [])
        text = " ".join(words)
        prompts.append(
            {
                "id": pid,
                "prompt": text,
                "category": CATS[i % 2],
                "source_id": f"s{i // 2}",
                "slice": "real" if i >= n - 4 else "main",
            }
        )
        ops = " ".join(["w"] * (6 + i))  # always shorter, drops "keep"
        sft = " ".join(words[: len(words) // 2]) if i % 5 else text + " longer"
        for m, t in (("ops_FHC", ops), ("sftx", sft)):
            cands.append({"prompt_id": pid, "candidate_id": f"{pid}:{m}", "method": m, "text": t})
            for tgt in (QWEN, PHI):
                if m == "sftx" and i == 3:
                    continue  # one unlabelled candidate
                ok = (i + (m == "ops_FHC")) % 3 != 0
                labels.append(
                    {
                        "prompt_id": pid,
                        "candidate_id": f"{pid}:{m}",
                        "target": tgt,
                        "label": "preserved" if ok else "degraded",
                        "aggregate": {"margin": 0.1 * (i % 7)},
                        "reason": None if ok else "judge_prefers_original",
                    }
                )
    return prompts, cands, labels


def _run(tmp_path: Path, **kw: Any) -> dict[str, Any]:
    prompts, cands, labels = _data()
    sft = [
        {
            "id": c["prompt_id"],
            "prompt": next(p["prompt"] for p in prompts if p["id"] == c["prompt_id"]),
            "raw_rewrite": c["text"],
            "latency_s": 0.4,
        }
        for c in cands
        if c["method"] == "sftx"
    ]
    bank = R.GateBank({g: FakeGate(g) for g in ("tuned", "untuned", "onnx")})
    clusters = {
        "k": 2,
        "assign": {p["id"]: int(p["id"][1:]) % 2 for p in prompts},
        "risk": {"0": 1.2, "1": 0.8},
        "any_high_risk": False,
        "slice_clusters": ["0"],
        "refit_agreement": 1.0,
    }
    return R.run(
        split="val",
        prompts=prompts,
        candidates=cands,
        labels=labels,
        count=count,
        pricing=PRICING,
        gates=bank,
        full_system="sftx",
        sft={"sftx": sft},
        meta={"ops_FHC": {"latency_mean_s": 0.002}},
        clusters=clusters,
        n_resamples=100,
        out_dir=tmp_path,
        **kw,
    )


def test_systems_and_definitions(tmp_path: Path) -> None:
    res = _run(tmp_path)
    by = res["by_name"]
    assert res["n_prompts"] == 20 and res["n_real"] == 4
    orig = by["Original"]
    assert orig["per_target"][QWEN]["median_savings"]["estimate"] == 0.0
    assert orig["preserved_pooled"]["estimate"] == 1.0 and orig["complete"]
    ops = by["Best operator (`ops_FHC`)"]
    assert ops["per_target"][QWEN]["fell_back"] == 0 and ops["complete"]
    assert ops["latency_s"] == pytest.approx(0.002)
    sft = by["SFT rewriter `sftx`"]
    # i % 5 == 0 rewrites are longer: the safeguard serves the original
    assert sft["per_target"][QWEN]["fell_back"] == 4
    assert sft["missing_labels"] == 2 and not sft["complete"]  # p3 unlabelled on both targets
    assert sft["path"] == "quality" and sft["latency_s"] == pytest.approx(0.4)
    # quality path pays the rewriter even on fallback: mean net < mean gross savings
    pt = sft["per_target"][QWEN]
    assert pt["mean_net_savings"]["estimate"] < pt["mean_savings"]
    full = by["Terser full (`sftx` + gate)"]
    assert full["per_target"][QWEN]["gate_rejected"] > 0
    ga = by["GBM tagger + gate"] if "GBM tagger + gate" in by else None
    assert ga is None  # no gbm_tagger candidates -> no row


def test_ablations_mcnemar_strata_and_files(tmp_path: Path) -> None:
    res = _run(tmp_path)
    assert res["ablation_order"][0].startswith("Terser full")
    assert len(res["ablation_order"]) == 5
    raw = res["by_name"]["Ablation: − fallback"]
    assert raw["per_target"][QWEN]["fell_back"] == 0  # longer raw rewrites are served
    assert res["top_systems"] and res["top_systems"][0] == "Best operator (`ops_FHC`)"
    assert all(m["n_pairs"] > 0 for m in res["mcnemar"])
    cat = res["strata"]["category"]["Original"][QWEN]
    assert set(cat) == set(CATS)
    assert res["strata"]["high_risk_category"]["Original"][QWEN]["fact"]["n"] == 10
    assert res["real"] and res["real"][0]["per_target"][QWEN]["n"] == 4
    assert res["criteria"]["Original"]["S2"]["pass"]
    md = (tmp_path / "final_results.md").read_text(encoding="utf-8")
    for heading in ("12.2 Master table", "12.4 McNemar", "12.6 Ablation", "12.7 Net savings"):
        assert heading in md
    assert (tmp_path / "figures" / "final_net_savings_by_length.png").stat().st_size > 1000
    assert "## Best" in (tmp_path / "gallery.md").read_text(encoding="utf-8")
    js = json.loads((tmp_path / "final_results.json").read_text(encoding="utf-8"))
    assert js["bootstrap"]["n_resamples"] == 100 and "by_name" not in js


def test_seed_groups_and_partial_cells(tmp_path: Path) -> None:
    res = _run(tmp_path)
    assert res["seeds"] == {}  # the SFT row and the gated full system are not two seeds
    d = {"estimate": 0.9, "low": 0.8, "high": 1.0}
    assert R._pres(d, 0, 100) == "90.0% [80.0%, 100.0%]"
    assert R._pres(d, 2, 100).endswith("†")
    assert R._pres(d, 20, 100) == "n/a (20 unlabelled)"


def test_gate_cache_keeps_scores_and_timing(tmp_path: Path) -> None:
    path = tmp_path / "gate_cache.json"
    bank = R.GateBank({"tuned": FakeGate("tuned")}, cache_path=path)
    assert bank.score("tuned", "a keep", "a", "fact", "qwen2.5") == 1.0
    bank.save()
    again = R.GateBank({"tuned": FakeGate("tuned")}, cache_path=path)
    assert again.mean_seconds("tuned") == bank.mean_seconds("tuned") is not None
    assert again.score("tuned", "a keep", "a", "fact", "qwen2.5") == 1.0
    assert again.calls["tuned"] == 1  # served from the cache, timing unchanged
    other = R.GateBank({"tuned": FakeGate("other")}, cache_path=path)  # gate changed
    assert other.cache == {} and other.mean_seconds("tuned") is None


def test_label_index_matches_identical_text() -> None:
    idx = R.LabelIndex(
        [{"prompt_id": "p", "candidate_id": "p:a", "target": QWEN, "label": "preserved"}],
        {"p:a": ("p", "short text"), "p:b": ("p", "short text")},
    )
    assert idx.get("p", "p:b", "short text", QWEN) is not None  # same text, other method
    assert idx.get("p", "p:a", "other", QWEN) is None


def test_length_bins_are_deciles() -> None:
    b = R.length_bins(np.arange(100, dtype=float))
    assert b.min() == 0 and b.max() == 9 and np.bincount(b).tolist() == [10] * 10
