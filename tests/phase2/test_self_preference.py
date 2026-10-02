"""Tests for the 2.2.5 self-preference control helpers."""

from __future__ import annotations

from terser.phase2.self_preference import agreement_table, filter_responses, select_pairs


def _rec(i: int, target: str, side: str | None, label: str = "preserved", identical: bool = False):
    return {
        "id": f"c{i}@{target}",
        "prompt_id": f"p{i}",
        "candidate_id": f"c{i}",
        "target": target,
        "label": label,
        "identical_outputs": identical,
        "aggregate": {"side": side},
    }


def test_select_pairs_stratified_and_skips_unjudged() -> None:
    labels = [_rec(i, "q", "tie") for i in range(10)] + [_rec(i, "p", "rewrite") for i in range(10)]
    labels += [_rec(99, "q", None), _rec(98, "p", "tie", identical=True)]
    chosen = select_pairs(labels, 8, seed=1)
    assert len(chosen) == 8
    assert sum(r["target"] == "q" for r in chosen) == 4
    assert all(r["id"] not in {"c99@q", "c98@p"} for r in chosen)
    assert chosen == select_pairs(labels, 8, seed=1)


def test_filter_responses_keeps_candidate_and_original() -> None:
    pairs = [_rec(1, "q", "tie")]
    resp = [
        {"target": "q", "prompt_id": "p1", "candidate_id": "c1"},
        {"target": "q", "prompt_id": "p1", "candidate_id": "original"},
        {"target": "q", "prompt_id": "p1", "candidate_id": "c2"},
        {"target": "p", "prompt_id": "p1", "candidate_id": "original"},
    ]
    assert len(filter_responses(resp, pairs)) == 2


def test_agreement_table_gap() -> None:
    main = [_rec(i, "q", "tie") for i in range(4)] + [_rec(i, "p", "tie") for i in range(4)]
    second = [_rec(i, "q", "tie") for i in range(4)]
    second += [_rec(0, "p", "tie"), _rec(1, "p", "tie"), _rec(2, "p", "original", "degraded")]
    second += [_rec(3, "p", "original", "degraded")]
    t = agreement_table(main, second)
    assert t["per_target"]["q"]["side_agreement"] == 1.0
    assert t["per_target"]["p"]["side_agreement"] == 0.5
    assert t["side_agreement_gap_pp"] == 50.0
    assert t["passes"] is False
