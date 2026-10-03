import csv
import json

import pytest

from terser.phase1.review_merge import main, merge


def _labels():
    return [
        {
            "id": "a:F",
            "original": "x",
            "rewrite": "y",
            "label": "preserved",
            "needs_human_review": False,
        },
        {
            "id": "a:H",
            "original": "x",
            "rewrite": "z",
            "label": "degraded",
            "needs_human_review": True,
        },
    ]


def _row(rid, label="", reviewer="", **kw):
    return {"id": rid, "human_label": label, "reviewer": reviewer, "notes": "", **kw}


def test_merge_partial_review():
    labels = _labels()
    s = merge(labels, [_row("a:F", "preserved", "guru"), _row("a:H")])
    assert labels[0]["human_label"] == "preserved" and "human_label" not in labels[1]
    assert (s.reviewed, s.flagged_reviewed, s.agree, s.complete) == (1, 0, 1, False)


def test_merge_complete_with_disagreement():
    s = merge(_labels(), [_row("a:F", "Degraded", "g"), _row("a:H", "degraded", "v")])
    assert s.complete and s.agree == 1 and s.reviewers == ("g", "v")


@pytest.mark.parametrize(
    "row",
    [
        _row("zzz", "preserved", "g"),
        _row("a:F", "maybe", "g"),
        _row("a:F", "preserved", ""),
        _row("a:F", "", "g"),
        _row("a:F", "preserved", "g", rewrite="other"),
    ],
)
def test_merge_rejects_bad_rows_without_modifying(row):
    labels = _labels()
    with pytest.raises(ValueError):
        merge(labels, [row])
    assert labels == _labels()


def test_cli_roundtrip(tmp_path):
    lp, rp = tmp_path / "labels.jsonl", tmp_path / "review.csv"
    lp.write_text("".join(json.dumps(r) + "\n" for r in _labels()), encoding="utf-8")
    with rp.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "human_label", "reviewer", "notes"])
        w.writeheader()
        w.writerow(_row("a:F", "preserved", "g"))
    assert main(["--labels", str(lp), "--review", str(rp), "--require-complete"]) == 1
    assert json.loads(lp.read_text(encoding="utf-8").splitlines()[0])["reviewer"] == "g"
