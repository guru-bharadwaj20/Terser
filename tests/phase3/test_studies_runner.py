"""Smoke test of the Phase 3 study runner on a synthetic stand-in table (no real results)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from terser.phase3.studies import STUDIES, load_table, main


def synthetic_table(path: Path, stand_in: bool = True) -> Path:
    rng = np.random.default_rng(0)
    rows = []
    cats = ["code", "fact", "reason", "explain", "creative", "transform"]
    for p in range(90):
        base = rng.normal(size=2)
        savings = float(np.clip(20 + 10 * base[0] + rng.normal(scale=3), 0, 60))
        prompt_len = 40 + 15 * base[0] + rng.normal(scale=5)
        for v in range(3):
            si = base[0] + rng.normal(scale=0.5)
            neg = float(rng.random() < 0.3)
            y = int(si + 2 * neg + rng.normal(scale=0.7) > 1.0)
            rows.append(
                {
                    "id": f"p{p}:{v}",
                    "prompt_id": f"p{p}",
                    "category": cats[p % 6],
                    "label": "degraded" if y else "preserved",
                    "max_si_removed": si,
                    "tokens_saved": base[1] + rng.normal(scale=0.3),
                    "negations_removed": neg,
                    "prompt_len": prompt_len,
                    "savings_pct": savings,
                    "overhead_pct": 5.0,
                    "stand_in": stand_in,
                }
            )
    out = path / "pairs.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def test_runner_writes_every_artifact(tmp_path: Path) -> None:
    data = synthetic_table(tmp_path)
    out, configs = tmp_path / "reports", tmp_path / "configs"
    code = main(
        [
            "--data",
            str(data),
            "--out",
            str(out),
            "--configs",
            str(configs),
            "--seeds",
            "0",
            "--quick",
            "--allow-stand-in",
            "--no-track",
        ]
    )
    assert code == 0
    for name in [
        "p3_cv_leakage.md",
        "p3_loco.md",
        "p3_tree_pruning.md",
        "p3_tree_rules.md",
        "p3_lr_grid.md",
        "p3_lr_coefficients.md",
        "p3_knn_grid.md",
        "p3_knn_regression.md",
        "p3_skip_rule.md",
        "p3_bias_variance.md",
    ]:
        text = (out / name).read_text(encoding="utf-8")
        assert "STAND-IN DATA" in text, name
    for fig in [
        "p3_tree_overfit.png",
        "p3_tree_boundary.png",
        "p3_lr_calibration.png",
        "p3_knn_boundaries.png",
        "p3_bias_variance.png",
        "p3_learning_curves.png",
    ]:
        assert (out / "figures" / fig).stat().st_size > 0, fig
    for model in ("tree", "knn", "lr"):
        doc = yaml.safe_load((configs / f"p3_{model}.yaml").read_text(encoding="utf-8"))
        assert doc["row"] == "3.6.4" and doc["params"]
    assert "baseline: always_preserved" in (out / "p3_knn_grid.md").read_text(encoding="utf-8")
    assert "knn_regression" in (out / "p3_bias_variance.md").read_text(encoding="utf-8")
    assert len(STUDIES) == 9  # knn_regression reruns the 3.5.2 / 3.5.4 part of knn alone


def test_runner_refuses_stand_in_without_flag(tmp_path: Path) -> None:
    data = synthetic_table(tmp_path)
    assert main(["--data", str(data), "--no-track"]) == 1
    with pytest.raises(ValueError, match="stand-in"):
        load_table(data)


def test_runner_refuses_test_split(tmp_path: Path) -> None:
    d = tmp_path / "data" / "processed"
    d.mkdir(parents=True)
    path = d / "test_pairs.csv"
    pd.DataFrame({"prompt_id": ["a"], "category": ["code"], "label": [0], "f": [1.0]}).to_csv(
        path, index=False
    )
    with pytest.raises(ValueError, match="test split"):
        load_table(path)


def test_runner_requires_columns(tmp_path: Path) -> None:
    path = tmp_path / "x.csv"
    pd.DataFrame({"prompt_id": ["a"], "label": [0], "f": [1.0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="category"):
        load_table(path)
