"""Registry metrics are read from the reports that measured them, not hard-coded."""

from __future__ import annotations

import json
from pathlib import Path

from terser.phase11.register import COMPONENTS, load_metrics

ROOT = Path(__file__).resolve().parents[2]


def src_path(src: str) -> str:
    return str(ROOT / src)


def test_load_metrics_key_and_fields(tmp_path: Path) -> None:
    f = tmp_path / "m.json"
    f.write_text(json.dumps({"a": {"x": 0.5, "y": 2}, "z": 0.25}), encoding="utf-8")
    assert load_metrics(str(f), "a", {"X": "x", "Y": "y"}) == {"X": 0.5, "Y": 2.0}
    assert load_metrics(str(f), None, {"Z": "z"}) == {"Z": 0.25}


def test_components_point_at_files() -> None:
    for c in COMPONENTS:
        src, _key, fields = c["metrics"]
        assert src.startswith("reports/") and src.endswith(".json")
        assert fields
        assert (ROOT / src).is_file(), f"{src} missing"
        # Every metric the registry reads must exist in the committed report.
        assert load_metrics(src_path(src), _key, fields).keys() == set(fields)
