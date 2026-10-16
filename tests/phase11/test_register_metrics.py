"""Registry metrics are read from the reports that measured them, not hard-coded."""

from __future__ import annotations

import json
from pathlib import Path

from terser.phase11.register import COMPONENTS, load_metrics


def test_load_metrics_key_and_fields(tmp_path: Path) -> None:
    f = tmp_path / "m.json"
    f.write_text(json.dumps({"a": {"x": 0.5, "y": 2}, "z": 0.25}))
    assert load_metrics(str(f), "a", {"X": "x", "Y": "y"}) == {"X": 0.5, "Y": 2.0}
    assert load_metrics(str(f), None, {"Z": "z"}) == {"Z": 0.25}


def test_components_point_at_files() -> None:
    for c in COMPONENTS:
        src, _key, fields = c["metrics"]
        assert src.startswith("reports/") and src.endswith(".json")
        assert fields
