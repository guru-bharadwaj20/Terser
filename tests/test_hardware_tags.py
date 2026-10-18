"""Hardware tag for MLflow runs and the flows CLI parameter parser."""

from __future__ import annotations

import pytest

from terser.flows.__main__ import _parse_params
from terser.hardware.tags import hardware_tag


def test_hardware_tag_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TERSER_HARDWARE", "rtx6000ada")
    assert hardware_tag() == "rtx6000ada"


def test_param_without_value_is_rejected() -> None:
    assert _parse_params(["n=3", "name=abc"]) == {"n": 3, "name": "abc"}
    with pytest.raises(SystemExit):
        _parse_params(["foo"])
