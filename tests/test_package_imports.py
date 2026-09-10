"""0.2.2: every package module imports cleanly."""

import importlib

import pytest

MODULES = [
    "terser",
    "terser.operators",
    "terser.concept",
    "terser.data",
    "terser.judge",
    "terser.features",
    "terser.models",
    "terser.models.classical",
    "terser.models.neural",
    "terser.models.ensembles",
    "terser.models.clustering",
    "terser.models.rewriter",
    "terser.models.rl",
    "terser.eval",
    "terser.hpo",
    "terser.serving",
    "terser.monitoring",
    "terser.pipelines",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name: str) -> None:
    module = importlib.import_module(name)
    assert module.__doc__, f"{name} needs a module docstring"
