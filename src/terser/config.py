"""Hydra config helpers: project paths, the ``terser_root`` resolver and compose-based loading.

Script convention (every ``python -m terser.x experiment=...`` entry point)::

    from omegaconf import DictConfig
    from terser.config import hydra_main

    @hydra_main()
    def main(cfg: DictConfig) -> None:
        ...

    if __name__ == "__main__":
        main()

``hydra_main`` is ``hydra.main(config_path=<abs configs dir>, config_name="config",
version_base="1.3")``, so config lookup never depends on the working directory. Notebooks and
tests use :func:`load_config` instead.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import hydra
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

ROOT_ENV_VAR = "TERSER_ROOT"
RESOLVER_NAME = "terser_root"
CONFIG_NAME = "config"
VERSION_BASE = "1.3"


def project_root() -> Path:
    """Return the project root: ``$TERSER_ROOT`` if set, else the repo containing this file."""
    env = os.environ.get(ROOT_ENV_VAR)
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def configs_dir() -> Path:
    """Return the absolute path of the Hydra ``configs/`` directory."""
    return project_root() / "configs"


def register_resolvers() -> None:
    """Register the ``terser_root`` OmegaConf resolver (idempotent)."""
    if not OmegaConf.has_resolver(RESOLVER_NAME):
        OmegaConf.register_new_resolver(RESOLVER_NAME, lambda: project_root().as_posix())


def load_config(overrides: list[str] | None = None) -> DictConfig:
    """Compose the root config with optional Hydra overrides, without ``@hydra.main``."""
    register_resolvers()
    with initialize_config_dir(config_dir=str(configs_dir()), version_base=VERSION_BASE):
        return compose(config_name=CONFIG_NAME, overrides=list(overrides or []))


def hydra_main() -> Callable[[Callable[[DictConfig], Any]], Callable[[], Any]]:
    """Return the standard ``@hydra.main`` decorator bound to the absolute configs dir."""
    register_resolvers()
    return hydra.main(
        config_path=str(configs_dir()), config_name=CONFIG_NAME, version_base=VERSION_BASE
    )


register_resolvers()
