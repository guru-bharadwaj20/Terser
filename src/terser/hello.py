"""Smoke entry point: ``python -m terser.hello experiment=hello hardware=cpu seed=3``.

Prints the fully resolved config as YAML and exits 0.
"""

from __future__ import annotations

from omegaconf import DictConfig, OmegaConf

from terser.config import hydra_main


@hydra_main()
def main(cfg: DictConfig) -> None:
    """Print the resolved config."""
    print(OmegaConf.to_yaml(cfg, resolve=True), end="")


if __name__ == "__main__":
    main()
