"""Print the fully composed Hydra config, then exit.

A verification tool: use it to check that group selection, interpolation, and
CLI overrides resolve to what you expect BEFORE spending a Kaggle GPU hour
finding out they did not.

    python scripts/show_config.py
    python scripts/show_config.py training.batch_size=4
    python scripts/show_config.py model=unet3d data.root_dir=/path/to/brats

This is a CLI entry point, so it prints to stdout on purpose — the
no-bare-print rule applies to library code under src/, not to scripts whose
whole job is producing terminal output.
"""

from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

# Relative to this file, so the script works from any working directory and on
# any machine — no absolute paths.
_CONFIG_DIR = str(Path(__file__).resolve().parent.parent / "configs")


@hydra.main(version_base="1.3", config_path=_CONFIG_DIR, config_name="config")
def main(cfg: DictConfig) -> None:
    """Print the composed config as YAML.

    Args:
        cfg: The config Hydra composed from configs/ plus any CLI overrides.
    """
    # resolve=True expands ${...} interpolations so you see the real values
    # rather than the references. Mandatory values still unset (Hydra's "???",
    # e.g. data.root_dir) would raise during resolution, so fall back to the
    # unresolved view and say why — being able to inspect an incomplete config
    # is exactly the point of this script.
    try:
        print(OmegaConf.to_yaml(cfg, resolve=True))
    except Exception as exc:  # noqa: BLE001 - want any resolution failure reported, not raised
        print(OmegaConf.to_yaml(cfg, resolve=False))
        print(f"# NOTE: interpolations left unresolved: {exc}")
        print("# Supply the missing mandatory value(s), e.g. data.root_dir=/path/to/brats")


if __name__ == "__main__":
    main()
