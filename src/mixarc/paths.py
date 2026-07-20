# src/mixarc/paths.py
"""Where MixArc lives on this machine.

A repo checkout (canonical/satellites) resolves to its own config.yaml and
repo-relative data exactly as before. A wheel install resolves to per-user
platform dirs; the repo branch self-disables there because _REPO_ROOT points
inside site-packages, where a config.yaml never exists.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from platformdirs import user_config_dir, user_data_dir

_REPO_ROOT = Path(__file__).resolve().parents[2]
_APP = "mixarc"


@dataclass(frozen=True)
class MixarcHome:
    config_path: Path   # where config.yaml is (or would be written)
    anchor_dir: Path    # base dir for relative data paths
    source: str         # "cli" | "env" | "repo" | "platformdirs"


def resolve_home(cli_config: str | None = None) -> MixarcHome:
    if cli_config:
        p = Path(cli_config)
        return MixarcHome(config_path=p, anchor_dir=p.parent, source="cli")
    env = os.environ.get("MIXARC_HOME", "").strip()
    if env:
        base = Path(env)
        return MixarcHome(config_path=base / "config.yaml", anchor_dir=base, source="env")
    repo_cfg = _REPO_ROOT / "config.yaml"
    if repo_cfg.exists():
        return MixarcHome(config_path=repo_cfg, anchor_dir=_REPO_ROOT, source="repo")
    cfg_dir = Path(user_config_dir(_APP, appauthor=False))
    data_dir = Path(user_data_dir(_APP, appauthor=False))
    return MixarcHome(config_path=cfg_dir / "config.yaml", anchor_dir=data_dir, source="platformdirs")
