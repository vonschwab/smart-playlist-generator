"""Drive the WSL-only Essentia energy extractor from the Windows pipeline.

The Windows runtime never imports essentia; this module shells out to
`wsl.exe ... /opt/ess/bin/python scripts/extract_energy_sidecar.py`.
Energy is a standalone pace-axis sidecar under <artifact>/energy/.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import yaml


@dataclass
class EnergyConfig:
    distro: str = "Ubuntu-22.04"
    python: str = "/opt/ess/bin/python"
    models_dir: str = "/opt/ess/models"
    workers: int = 14


def load_energy_config(config_path: str) -> EnergyConfig:
    """Read analyze.energy from config.yaml; defaults on any miss."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception:
        cfg = {}
    block = ((cfg.get("analyze") or {}).get("energy") or {})
    default = EnergyConfig()
    return EnergyConfig(
        distro=str(block.get("distro", default.distro)),
        python=str(block.get("python", default.python)),
        models_dir=str(block.get("models_dir", default.models_dir)),
        workers=int(block.get("workers", default.workers)),
    )


def win_path_to_wsl(p: str) -> str:
    p = str(p).replace("\\", "/")
    if len(p) > 1 and p[1] == ":":
        return "/mnt/" + p[0].lower() + p[2:]
    return p


def energy_paths(out_dir: Path) -> tuple[Path, Path, Path]:
    out_dir = Path(out_dir)
    return (
        out_dir / "data_matrices_step1.npz",
        out_dir / "energy" / "checkpoint.jsonl",
        out_dir / "energy" / "energy_sidecar.npz",
    )


def pending_energy(out_dir: Path) -> tuple[int, int]:
    artifact_npz, ckpt, _ = energy_paths(out_dir)
    if not artifact_npz.exists():
        return (0, 0)
    track_ids = [str(t) for t in np.load(artifact_npz, allow_pickle=True)["track_ids"]]
    done: set[str] = set()
    if ckpt.exists():
        with open(ckpt, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    done.add(json.loads(line)["track_id"])
                except Exception:
                    continue
    pending = sum(1 for t in track_ids if t not in done)
    return (pending, len(track_ids))


def preflight_wsl(cfg: EnergyConfig, *, runner: Callable = subprocess.run) -> None:
    """Raise RuntimeError if the WSL distro, venv, or models are unreachable."""
    probe = f"test -x {cfg.python} && test -f {cfg.models_dir}/msd-musicnn-1.pb"
    cmd = ["wsl.exe", "-d", cfg.distro, "-u", "root", "--", "bash", "-c", probe]
    try:
        res = runner(cmd, capture_output=True, text=True, timeout=60)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "WSL not available (wsl.exe not found). The energy stage needs WSL2 + "
            "the Essentia venv at /opt/ess. See project_energy_feature_exploration."
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"WSL preflight failed to run: {exc!r}") from exc
    if getattr(res, "returncode", 1) != 0:
        raise RuntimeError(
            f"WSL energy environment not ready (distro={cfg.distro}, python={cfg.python}, "
            f"models={cfg.models_dir}). Set up the /opt/ess venv + models, or fix "
            f"analyze.energy in config.yaml. stderr: {getattr(res, 'stderr', '')!r}"
        )
