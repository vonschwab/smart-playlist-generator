from pathlib import Path

import numpy as np
import pytest

from src.analyze import energy_runner as er


def test_win_path_to_wsl():
    assert er.win_path_to_wsl(r"C:\Users\Dylan\proj") == "/mnt/c/Users/Dylan/proj"


def test_load_energy_config_defaults(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("library:\n  music_directory: /x\n", encoding="utf-8")
    cfg = er.load_energy_config(str(cfg_file))
    assert cfg.distro == "Ubuntu-22.04"
    assert cfg.python == "/opt/ess/bin/python"
    assert cfg.workers == 14


def test_load_energy_config_overrides(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        "analyze:\n  energy:\n    distro: Ubuntu-24.04\n    workers: 8\n",
        encoding="utf-8",
    )
    cfg = er.load_energy_config(str(cfg_file))
    assert cfg.distro == "Ubuntu-24.04"
    assert cfg.workers == 8
    assert cfg.python == "/opt/ess/bin/python"  # untouched default


def test_pending_energy(tmp_path):
    out = tmp_path
    np.savez(
        out / "data_matrices_step1.npz",
        track_ids=np.array(["a", "b", "c"], dtype=object),
    )
    (out / "energy").mkdir()
    (out / "energy" / "checkpoint.jsonl").write_text(
        '{"track_id": "a", "arousal_p50": 4.5}\n', encoding="utf-8"
    )
    pending, total = er.pending_energy(out)
    assert (pending, total) == (2, 3)


def test_pending_energy_no_artifact(tmp_path):
    assert er.pending_energy(tmp_path) == (0, 0)


def test_preflight_wsl_ok():
    def fake_runner(cmd, **kw):
        class R:
            returncode = 0
            stdout = ""
            stderr = ""
        return R()

    er.preflight_wsl(er.EnergyConfig(), runner=fake_runner)  # no raise


def test_preflight_wsl_missing_raises():
    def fake_runner(cmd, **kw):
        class R:
            returncode = 1
            stdout = ""
            stderr = "not found"
        return R()

    with pytest.raises(RuntimeError, match="WSL"):
        er.preflight_wsl(er.EnergyConfig(), runner=fake_runner)
