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


import logging
import re


class _FakeProc:
    def __init__(self, lines, returncode=0, raise_on_line=None):
        self._lines = list(lines)
        self.returncode = returncode
        self.stdout = iter(self._lines)
        self.terminated = False
        self.killed = False

    def wait(self, timeout=None):
        return self.returncode

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


def test_run_energy_scan_parses_result():
    lines = [
        "artifact=3 done=0 todo=3 workers=2\n",
        "  100/3  1.0 trk/s  ETA 0.0h\n",
        "scan pass done: 3 tracks in 0.00h\n",
        "wrote /x/energy/energy_sidecar.npz: ok=2 missing=1 error=0 total=3\n",
    ]
    captured = {}

    def fake_popen(cmd, **kw):
        captured["cmd"] = cmd
        return _FakeProc(lines, returncode=0)

    res = er.run_energy_scan(
        er.EnergyConfig(workers=2),
        repo_root=Path(r"C:\repo"),
        force=False,
        logger=logging.getLogger("test"),
        popen=fake_popen,
    )
    assert res == {"ok": 2, "missing": 1, "error": 0, "total": 3,
                   "sidecar": "/x/energy/energy_sidecar.npz"}
    # command shells to wsl with the translated repo path
    joined = " ".join(captured["cmd"])
    assert "wsl.exe" in captured["cmd"][0]
    assert "/mnt/c/repo" in joined
    assert "--workers 2" in joined
    assert "--force" not in joined


def test_run_energy_scan_force_flag():
    def fake_popen(cmd, **kw):
        assert "--force" in " ".join(cmd)
        return _FakeProc(["wrote x: ok=0 missing=0 error=0 total=0\n"], 0)

    er.run_energy_scan(er.EnergyConfig(), repo_root=Path("C:/r"), force=True,
                       logger=logging.getLogger("test"), popen=fake_popen)


def test_run_energy_scan_nonzero_raises():
    def fake_popen(cmd, **kw):
        return _FakeProc(["boom\n"], returncode=2)

    with pytest.raises(RuntimeError, match="exit"):
        er.run_energy_scan(er.EnergyConfig(), repo_root=Path("C:/r"), force=False,
                           logger=logging.getLogger("test"), popen=fake_popen)


def test_run_energy_scan_cancel_terminates():
    proc_holder = {}

    def fake_popen(cmd, **kw):
        p = _FakeProc(["line1\n", "line2\n"], returncode=0)
        proc_holder["p"] = p
        return p

    calls = {"n": 0}

    def cancel():
        calls["n"] += 1
        raise KeyboardInterrupt("cancelled")

    with pytest.raises(KeyboardInterrupt):
        er.run_energy_scan(er.EnergyConfig(), repo_root=Path("C:/r"), force=False,
                           logger=logging.getLogger("test"),
                           cancellation_check=cancel, popen=fake_popen)
    assert proc_holder["p"].terminated is True
