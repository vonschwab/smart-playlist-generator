from unittest.mock import MagicMock

from src.analyze.energy_runner import EnergyConfig, preflight_wsl


def _cfg():
    return EnergyConfig()  # defaults are fine; we only inspect the probe string


def test_preflight_probe_includes_voice_instrumental_model():
    captured = {}

    def fake_runner(cmd, **kwargs):
        captured["cmd"] = cmd
        m = MagicMock()
        m.returncode = 0
        return m

    preflight_wsl(_cfg(), runner=fake_runner)
    probe = " ".join(captured["cmd"])
    assert "voice_instrumental-musicnn-msd-2.pb" in probe
