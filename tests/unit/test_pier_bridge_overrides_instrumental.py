"""Instrumental lean: PierBridgeConfig fields + override parsing (Task 6).

instrumental_enabled is per-request (set by the policy layer, e.g. from a GUI
checkbox); instrumental_penalty_weight is a static tuning value from
config.yaml. Both default off (False / 0.0) so the feature is inert until a
later task (the beam, Task 7) reads them and a request opts in.

Mirrors the apply_pier_bridge_overrides call pattern used by
tests/unit/test_mini_pier_overrides.py and
tests/unit/test_generation_budget_override.py.
"""
from src.playlist.config import default_ds_config
from src.playlist.pier_bridge.config import PierBridgeConfig
from src.playlist.pipeline.pier_bridge_overrides import apply_pier_bridge_overrides


def _apply(overrides: dict) -> PierBridgeConfig:
    pb_cfg, _tuning, _sources = apply_pier_bridge_overrides(
        pier_bridge_config=PierBridgeConfig(),
        cfg=default_ds_config("dynamic", playlist_len=3),
        overrides=overrides,
        pb_overrides=overrides.get("pier_bridge", {}),
        artist_playlist=False,
        dry_run=True,
        audit_cfg=None,
    )
    return pb_cfg


def test_instrumental_fields_default_off():
    cfg = PierBridgeConfig()
    assert cfg.instrumental_enabled is False
    assert cfg.instrumental_penalty_weight == 0.0


def test_overrides_absent_keeps_dataclass_default():
    cfg = _apply({"pier_bridge": {}})
    assert cfg.instrumental_enabled is False
    assert cfg.instrumental_penalty_weight == 0.0


def test_overrides_apply_instrumental_fields():
    cfg = _apply({
        "pier_bridge": {
            "instrumental_enabled": True,
            "instrumental_penalty_weight": 0.6,
        }
    })
    assert cfg.instrumental_enabled is True
    assert abs(cfg.instrumental_penalty_weight - 0.6) < 1e-9
