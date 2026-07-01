from src.playlist.config import default_ds_config
from src.playlist.pier_bridge.config import PierBridgeConfig
from src.playlist.pipeline.pier_bridge_overrides import apply_pier_bridge_overrides

def _apply(overrides):
    cfg, _, _, _ = apply_pier_bridge_overrides(
        pier_bridge_config=PierBridgeConfig(),
        cfg=default_ds_config("dynamic", playlist_len=3),
        overrides=overrides, pb_overrides=overrides.get("pier_bridge", {}),
        artist_playlist=False, dry_run=True, audit_cfg=None, resolved_variant="raw")
    return cfg

def test_mini_pier_defaults_off():
    c = _apply({"pier_bridge": {}})
    assert c.mini_pier_enabled is False
    assert c.mini_pier_max_interior == 5
    assert c.mini_pier_smoothness_margin == 0.12

def test_mini_pier_overrides_propagate():
    c = _apply({"pier_bridge": {"mini_pier_enabled": True,
                                "mini_pier_max_interior": 4,
                                "mini_pier_smoothness_margin": 0.1}})
    assert c.mini_pier_enabled is True
    assert c.mini_pier_max_interior == 4
    assert c.mini_pier_smoothness_margin == 0.1
