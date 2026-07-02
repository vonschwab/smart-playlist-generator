"""Roam corridors: overrides -> PierBridgeConfig plumbing (Phase-1 opt-in)."""
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


def test_roam_override_enables_and_sets_all_knobs():
    cfg = _apply({"pier_bridge": {"roam": {
        "enabled": True, "knn_k": 40, "mutual_proximity": False,
        "width_sonic": 2.0, "width_genre": 1.0, "width_energy": 0.5,
        "penalty_slope": 3.0, "worst_edge_minimax": True,
    }}})
    assert cfg.roam_corridors_enabled is True
    assert cfg.roam_knn_k == 40
    assert cfg.roam_mutual_proximity is False
    assert cfg.roam_width_sonic == 2.0
    assert cfg.roam_width_genre == 1.0
    assert cfg.roam_width_energy == 0.5
    assert cfg.roam_penalty_slope == 3.0
    assert cfg.worst_edge_minimax_enabled is True


def test_roam_override_absent_leaves_defaults_off():
    cfg = _apply({"pier_bridge": {}})
    assert cfg.roam_corridors_enabled is False
    assert cfg.roam_width_sonic == 0.0
    assert cfg.worst_edge_minimax_enabled is False


def test_roam_override_partial_only_sets_given_fields():
    cfg = _apply({"pier_bridge": {"roam": {"enabled": True, "width_sonic": 1.5}}})
    assert cfg.roam_corridors_enabled is True
    assert cfg.roam_width_sonic == 1.5
    assert cfg.roam_width_genre == 0.0      # untouched default
    assert cfg.roam_knn_k == 25             # untouched default
