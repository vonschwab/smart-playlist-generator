from src.playlist.config import default_ds_config
from src.playlist.pier_bridge.config import PierBridgeConfig
from src.playlist.pipeline.pier_bridge_overrides import apply_pier_bridge_overrides


def _apply(overrides: dict):
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


def test_layered_genre_graph_source_enables_layered_transition_scoring():
    cfg = _apply(
        {
            "genre_graph": {
                "source": "layered",
                "transition_weight": 0.25,
            }
        }
    )

    assert cfg.layered_transition_scoring_enabled is True
    assert cfg.layered_transition_weight == 0.25
    assert cfg.layered_transition_mode == "dynamic"


def test_layered_shadow_does_not_enable_layered_transition_scoring():
    cfg = _apply({"genre_graph": {"source": "layered_shadow"}})

    assert cfg.layered_transition_scoring_enabled is False
    assert cfg.layered_transition_weight == 0.0


def test_layered_source_disables_legacy_flat_genre_bridge_routing():
    cfg = _apply(
        {
            "genre_graph": {"source": "layered"},
            "pier_bridge": {
                "genre_steering_enabled": True,
                "weight_genre": 0.25,
                "dj_bridging": {
                    "enabled": True,
                    "pooling": {"k_genre": 25},
                },
            },
        }
    )

    assert cfg.layered_transition_scoring_enabled is True
    assert cfg.genre_steering_enabled is False
    assert cfg.weight_genre == 0.0
    assert cfg.dj_bridging_enabled is False
    assert cfg.dj_pooling_k_genre == 0
