import dataclasses
from src.playlist.pier_bridge.config import PierBridgeConfig


def test_popularity_penalty_strength_defaults_off():
    cfg = PierBridgeConfig()
    assert cfg.popularity_penalty_strength == 0.0


def test_popularity_penalty_strength_flows_through_replace():
    # the relaxation cascade builds attempt configs via dataclasses.replace
    cfg = dataclasses.replace(PierBridgeConfig(), popularity_penalty_strength=0.3)
    assert cfg.popularity_penalty_strength == 0.3
    # other defaults untouched
    assert cfg.genre_penalty_strength == 0.10


def test_pier_bridge_config_popularity_rank_cutoff_defaults_none():
    from src.playlist.pier_bridge.config import PierBridgeConfig
    cfg = PierBridgeConfig()
    assert cfg.popularity_rank_cutoff is None
