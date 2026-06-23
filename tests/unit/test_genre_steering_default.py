"""Genre steering canonicalization: taxonomy is the default source, and a
configured-but-unusable source fails loudly instead of silently producing no
targets every segment (the dead-dense regression class).
"""
import numpy as np
import pytest

from src.playlist.pier_bridge_builder import (
    PierBridgeConfig,
    _require_usable_genre_steering,
)
from src.playlist.pier_bridge.config import resolve_pier_bridge_tuning


def test_pier_bridge_config_default_source_is_taxonomy():
    assert PierBridgeConfig().genre_steering_source == "taxonomy"


def test_resolve_tuning_defaults_to_taxonomy_when_unset():
    t = resolve_pier_bridge_tuning(None, "dynamic")
    assert t["genre_steering_source"] == "taxonomy"


def test_resolve_tuning_invalid_source_falls_to_taxonomy():
    t = resolve_pier_bridge_tuning(
        {"pier_bridge": {"genre_steering_source": "nonsense"}}, "dynamic"
    )
    assert t["genre_steering_source"] == "taxonomy"


def test_resolve_tuning_respects_explicit_dense():
    # Explicit opt-in is still honored (so a rebuilt dense sidecar can be used);
    # only the *default* changed.
    t = resolve_pier_bridge_tuning(
        {"pier_bridge": {"genre_steering_source": "dense"}}, "dynamic"
    )
    assert t["genre_steering_source"] == "dense"


def test_dense_steering_without_dense_data_raises():
    cfg = PierBridgeConfig(genre_steering_enabled=True, genre_steering_source="dense")
    with pytest.raises(ValueError, match="dense"):
        _require_usable_genre_steering(cfg, None)


def test_dense_steering_with_dense_data_ok():
    cfg = PierBridgeConfig(genre_steering_enabled=True, genre_steering_source="dense")
    _require_usable_genre_steering(cfg, np.zeros((3, 4), dtype=np.float32))


def test_taxonomy_steering_without_dense_data_ok():
    # Taxonomy uses in-artifact X_genre_raw, so it never needs the dense sidecar.
    cfg = PierBridgeConfig(
        genre_steering_enabled=True, genre_steering_source="taxonomy"
    )
    _require_usable_genre_steering(cfg, None)


def test_disabled_steering_never_raises():
    cfg = PierBridgeConfig(genre_steering_enabled=False, genre_steering_source="dense")
    _require_usable_genre_steering(cfg, None)
