"""Regression guard for the artist-style pier-config silent-drop bug (2026-07-02).

The artist-style entry points build a *pre-built* ``PierBridgeConfig`` and pass it into
the pipeline, which short-circuits ``apply_pier_bridge_overrides``' own construction. So
every mode-tuned field the beam reads must be threaded from resolved tuning in
``_build_artist_pier_config`` — a field omitted there silently falls to a
``PierBridgeConfig`` dataclass default. Before this guard, nine fields were dropped: the
six beam/pool sizes ran at HALF config width and the two genre guardrails were off on
every artist playlist. See docs/CLEANUP_LIST.md.
"""
from types import SimpleNamespace

from src.playlist_generator import _build_artist_pier_config
from src.playlist.pier_bridge_builder import resolve_pier_bridge_tuning

# The fields that were silently dropping to dataclass defaults. Each MUST be threaded
# from resolved tuning; add any new mode-tuned beam/genre knob here when you add it to
# config + the helper.
PREVIOUSLY_DROPPED_FIELDS = (
    "initial_beam_width",
    "max_beam_width",
    "initial_neighbors_m",
    "max_neighbors_m",
    "initial_bridge_helpers",
    "max_bridge_helpers",
    "segment_pool_genre_weight",
    "genre_pair_floor",
    "genre_pair_penalty",
)


def test_artist_pier_config_threads_resolved_tuning_no_silent_drop():
    # Distinctive, non-default values so a dropped field fails loudly instead of
    # coincidentally matching its dataclass default.
    ds_cfg = {
        "pier_bridge": {
            "initial_beam_width": 37,
            "max_beam_width": 199,
            "initial_neighbors_m": 201,
            "max_neighbors_m": 799,
            "initial_bridge_helpers": 101,
            "max_bridge_helpers": 399,
            "segment_pool_genre_weight": 0.23,
            "genre_pair_floor_dynamic": 0.11,
        }
    }
    pb_tuning = resolve_pier_bridge_tuning(ds_cfg, "dynamic")
    ds_defaults = SimpleNamespace(
        construct=SimpleNamespace(transition_floor=0.2, center_transitions=True)
    )

    cfg = _build_artist_pier_config(
        pb_tuning=pb_tuning,
        ds_defaults=ds_defaults,
        ds_cfg=ds_cfg,
        weight_bridge=0.6,
        weight_transition=0.4,
        genre_tiebreak_weight=0.05,
        bridge_floor=0.02,
    )

    for field in PREVIOUSLY_DROPPED_FIELDS:
        assert getattr(cfg, field) == pb_tuning[field], (
            f"{field}: pier config has {getattr(cfg, field)!r} but resolved tuning has "
            f"{pb_tuning[field]!r} — the field was dropped to its dataclass default "
            "(the silent-drop bug this guard exists to catch)."
        )

    # Spot-check the specific consequences of the original bug: full beam width, not half,
    # and the genre guardrail floor is armed, not zero.
    assert cfg.initial_beam_width == 37
    assert cfg.max_beam_width == 199
    assert cfg.segment_pool_genre_weight == 0.23
    assert cfg.genre_pair_floor == 0.11


def test_artist_pier_config_popularity_defaults_off():
    # The plain artist path calls the helper without popularity args; they must default
    # to off (the pre-helper behavior of that constructor).
    ds_cfg = {"pier_bridge": {}}
    pb_tuning = resolve_pier_bridge_tuning(ds_cfg, "dynamic")
    ds_defaults = SimpleNamespace(
        construct=SimpleNamespace(transition_floor=0.2, center_transitions=True)
    )

    cfg = _build_artist_pier_config(
        pb_tuning=pb_tuning,
        ds_defaults=ds_defaults,
        ds_cfg=ds_cfg,
        weight_bridge=0.6,
        weight_transition=0.4,
        genre_tiebreak_weight=0.05,
        bridge_floor=0.02,
    )
    assert cfg.popularity_penalty_strength == 0.0
    assert cfg.popularity_rank_cutoff is None
