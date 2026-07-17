"""
Phase 0 Task 3: loud startup warnings for retired pool knobs.

"A configured knob that can't act is a startup error, not a silent no-op."
(CLAUDE.md, "Project-specific gotchas"). Task 1 deleted the per-artist cap
walk + never-starve backstop in src/playlist/candidate_pool.py; Task 2
deleted the min_pool_size preset/resolution chains that fed both the
CandidatePoolConfig.min_pool_size field and the reviewer-confirmed-dead
PierBridgeTuning.min_pool_size field. This test locks in that
_warn_retired_keys finds every yaml key that used to reach one of those
now-deleted code paths, in both locations they can appear:
  - playlists.ds_pipeline.candidate_pool.* (candidates_per_artist,
    target_artists, max_pool_size, min_pool_size, seed_artist_bonus)
  - playlists.ds_pipeline.pier_bridge.min_pool_size and its cohesion_mode-
    suffixed variants (min_pool_size_strict/_narrow/_dynamic/_discover)
"""

from src.playlist_gui.worker import _warn_retired_keys, _WARNED_RETIRED_KEYS


def test_dedup_warnings_per_process(caplog):
    """Verify that retired-key warnings are deduped per process: same key
    only logs once even if _warn_retired_keys is called multiple times.
    Both calls must still return the found key."""
    _WARNED_RETIRED_KEYS.clear()
    cfg = {"playlists": {"ds_pipeline": {"candidate_pool": {
        "min_pool_size": 0,
    }}}}

    with caplog.at_level("WARNING"):
        found1 = _warn_retired_keys(cfg)
        found2 = _warn_retired_keys(cfg)

    # Both calls must return the key
    assert "candidate_pool.min_pool_size" in found1
    assert "candidate_pool.min_pool_size" in found2

    # Warning text must appear exactly once across both calls
    warning_records = [r for r in caplog.records if r.levelname == "WARNING"]
    min_pool_size_warnings = [r for r in warning_records
                               if "candidate_pool.min_pool_size" in r.message]
    assert len(min_pool_size_warnings) == 1


def test_retired_candidate_pool_knobs_warn():
    cfg = {"playlists": {"ds_pipeline": {"candidate_pool": {
        "candidates_per_artist": 6,
        "target_artists": 22,
        "max_pool_size": 2400,
        "min_pool_size": 0,
        "seed_artist_bonus": 2,
    }}}}
    found = _warn_retired_keys(cfg)
    assert set(found) >= {
        "candidate_pool.candidates_per_artist",
        "candidate_pool.target_artists",
        "candidate_pool.max_pool_size",
        "candidate_pool.min_pool_size",
        "candidate_pool.seed_artist_bonus",
    }


def test_retired_pier_bridge_min_pool_size_family_warns():
    cfg = {"playlists": {"ds_pipeline": {"pier_bridge": {
        "min_pool_size": 20,
        "min_pool_size_strict": 12,
        "min_pool_size_narrow": 16,
        "min_pool_size_dynamic": 20,
        "min_pool_size_discover": 24,
    }}}}
    found = _warn_retired_keys(cfg)
    assert set(found) >= {
        "pier_bridge.min_pool_size",
        "pier_bridge.min_pool_size_strict",
        "pier_bridge.min_pool_size_narrow",
        "pier_bridge.min_pool_size_dynamic",
        "pier_bridge.min_pool_size_discover",
    }


def test_live_pool_gate_keys_do_not_warn():
    """Keys that still act (e.g. the still-live segment_pool_max / bridge_floor
    family) must never be flagged -- a false positive here is exactly the
    noise the task brief warns against."""
    cfg = {"playlists": {"ds_pipeline": {
        "candidate_pool": {
            "similarity_floor": 0.2,
            "max_artist_fraction": 0.125,
        },
        "pier_bridge": {
            "bridge_floor_dynamic": 0.02,
            "bridge_floor_narrow": 0.05,
            "bridge_floor_strict": 0.1,
        },
    }}}
    assert _warn_retired_keys(cfg) == []


def test_clean_config_warns_nothing():
    assert _warn_retired_keys({"playlists": {}}) == []


def test_empty_config_warns_nothing():
    assert _warn_retired_keys({}) == []
