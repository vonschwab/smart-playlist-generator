import types
from pathlib import Path

import numpy as np
import pytest

from src.playlist.artist_style import (
    ArtistStyleConfig,
    build_balanced_candidate_pool,
    build_genre_neighbor_candidate_pool,
    cluster_artist_tracks,
    order_clusters,
    _finite_median,
    _robust_energy_span,
    _slot_targets_by_quantile,
    _slot_proximity,
    _medoids_for_cluster,
    _dedupe_artist_indices,
    load_artist_energy_values,
    compute_pier_bridgeability,
)


class DummyBundle:
    def __init__(self, X_sonic, artist_keys, track_ids, track_artists=None, track_titles=None):
        self.X_sonic = X_sonic
        self.X_sonic_start = None
        self.X_sonic_mid = None
        self.X_sonic_end = None
        self.artist_keys = artist_keys
        self.track_ids = track_ids
        self.track_artists = track_artists
        self.track_titles = track_titles
        self.track_id_to_index = {str(t): i for i, t in enumerate(track_ids)}
        # ArtifactBundle has Optional[np.ndarray] = None for these; the prod
        # cluster_artist_tracks code path checks them, so the mock must mirror.
        self.durations_ms = None
        self.X_genre_raw = None
        self.X_genre_smoothed = None
        self.genre_vocab = None


def test_artist_style_config_has_energy_defaults():
    cfg = ArtistStyleConfig()
    assert cfg.medoid_energy_weight == 0.0          # opt-in: off by default
    assert cfg.energy_feature == "arousal_p50"
    assert cfg.energy_slot_lo_pct == 10.0
    assert cfg.energy_slot_hi_pct == 90.0


def test_selects_multiple_clusters_and_medoids():
    # Two clear clusters on unit vectors
    artist_keys = np.array(["a"] * 6)
    track_ids = np.array([str(i) for i in range(6)])
    X = np.array([
        [1.0, 0.0], [0.9, 0.1], [0.95, -0.05],  # cluster 1
        [0.0, 1.0], [0.1, 0.9], [-0.05, 0.95],  # cluster 2
    ])
    bundle = DummyBundle(X_sonic=X, artist_keys=artist_keys, track_ids=track_ids)
    cfg = ArtistStyleConfig(cluster_k_min=2, cluster_k_max=2, piers_per_cluster=1, enabled=True)
    clusters, medoids, medoids_by_cluster, X_norm = cluster_artist_tracks(
        bundle=bundle,
        artist_name="A",
        cfg=cfg,
        random_seed=0,
    )
    assert len(clusters) == 2
    assert len(medoids_by_cluster) == 2
    ordered = order_clusters(medoids, X_norm)
    assert set(ordered) == set(medoids)


def test_balanced_candidate_pool_respects_per_cluster_limits():
    artist_keys = np.array(["a", "a", "b", "c", "d", "e"])
    track_ids = np.array([str(i) for i in range(6)])
    X = np.array([
        [1.0, 0.0],
        [0.9, 0.0],
        [0.0, 1.0],
        [0.0, 0.9],
        [0.7, 0.7],
        [0.6, 0.6],
    ])
    bundle = DummyBundle(X_sonic=X, artist_keys=artist_keys, track_ids=track_ids)
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    # Two clusters with single pier each
    cluster_piers = [[0], [2]]
    cfg = ArtistStyleConfig(per_cluster_candidate_pool_size=1)
    pool = build_balanced_candidate_pool(
        bundle=bundle,
        cluster_piers=cluster_piers,
        X_norm=X_norm,
        per_cluster_size=1,
        pool_balance_mode="equal",
        global_floor=0.0,
        artist_key="a",
    )
    # Should take one from each cluster
    assert len(pool) == 2
    assert any(pid in pool for pid in ["4", "5"])
    assert any(pid in pool for pid in ["2", "3"])


def test_genre_neighbor_pool_recovers_low_sonic_genre_match():
    artist_keys = np.array(["seed", "seed", "genre-match", "conflict"])
    track_ids = np.array(["s0", "s1", "genre_match", "conflict"])
    X = np.array([
        [1.0, 0.0],
        [0.9, 0.1],
        [0.0, 1.0],  # low sonic similarity to seed piers
        [0.0, 1.0],
    ])
    bundle = DummyBundle(X_sonic=X, artist_keys=artist_keys, track_ids=track_ids)
    bundle.genre_vocab = np.array(["indie pop", "punk", "rnb", "house", "soul"])
    bundle.X_genre_raw = np.array([
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 0, 1, 1, 1],
    ], dtype=float)
    bundle.X_genre_smoothed = bundle.X_genre_raw.copy()

    pool = build_genre_neighbor_candidate_pool(
        bundle=bundle,
        pier_indices=[0, 1],
        artist_key="seed",
        pool_size=10,
        min_similarity=0.3,
        min_confidence=0.5,
        compatible_threshold=0.35,
        conflict_threshold=0.15,
        genre_method="weighted_jaccard",
    )

    assert "genre_match" in pool
    assert "conflict" not in pool


def test_allowed_invariant_helper():
    from src.playlist.pipeline import enforce_allowed_invariant
    allowed = {"1", "2"}
    enforce_allowed_invariant(["1", "2"], allowed, context="test")
    with pytest.raises(ValueError):
        enforce_allowed_invariant(["1", "3"], allowed, context="test")


def test_bridge_endpoint_gate_blocks_low_sim():
    from src.playlist.pier_bridge_builder import _build_segment_candidate_pool_scored
    X = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.9, 0.1],  # high to A, low to B
    ])
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    bundle = DummyBundle(
        X_sonic=X,
        artist_keys=np.array(["a", "b", "c"]),
        track_ids=np.array(["a0", "b0", "c0"]),
        track_artists=np.array(["a", "b", "c"]),
        track_titles=np.array(["A", "B", "C"]),
    )
    pool, _artist_keys, _title_keys = _build_segment_candidate_pool_scored(
        pier_a=0,
        pier_b=1,
        X_full_norm=X_norm,
        universe_indices=[2],
        used_track_ids=set(),
        bundle=bundle,
        bridge_floor=0.2,
        segment_pool_max=50,
    )
    assert pool == []


def test_narrow_bridge_floor_default_blocks_low_sim():
    # Candidate min(simA, simB) below 0.08 should be excluded by bridge gate.
    from src.playlist.pier_bridge_builder import _build_segment_candidate_pool_scored

    X = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.99, 0.06],  # simB ~= 0.06 < 0.08
    ])
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    bundle = DummyBundle(
        X_sonic=X,
        artist_keys=np.array(["a", "b", "c"]),
        track_ids=np.array(["a0", "b0", "c0"]),
        track_artists=np.array(["a", "b", "c"]),
        track_titles=np.array(["A", "B", "C"]),
    )
    pool, _artist_keys, _title_keys = _build_segment_candidate_pool_scored(
        pier_a=0,
        pier_b=1,
        X_full_norm=X_norm,
        universe_indices=[2],
        used_track_ids=set(),
        bundle=bundle,
        bridge_floor=0.08,
        segment_pool_max=50,
    )
    assert pool == []


def test_seed_artist_disallowed_in_interiors_when_enabled():
    # For artist playlist runs, we want "seed artist = piers only" when the policy is enabled,
    # even if seed-artist candidates would otherwise win.
    from src.playlist.pier_bridge_builder import PierBridgeConfig, build_pier_bridge_playlist

    X = np.array([
        [1.0, 0.0],                   # pier A (seed artist)
        [0.0, 1.0],                   # pier B (seed artist)
        [np.sqrt(0.5), np.sqrt(0.5)], # seed-artist candidate (would win on bridge score)
        [0.6, 0.8],                   # other-artist candidate
    ])
    bundle = DummyBundle(
        X_sonic=X,
        artist_keys=np.array(["seed", "seed", "seed", "other"]),
        track_ids=np.array(["a0", "b0", "seed_cand", "other_cand"]),
        track_artists=np.array(["Seed Artist", "Seed Artist", "Seed Artist feat. X", "Other Artist"]),
        track_titles=np.array(["A", "B", "Bridge Seed", "Bridge Other"]),
    )
    result = build_pier_bridge_playlist(
        seed_track_ids=["a0", "b0"],
        total_tracks=3,
        bundle=bundle,
        candidate_pool_indices=[2, 3],
        cfg=PierBridgeConfig(
            transition_floor=0.0,
            bridge_floor=0.0,
            weight_bridge=1.0,
            weight_transition=0.0,
            eta_destination_pull=0.0,
            disallow_seed_artist_in_interiors=True,
        ),
        allowed_track_ids_set={"a0", "b0", "seed_cand", "other_cand"},
    )
    assert result.success
    assert result.track_ids[1] == "other_cand"


def test_builder_wires_roam_detour_when_enabled(monkeypatch):
    # Roam corridors (Phase 1): when the flag is on, the builder must compute the
    # per-segment on-manifold sonic detour and pass it to the beam; when off, the
    # beam gets None. Both must still produce a valid playlist (never-fail).
    import src.playlist.pier_bridge_builder as pbb

    X = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [np.sqrt(0.5), np.sqrt(0.5)],
        [0.6, 0.8],
    ])
    bundle = DummyBundle(
        X_sonic=X,
        artist_keys=np.array(["a", "b", "c", "d"]),
        track_ids=np.array(["a0", "b0", "c0", "d0"]),
        track_artists=np.array(["A", "B", "C", "D"]),
        track_titles=np.array(["A", "B", "C", "D"]),
    )
    captured = {}
    real_beam = pbb._beam_search_segment

    def _spy(*args, **kwargs):
        captured["roam"] = kwargs.get("roam_detour_sonic")
        return real_beam(*args, **kwargs)

    monkeypatch.setattr(pbb, "_beam_search_segment", _spy)

    def _run(enabled):
        captured.clear()
        return pbb.build_pier_bridge_playlist(
            seed_track_ids=["a0", "b0"],
            total_tracks=3,
            bundle=bundle,
            candidate_pool_indices=[2, 3],
            cfg=pbb.PierBridgeConfig(
                transition_floor=0.0,
                bridge_floor=0.0,
                weight_bridge=1.0,
                weight_transition=0.0,
                eta_destination_pull=0.0,
                roam_corridors_enabled=enabled,
                roam_width_sonic=0.0,
                roam_penalty_slope=5.0,
            ),
            allowed_track_ids_set={"a0", "b0", "c0", "d0"},
        )

    res_on = _run(enabled=True)
    assert res_on.success
    assert captured["roam"] is not None          # detour wired through when enabled
    res_off = _run(enabled=False)
    assert res_off.success
    assert captured["roam"] is None              # not computed when disabled


def test_one_artist_per_segment_collapses_feat_with_variants():
    from src.playlist.pier_bridge_builder import _build_segment_candidate_pool_scored

    X = np.array([
        [1.0, 0.0],   # pier A
        [0.0, 1.0],   # pier B
        [0.7, 0.7],   # Mount Eerie (candidate 1)
        [0.7, 0.7],   # Mount Eerie with ... (candidate 2)
    ])
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    bundle = DummyBundle(
        X_sonic=X,
        artist_keys=np.array(["a", "b", "c", "d"]),
        track_ids=np.array(["a0", "b0", "c0", "d0"]),
        track_artists=np.array([
            "Pier A",
            "Pier B",
            "Mount Eerie",
            "Mount Eerie with Julie Doiron & Fred Squire",
        ]),
        track_titles=np.array(["A", "B", "Moon", "Swan"]),
    )
    diag = {}
    pool, _artist_keys, _title_keys = _build_segment_candidate_pool_scored(
        pier_a=0,
        pier_b=1,
        X_full_norm=X_norm,
        universe_indices=[2, 3],
        used_track_ids=set(),
        bundle=bundle,
        bridge_floor=0.0,
        segment_pool_max=50,
        diagnostics=diag,
    )
    assert len(pool) == 1
    assert diag.get("collapsed_by_artist_key") == 1


def test_seed_track_key_collision_excludes_duplicate_song():
    # Prevent "same song twice" when a pier seed and a candidate share (artist_key,title_key)
    # but have different track_ids (Yo La Tengo "Autumn Sweater" style).
    from src.playlist.pier_bridge_builder import PierBridgeConfig, build_pier_bridge_playlist

    X = np.array([
        [1.0, 0.0],                   # seed 1
        [0.0, 1.0],                   # seed 2
        [np.sqrt(0.5), np.sqrt(0.5)], # duplicate of seed 1 by track_key
        [0.6, 0.8],                   # valid bridge candidate
    ])
    bundle = DummyBundle(
        X_sonic=X,
        artist_keys=np.array(["yo la tengo", "other", "yo la tengo", "x"]),
        track_ids=np.array(["seed_autumn", "seed_other", "dup_autumn", "x_song"]),
        track_artists=np.array(["Yo La Tengo", "Other", "Yo La Tengo", "X"]),
        track_titles=np.array(["Autumn Sweater", "Other Seed", "Autumn Sweater", "Bridge"]),
    )
    result = build_pier_bridge_playlist(
        seed_track_ids=["seed_autumn", "seed_other"],
        total_tracks=3,
        bundle=bundle,
        candidate_pool_indices=[2, 3],
        cfg=PierBridgeConfig(
            transition_floor=0.0,
            bridge_floor=0.0,
            weight_bridge=1.0,
            weight_transition=0.0,
            eta_destination_pull=0.0,
        ),
        allowed_track_ids_set={"seed_autumn", "seed_other", "dup_autumn", "x_song"},
    )
    assert result.success
    assert "dup_autumn" not in result.track_ids
    assert result.track_ids[1] == "x_song"


def test_one_per_artist_cap_applies_across_segments_but_exempts_seeds():
    from src.playlist.pier_bridge_builder import PierBridgeConfig, build_pier_bridge_playlist

    X = np.array([
        [1.0, 0.0],                   # seed A, artist seed-a
        [0.0, 1.0],                   # seed B, artist seed-b
        [-1.0, 0.0],                  # seed C, artist repeat
        [np.sqrt(0.5), np.sqrt(0.5)], # repeat artist, segment A->B winner
        [-0.6, 0.8],                  # repeat artist, segment B->C would win without global cap
        [-0.4, 0.9],                  # other artist, valid fallback for B->C
    ])
    bundle = DummyBundle(
        X_sonic=X,
        artist_keys=np.array(["seed-a", "seed-b", "repeat", "repeat", "repeat", "other"]),
        track_ids=np.array(["seed_a", "seed_b", "seed_c", "repeat_1", "repeat_2", "other_1"]),
        track_artists=np.array(["Seed A", "Seed B", "Repeat", "Repeat", "Repeat", "Other"]),
        track_titles=np.array(["A", "B", "C", "Repeat 1", "Repeat 2", "Other 1"]),
    )

    result = build_pier_bridge_playlist(
        seed_track_ids=["seed_a", "seed_b", "seed_c"],
        total_tracks=5,
        bundle=bundle,
        candidate_pool_indices=[3, 4, 5],
        cfg=PierBridgeConfig(
            transition_floor=0.0,
            bridge_floor=0.0,
            weight_bridge=1.0,
            weight_transition=0.0,
            eta_destination_pull=0.0,
            max_non_seed_tracks_per_artist=1,
        ),
        allowed_track_ids_set={"seed_a", "seed_b", "seed_c", "repeat_1", "repeat_2", "other_1"},
    )

    assert result.success
    assert result.track_ids == ["seed_a", "repeat_1", "seed_b", "other_1", "seed_c"]
    assert result.stats["artist_counts"]["repeat"] == 2
    assert result.stats["non_seed_artist_counts"]["repeat"] == 1


def test_segment_pool_is_endpoint_local_not_global_seed_max():
    # A candidate can be identical to some other seed C, but should still be excluded
    # from a segment A->B if it fails the bridge gate vs (A,B).
    from src.playlist.pier_bridge_builder import _build_segment_candidate_pool_scored

    X = np.array([
        [1.0, 0.0],   # pier A
        [0.0, 1.0],   # pier B
        [-1.0, 0.0],  # other seed C (not used by this segment)
        [-1.0, 0.0],  # candidate identical to C
    ])
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    bundle = DummyBundle(
        X_sonic=X,
        artist_keys=np.array(["a", "b", "c", "d"]),
        track_ids=np.array(["a0", "b0", "c0", "d0"]),
        track_artists=np.array(["A", "B", "C", "Cand"]),
        track_titles=np.array(["A", "B", "C", "Cand"]),
    )
    pool, _artist_keys, _title_keys = _build_segment_candidate_pool_scored(
        pier_a=0,
        pier_b=1,
        X_full_norm=X_norm,
        universe_indices=[3],
        used_track_ids=set(),
        bundle=bundle,
        bridge_floor=0.10,
        segment_pool_max=50,
    )
    assert pool == []


def test_progress_monotonicity_in_beam_search_segment():
    from src.playlist.pier_bridge_builder import PierBridgeConfig, _beam_search_segment

    X = np.array([
        [1.0, 0.0],  # pier A
        [0.0, 1.0],  # pier B
        [0.9, 0.4],  # early-ish
        [0.7, 0.7],  # mid
        [0.4, 0.9],  # late-ish
    ])
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    candidates = [2, 3, 4]
    cfg = PierBridgeConfig(
        transition_floor=0.0,
        bridge_floor=0.0,
        weight_bridge=0.0,
        weight_transition=0.0,
        eta_destination_pull=0.0,
        progress_enabled=True,
        progress_monotonic_epsilon=0.0,
        progress_penalty_weight=1.0,
    )
    path, _hits, _edges, err = _beam_search_segment(
        0,
        1,
        2,
        candidates,
        X_norm,  # X_full (transition space)
        X_norm,  # X_full_norm (sonic space)
        None,
        None,
        None,
        None,
        cfg,
        20,
    )
    assert err is None
    assert path is not None and len(path) == 2

    vec_a = X_norm[0]
    vec_b = X_norm[1]
    d = vec_b - vec_a
    denom = float(np.dot(d, d))
    assert denom > 1e-9

    ts = []
    for idx in path:
        t_raw = float(np.dot((X_norm[int(idx)] - vec_a), d) / denom)
        ts.append(max(0.0, min(1.0, t_raw)))
    assert ts[0] <= ts[1] + 1e-9


def test_transition_floor_no_longer_hard_gates_low_t():
    # Roam-only design (2026-06-25 centered-transition fix, design §5): the
    # ``transition_floor`` HARD GATE is removed. A low-T edge that the legacy floor
    # would have rejected (cosine ~0.20 < the old 0.35 floor) is now ACCEPTED in the
    # legacy beam path — roam's worst-edge minimax, not a hard floor, is the quality
    # mechanism (and removing the gate is strictly fewer cascade triggers → budget-safe).
    # ``transition_floor`` is retained only as an inert param for cascade callers.
    from src.playlist.pier_bridge_builder import PierBridgeConfig, build_pier_bridge_playlist
    from src.playlist.run_audit import InfeasibleHandlingConfig

    X = np.array([
        [1.0, 0.0],         # pier A
        [0.0, 1.0],         # pier B
        [0.2, 0.98],        # candidate: A->cand cosine ~0.20, below the old 0.35 floor
    ])
    bundle = DummyBundle(
        X_sonic=X,
        artist_keys=np.array(["a", "b", "c"]),
        track_ids=np.array(["a0", "b0", "c0"]),
        track_titles=np.array(["A", "B", "C"]),
    )
    result = build_pier_bridge_playlist(
        seed_track_ids=["a0", "b0"],
        total_tracks=3,
        bundle=bundle,
        candidate_pool_indices=[2],
        cfg=PierBridgeConfig(transition_floor=0.35, bridge_floor=0.0, eta_destination_pull=0.0),
        allowed_track_ids_set={"a0", "b0", "c0"},
        infeasible_handling=InfeasibleHandlingConfig(guarantee_feasible=False),
    )
    # The old hard gate would have failed this build (no edge ≥ 0.35); it no longer gates.
    assert result.success
    assert "c0" in result.track_ids


def test_soft_genre_penalty_changes_ranking_without_gating():
    # Candidate "low" is slightly better on transitions, but has very low genre sim (< threshold),
    # so with a soft penalty enabled it should be outranked by "high".
    from src.playlist.pier_bridge_builder import PierBridgeConfig, build_pier_bridge_playlist

    X = np.array([
        [1.0, 0.0],   # pier A
        [0.0, 1.0],   # pier B
        [0.71, 0.71],   # cand low-genre: slightly better transitions overall
        [0.65, 0.76],   # cand high-genre: slightly worse transitions, better genre
    ])
    X_genre = np.array([
        [1.0, 0.0],                              # A
        [1.0, 0.0],                              # B
        [0.10, float(np.sqrt(1 - 0.10**2))],      # low genre sim to A
        [0.30, float(np.sqrt(1 - 0.30**2))],      # high genre sim to A
    ])
    bundle = DummyBundle(
        X_sonic=X,
        artist_keys=np.array(["a", "b", "low", "high"]),
        track_ids=np.array(["a0", "b0", "c_low", "c_high"]),
        track_titles=np.array(["A", "B", "Low", "High"]),
    )
    base_cfg = PierBridgeConfig(
        transition_floor=0.0,
        bridge_floor=0.0,
        eta_destination_pull=0.0,
        weight_bridge=0.0,
        weight_transition=1.0,
        genre_tiebreak_weight=0.0,
        genre_penalty_threshold=0.20,
        genre_penalty_strength=0.0,
    )
    base = build_pier_bridge_playlist(
        seed_track_ids=["a0", "b0"],
        total_tracks=3,
        bundle=bundle,
        candidate_pool_indices=[2, 3],
        cfg=base_cfg,
        X_genre_smoothed=X_genre,
        allowed_track_ids_set={"a0", "b0", "c_low", "c_high"},
    )
    assert base.success
    assert base.track_ids[1] == "c_low"

    penalized = build_pier_bridge_playlist(
        seed_track_ids=["a0", "b0"],
        total_tracks=3,
        bundle=bundle,
        candidate_pool_indices=[2, 3],
        cfg=PierBridgeConfig(**{**base_cfg.__dict__, "genre_penalty_strength": 0.10}),
        X_genre_smoothed=X_genre,
        allowed_track_ids_set={"a0", "b0", "c_low", "c_high"},
    )
    assert penalized.success
    assert penalized.track_ids[1] == "c_high"


def test_soft_genre_penalty_per_mode_resolution():
    """Per-mode soft_genre_penalty keys override the legacy flat defaults."""
    from src.playlist.config import resolve_pier_bridge_tuning

    overrides = {
        "pier_bridge": {
            "soft_genre_penalty_threshold_strict": 0.82,
            "soft_genre_penalty_threshold_narrow": 0.78,
            "soft_genre_penalty_threshold_dynamic": 0.55,
            "soft_genre_penalty_strength_strict": 0.40,
            "soft_genre_penalty_strength_narrow": 0.30,
            "soft_genre_penalty_strength_dynamic": 0.15,
        }
    }

    for mode, expected_threshold, expected_strength in [
        ("strict", 0.82, 0.40),
        ("narrow", 0.78, 0.30),
        ("dynamic", 0.55, 0.15),
    ]:
        tuning, sources = resolve_pier_bridge_tuning(
            mode=mode, similarity_floor=0.20, overrides=overrides
        )
        assert tuning.genre_penalty_threshold == expected_threshold, (
            f"mode={mode}: expected threshold {expected_threshold}, "
            f"got {tuning.genre_penalty_threshold}"
        )
        assert tuning.genre_penalty_strength == expected_strength, (
            f"mode={mode}: expected strength {expected_strength}, "
            f"got {tuning.genre_penalty_strength}"
        )
        assert sources["genre_penalty_threshold"] == (
            f"pier_bridge.soft_genre_penalty_threshold_{mode}"
        )
        assert sources["genre_penalty_strength"] == (
            f"pier_bridge.soft_genre_penalty_strength_{mode}"
        )

    # Modes without per-mode keys fall back to the flat default (0.20 / 0.10)
    tuning_off, sources_off = resolve_pier_bridge_tuning(
        mode="off", similarity_floor=0.20, overrides=overrides
    )
    assert tuning_off.genre_penalty_threshold == 0.20
    assert tuning_off.genre_penalty_strength == 0.10
    assert sources_off["genre_penalty_threshold"] == "default"
    assert sources_off["genre_penalty_strength"] == "default"


def test_local_sonic_edge_penalty_changes_ranking_without_gating():
    from src.playlist.pier_bridge_builder import PierBridgeConfig, _beam_search_segment

    X_transition = np.array([
        [1.0, 0.0],   # pier A
        [1.0, 0.0],   # pier B
        [1.0, 0.0],   # low-sonic candidate: best transition score
        [0.9, 0.4],   # coherent candidate: slightly weaker transition score
    ])
    X_transition = X_transition / (np.linalg.norm(X_transition, axis=1, keepdims=True) + 1e-12)
    X_sonic = np.array([
        [1.0, 0.0],   # pier A
        [1.0, 0.0],   # pier B
        [-1.0, 0.0],  # low-sonic candidate: locally opposed to both piers
        [1.0, 0.0],   # coherent candidate
    ])
    X_sonic = X_sonic / (np.linalg.norm(X_sonic, axis=1, keepdims=True) + 1e-12)
    base_cfg = PierBridgeConfig(
        transition_floor=-1.0,
        bridge_floor=-1.0,
        weight_bridge=0.0,
        weight_transition=1.0,
        eta_destination_pull=0.0,
        progress_enabled=False,
    )

    base_path, _hits, _edges, err = _beam_search_segment(
        0,
        1,
        1,
        [2, 3],
        X_transition,
        X_sonic,
        None,
        None,
        None,
        None,
        base_cfg,
        10,
    )
    assert err is None
    assert base_path == [2]

    local_stats = {}
    penalized_path, _hits, _edges, err = _beam_search_segment(
        0,
        1,
        1,
        [2, 3],
        X_transition,
        X_sonic,
        None,
        None,
        None,
        None,
        PierBridgeConfig(
            **{
                **base_cfg.__dict__,
                "local_sonic_edge_penalty_enabled": True,
                "local_sonic_edge_penalty_threshold": 0.5,
                "local_sonic_edge_penalty_strength": 1.0,
            }
        ),
        10,
        local_sonic_stats=local_stats,
    )
    assert err is None
    assert penalized_path == [3]
    assert local_stats["local_sonic_penalty_hits"] >= 2
    assert local_stats["local_sonic_gate_rejected"] == 0


def test_local_sonic_edge_floor_applies_to_final_destination_edge():
    from src.playlist.pier_bridge_builder import PierBridgeConfig, _beam_search_segment

    X_transition = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
    ])
    X_sonic = np.array([
        [1.0, 0.0],   # pier A
        [-1.0, 0.0],  # pier B
        [1.0, 0.0],   # candidate: good after A, bad into B
    ])
    local_stats = {}

    path, _hits, _edges, err = _beam_search_segment(
        0,
        1,
        1,
        [2],
        X_transition,
        X_sonic,
        None,
        None,
        None,
        None,
        PierBridgeConfig(
            transition_floor=-1.0,
            bridge_floor=-1.0,
            weight_bridge=0.0,
            weight_transition=1.0,
            eta_destination_pull=0.0,
            progress_enabled=False,
            local_sonic_edge_floor=0.0,
        ),
        10,
        local_sonic_stats=local_stats,
    )
    assert path is None
    assert err == "no valid final connection to destination"
    assert local_stats["local_sonic_gate_rejected"] == 1


# ---------------------------------------------------------------------------
# Task 5: energy-slot wiring in cluster_artist_tracks
# ---------------------------------------------------------------------------

def _two_cluster_bundle():
    artist_keys = np.array(["a"] * 6)
    track_ids = np.array([str(i) for i in range(6)])
    X = np.array([
        [1.0, 0.0], [0.9, 0.1], [0.95, -0.05],   # cluster 1 (indices 0,1,2)
        [0.0, 1.0], [0.1, 0.9], [-0.05, 0.95],   # cluster 2 (indices 3,4,5)
    ])
    return DummyBundle(X_sonic=X, artist_keys=artist_keys, track_ids=track_ids)


def test_cluster_artist_tracks_energy_weight_zero_matches_none():
    bundle = _two_cluster_bundle()
    cfg_off = ArtistStyleConfig(cluster_k_min=2, cluster_k_max=2, enabled=True)
    cfg_zero = ArtistStyleConfig(
        cluster_k_min=2, cluster_k_max=2, enabled=True, medoid_energy_weight=0.0
    )
    energy = np.array([2.0, -2.0, 0.0, 2.0, -2.0, 0.0])
    base = cluster_artist_tracks(bundle=bundle, artist_name="A", cfg=cfg_off, random_seed=0)
    zeroed = cluster_artist_tracks(
        bundle=bundle, artist_name="A", cfg=cfg_zero, random_seed=0, energy_values=energy
    )
    assert sorted(base[1]) == sorted(zeroed[1])   # identical medoids


def test_cluster_artist_tracks_energy_runs_and_returns_medoids():
    bundle = _two_cluster_bundle()
    cfg = ArtistStyleConfig(
        cluster_k_min=2, cluster_k_max=2, enabled=True, medoid_energy_weight=5.0
    )
    energy = np.array([2.0, -2.0, 0.0, 2.0, -2.0, 0.0])
    clusters, medoids, by_cluster, X_norm = cluster_artist_tracks(
        bundle=bundle, artist_name="A", cfg=cfg, random_seed=0, energy_values=energy
    )
    assert len(clusters) == 2
    assert len(medoids) == 2


def test_cluster_artist_tracks_inert_on_flat_energy():
    bundle = _two_cluster_bundle()
    cfg = ArtistStyleConfig(
        cluster_k_min=2, cluster_k_max=2, enabled=True, medoid_energy_weight=5.0
    )
    flat = np.zeros(6)   # zero span => energy term inert, must not crash
    base = cluster_artist_tracks(bundle=bundle, artist_name="A", cfg=cfg, random_seed=0)
    flatted = cluster_artist_tracks(
        bundle=bundle, artist_name="A", cfg=cfg, random_seed=0, energy_values=flat
    )
    assert sorted(base[1]) == sorted(flatted[1])


def test_transition_floor_config_override_per_mode():
    from src.playlist.config import default_ds_config

    cfg = default_ds_config(
        "dynamic",
        playlist_len=30,
        overrides={"constraints": {"transition_floor_dynamic": 0.55}},
    )
    assert cfg.construct.transition_floor == 0.55

    cfg2 = default_ds_config(
        "narrow",
        playlist_len=30,
        overrides={"constraints": {"transition_floor_narrow": 0.66}},
    )
    assert cfg2.construct.transition_floor == 0.66


def test_pier_bridge_tuning_defaults_per_mode():
    from src.playlist.config import resolve_pier_bridge_tuning

    # Phase 1-3A relaxed bridge_floor: dynamic 0.03 -> 0.02, narrow 0.08 -> 0.05
    # (commit 34f2948).
    dyn, _ = resolve_pier_bridge_tuning(mode="dynamic", similarity_floor=0.2, overrides={})
    assert dyn.transition_floor == 0.35
    assert dyn.bridge_floor == 0.02
    assert dyn.weight_bridge == 0.6
    assert dyn.weight_transition == 0.4
    assert dyn.genre_tiebreak_weight == 0.05
    assert dyn.genre_penalty_threshold == 0.20
    assert dyn.genre_penalty_strength == 0.10

    nar, _ = resolve_pier_bridge_tuning(mode="narrow", similarity_floor=0.2, overrides={})
    assert nar.transition_floor == 0.45
    assert nar.bridge_floor == 0.05
    assert nar.weight_bridge == 0.7
    assert nar.weight_transition == 0.3
    assert nar.genre_tiebreak_weight == 0.05
    assert nar.genre_penalty_threshold == 0.20
    assert nar.genre_penalty_strength == 0.10


def test_pier_bridge_tuning_overrides_and_penalty_clamp():
    from src.playlist.config import resolve_pier_bridge_tuning

    tuning, sources = resolve_pier_bridge_tuning(
        mode="narrow",
        similarity_floor=0.2,
        overrides={
            "pier_bridge": {
                "bridge_floor_narrow": 0.12,
                "soft_genre_penalty_strength": 1.5,  # should clamp to 1.0
            }
        },
    )
    assert tuning.bridge_floor == 0.12
    assert sources["bridge_floor"] == "pier_bridge.bridge_floor_narrow"
    assert tuning.genre_penalty_strength == 1.0
    assert sources["genre_penalty_strength"] == "pier_bridge.soft_genre_penalty_strength"


def test_build_ds_overrides_includes_pier_bridge():
    from src.playlist_generator import build_ds_overrides

    ds_cfg = {
        "constraints": {"transition_floor_dynamic": 0.55},
        "pier_bridge": {"bridge_floor_dynamic": 0.04},
    }
    overrides = build_ds_overrides(ds_cfg)
    assert overrides["constraints"]["transition_floor_dynamic"] == 0.55
    assert overrides["pier_bridge"]["bridge_floor_dynamic"] == 0.04


def test_high_bridge_floor_fails_segment():
    # guarantee_feasible=False restores the legacy failure path so the high bridge_floor gate is exercised.
    from src.playlist.pier_bridge_builder import build_pier_bridge_playlist, PierBridgeConfig
    from src.playlist.run_audit import InfeasibleHandlingConfig
    X = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.5, 0.5],
    ])
    bundle = DummyBundle(
        X_sonic=X,
        artist_keys=np.array(["a", "b", "c"]),
        track_ids=np.array(["a0", "b0", "c0"]),
        track_titles=np.array(["A", "B", "C"]),
    )
    result = build_pier_bridge_playlist(
        seed_track_ids=["a0", "b0"],
        total_tracks=2,
        bundle=bundle,
        candidate_pool_indices=[2],
        cfg=PierBridgeConfig(bridge_floor=0.9, transition_floor=0.8),
        allowed_track_ids_set={"a0", "b0", "c0"},
        infeasible_handling=InfeasibleHandlingConfig(guarantee_feasible=False),
    )
    assert not result.success
    assert result.failure_reason


def test_bridge_floor_backoff_disabled_infeasible_segment_fails():
    # Initial bridge_floor is too strict for the only candidate; without backoff,
    # segment remains infeasible (default behavior).
    # guarantee_feasible=False restores the legacy failure path so the bridge_floor gate is exercised.
    from src.playlist.pier_bridge_builder import build_pier_bridge_playlist, PierBridgeConfig
    from src.playlist.run_audit import InfeasibleHandlingConfig

    x = 0.04  # min(simA, simB)
    X = np.array([
        [1.0, 0.0],                       # pier A
        [0.0, 1.0],                       # pier B
        [x, float(np.sqrt(1 - x**2))],    # candidate: very low sim to A, high to B
    ])
    bundle = DummyBundle(
        X_sonic=X,
        artist_keys=np.array(["a", "b", "c"]),
        track_ids=np.array(["a0", "b0", "c0"]),
        track_titles=np.array(["A", "B", "C"]),
    )
    result = build_pier_bridge_playlist(
        seed_track_ids=["a0", "b0"],
        total_tracks=3,
        bundle=bundle,
        candidate_pool_indices=[2],
        cfg=PierBridgeConfig(bridge_floor=0.08, transition_floor=0.0, eta_destination_pull=0.0),
        allowed_track_ids_set={"a0", "b0", "c0"},
        infeasible_handling=InfeasibleHandlingConfig(guarantee_feasible=False),
    )
    assert not result.success
    assert result.failure_reason and "bridge_floor=0.08" in result.failure_reason


def test_bridge_floor_backoff_enabled_succeeds_and_records_attempts():
    from src.playlist.pier_bridge_builder import build_pier_bridge_playlist, PierBridgeConfig
    from src.playlist.run_audit import InfeasibleHandlingConfig, RunAuditConfig

    x = 0.04
    X = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [x, float(np.sqrt(1 - x**2))],
    ])
    bundle = DummyBundle(
        X_sonic=X,
        artist_keys=np.array(["a", "b", "c"]),
        track_ids=np.array(["a0", "b0", "c0"]),
        track_titles=np.array(["A", "B", "C"]),
    )
    events = []
    result = build_pier_bridge_playlist(
        seed_track_ids=["a0", "b0"],
        total_tracks=3,
        bundle=bundle,
        candidate_pool_indices=[2],
        cfg=PierBridgeConfig(bridge_floor=0.08, transition_floor=0.0, eta_destination_pull=0.0),
        allowed_track_ids_set={"a0", "b0", "c0"},
        infeasible_handling=InfeasibleHandlingConfig(
            enabled=True,
            backoff_steps=(0.05, 0.03),
            min_bridge_floor=0.0,
            max_attempts_per_segment=8,
        ),
        audit_config=RunAuditConfig(enabled=True, include_top_k=5),
        audit_events=events,
    )
    assert result.success
    assert result.segment_diagnostics
    assert result.segment_diagnostics[0].bridge_floor_used < 0.08
    assert result.segment_diagnostics[0].bridge_floor_used == pytest.approx(0.03)
    attempt_nos = [
        int(ev.payload.get("attempt_number"))
        for ev in events
        if ev.kind == "segment_attempt" and int(ev.payload.get("segment_index", -1)) == 0
    ]
    assert attempt_nos == [1, 2, 3]


def test_run_audit_writer_creates_markdown_report(tmp_path):
    from src.playlist.pier_bridge_builder import build_pier_bridge_playlist, PierBridgeConfig
    from src.playlist.run_audit import (
        InfeasibleHandlingConfig,
        RunAuditConfig,
        RunAuditContext,
        RunAuditEvent,
        now_utc_iso,
        write_markdown_report,
    )

    x = 0.04
    X = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [x, float(np.sqrt(1 - x**2))],
    ])
    bundle = DummyBundle(
        X_sonic=X,
        artist_keys=np.array(["a", "b", "c"]),
        track_ids=np.array(["a0", "b0", "c0"]),
        track_artists=np.array(["A", "B", "C"]),
        track_titles=np.array(["A", "B", "C"]),
    )
    events = [
        RunAuditEvent(
            kind="preflight",
            ts_utc=now_utc_iso(),
            payload={
                "tuning": {"mode": "narrow", "bridge_floor": 0.08, "transition_floor": 0.0},
                "tuning_sources": {},
                "pool_summary": {"allowed_ids_count": 3},
            },
        )
    ]
    result = build_pier_bridge_playlist(
        seed_track_ids=["a0", "b0"],
        total_tracks=3,
        bundle=bundle,
        candidate_pool_indices=[2],
        cfg=PierBridgeConfig(bridge_floor=0.08, transition_floor=0.0, eta_destination_pull=0.0),
        allowed_track_ids_set={"a0", "b0", "c0"},
        infeasible_handling=InfeasibleHandlingConfig(enabled=True, backoff_steps=(0.03,)),
        audit_config=RunAuditConfig(enabled=True, include_top_k=5),
        audit_events=events,
    )
    assert result.success
    events.append(
        RunAuditEvent(
            kind="final_success",
            ts_utc=now_utc_iso(),
            payload={
                "playlist_tracks": [
                    {"track_id": tid, "artist": "", "title": ""} for tid in result.track_ids
                ],
                "weakest_edges": [],
                "summary_stats": {"final_playlist_size": len(result.track_ids)},
            },
        )
    )
    ctx = RunAuditContext(
        timestamp_utc=now_utc_iso(),
        run_id="test_run",
        cohesion_mode="narrow",
        seed_track_id="a0",
        seed_artist="A",
        dry_run=True,
        artifact_path="dummy",
        sonic_variant="raw",
        allowed_ids_count=3,
        pool_source="test",
        artist_style_enabled=False,
        extra={},
    )
    out_path = write_markdown_report(
        context=ctx,
        events=events,
        path=tmp_path / "audit.md",
        max_bytes=200000,
    )
    text = out_path.read_text(encoding="utf-8")
    assert "## 2) Effective Tuning" in text
    assert "## 4) Segment Diagnostics" in text
    assert "### Segment 0" in text
    assert "#### Attempt 1" in text


def test_finite_median_ignores_nan():
    assert _finite_median(np.array([1.0, np.nan, 3.0])) == 2.0
    assert np.isnan(_finite_median(np.array([np.nan, np.nan])))


def test_robust_energy_span_uses_percentiles():
    vals = np.arange(10, dtype=float)  # 0..9
    span = _robust_energy_span(vals, 10.0, 90.0)
    assert span is not None
    lo, hi = span
    assert lo == pytest.approx(0.9)
    assert hi == pytest.approx(8.1)


def test_robust_energy_span_none_when_flat_or_sparse():
    assert _robust_energy_span(np.array([5.0, 5.0, 5.0]), 10.0, 90.0) is None  # zero span
    assert _robust_energy_span(np.array([np.nan, 1.0]), 10.0, 90.0) is None     # <2 finite


def test_slot_targets_quantile_uniform_is_evenly_spaced():
    # Uniform distribution: quantiles map ~linearly to values, so quantile-spacing
    # reduces to even value-spacing. medians order: cluster0 high, cluster1 low, cluster2 mid.
    energy = np.linspace(0.0, 10.0, 101)
    targets = _slot_targets_by_quantile([2.0, 0.0, 1.0], energy, 10.0, 90.0)
    assert targets[1] == pytest.approx(1.0, abs=0.2)    # lowest cluster -> p10 -> ~1.0
    assert targets[2] == pytest.approx(5.0, abs=0.2)    # mid -> p50 -> ~5.0
    assert targets[0] == pytest.approx(9.0, abs=0.2)    # highest cluster -> p90 -> ~9.0


def test_slot_targets_quantile_follows_density_not_range():
    # 8 high-energy tracks, 2 low: evenly-spaced quantiles land mostly in the dense
    # high mass, NOT evenly across the 0..5 value range (the representativeness fix).
    energy = np.array([0.0, 0.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
    # 4 clusters ranked low->high by median energy
    targets = _slot_targets_by_quantile([0.0, 2.5, 5.0, 5.0], energy, 10.0, 90.0)
    assert targets[0] < 1.0                              # only the lowest reaches the sparse low end
    assert all(t == pytest.approx(5.0) for t in targets[1:])  # the rest follow the dense mass
    # (even value-spacing would have put a target at ~1.67 in the sparse middle)


def test_slot_targets_quantile_single_cluster_is_mid_quantile():
    energy = np.linspace(0.0, 10.0, 101)
    assert _slot_targets_by_quantile([3.0], energy, 10.0, 90.0) == [pytest.approx(5.0, abs=0.2)]


def test_slot_targets_quantile_nan_median_stays_nan():
    energy = np.linspace(0.0, 10.0, 101)
    targets = _slot_targets_by_quantile([np.nan, 1.0], energy, 10.0, 90.0)
    assert np.isnan(targets[0])


def test_slot_targets_quantile_sparse_energy_is_inert():
    targets = _slot_targets_by_quantile([0.0, 5.0], np.array([1.0]), 10.0, 90.0)
    assert all(np.isnan(t) for t in targets)   # <2 finite values -> all NaN (inert)


def test_slot_proximity_peaks_at_target_and_zeros_for_nan():
    z = np.array([5.0, 0.0, 10.0, np.nan])
    prox = _slot_proximity(z, target=5.0, span_width=10.0)
    assert prox[0] == pytest.approx(1.0)     # at target
    assert prox[1] == pytest.approx(0.5)     # half a span away
    assert prox[2] == pytest.approx(0.5)
    assert prox[3] == 0.0                     # NaN energy -> neutral (no bonus)


def test_slot_proximity_inert_when_target_nan():
    z = np.array([1.0, 2.0])
    assert np.all(_slot_proximity(z, target=np.nan, span_width=10.0) == 0.0)
    # span_width <= 0 is also inert (all zeros)
    assert np.all(_slot_proximity(z, target=5.0, span_width=0.0) == 0.0)


def test_dedupe_artist_indices_prefers_studio_and_collapses_dupes():
    # 0 studio "On a Plain" | 1 live version | 2 duplicate studio (lowercase)
    # | 3 unique song | 4 a song that exists ONLY as a live cut
    titles = np.array([
        "On a Plain",
        "On A Plain (Live In Tokyo)",
        "On a plain",
        "About a Girl",
        "Scoff (Live at Pine Street)",
    ])
    durations = np.array([200000, 210000, 200000, 150000, 180000], dtype=float)
    kept = _dedupe_artist_indices([0, 1, 2, 3, 4], titles, durations)
    assert 1 not in kept            # live "On a Plain" demoted (studio present)
    assert 2 not in kept            # duplicate studio collapsed
    assert 0 in kept                # one canonical studio "On a Plain" survives
    assert 3 in kept                # unique song kept
    assert 4 in kept                # sole-version live cut kept
    assert kept == sorted(kept)


def test_dedupe_artist_indices_no_titles_passthrough():
    # No titles -> can't group -> every index passes through unchanged.
    assert _dedupe_artist_indices([0, 1, 2], None, None) == [0, 1, 2]


def _centroid_for(X, indices):
    c = X[indices].mean(axis=0)
    return c / (np.linalg.norm(c) + 1e-12)


def test_medoid_energy_term_pulls_to_slot():
    # 3 candidates, near-identical sonic centrality; energy proximity favors index 1.
    X = np.array([[1.0, 0.0], [0.98, 0.02], [0.99, 0.01]])
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    indices = [0, 1, 2]
    centroid = _centroid_for(X, indices)
    rng = np.random.default_rng(0)

    # Baseline (no energy): pick by sonic alone, top_k=1 => deterministic argmax.
    base = _medoids_for_cluster(
        X, indices, centroid, ["t0", "t1", "t2"], 1, rng, 1,
        None, None, 0.7, 0.3,
    )
    # Energy strongly favors index 1.
    rng2 = np.random.default_rng(0)
    energized = _medoids_for_cluster(
        X, indices, centroid, ["t0", "t1", "t2"], 1, rng2, 1,
        None, None, 0.7, 0.3,
        10.0, np.array([0.0, 1.0, 0.0]),   # energy_weight, energy_proximity
    )
    assert energized == [1]


def test_medoid_energy_weight_zero_is_regression_safe():
    X = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    indices = [0, 1, 2]
    centroid = _centroid_for(X, indices)
    base = _medoids_for_cluster(
        X, indices, centroid, ["t0", "t1", "t2"], 1, np.random.default_rng(3), 1,
        None, None, 0.7, 0.3,
    )
    with_zero = _medoids_for_cluster(
        X, indices, centroid, ["t0", "t1", "t2"], 1, np.random.default_rng(3), 1,
        None, None, 0.7, 0.3,
        0.0, np.array([1.0, 0.0, 0.0]),   # weight 0 => proximity ignored
    )
    assert with_zero == base


def _write_energy_sidecar(tmp_path, track_ids, arousal):
    energy_dir = Path(tmp_path) / "energy"
    energy_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        energy_dir / "energy_sidecar.npz",
        track_ids=np.array(track_ids, dtype=object),
        arousal_p50=np.array(arousal, dtype=np.float32),
    )


def test_load_artist_energy_values_returns_zscored():
    import tempfile
    import shutil
    tmp_path = tempfile.mkdtemp()
    try:
        track_ids = ["a", "b", "c"]
        _write_energy_sidecar(tmp_path, track_ids, [1.0, 3.0, 5.0])
        bundle = types.SimpleNamespace(
            track_ids=np.array(track_ids), artifact_path=Path(tmp_path) / "artifact.npz"
        )
        cfg = ArtistStyleConfig(medoid_energy_weight=1.0, energy_feature="arousal_p50")
        vals = load_artist_energy_values(bundle, cfg)
        assert vals is not None and vals.shape == (3,)
        assert vals[0] < vals[1] < vals[2]            # preserves ordering
        assert abs(float(np.mean(vals))) < 1e-6        # z-scored => ~zero mean
    finally:
        shutil.rmtree(tmp_path)


def test_load_artist_energy_values_inert_when_weight_zero():
    import tempfile
    import shutil
    tmp_path = tempfile.mkdtemp()
    try:
        bundle = types.SimpleNamespace(
            track_ids=np.array(["a"]), artifact_path=Path(tmp_path) / "artifact.npz"
        )
        assert load_artist_energy_values(bundle, ArtistStyleConfig()) is None
    finally:
        shutil.rmtree(tmp_path)


def test_load_artist_energy_values_warns_when_sidecar_missing(caplog):
    import tempfile
    import shutil
    import logging
    tmp_path = tempfile.mkdtemp()
    try:
        bundle = types.SimpleNamespace(
            track_ids=np.array(["a"]), artifact_path=Path(tmp_path) / "artifact.npz"
        )
        cfg = ArtistStyleConfig(medoid_energy_weight=0.5)
        with caplog.at_level(logging.WARNING):
            assert load_artist_energy_values(bundle, cfg) is None
        assert any("energy sidecar missing" in r.message for r in caplog.records)
    finally:
        shutil.rmtree(tmp_path)


def test_load_artist_energy_values_warns_when_no_finite(caplog):
    # Sidecar exists but its track_ids do NOT overlap the bundle's track_ids,
    # so load_energy_matrix returns an all-NaN column. This is the production
    # "configured-knob-must-act" path: an artist whose tracks aren't in the
    # sidecar. Must return None AND log a "no finite" WARNING.
    import tempfile
    import shutil
    import logging
    tmp_path = tempfile.mkdtemp()
    try:
        _write_energy_sidecar(tmp_path, ["x", "y", "z"], [1.0, 3.0, 5.0])
        bundle = types.SimpleNamespace(
            track_ids=np.array(["a", "b", "c"]),  # disjoint from sidecar ids
            artifact_path=Path(tmp_path) / "artifact.npz",
        )
        cfg = ArtistStyleConfig(medoid_energy_weight=0.5, energy_feature="arousal_p50")
        with caplog.at_level(logging.WARNING):
            assert load_artist_energy_values(bundle, cfg) is None
        assert any("no finite" in r.message for r in caplog.records)
    finally:
        shutil.rmtree(tmp_path)


def test_medoid_popularity_term_breaks_tie_toward_popular():
    X = np.array([[1.0, 0.0], [0.98, 0.02], [0.99, 0.01]])
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    indices = [0, 1, 2]; centroid = _centroid_for(X, indices)
    base = _medoids_for_cluster(X, indices, centroid, ["t0","t1","t2"], 1,
        np.random.default_rng(0), 1, None, None, 0.7, 0.3)
    pop = _medoids_for_cluster(X, indices, centroid, ["t0","t1","t2"], 1,
        np.random.default_rng(0), 1, None, None, 0.7, 0.3, 0.0, None,
        5.0, np.array([0.0, 1.0, 0.0]))   # popularity_weight, popularity_values
    assert pop == [1]                       # strong popularity on index 1 wins the pick
    del base                                # baseline computed only to mirror the call shape


def test_medoid_popularity_weight_zero_is_regression_safe():
    X = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    indices = [0, 1, 2]; centroid = _centroid_for(X, indices)
    base = _medoids_for_cluster(X, indices, centroid, ["t0","t1","t2"], 1,
        np.random.default_rng(3), 1, None, None, 0.7, 0.3)
    z = _medoids_for_cluster(X, indices, centroid, ["t0","t1","t2"], 1,
        np.random.default_rng(3), 1, None, None, 0.7, 0.3, 0.0, None,
        0.0, np.array([1.0, 0.0, 0.0]))   # popularity weight 0 -> ignored
    assert z == base


def _unit_rows(rows):
    X = np.array(rows, dtype=float)
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def test_bridgeability_kth_rank_same_artist_exclusion():
    # 5-dim basis geometry. Artist 'a' rows: 0 (e1), 1 (e2), 2 (e3), 9 (e2 twin of 1).
    # Library rows: 3,4 (e1 — two neighbors for member 0), 5 (e3 — ONE neighbor for
    # member 2), 6,7 (e4), 8 (e5) filler.
    e = np.eye(5)
    X = _unit_rows([e[0], e[1], e[2],          # members 0,1,2 (artist 'a')
                    e[0], e[0], e[2],          # lib 3,4,5
                    e[3], e[3], e[4],          # lib 6,7,8
                    e[1]])                     # 9 = same-artist twin of member 1
    # muq calibration band (what resolve_transition_calib(None) returns)
    from src.playlist.transition_metrics import resolve_transition_calib
    c, s, g = resolve_transition_calib(None)

    t = compute_pier_bridgeability(X, [0, 1, 2], [0, 1, 2, 9], k=2,
                                   calib_center=c, calib_scale=s, calib_gain=g)
    assert t.shape == (3,)
    assert t[0] > 0.9        # member 0: two e1 library neighbors -> kth cos = 1.0
    assert t[1] < 0.05       # member 1: only close row is same-artist twin 9 -> excluded
    assert t[2] < 0.05       # member 2: ONE library neighbor but k=2 -> kth cos = 0.0

    # Same geometry, k=1, excluding ONLY self (callers must always exclude the
    # member's own row — production passes all artist rows): the same-artist twin
    # at row 9 now counts and rescues member 1 -> proves the artist mask (not the
    # geometry) drove the failure above.
    t_selfonly = compute_pier_bridgeability(X, [1], [1], k=1,
                                            calib_center=c, calib_scale=s, calib_gain=g)
    assert t_selfonly[0] > 0.9


def test_bridgeability_empty_members_and_k_clamp():
    from src.playlist.transition_metrics import resolve_transition_calib
    c, s, g = resolve_transition_calib(None)
    e = np.eye(3)
    X = _unit_rows([e[0], e[0], e[1]])
    assert compute_pier_bridgeability(X, [], [0], k=10,
                                      calib_center=c, calib_scale=s, calib_gain=g).shape == (0,)
    # k larger than available non-excluded columns clamps to what's there (2 columns).
    t = compute_pier_bridgeability(X, [0], [0], k=10,
                                   calib_center=c, calib_scale=s, calib_gain=g)
    # kth clamps to 2nd best = cos(e0, e1) = 0 -> low T, but finite (no crash/inf)
    assert np.isfinite(t[0]) and t[0] < 0.05


# ---------------------------------------------------------------------------
# Task 2: bridgeability veto + slot reallocation wired into cluster_artist_tracks
# ---------------------------------------------------------------------------

def _bridgeability_fixture():
    """Artist 'a' with 2 sonic clusters; only cluster A has library mass near it.

    Rows 0-2: artist cluster A (~e1, tiny jitter so kmeans is stable).
    Rows 3-5: artist cluster B (~e2) — isolated, no library neighbors.
    Rows 6-11: library tracks near e1 (cluster A's bridges).
    Rows 12-13: library filler on e3.
    """
    e = np.eye(4)
    jit = [0.00, 0.01, -0.01]
    a_rows = [e[0] + j * e[3] for j in jit] + [e[1] + j * e[3] for j in jit]
    lib_rows = [e[0] + j * e[3] for j in (0.02, -0.02, 0.03, -0.03, 0.04, -0.04)]
    fill = [e[2], e[2] + 0.01 * e[3]]
    X = _unit_rows(a_rows + lib_rows + fill)
    artist_keys = np.array(["a"] * 6 + ["lib"] * 8)
    track_ids = np.array([f"t{i}" for i in range(14)])
    return DummyBundle(X_sonic=X, artist_keys=artist_keys, track_ids=track_ids)


def _cfg_bridge(**kw):
    base = dict(cluster_k_min=2, cluster_k_max=2, piers_per_cluster=1, enabled=True,
                dedupe_versions=False, pier_bridgeability_k=3,
                pier_bridgeability_floor_t=0.30)
    base.update(kw)
    return ArtistStyleConfig(**base)


def test_outlier_cluster_contributes_no_piers_and_slots_reallocate():
    bundle = _bridgeability_fixture()
    clusters, medoids, by_cluster, X_norm = cluster_artist_tracks(
        bundle=bundle, artist_name="A", cfg=_cfg_bridge(), random_seed=0,
        medoid_top_k=2, target_pier_count=4,
    )
    assert len(clusters) == 2                      # cluster membership untouched
    assert sorted(len(m) for m in by_cluster) == [0, 3]   # B empty; A bumped to all 3
    assert all(m in {0, 1, 2} for m in medoids)    # every pier from cluster A
    assert len(medoids) == 3                       # ceil(4/1)=4 capped by cluster size


def test_bridgeability_all_fail_falls_back_unchecked():
    # Shrink the library to filler only: no artist track has 3 neighbors anywhere.
    e = np.eye(4)
    X = _unit_rows([e[0], e[0] + 0.01 * e[3], e[1], e[1] + 0.01 * e[3], e[2], e[3]])
    bundle = DummyBundle(X_sonic=X, artist_keys=np.array(["a"] * 4 + ["lib"] * 2),
                         track_ids=np.array([f"t{i}" for i in range(6)]))
    checked = cluster_artist_tracks(
        bundle=bundle, artist_name="A", cfg=_cfg_bridge(), random_seed=0, medoid_top_k=1)
    unchecked = cluster_artist_tracks(
        bundle=bundle, artist_name="A",
        cfg=_cfg_bridge(pier_bridgeability_enabled=False), random_seed=0, medoid_top_k=1)
    assert checked[1] == unchecked[1]              # same medoids: never-fail fallback


def test_bridgeability_no_veto_is_byte_identical():
    bundle = _bridgeability_fixture()
    # Floor 0.0 => nothing vetoed => identical to disabled.
    on = cluster_artist_tracks(
        bundle=bundle, artist_name="A",
        cfg=_cfg_bridge(pier_bridgeability_floor_t=0.0), random_seed=0,
        medoid_top_k=2, target_pier_count=4)
    off = cluster_artist_tracks(
        bundle=bundle, artist_name="A",
        cfg=_cfg_bridge(pier_bridgeability_enabled=False), random_seed=0,
        medoid_top_k=2, target_pier_count=4)
    assert on[1] == off[1] and on[2] == off[2]


def test_artist_style_config_has_bridgeability_defaults():
    cfg = ArtistStyleConfig()
    assert cfg.pier_bridgeability_enabled is True   # live default (activate-fixes rule)
    assert cfg.pier_bridgeability_floor_t == 0.30
    assert cfg.pier_bridgeability_k == 10


def test_bridgeability_config_keys_parse():
    """Guards the exact key names playlist_generator.py parses (both sites)."""
    raw = {"pier_bridgeability_enabled": False,
           "pier_bridgeability_floor_t": 0.42,
           "pier_bridgeability_k": 7}
    cfg = ArtistStyleConfig(
        pier_bridgeability_enabled=bool(raw.get("pier_bridgeability_enabled", True)),
        pier_bridgeability_floor_t=float(raw.get("pier_bridgeability_floor_t", 0.30)),
        pier_bridgeability_k=int(raw.get("pier_bridgeability_k", 10)),
    )
    assert (cfg.pier_bridgeability_enabled, cfg.pier_bridgeability_floor_t,
            cfg.pier_bridgeability_k) == (False, 0.42, 7)
