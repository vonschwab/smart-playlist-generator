import numpy as np
import pytest

from src.playlist.artist_style import (
    ArtistStyleConfig,
    build_balanced_candidate_pool,
    build_genre_neighbor_candidate_pool,
    cluster_artist_tracks,
    order_clusters,
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


def test_dynamic_transition_floor_blocks_low_t():
    # With dynamic transition_floor=0.35, a segment with no valid transitions should fail.
    # guarantee_feasible=False restores the legacy failure path so this gate is still exercised.
    from src.playlist.pier_bridge_builder import PierBridgeConfig, build_pier_bridge_playlist
    from src.playlist.run_audit import InfeasibleHandlingConfig

    X = np.array([
        [1.0, 0.0],         # pier A
        [0.0, 1.0],         # pier B
        [0.2, 0.98],        # candidate: A->cand cosine ~0.20 < 0.35
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
    assert not result.success


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
