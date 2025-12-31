import numpy as np
import pytest

from src.playlist.candidate_pool import build_candidate_pool
from src.playlist.config import CandidatePoolConfig
from src.genre_similarity_v2 import GenreSimilarityV2


def test_narrow_sonic_floor_rejects_negative():
    # Seed + two candidates; floor=0.10 rejects 0.09 but admits 0.10+
    embedding = np.array([
        [1.0, 0.0],  # seed
        [0.09, np.sqrt(1 - 0.09**2)],  # 0.09 cosine to seed
        [0.10, np.sqrt(1 - 0.10**2)],  # 0.10 cosine to seed
    ])
    X_sonic = embedding.copy()
    artist_keys = np.array(["seed", "low", "pass"])
    track_ids = np.array(["seed", "low_id", "pass_id"])
    cfg = CandidatePoolConfig(
        similarity_floor=-1.0,
        min_sonic_similarity=0.10,
        max_pool_size=10,
        target_artists=3,
        candidates_per_artist=5,
        seed_artist_bonus=0,
        max_artist_fraction_final=1.0,
        broad_filters=(),
    )

    result = build_candidate_pool(
        seed_idx=0,
        embedding=embedding,
        artist_keys=artist_keys,
        track_ids=track_ids,
        cfg=cfg,
        random_seed=0,
        X_sonic=X_sonic,
        min_genre_similarity=None,
        genre_method="ensemble",
        genre_vocab=[],
        broad_filters=(),
        mode="narrow",
    )

    assert result.pool_indices.tolist() == [2]
    assert result.stats["below_sonic_similarity"] == 1
    assert result.stats["seed_sonic_sim_track_ids"]["pass_id"] == result.sonic_sim[0]


def test_dynamic_sonic_floor_rejects_negative():
    embedding = np.array([
        [1.0, 0.0],
        [-0.01, np.sqrt(1 - 0.01**2)],  # -0.01 cosine
        [0.0, 1.0],  # 0.0 cosine
    ])
    X_sonic = embedding.copy()
    artist_keys = np.array(["seed", "neg", "zero"])
    cfg = CandidatePoolConfig(
        similarity_floor=-1.0,
        min_sonic_similarity=0.0,
        max_pool_size=10,
        target_artists=3,
        candidates_per_artist=5,
        seed_artist_bonus=0,
        max_artist_fraction_final=1.0,
        broad_filters=(),
    )
    result = build_candidate_pool(
        seed_idx=0,
        embedding=embedding,
        artist_keys=artist_keys,
        cfg=cfg,
        random_seed=0,
        X_sonic=X_sonic,
        min_genre_similarity=None,
        genre_method="ensemble",
        genre_vocab=[],
        broad_filters=(),
        mode="dynamic",
    )
    assert "neg" not in [artist_keys[i] for i in result.pool_indices.tolist()]
    assert "zero" in [artist_keys[i] for i in result.pool_indices.tolist()]


def test_dynamic_floor_config_override():
    # Floor override to 0.20 should reject 0.19 and admit 0.20
    embedding = np.array([
        [1.0, 0.0],
        [0.19, np.sqrt(1 - 0.19**2)],
        [0.20, np.sqrt(1 - 0.20**2)],
    ])
    X_sonic = embedding.copy()
    artist_keys = np.array(["seed", "low", "high"])
    cfg_override = CandidatePoolConfig(
        similarity_floor=-1.0,
        min_sonic_similarity=0.20,
        max_pool_size=10,
        target_artists=3,
        candidates_per_artist=5,
        seed_artist_bonus=0,
        max_artist_fraction_final=1.0,
        broad_filters=(),
    )
    result = build_candidate_pool(
        seed_idx=0,
        embedding=embedding,
        artist_keys=artist_keys,
        cfg=cfg_override,
        random_seed=0,
        X_sonic=X_sonic,
        min_genre_similarity=None,
        genre_method="ensemble",
        genre_vocab=[],
        broad_filters=(),
        mode="narrow",
    )
    assert "low" not in [artist_keys[i] for i in result.pool_indices.tolist()]
    assert "high" in [artist_keys[i] for i in result.pool_indices.tolist()]


def test_multi_seed_max_similarity_used_for_floor():
    embedding = np.array([
        [1.0, 0.0],   # seed A
        [0.0, 1.0],   # seed B
        [0.0, 1.0],   # candidate similar only to seed B
        [-1.0, 0.0],  # candidate below floor to both
    ])
    X_sonic = embedding.copy()
    artist_keys = np.array(["seed_a", "seed_b", "cand_pass", "cand_drop"])
    cfg = CandidatePoolConfig(
        similarity_floor=-1.0,
        min_sonic_similarity=0.5,
        max_pool_size=10,
        target_artists=4,
        candidates_per_artist=5,
        seed_artist_bonus=0,
        max_artist_fraction_final=1.0,
        broad_filters=(),
    )
    result = build_candidate_pool(
        seed_idx=0,
        seed_indices=[0, 1],
        embedding=embedding,
        artist_keys=artist_keys,
        cfg=cfg,
        random_seed=0,
        X_sonic=X_sonic,
        min_genre_similarity=None,
        genre_method="ensemble",
        genre_vocab=[],
        broad_filters=(),
        mode="narrow",
    )
    artists = [artist_keys[i] for i in result.pool_indices.tolist()]
    assert "cand_pass" in artists
    assert "cand_drop" not in artists


def test_broad_genre_mask_prevents_inflation():
    # Seed genres: rock, folk; candidate: indie rock
    # Broad filter removes "rock", leaving no overlap => genre sim zero => rejected by gate
    embedding = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
    ])
    X_sonic = embedding.copy()
    artist_keys = np.array(["a", "b"])

    genre_vocab = ["rock", "folk", "indie rock"]
    X_genre_raw = np.array([
        [1, 1, 0],  # seed
        [1, 0, 1],  # candidate
    ], dtype=float)

    cfg = CandidatePoolConfig(
        similarity_floor=-1.0,
        min_sonic_similarity=None,
        max_pool_size=10,
        target_artists=2,
        candidates_per_artist=5,
        seed_artist_bonus=0,
        max_artist_fraction_final=1.0,
        broad_filters=("rock",),
    )

    result = build_candidate_pool(
        seed_idx=0,
        embedding=embedding,
        artist_keys=artist_keys,
        cfg=cfg,
        random_seed=0,
        X_sonic=X_sonic,
        X_genre_raw=X_genre_raw,
        X_genre_smoothed=None,
        min_genre_similarity=0.3,
        genre_method="weighted_jaccard",
        genre_vocab=genre_vocab,
        broad_filters=cfg.broad_filters,
        mode="narrow",
    )

    assert result.pool_indices.tolist() == []  # candidate should be rejected by genre gate


def test_genre_explainer_returns_filtered_pairs():
    calc = GenreSimilarityV2()
    score, details = calc.calculate_similarity_with_explain(
        ["rock", "dream pop"],
        ["indie rock", "shoegaze"],
        broad_filters=["rock"],
        top_k=3,
    )
    assert "rock" in details["seed_broad_removed"]
    assert "rock" not in [g.lower() for g in details["seed_genres_filtered"]]
    assert isinstance(score, float)
    assert details["top_pairs"]  # should have at least one contributing pair   


def test_allowed_set_applies_excluded_track_ids_in_pipeline(tmp_path):
    """
    Regression test: style-aware runs pass both `allowed_track_ids` and `excluded_track_ids`.
    We must apply the exclusions even when the bundle is clamped to the allowed set.
    """
    from src.playlist.pipeline import generate_playlist_ds
    from src.playlist.pier_bridge_builder import PierBridgeConfig

    # Minimal 3-track artifact:
    # - t0 = seed pier A
    # - t1 = anchor pier B
    # - t2 = only possible interior track (but excluded)
    track_ids = np.array(["t0", "t1", "t2"])
    artist_keys = np.array(["a0", "a1", "a2"])
    track_artists = np.array(["Artist 0", "Artist 1", "Artist 2"])
    track_titles = np.array(["Song 0", "Song 1", "Song 2"])

    X_sonic = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )
    # Give genre some variance to avoid PCA "total_var=0" warnings.
    X_genre_raw = np.array([[0.0], [1.0], [0.5]], dtype=float)
    X_genre_smoothed = X_genre_raw.copy()
    genre_vocab = np.array(["g0"])

    artifact_path = tmp_path / "tiny_artifact.npz"
    np.savez(
        artifact_path,
        track_ids=track_ids,
        artist_keys=artist_keys,
        track_artists=track_artists,
        track_titles=track_titles,
        X_sonic=X_sonic,
        X_genre_raw=X_genre_raw,
        X_genre_smoothed=X_genre_smoothed,
        genre_vocab=genre_vocab,
    )

    result = generate_playlist_ds(
        artifact_path=str(artifact_path),
        seed_track_id="t0",
        anchor_seed_ids=["t1"],
        num_tracks=3,
        mode="narrow",
        random_seed=0,
        allowed_track_ids=["t0", "t1", "t2"],
        excluded_track_ids={"t2"},
        pier_bridge_config=PierBridgeConfig(
            transition_floor=0.0,
            bridge_floor=0.0,
            progress_enabled=False,
        ),
        sonic_weight=1.0,
        genre_weight=0.0,
        min_genre_similarity=None,
        genre_method=None,
    )

    # Excluded track_id must not appear in the result.
    assert "t2" not in result.track_ids


def test_excluded_set_does_not_remove_piers_in_pipeline(tmp_path):
    """
    Piers (seed + anchor seeds) must be exempt from excluded_track_ids so DS can
    still place them as fixed endpoints.
    """
    from src.playlist.pipeline import generate_playlist_ds
    from src.playlist.pier_bridge_builder import PierBridgeConfig

    track_ids = np.array(["t0", "t1", "t2"])
    artist_keys = np.array(["a0", "a1", "a2"])
    track_artists = np.array(["Artist 0", "Artist 1", "Artist 2"])
    track_titles = np.array(["Song 0", "Song 1", "Song 2"])

    X_sonic = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ],
        dtype=float,
    )
    X_genre_raw = np.array([[0.0], [1.0], [0.5]], dtype=float)
    X_genre_smoothed = X_genre_raw.copy()
    genre_vocab = np.array(["g0"])

    artifact_path = tmp_path / "tiny_artifact_pier_exempt.npz"
    np.savez(
        artifact_path,
        track_ids=track_ids,
        artist_keys=artist_keys,
        track_artists=track_artists,
        track_titles=track_titles,
        X_sonic=X_sonic,
        X_genre_raw=X_genre_raw,
        X_genre_smoothed=X_genre_smoothed,
        genre_vocab=genre_vocab,
    )

    result = generate_playlist_ds(
        artifact_path=str(artifact_path),
        seed_track_id="t0",
        anchor_seed_ids=["t1"],
        num_tracks=3,
        mode="narrow",
        random_seed=0,
        allowed_track_ids=["t0", "t1", "t2"],
        # Exclude everything including piers; piers must still be present.
        excluded_track_ids={"t0", "t1", "t2"},
        pier_bridge_config=PierBridgeConfig(
            transition_floor=0.0,
            bridge_floor=0.0,
            progress_enabled=False,
        ),
        sonic_weight=1.0,
        genre_weight=0.0,
        min_genre_similarity=None,
        genre_method=None,
    )
    assert "t0" in result.track_ids
    assert "t1" in result.track_ids
    assert "t2" not in result.track_ids


def test_recency_filters_require_candidate_pool_stage():
    from src.playlist.filtering import filter_by_recently_played, filter_by_scrobbles

    with pytest.raises(ValueError, match="Recency filter must not run after ordering"):
        filter_by_scrobbles(
            tracks=[],
            scrobbles=[],
            lookback_days=14,
            stage="post_order_validation",
        )

    with pytest.raises(ValueError, match="Recency filter must not run after ordering"):
        filter_by_recently_played(
            tracks=[],
            play_history=[],
            lookback_days=14,
            stage="post_order_validation",
        )


def test_post_order_validation_fails_loudly_on_recency_overlap():
    import logging
    from src.playlist_generator import PlaylistGenerator

    class DummyLibrary:
        similarity_calc = object()

        def get_all_tracks(self, library_id=None):
            return []

    class DummyConfig:
        recently_played_filter_enabled = True
        recently_played_lookback_days = 14
        recently_played_min_playcount = 0

        def get(self, section, key=None, default=None):
            return default

    gen = PlaylistGenerator(DummyLibrary(), DummyConfig())
    with pytest.raises(ValueError, match="post_order_validation_failed"):
        gen._post_order_validate_ds_output(
            ordered_tracks=[{"rating_key": "bad_id", "artist": "A", "title": "T"}],
            expected_length=1,
            excluded_track_ids={"bad_id"},
            exempt_pier_track_ids=set(),
            audit_path="docs/run_audits/test.md",
        )


def test_playlist_generator_does_not_post_filter_ds_results(monkeypatch, caplog):
    import logging
    from src.playlist_generator import PlaylistGenerator

    class DummyLibrary:
        similarity_calc = object()

        def __init__(self, tracks):
            self._tracks = list(tracks)

        def get_all_tracks(self, library_id=None):
            return list(self._tracks)

    class DummyConfig:
        recently_played_filter_enabled = True
        recently_played_lookback_days = 14
        recently_played_min_playcount = 0

        config = {"playlists": {}}

        def get(self, section, key=None, default=None):
            if key is None:
                return self.config.get(section, default)
            return self.config.get(section, {}).get(key, default)

    artist = "Test Artist"
    artist_key = "test artist"
    library_tracks = [
        {"rating_key": f"s{i}", "artist": artist, "artist_key": artist_key, "title": f"Seed {i}"}
        for i in range(4)
    ]

    gen = PlaylistGenerator(DummyLibrary(library_tracks), DummyConfig(), lastfm_client=None)

    # If any post-order recency filtering is attempted, fail the test.
    monkeypatch.setattr(
        gen,
        "filter_tracks",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("post-order filter_tracks called")),
    )
    monkeypatch.setattr(
        gen,
        "_filter_by_scrobbles",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("post-order _filter_by_scrobbles called")),
    )
    monkeypatch.setattr(gen, "_print_playlist_report", lambda *args, **kwargs: None)
    monkeypatch.setattr(gen, "_compute_edge_scores_from_artifact", lambda *args, **kwargs: [])

    def _stub_ds(*args, **kwargs):
        gen._last_ds_report = {
            "metrics": {"strategy": "pier_bridge"},
            "playlist_stats": {"playlist": {}},
        }
        return [
            {"rating_key": "p0", "artist": artist, "title": "Pier 0"},
            {"rating_key": "x1", "artist": "Other", "title": "Bridge 1"},
            {"rating_key": "p2", "artist": artist, "title": "Pier 2"},
        ]

    monkeypatch.setattr(gen, "_maybe_generate_ds_playlist", _stub_ds)

    caplog.set_level(logging.INFO)
    result = gen.create_playlist_for_artist(artist, track_count=3)
    assert len(result["tracks"]) == 3
    assert any("stage=post_order_validation" in rec.getMessage() for rec in caplog.records)
