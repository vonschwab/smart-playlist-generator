"""IDF-weighted admission genre similarity ranks rare-tag matches higher."""
import numpy as np
from src.playlist.candidate_pool import _compute_genre_similarity, build_candidate_pool
from src.playlist.config import CandidatePoolConfig


def _make_cfg(*, idf_enabled: bool):
    return CandidatePoolConfig(
        similarity_floor=-1.0,
        min_sonic_similarity=None,
        max_pool_size=10,
        target_artists=5,
        candidates_per_artist=5,
        seed_artist_bonus=0,
        max_artist_fraction_final=1.0,
        title_hard_exclude_flags=frozenset(),
        genre_idf_enabled=idf_enabled,
    )


def test_idf_disabled_produces_equal_scores_for_equal_cosine():
    seed = np.array([1.0, 1.0, 0.0, 0.0])
    a = np.array([1.0, 0.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0, 0.0])
    candidates = np.stack([a, b])
    sim = _compute_genre_similarity(seed, candidates, method="cosine", idf_weights=None)
    assert sim[0] == sim[1]


def test_idf_enabled_ranks_rare_tag_match_higher():
    seed = np.array([1.0, 1.0, 0.0, 0.0])
    a = np.array([1.0, 0.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0, 0.0])
    candidates = np.stack([a, b])
    idf = np.array([0.2, 1.0, 0.5, 0.5])  # tag 1 is rare
    sim = _compute_genre_similarity(seed, candidates, method="cosine", idf_weights=idf)
    assert sim[1] > sim[0]


def test_idf_zero_weights_produce_zero_similarity():
    seed = np.array([1.0, 1.0, 0.0, 0.0])
    cand = np.array([1.0, 1.0, 0.0, 0.0])
    candidates = np.stack([cand])
    idf = np.zeros(4, dtype=float)
    sim = _compute_genre_similarity(seed, candidates, method="cosine", idf_weights=idf)
    assert sim[0] == 0.0


def test_idf_applied_in_build_candidate_pool_when_enabled():
    # 4 tracks, 3 genres. Genre 1 is rare (appears in only 1 of 4 tracks).
    # Seed: genres 0 and 1. Candidate a: genre 0 only (common match).
    # Candidate b: genre 1 only (rare match). With IDF, b should rank ahead.
    X_genre = np.array([
        [1.0, 1.0, 0.0],  # seed
        [1.0, 0.0, 1.0],  # noisy track (makes genre 0/2 common, genre 1 rare)
        [1.0, 0.0, 0.0],  # candidate a: common match on genre 0
        [0.0, 1.0, 0.0],  # candidate b: rare match on genre 1
    ], dtype=float)
    emb = np.array([[1.0, 0.0]] * 4, dtype=float)
    artist_keys = np.array(["seed", "noise", "cand_common", "cand_rare"])
    cfg = _make_cfg(idf_enabled=True)

    result = build_candidate_pool(
        seed_idx=0,
        embedding=emb,
        artist_keys=artist_keys,
        cfg=cfg,
        random_seed=0,
        X_genre_raw=X_genre,
        X_genre_smoothed=X_genre,
        min_genre_similarity=0.3,
        genre_method="cosine",
        genre_vocab=["common", "rare", "filler"],
        mode="dynamic",
    )

    pool_artists = [artist_keys[int(i)] for i in result.pool_indices]
    assert "cand_rare" in pool_artists
    assert pool_artists.index("cand_rare") < pool_artists.index("cand_common")
