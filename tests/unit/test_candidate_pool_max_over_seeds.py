"""Genre filter and overlap guard must use max-over-seeds, not primary-seed-only."""
import numpy as np
from src.playlist.candidate_pool import build_candidate_pool
from src.playlist.config import CandidatePoolConfig


def _make_cfg(**overrides):
    base = dict(
        similarity_floor=-1.0,
        min_sonic_similarity=None,
        max_pool_size=10,
        target_artists=5,
        candidates_per_artist=5,
        seed_artist_bonus=0,
        max_artist_fraction_final=1.0,
        title_hard_exclude_flags=frozenset(),
    )
    base.update(overrides)
    return CandidatePoolConfig(**base)


def test_genre_filter_admits_candidate_aligned_to_secondary_seed():
    # Seed 0: genre 0. Seed 1: genre 2. Candidate b: genre 2 only.
    # Without fix, candidate b is rejected (no match to seed 0).
    # With fix (max over seeds), candidate b is admitted via seed 1.
    X_genre = np.array([
        [1.0, 0.0, 0.0],  # seed 0
        [0.0, 0.0, 1.0],  # seed 1
        [1.0, 0.0, 0.0],  # candidate a: matches seed 0
        [0.0, 0.0, 1.0],  # candidate b: matches seed 1 only
    ], dtype=float)
    emb = np.array([[1.0, 0.0]] * 4, dtype=float)
    artist_keys = np.array(["seed_a", "seed_b", "cand_a", "cand_b"])
    cfg = _make_cfg()

    result = build_candidate_pool(
        seed_idx=0,
        seed_indices=[1],
        embedding=emb,
        artist_keys=artist_keys,
        cfg=cfg,
        random_seed=0,
        X_genre_raw=X_genre,
        X_genre_smoothed=X_genre,
        min_genre_similarity=0.5,
        genre_method="cosine",
        genre_vocab=["electronic", "rock", "indie"],
        mode="dynamic",
    )

    admitted = {artist_keys[int(i)] for i in result.pool_indices}
    assert "cand_a" in admitted
    assert "cand_b" in admitted, "candidate aligned to secondary seed must be admitted"


def test_overlap_guard_admits_candidate_with_secondary_seed_tag_overlap():
    # Overlap guard in narrow mode: requires >=1 shared raw tag with the seed.
    # Seed 0: genre 0. Seed 1: genre 2. Candidate: genre 2 only.
    # Without fix, overlap guard rejects (no overlap with seed 0).
    # With fix, overlap guard sees overlap with seed 1.
    X_genre = np.array([
        [1.0, 0.0, 0.0],  # seed 0
        [0.0, 0.0, 1.0],  # seed 1
        [0.0, 0.0, 1.0],  # candidate: overlaps with seed 1
        [0.0, 1.0, 0.0],  # unrelated
    ], dtype=float)
    emb = np.array([[1.0, 0.0]] * 4, dtype=float)
    sonic = np.array([[1.0, 0.0]] * 4, dtype=float)
    artist_keys = np.array(["seed_a", "seed_b", "cand_overlap_b", "unrelated"])
    cfg = _make_cfg()

    result = build_candidate_pool(
        seed_idx=0,
        seed_indices=[1],
        embedding=emb,
        artist_keys=artist_keys,
        cfg=cfg,
        random_seed=0,
        X_sonic=sonic,
        X_genre_raw=X_genre,
        X_genre_smoothed=X_genre,
        min_genre_similarity=0.4,
        genre_method="ensemble",
        genre_vocab=["electronic", "rock", "indie"],
        broad_filters=("electronic",),
        mode="narrow",
    )

    admitted = {artist_keys[int(i)] for i in result.pool_indices}
    assert "cand_overlap_b" in admitted, "candidate overlapping secondary seed must pass overlap guard"
