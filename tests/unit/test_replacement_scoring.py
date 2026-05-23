import numpy as np
import pytest

from src.playlist.replacement import (
    ReplacementContext,
    SUPPORTED_MODES,
    find_replacement_candidates,
)


def _ctx(N=20, dim=32):
    """Synthetic context with predictable embeddings."""
    rng = np.random.default_rng(42)
    X_sonic = rng.standard_normal((N, dim))
    X_genre = np.eye(N, 8)
    perceptual_bpm = np.linspace(60.0, 180.0, N)
    return ReplacementContext(
        X_sonic=X_sonic,
        X_full=X_sonic,
        X_start=X_sonic,
        X_end=X_sonic,
        X_mid=X_sonic,
        X_genre_smoothed=X_genre,
        perceptual_bpm=perceptual_bpm,
        tempo_stability=np.ones(N),
        track_ids=np.array([f"t{i}" for i in range(N)], dtype=object),
        artist_keys=np.array([f"a{i // 2}" for i in range(N)], dtype=object),
        candidate_pool_indices=np.arange(N),
        tower_pca_dims=(8, 16, 8),
        idf_weights=None,
    )


def test_supported_modes_are_four():
    assert SUPPORTED_MODES == ("best", "different_pace", "different_genre", "different_sound")


def test_best_mode_returns_top_k():
    ctx = _ctx()
    result = find_replacement_candidates(
        prev_idx=2,
        next_idx=10,
        current_idx=5,
        playlist_indices=[0, 2, 5, 10, 15],
        ctx=ctx,
        mode="best",
        top_k=5,
    )
    assert len(result) <= 5
    for candidate in result:
        assert "track_id" in candidate
        assert "t_prev" in candidate
        assert "t_next" in candidate


def test_excludes_current_track():
    ctx = _ctx()
    result = find_replacement_candidates(
        prev_idx=2,
        next_idx=10,
        current_idx=5,
        playlist_indices=[0, 2, 5, 10, 15],
        ctx=ctx,
        mode="best",
        top_k=20,
    )
    assert all(candidate["index"] != 5 for candidate in result)


def test_excludes_playlist_tracks():
    ctx = _ctx()
    playlist = [0, 2, 5, 10, 15]
    result = find_replacement_candidates(
        prev_idx=2,
        next_idx=10,
        current_idx=5,
        playlist_indices=playlist,
        ctx=ctx,
        mode="best",
        top_k=20,
    )
    returned_idx = {candidate["index"] for candidate in result}
    assert returned_idx.isdisjoint(set(playlist))


def test_excludes_neighbor_artist():
    ctx = _ctx()
    result = find_replacement_candidates(
        prev_idx=2,
        next_idx=4,
        current_idx=3,
        playlist_indices=[0, 2, 3, 4, 8],
        ctx=ctx,
        mode="best",
        top_k=20,
    )
    returned_idx = {candidate["index"] for candidate in result}
    assert 3 not in returned_idx
    assert 5 not in returned_idx


def test_different_pace_diverges_from_current_bpm():
    ctx = _ctx()
    current_idx = 10
    best = find_replacement_candidates(
        prev_idx=2,
        next_idx=18,
        current_idx=current_idx,
        playlist_indices=[2, current_idx, 18],
        ctx=ctx,
        mode="best",
        top_k=5,
    )
    pace = find_replacement_candidates(
        prev_idx=2,
        next_idx=18,
        current_idx=current_idx,
        playlist_indices=[2, current_idx, 18],
        ctx=ctx,
        mode="different_pace",
        top_k=5,
    )
    current_bpm = ctx.perceptual_bpm[current_idx]
    best_avg_div = np.mean([abs(np.log2(candidate["perceptual_bpm"] / current_bpm)) for candidate in best])
    pace_avg_div = np.mean([abs(np.log2(candidate["perceptual_bpm"] / current_bpm)) for candidate in pace])
    assert pace_avg_div > best_avg_div


def test_unknown_mode_raises():
    ctx = _ctx()
    with pytest.raises(ValueError, match="Unknown replacement mode"):
        find_replacement_candidates(
            prev_idx=0,
            next_idx=5,
            current_idx=2,
            playlist_indices=[0, 2, 5],
            ctx=ctx,
            mode="bogus",
            top_k=5,
        )


def test_empty_pool_returns_empty():
    ctx = _ctx()
    ctx_empty = ReplacementContext(
        **{**ctx.__dict__, "candidate_pool_indices": np.array([], dtype=int)},
    )
    result = find_replacement_candidates(
        prev_idx=0,
        next_idx=5,
        current_idx=2,
        playlist_indices=[0, 2, 5],
        ctx=ctx_empty,
        mode="best",
        top_k=10,
    )
    assert result == []
