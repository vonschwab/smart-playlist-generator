import numpy as np

from src.playlist.replacement import ReplacementContext, find_replacement_candidates


def _replacement_ctx() -> ReplacementContext:
    n = 24
    dim = 32
    x = np.zeros((n, dim), dtype=float)
    for idx in range(n):
        x[idx, idx % dim] = 1.0
        x[idx, (idx * 3) % dim] += 0.5
    genres = np.eye(n, 12)
    return ReplacementContext(
        X_sonic=x,
        X_full=x,
        X_start=x,
        X_end=x,
        X_mid=x,
        X_genre_smoothed=genres,
        perceptual_bpm=np.linspace(70.0, 170.0, n),
        tempo_stability=np.ones(n),
        track_ids=np.array([f"track-{idx}" for idx in range(n)], dtype=object),
        artist_keys=np.array([f"artist-{idx}" for idx in range(n)], dtype=object),
        candidate_pool_indices=np.arange(n),
        tower_pca_dims=(8, 16, 8),
        transition_floor=0.0,
    )


def test_replacement_flow_smoke_for_each_mode():
    ctx = _replacement_ctx()
    playlist = [0, 4, 8, 12, 16]

    for mode in ("best", "different_pace", "different_genre", "different_sound"):
        candidates = find_replacement_candidates(
            prev_idx=4,
            next_idx=12,
            current_idx=8,
            playlist_indices=playlist,
            ctx=ctx,
            mode=mode,
            top_k=10,
        )

        assert len(candidates) <= 10
        returned_indices = {candidate["index"] for candidate in candidates}
        assert returned_indices.isdisjoint(playlist)
        assert all(candidate["artist_key"] not in {"artist-4", "artist-12"} for candidate in candidates)
        assert all("t_prev" in candidate and "t_next" in candidate for candidate in candidates)


def test_different_modes_include_axis_specific_fields_for_gui():
    ctx = _replacement_ctx()
    candidates = find_replacement_candidates(
        prev_idx=4,
        next_idx=12,
        current_idx=8,
        playlist_indices=[4, 8, 12],
        ctx=ctx,
        mode="different_pace",
        top_k=3,
    )

    assert candidates
    assert all("track_id" in candidate for candidate in candidates)
    assert all("perceptual_bpm" in candidate for candidate in candidates)
