"""The live builder must thread rhythm_matrix into the beam's soft pace penalty.

Regression test: `_beam_search_segment` computes `pace_sim_for_penalty` when
`rhythm_matrix is not None`, but `pier_bridge_builder.py` historically never
extracted rhythm_matrix unless `pace_bridge_floor > 0` (the old hard gate).
After the 2026-06-12 retune the hard gate is gone; rhythm_matrix must now also
be extracted when `rhythm_soft_penalty_strength > 0`.

Fixture: 6-dim sonic space with tower_dims (2, 2, 2) → rhythm = dims [0:2].
The off-rhythm candidate is engineered to win on full-vector similarity, so it
is chosen when no penalty is active and demoted when the soft penalty is live.
"""
from pathlib import Path

import numpy as np

from src.features.artifacts import ArtifactBundle
from src.playlist.pier_bridge.config import PierBridgeConfig
from src.playlist.pier_bridge_builder import build_pier_bridge_playlist


def _bundle(X_sonic: np.ndarray, tower_dims: tuple[int, int, int]) -> ArtifactBundle:
    n = int(X_sonic.shape[0])
    track_ids = np.array([f"t{i}" for i in range(n)], dtype=object)
    return ArtifactBundle(
        artifact_path=Path("fake.npz"),
        track_ids=track_ids,
        artist_keys=np.array([f"artist-{i}" for i in range(n)], dtype=object),
        track_artists=np.array([f"Artist {i}" for i in range(n)], dtype=object),
        track_titles=np.array([f"Track {i}" for i in range(n)], dtype=object),
        X_sonic=X_sonic,
        X_sonic_start=None,
        X_sonic_mid=None,
        X_sonic_end=None,
        X_genre_raw=np.eye(n, dtype=float),
        X_genre_smoothed=np.eye(n, dtype=float),
        genre_vocab=np.array([f"g{i}" for i in range(n)], dtype=object),
        track_id_to_index={str(tid): i for i, tid in enumerate(track_ids)},
        tower_dims=tower_dims,
    )


# Layout: [rhythm0, rhythm1, timbre0, timbre1, harmony0, harmony1]
_X = np.array(
    [
        [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],  # t0: pier A (rhythm [1,0])
        [1.0, 0.0, 0.6, 0.8, 0.0, 1.0],  # t1: on-rhythm, weaker timbre/harmony
        [0.0, 1.0, 1.0, 0.0, 1.0, 0.0],  # t2: ORTHOGONAL rhythm, otherwise = piers
        [1.0, 0.0, 1.0, 0.0, 1.0, 0.0],  # t3: pier B (rhythm [1,0])
    ],
    dtype=float,
)


def _build(
    rhythm_soft_penalty_threshold: float = 0.0,
    rhythm_soft_penalty_strength: float = 0.0,
):
    bundle = _bundle(_X, tower_dims=(2, 2, 2))
    cfg = PierBridgeConfig(
        transition_floor=-1.0,
        bridge_floor=-1.0,
        pace_bridge_floor=0.0,
        rhythm_soft_penalty_threshold=rhythm_soft_penalty_threshold,
        rhythm_soft_penalty_strength=rhythm_soft_penalty_strength,
        progress_enabled=False,
        center_transitions=False,
        collapse_segment_pool_by_artist=False,
    )
    return build_pier_bridge_playlist(
        seed_track_ids=["t0", "t3"],
        total_tracks=3,
        bundle=bundle,
        candidate_pool_indices=[1, 2],
        cfg=cfg,
        min_genre_similarity=None,
        X_genre_smoothed=bundle.X_genre_smoothed,
    )


def test_off_rhythm_candidate_wins_when_penalty_disabled():
    """Sanity baseline: with no soft penalty, t2 wins on full-vector similarity."""
    result = _build()
    assert result.success
    assert result.track_ids == ["t0", "t2", "t3"]


def test_soft_penalty_demotes_off_rhythm_candidate_in_live_builder_path():
    """Builder must extract rhythm_matrix when rhythm_soft_penalty_strength > 0
    so the beam's soft penalty fires. t2 rhythm cosine = 0.0 < threshold 0.5,
    so its score is zeroed; on-rhythm t1 wins despite weaker full-vector sim."""
    result = _build(rhythm_soft_penalty_threshold=0.5, rhythm_soft_penalty_strength=1.0)
    assert result.success
    assert result.track_ids == ["t0", "t1", "t3"]
