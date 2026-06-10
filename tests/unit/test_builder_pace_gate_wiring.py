"""The live builder must thread rhythm_matrix into the beam's pace gate.

Regression test for the silent no-op found 2026-06-10: `_beam_search_segment`
implements a rhythm-cosine gate (`pace_bridge_floor`), but the production call
site in `pier_bridge_builder.py` never passed `rhythm_matrix`, so the gate was
inert in every real generation (only the dead `assemble.py` wired it).

Fixture: 6-dim sonic space with tower_dims (2, 2, 2) → rhythm = dims [0:2].
The off-rhythm candidate is engineered to win on full-vector similarity, so it
is chosen when the gate is dead and rejected when the gate is live.
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


def _build(pace_bridge_floor: float):
    bundle = _bundle(_X, tower_dims=(2, 2, 2))
    cfg = PierBridgeConfig(
        transition_floor=-1.0,
        bridge_floor=-1.0,
        pace_bridge_floor=pace_bridge_floor,
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


def test_off_rhythm_candidate_wins_when_gate_disabled():
    """Sanity baseline: with the gate off, t2 wins on full-vector similarity."""
    result = _build(pace_bridge_floor=0.0)
    assert result.success
    assert result.track_ids == ["t0", "t2", "t3"]


def test_pace_gate_rejects_off_rhythm_candidate_in_live_builder_path():
    """With pace_bridge_floor set, the builder must supply rhythm vectors so the
    beam rejects t2 (rhythm cosine 0.0 to the pier rhythm target) and falls back
    to the on-rhythm t1 — despite t2's higher full-vector similarity."""
    result = _build(pace_bridge_floor=0.5)
    assert result.success
    assert result.track_ids == ["t0", "t1", "t3"]
