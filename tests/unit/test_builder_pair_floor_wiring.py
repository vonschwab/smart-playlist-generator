"""The live builder must supply the tag-level pair provider to the beam's
pairwise genre-edge floor under taxonomy steering.

Without the provider the gate would fall back to the smoothed-vector cosine,
which calibration (2026-06-10) proved cannot separate bad edges from good
(Sharp Pins->Springsteen 0.693 vs the praised YYY->StVincent 0.677). This test
pins the end-to-end wiring: config floor -> builder provider -> beam rejection.

Fixture: real taxonomy genre names so canonicalization works. The off-genre
candidate (funk) is sonically IDENTICAL to the piers, so it wins when the gate
is dead and is rejected (in favor of the sonically-weaker dream pop candidate)
when the gate is live: shoegaze~dream pop = 0.69, shoegaze~funk ~ 0.0.
"""
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.features.artifacts import ArtifactBundle
from src.playlist.pier_bridge.config import PierBridgeConfig
from src.playlist.pier_bridge_builder import build_pier_bridge_playlist

_VOCAB = np.array(["shoegaze", "dream pop", "funk"], dtype=object)

# Genre rows: t0/t3 piers = shoegaze; t1 = dream pop (compatible);
# t2 = funk (incompatible with shoegaze in the taxonomy).
_G = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=float,
)

# Sonic: t2 identical to the piers (wins without the gate); t1 weaker.
_X = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.8, 0.6, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=float,
)


def _bundle() -> ArtifactBundle:
    n = _X.shape[0]
    track_ids = np.array([f"t{i}" for i in range(n)], dtype=object)
    return ArtifactBundle(
        artifact_path=Path("fake.npz"),
        track_ids=track_ids,
        artist_keys=np.array([f"artist-{i}" for i in range(n)], dtype=object),
        track_artists=np.array([f"Artist {i}" for i in range(n)], dtype=object),
        track_titles=np.array([f"Track {i}" for i in range(n)], dtype=object),
        X_sonic=_X,
        X_sonic_start=None,
        X_sonic_mid=None,
        X_sonic_end=None,
        X_genre_raw=_G,
        X_genre_smoothed=_G,
        genre_vocab=_VOCAB,
        track_id_to_index={str(tid): i for i, tid in enumerate(track_ids)},
    )


def _build(genre_pair_floor: float):
    cfg = PierBridgeConfig(
        transition_floor=-1.0,
        bridge_floor=-1.0,
        progress_enabled=False,
        center_transitions=False,
        collapse_segment_pool_by_artist=False,
        genre_steering_enabled=True,
        genre_steering_source="taxonomy",
        weight_genre=0.0,                # arc vote inert: isolate the pair gate
        genre_arc_floor=0.0,
        genre_arc_floor_percentile=0.0,
        genre_pair_floor=genre_pair_floor,
        # Phase 1 Task 8: corridor is the sole pooling path. bridge_floor=-1.0
        # was this fixture's "admit everyone" sentinel under legacy's fixed-
        # floor gate; corridor's percentile-based membership needs the
        # equivalent -- width_percentile=0.0 admits the full 2-candidate
        # universe (t1, t2) so the beam-level pair-floor demotion this test
        # exists to demonstrate can actually compete between both.
        corridor_width_percentile=0.0,
    )
    bundle = _bundle()
    return build_pier_bridge_playlist(
        seed_track_ids=["t0", "t3"],
        total_tracks=3,
        bundle=bundle,
        candidate_pool_indices=[1, 2],
        cfg=cfg,
        X_genre_smoothed=bundle.X_genre_smoothed,
    )


def test_off_genre_candidate_wins_when_pair_floor_disabled():
    """Sanity baseline: gate off, the sonically-identical funk track wins."""
    result = _build(genre_pair_floor=0.0)
    assert result.success
    assert result.track_ids == ["t0", "t2", "t3"]


def test_pair_floor_demotes_off_genre_candidate_in_live_builder_path():
    """With genre_pair_floor set, the builder must build + pass the taxonomy
    pair provider so the beam demotes funk (shoegaze~funk ~ 0) below the
    sonically-weaker dream pop track (shoegaze~dream pop = 0.69), which then wins."""
    result = _build(genre_pair_floor=0.3)
    assert result.success
    assert result.track_ids == ["t0", "t1", "t3"]


def test_pair_floor_does_not_brick_when_every_edge_is_below_floor():
    """A zero-sim provider penalizes every candidate edge equally, so the floor
    can never make the segment infeasible — generation still succeeds and sonic
    decides. (A hard gate here would return infeasible and detonate the
    relaxation cascade: the multi-minute hang the soft penalty replaced.)

    t2 (funk) is sonically identical to the piers, so with all edges demoted by
    the same amount it wins on sonic."""
    class _ZeroSimProvider:
        def sim(self, a: int, b: int):
            return 0.0

    with patch(
        "src.playlist.pier_bridge.taxonomy_steering.build_taxonomy_pair_provider",
        return_value=_ZeroSimProvider(),
    ):
        result = _build(genre_pair_floor=0.3)

    assert result.success, "pair penalty must never make a segment infeasible"
    assert result.track_ids == ["t0", "t2", "t3"]
