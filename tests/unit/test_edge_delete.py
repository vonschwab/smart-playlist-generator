# tests/unit/test_edge_delete.py
import math
from pathlib import Path

import numpy as np

from src.features.artifacts import ArtifactBundle
from src.playlist.config import default_ds_config
from src.playlist.pier_bridge.config import PierBridgeConfig
from src.playlist.pier_bridge_builder import build_pier_bridge_playlist
from src.playlist.pipeline.pier_bridge_overrides import apply_pier_bridge_overrides
from src.playlist.repair.edge_delete import delete_broken_edges


def _score(pairs, default=0.9):
    """edge_score backed by a symmetric dict; default for unspecified pairs."""
    def score(a, b):
        return pairs.get((a, b), pairs.get((b, a), default))
    return score


def test_deletes_worse_interior_endpoint_and_merges():
    # piers 10,13 protected; edge 11->12 broken (0.05). Deleting 11 merges 10->12=0.7;
    # deleting 12 merges 11->13=0.6. Best = delete 11.
    score = _score({(11, 12): 0.05, (10, 12): 0.70, (11, 13): 0.60, (10, 11): 0.8, (12, 13): 0.8})
    r = delete_broken_edges([10, 11, 12, 13], edge_score=score, floor=0.30,
                            protected_indices={10, 13}, max_deletions=4)
    assert r.indices == [10, 12, 13]
    assert len(r.delete_log) == 1 and r.delete_log[0]["deleted_idx"] == 11


def test_leaves_edge_when_no_deletion_improves():
    # broken 11->12=0.05; both merges also below the broken value -> never-worse blocks it.
    score = _score({(11, 12): 0.05, (10, 12): 0.02, (11, 13): 0.01})
    r = delete_broken_edges([10, 11, 12, 13], edge_score=score, floor=0.30,
                            protected_indices={10, 13}, max_deletions=4)
    assert r.indices == [10, 11, 12, 13]
    assert r.delete_log == []


def test_never_deletes_between_two_piers():
    # broken edge 10->13 is directly between two protected piers -> cannot delete either.
    score = _score({(10, 13): 0.05})
    r = delete_broken_edges([10, 13], edge_score=score, floor=0.30,
                            protected_indices={10, 13}, max_deletions=4)
    assert r.indices == [10, 13]
    assert r.delete_log == []


def test_noop_when_nothing_broken():
    score = _score({}, default=0.8)  # all edges 0.8 >= floor
    r = delete_broken_edges([10, 11, 12, 13], edge_score=score, floor=0.30,
                            protected_indices={10, 13}, max_deletions=4)
    assert r.indices == [10, 11, 12, 13]
    assert r.delete_log == []


def test_respects_max_deletions():
    # two broken interior edges; cap at 1 deletion.
    score = _score({(11, 12): 0.05, (12, 13): 0.05, (10, 12): 0.7, (11, 13): 0.7,
                    (10, 11): 0.8, (13, 14): 0.8}, default=0.8)
    r = delete_broken_edges([10, 11, 12, 13, 14], edge_score=score, floor=0.30,
                            protected_indices={10, 14}, max_deletions=1)
    assert len(r.delete_log) == 1


# --- bystander min_gap guard -------------------------------------------------


def test_skips_deletion_that_breaches_bystander_min_gap():
    # positions == bundle indices 0..9; piers 0 and 9. Artist "Q" sits at positions
    # 2 and 5 (distance 3 == min_gap+1, exactly legal). Broken edge is 3->4 (0.05);
    # both candidate deletions (pos 3 or pos 4) sit strictly between the Q pair, so
    # either one would shift Q@5 down to position 4, shrinking the pair to distance
    # 2 == min_gap -> a fresh bystander violation. Both merges score well (0.9), so
    # WITHOUT the guard the old code deletes pos 3 (tie -> lower del_pos wins) and
    # breaches the hard diversity constraint. WITH the guard both candidates are
    # blocked and the edge is left broken (never-worse: nothing safe improves it).
    artists = {2: "Q", 5: "Q"}

    def akey(i):
        return artists.get(i, f"u{i}")

    score = _score({(3, 4): 0.05, (2, 4): 0.9, (3, 5): 0.9}, default=0.9)
    r = delete_broken_edges(
        list(range(10)), edge_score=score, floor=0.30,
        protected_indices={0, 9}, max_deletions=4,
        artist_key_of=akey, min_gap=2,
    )
    # Key invariant: no same-artist pair in the surviving order sits at distance
    # <= min_gap (the hard diversity constraint the deletion must never breach).
    q_positions = [p for p, i in enumerate(r.indices) if akey(i) == "Q"]
    for a, b in zip(q_positions, q_positions[1:]):
        assert b - a > 2  # > min_gap
    # In this scenario both candidate deletions straddle the Q pair, so the guard
    # blocks the deletion outright and the broken edge is left (never-worse).
    assert r.indices == list(range(10))
    assert r.delete_log == []


def test_without_artist_key_of_gap_blind_still_deletes():
    # Companion contrast test: the same scenario as above, but WITHOUT
    # artist_key_of/min_gap (the defaults) -- old gap-blind behavior is
    # unchanged, and the deletion proceeds (proving the guard above is the
    # thing that changed the outcome, not some other side effect).
    score = _score({(3, 4): 0.05, (2, 4): 0.9, (3, 5): 0.9}, default=0.9)
    r = delete_broken_edges(
        list(range(10)), edge_score=score, floor=0.30,
        protected_indices={0, 9}, max_deletions=4,
    )
    assert r.indices == [0, 1, 2, 4, 5, 6, 7, 8, 9]  # pos 3 deleted, tie -> lower del_pos
    assert len(r.delete_log) == 1 and r.delete_log[0]["deleted_idx"] == 3


# --- Task 2: config knobs + override threading ------------------------------


def test_edge_delete_knobs_default_and_override():
    c = PierBridgeConfig()
    assert c.edge_delete_enabled is True
    assert c.edge_delete_floor == 0.30
    assert c.edge_delete_max_deletions == 4


def test_edge_delete_nested_config_overrides_are_parsed():
    pb_cfg, _tuning, _sources = apply_pier_bridge_overrides(
        pier_bridge_config=PierBridgeConfig(),
        cfg=default_ds_config("dynamic", playlist_len=3),
        overrides={},
        pb_overrides={
            "edge_delete": {
                "enabled": False,
                "floor": 0.42,
                "max_deletions": 2,
            }
        },
        artist_playlist=False,
        dry_run=True,
        audit_cfg=None,
    )

    assert pb_cfg.edge_delete_enabled is False
    assert pb_cfg.edge_delete_floor == 0.42
    assert pb_cfg.edge_delete_max_deletions == 2


def test_edge_delete_overrides_absent_keep_dataclass_defaults():
    pb_cfg, _tuning, _sources = apply_pier_bridge_overrides(
        pier_bridge_config=PierBridgeConfig(),
        cfg=default_ds_config("dynamic", playlist_len=3),
        overrides={},
        pb_overrides={},
        artist_playlist=False,
        dry_run=True,
        audit_cfg=None,
    )

    assert pb_cfg.edge_delete_enabled is True
    assert pb_cfg.edge_delete_floor == 0.30
    assert pb_cfg.edge_delete_max_deletions == 4


# --- Task 2: builder integration (real multi-pier trigger, not single-seed) -
#
# NOTE on harness choice: tests/support/gui_fidelity drives a full production
# config.yaml -> generate_playlist_ds chain against the live artifact
# (integration/@slow, skipped without data/artifacts/.../data_matrices_step1.npz).
# Engineering a *deterministic* real-artifact deletion trigger there is not
# reliable (live sonic vectors, pool composition, and beam scoring are not
# something a unit test can pin). Per the task brief's authorized fallback,
# this test instead drives the real `build_pier_bridge_playlist` entry point
# directly (same pattern as tests/unit/test_builder_pair_floor_wiring.py) with
# a hand-built ArtifactBundle -- but crucially with TWO piers (a real pier-
# bridge segment, not the single-seed-arc special case: num_seeds=1 triggers
# `is_single_seed_arc` and a different code path). The sonic vectors below are
# picked geometrically (unit circle placements) so the beam is *forced* to
# route through a genuinely broken interior edge that break-glass repair
# cannot fix (repair is disabled here; its candidate pool would be exhausted
# anyway since the only two candidates are already both in the playlist).
# This exercises the real end-to-end deletion, not a mock.


def _outlier_bundle() -> ArtifactBundle:
    def vec(deg: float) -> list[float]:
        r = math.radians(deg)
        return [math.cos(r), math.sin(r)]

    # t0=pier A (0deg), t1=outlier interior (250deg), t2=close interior (5deg),
    # t3=pier B (15deg). The beam places these in seed-track order [t0,t1,t2,t3]:
    # edge t0->t1 = -0.342 (broken), t1->t2 = -0.423 (broken, worst), t2->t3 =
    # 0.985 (healthy). Deleting the outlier t1 merges t0->t2 = 0.996 (healthy),
    # which strictly beats the worst broken edge -> a real deletion fires,
    # leaving [t0, t2, t3] with both edges above the 0.30 floor.
    pts = [vec(0), vec(250), vec(5), vec(15)]
    X = np.array(pts, dtype=float)
    n = X.shape[0]
    track_ids = np.array([f"t{i}" for i in range(n)], dtype=object)
    return ArtifactBundle(
        artifact_path=Path("fake.npz"),
        track_ids=track_ids,
        artist_keys=np.array([f"artist-{i}" for i in range(n)], dtype=object),
        track_artists=np.array([f"Artist {i}" for i in range(n)], dtype=object),
        track_titles=np.array([f"Track {i}" for i in range(n)], dtype=object),
        X_sonic=X,
        X_sonic_start=X,
        X_sonic_mid=X,
        X_sonic_end=X,
        X_genre_raw=np.eye(n, dtype=float),
        X_genre_smoothed=np.eye(n, dtype=float),
        genre_vocab=np.array([f"g{i}" for i in range(n)], dtype=object),
        track_id_to_index={str(tid): i for i, tid in enumerate(track_ids)},
    )


def _build_outlier(*, edge_delete_enabled: bool):
    cfg = PierBridgeConfig(
        transition_floor=-1.0,
        bridge_floor=-1.0,
        progress_enabled=False,
        center_transitions=False,
        collapse_segment_pool_by_artist=False,
        genre_steering_enabled=False,
        edge_repair_enabled=False,
        edge_delete_enabled=edge_delete_enabled,
        edge_delete_floor=0.30,
        edge_delete_max_deletions=4,
    )
    bundle = _outlier_bundle()
    return build_pier_bridge_playlist(
        seed_track_ids=["t0", "t3"],
        total_tracks=4,
        bundle=bundle,
        candidate_pool_indices=[1, 2],
        cfg=cfg,
        min_genre_similarity=None,
        X_genre_smoothed=bundle.X_genre_smoothed,
    )


def test_edge_delete_removes_outlier_track_and_merges_pier_bridge():
    result = _build_outlier(edge_delete_enabled=True)

    assert result.success
    assert result.track_ids == ["t0", "t2", "t3"]  # outlier t1 removed
    assert result.stats["edge_delete_enabled"] is True
    assert result.stats["edge_delete_applied"] is True
    log = result.stats["edge_delete_log"]
    assert len(log) == 1
    assert log[0]["deleted_idx"] == 1  # bundle index of t1
    # No pier was touched -- both seed track_ids survive.
    assert "t0" in result.track_ids and "t3" in result.track_ids
    # The playlist is shorter by exactly the number of delete_log entries.
    assert len(result.track_ids) == 4 - len(log)


def test_edge_delete_disabled_leaves_broken_edge_and_full_length():
    result = _build_outlier(edge_delete_enabled=False)

    assert result.success
    assert result.track_ids == ["t0", "t1", "t2", "t3"]
    assert result.stats["edge_delete_enabled"] is False
    assert result.stats["edge_delete_applied"] is False
    assert result.stats["edge_delete_log"] == []
