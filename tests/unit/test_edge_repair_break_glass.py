"""Break-glass edge repair (spec 2026-07-01): T < t_floor triggers repair;
best-effort/never-worse acceptance; positional min_gap refusal; worst-first order.
The old trigger (centered_cos < -0.5) never fires on mildly-bad edges — these
tests use orthogonal (cos 0) vectors that only the new t_floor catches."""
from pathlib import Path

import numpy as np

from src.features.artifacts import ArtifactBundle
from src.playlist.repair.edge_repair import repair_playlist_edges
from src.playlist.transition_metrics import build_transition_metric_context, score_transition_edge

C25 = [0.25, 0.9682458365518543]   # cos 0.25 with [1,0]
C90 = [0.90, 0.4358898943540674]   # cos 0.90 with [1,0]


def _bundle(X: list[list[float]], artists: list[str] | None = None) -> ArtifactBundle:
    X_arr = np.array(X, dtype=float)
    n = int(X_arr.shape[0])
    track_ids = np.array([f"t{i}" for i in range(n)], dtype=object)
    return ArtifactBundle(
        artifact_path=Path("fake.npz"),
        track_ids=track_ids,
        artist_keys=np.array(artists or [f"artist-{i}" for i in range(n)], dtype=object),
        track_artists=np.array(artists or [f"Artist {i}" for i in range(n)], dtype=object),
        track_titles=np.array([f"Track {i}" for i in range(n)], dtype=object),
        X_sonic=X_arr,
        X_sonic_start=X_arr,
        X_sonic_mid=X_arr,
        X_sonic_end=X_arr,
        X_genre_raw=np.eye(n, dtype=float),
        X_genre_smoothed=np.eye(n, dtype=float),
        genre_vocab=np.array([f"g{i}" for i in range(n)], dtype=object),
        track_id_to_index={str(tid): i for i, tid in enumerate(track_ids)},
    )


def _ctx(bundle: ArtifactBundle):
    return build_transition_metric_context(
        X_sonic=bundle.X_sonic, X_start=bundle.X_sonic_start,
        X_mid=bundle.X_sonic_mid, X_end=bundle.X_sonic_end,
        X_genre=bundle.X_genre_smoothed, center_transitions=False,
    )


def _run(bundle, indices, candidates, **kw):
    defaults = dict(
        final_indices=indices, candidate_indices=candidates,
        metric_context=_ctx(bundle), bundle=bundle,
        seed_indices={indices[0], indices[-1]}, pier_positions={0, len(indices) - 1},
        transition_floor=0.2, centered_cos_floor=-0.5, margin=0.05,
    )
    defaults.update(kw)
    return repair_playlist_edges(**defaults)


def test_t_floor_zero_is_todays_noop():
    # Orthogonal edge (T=0) does NOT trip centered_cos -0.5; with t_floor=0 nothing fires.
    b = _bundle([[1, 0], [0, 1], [1, 0], C90])
    res = _run(b, [0, 1, 2], [3], t_floor=0.0)
    assert res.indices == [0, 1, 2]
    assert not any("new_idx" in e for e in res.swap_log)


def test_weak_edge_fires_and_swaps_to_best():
    # [1,0] -> [0,1] -> [1,0]: both interior edges T=0 < 0.30. Candidate 3 (cos .90
    # to both piers) lifts worst edge 0.0 -> 0.9.
    b = _bundle([[1, 0], [0, 1], [1, 0], C90])
    res = _run(b, [0, 1, 2], [3], t_floor=0.30)
    assert res.indices == [0, 3, 2]
    swap = next(e for e in res.swap_log if "new_idx" in e)
    assert swap["new_idx"] == 3 and swap["old_idx"] == 1
    worst = min(
        score_transition_edge(_ctx(b), res.indices[i - 1], res.indices[i])["T"]
        for i in range(1, len(res.indices))
    )
    assert worst > 0.8


def test_partial_lift_below_floor_is_accepted():
    # Best available candidate reaches only T=0.25 (< floor 0.30) — break-glass
    # accepts it anyway (0.0 -> 0.25 beats leaving 0.0).
    b = _bundle([[1, 0], [0, 1], [1, 0], C25])
    res = _run(b, [0, 1, 2], [3], t_floor=0.30)
    assert res.indices == [0, 3, 2]


def test_left_alone_when_no_candidate_clears_margin():
    # Only candidate is ALSO orthogonal to the piers: no improvement >= margin ->
    # leave as-is (never worse, no swap entries).
    b = _bundle([[1, 0], [0, 1], [1, 0], [0, 1]])
    res = _run(b, [0, 1, 2], [3], t_floor=0.30)
    assert res.indices == [0, 1, 2]
    assert not any("new_idx" in e for e in res.swap_log)


def test_min_gap_refuses_nearby_same_artist():
    # Candidate 5 shares an artist with position 1 (distance 1 < min_gap 3) -> refused;
    # fallback candidate 6 (distinct artist) is used instead.
    X = [[1, 0], C90, [0, 1], [1, 0], [1, 0], C90, C90]
    artists = ["P0", "SameA", "Bad", "P3", "unused", "SameA", "Fresh"]
    b = _bundle(X, artists)
    res = _run(b, [0, 1, 2, 3], [5, 6], t_floor=0.30,
               seed_indices={0, 3}, pier_positions={0, 3}, min_gap=3)
    assert res.indices == [0, 1, 6, 3]
    assert any(e.get("reason") == "min_gap" and e.get("candidate_idx") == 5 for e in res.swap_log)


def test_worst_first_ordering():
    # Two triggered edges; the worse one (T=0 at positions 2->3) must be processed
    # before the milder one (T=0.25 at 1->2): first executed swap targets pos 3.
    # Layout: piers 0,4; interior 1,2,3. Edge 1->2 has cos .25, edge 2->3 cos 0.
    X = [[1, 0], [1, 0], C25, [0, 1], [1, 0], C90]
    b = _bundle(X, ["P", "A", "B", "C", "Q", "R"])
    res = _run(b, [0, 1, 2, 3, 4], [5], t_floor=0.30,
               seed_indices={0, 4}, pier_positions={0, 4})
    swaps = [e for e in res.swap_log if "new_idx" in e]
    assert swaps, "expected at least one executed swap"
    assert swaps[0]["position"] == 3  # worst edge repaired first


def test_edge_repair_t_floor_default_and_override():
    from src.playlist.config import default_ds_config
    from src.playlist.pier_bridge.config import PierBridgeConfig
    from src.playlist.pipeline.pier_bridge_overrides import apply_pier_bridge_overrides

    assert PierBridgeConfig().edge_repair_t_floor == 0.30

    pb_cfg, _tuning, _sources = apply_pier_bridge_overrides(
        pier_bridge_config=PierBridgeConfig(),
        cfg=default_ds_config("dynamic", playlist_len=3),
        overrides={},
        pb_overrides={"edge_repair": {"t_floor": 0.42}},
        artist_playlist=False,
        dry_run=True,
        audit_cfg=None,
    )

    assert pb_cfg.edge_repair_t_floor == 0.42
