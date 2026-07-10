import numpy as np
from src.playlist.tag_steering import select_on_tag_anchors


def test_selects_bridgeable_tag_central_diverse():
    # 6 tracks. piers = [0]. on-tag = [1,2,3,4,5].
    # track 5 is an ISLAND (far from pier 0) -> excluded by min_bridge.
    X = np.array([
        [1.0, 0.0, 0.0],   # 0 pier
        [0.9, 0.1, 0.0],   # 1 bridgeable, artist A
        [0.85, 0.2, 0.0],  # 2 bridgeable, artist A
        [0.8, 0.3, 0.0],   # 3 bridgeable, artist B
        [0.75, 0.4, 0.0],  # 4 bridgeable, artist C
        [0.0, 0.0, 1.0],   # 5 ISLAND, artist D
    ], dtype=np.float64)
    tag_centrality = np.array([0.0, 0.9, 0.8, 0.7, 0.6, 0.99])  # 5 most central but island
    artist_keys = np.array(["p", "a", "a", "b", "c", "d"])
    track_ids = np.array([f"t{i}" for i in range(6)])
    got = select_on_tag_anchors(
        on_tag_indices=[1, 2, 3, 4, 5], pier_indices=[0], X_sonic=X,
        tag_centrality=tag_centrality, artist_keys=artist_keys, track_ids=track_ids,
        max_anchors=3, min_bridge=0.5, per_artist=1,
    )
    assert 5 not in got                 # island excluded (max sim to pier 0 = 0.0 < 0.5)
    assert got == [1, 3, 4]             # per-artist cap 1: A->only track1 (higher centrality), then B(3), C(4); capped at 3


def test_empty_when_no_bridgeable():
    X = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    assert select_on_tag_anchors(
        on_tag_indices=[1], pier_indices=[0], X_sonic=X, tag_centrality=None,
        artist_keys=np.array(["p", "a"]), track_ids=np.array(["t0", "t1"]),
        max_anchors=3, min_bridge=0.5, per_artist=1,
    ) == []


def test_max_zero_returns_empty():
    X = np.array([[1.0, 0.0], [0.9, 0.1]], dtype=np.float64)
    assert select_on_tag_anchors(
        on_tag_indices=[1], pier_indices=[0], X_sonic=X, tag_centrality=None,
        artist_keys=np.array(["p", "a"]), track_ids=np.array(["t0", "t1"]),
        max_anchors=0, min_bridge=0.5, per_artist=1,
    ) == []
