import numpy as np

from src.playlist.candidate_pool import build_candidate_pool
from src.playlist.config import CandidatePoolConfig


def _cfg() -> CandidatePoolConfig:
    return CandidatePoolConfig(
        similarity_floor=-1.0,
        min_sonic_similarity=None,
        max_pool_size=10,
        target_artists=5,
        candidates_per_artist=5,
        seed_artist_bonus=0,
        max_artist_fraction_final=1.0,
        title_hard_exclude_flags=frozenset(),
    )


def test_layered_candidate_pool_shadow_diagnostics_do_not_change_pool_membership():
    emb = np.array([[1.0, 0.0]] * 4, dtype=float)
    artist_keys = np.array(["seed", "leaf_match", "broad_only", "bridge"])
    track_ids = np.array(["seed-id", "leaf-id", "broad-id", "bridge-id"])

    baseline = build_candidate_pool(
        seed_idx=0,
        embedding=emb,
        artist_keys=artist_keys,
        track_ids=track_ids,
        cfg=_cfg(),
        random_seed=0,
        mode="strict",
    )

    # Leaf dimensions: [jangle pop, synth-pop]
    X_leaf = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=float,
    )
    X_family = np.array(
        [
            [1.0],
            [1.0],
            [1.0],
            [1.0],
        ],
        dtype=float,
    )
    X_bridge = np.array(
        [
            [0.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=float,
    )
    X_facet = np.array(
        [
            [1.0],
            [1.0],
            [0.0],
            [1.0],
        ],
        dtype=float,
    )

    with_shadow = build_candidate_pool(
        seed_idx=0,
        embedding=emb,
        artist_keys=artist_keys,
        track_ids=track_ids,
        cfg=_cfg(),
        random_seed=0,
        mode="strict",
        layered_genre_diagnostics=True,
        X_genre_leaf_idf=X_leaf,
        X_genre_family=X_family,
        X_genre_bridge=X_bridge,
        X_facet=X_facet,
    )

    assert with_shadow.pool_indices.tolist() == baseline.pool_indices.tolist()
    shadow = with_shadow.stats["layered_genre_shadow"]
    assert shadow["enabled"] is True
    assert shadow["evaluated_count"] == 3
    assert shadow["would_admit_count"] == 2
    assert shadow["broad_only_reject_count"] == 1
    assert shadow["bridge_supported_count"] == 1

    by_track_id = {row["track_id"]: row for row in shadow["samples"]}
    assert by_track_id["leaf-id"]["reason"] == "layered_score_threshold"
    assert by_track_id["broad-id"]["reason"] == "broad_only_without_leaf_support"
    assert by_track_id["bridge-id"]["reason"] == "bridge_supported"


def test_layered_candidate_pool_shadow_diagnostics_require_complete_matrices():
    emb = np.array([[1.0, 0.0]] * 2, dtype=float)
    artist_keys = np.array(["seed", "candidate"])

    result = build_candidate_pool(
        seed_idx=0,
        embedding=emb,
        artist_keys=artist_keys,
        cfg=_cfg(),
        random_seed=0,
        mode="dynamic",
        layered_genre_diagnostics=True,
        X_genre_leaf_idf=np.zeros((2, 1), dtype=float),
    )

    shadow = result.stats["layered_genre_shadow"]
    assert shadow == {
        "enabled": False,
        "reason": "missing_layered_matrices",
    }
