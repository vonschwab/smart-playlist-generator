import numpy as np

from src.playlist.config import resolve_pier_bridge_tuning
from src.playlist.pier_bridge.beam import _beam_search_segment
from src.playlist.pier_bridge.config import PierBridgeConfig


def _diag4():
    # 4 tracks: 0=pierA, 1=genre-OK candidate, 2=genre-misfit candidate, 3=pierB.
    # Sonic: all identical so sonic never decides (forces genre to break the tie).
    X = np.ones((4, 3), dtype=float)
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    # Dense genre (L2-normalized rows): cand 1 aligns with piers; cand 2 is orthogonal.
    dense = np.array([
        [1.0, 0.0, 0.0],   # pierA
        [1.0, 0.0, 0.0],   # cand 1: same genre as piers (sim 1.0)
        [0.0, 1.0, 0.0],   # cand 2: orthogonal genre (sim 0.0)
        [1.0, 0.0, 0.0],   # pierB
    ], dtype=float)
    return Xn, dense


def test_beam_floor_rejects_off_genre_candidate():
    Xn, dense = _diag4()
    cfg = PierBridgeConfig(
        bridge_floor=-1.0, transition_floor=-1.0, progress_enabled=False,
        genre_steering_enabled=True, weight_genre=0.2, genre_arc_floor=0.5,
        weight_bridge=0.5, weight_transition=0.3,
    )
    # Arc target == piers' genre; cand 1 is on-arc (sim 1.0), cand 2 off (sim 0.0).
    g_targets = [np.array([1.0, 0.0, 0.0])]
    path, _h, _e, err = _beam_search_segment(
        0, 3, 1, [2, 1], Xn, Xn, None, None, None, None, cfg, 5,
        X_genre_dense=dense, g_targets_override=g_targets,
    )
    assert err is None
    assert path == [1], f"expected genre-OK cand 1, got {path}"


def test_beam_steering_prefers_higher_genre_when_sonic_tied():
    Xn, dense = _diag4()
    # No floor (0.0) so cand 2 is allowed; steering weight should still rank cand 1 first.
    cfg = PierBridgeConfig(
        bridge_floor=-1.0, transition_floor=-1.0, progress_enabled=False,
        genre_steering_enabled=True, weight_genre=0.3, genre_arc_floor=0.0,
        weight_bridge=0.4, weight_transition=0.3,
    )
    # Arc target == piers' genre; cand 1 (sim 1.0) outranks cand 2 (sim 0.0).
    g_targets = [np.array([1.0, 0.0, 0.0])]
    path, _h, _e, err = _beam_search_segment(
        0, 3, 1, [2, 1], Xn, Xn, None, None, None, None, cfg, 5,
        X_genre_dense=dense, g_targets_override=g_targets,
    )
    assert err is None
    assert path == [1]


def test_beam_legacy_unchanged_when_steering_off():
    Xn, dense = _diag4()
    cfg = PierBridgeConfig(
        bridge_floor=-1.0, transition_floor=-1.0, progress_enabled=False,
        genre_steering_enabled=False,
    )
    # Steering off: floor must NOT reject; both candidates valid, no crash.
    path, _h, _e, err = _beam_search_segment(
        0, 3, 1, [2, 1], Xn, Xn, None, None, None, None, cfg, 5,
        X_genre_dense=dense,
    )
    assert err is None
    assert len(path) == 1


def test_tuning_genre_steering_defaults_off():
    """No overrides -> steering off, inert genre knobs, legacy weights intact."""
    t, _ = resolve_pier_bridge_tuning(mode="narrow", similarity_floor=0.35, overrides=None)
    assert t.genre_steering_enabled is False
    assert t.weight_genre == 0.0
    assert t.genre_arc_floor == 0.0
    # Legacy narrow weights unchanged when steering off
    assert abs(t.weight_bridge - 0.7) < 1e-6
    assert abs(t.weight_transition - 0.3) < 1e-6


def test_tuning_genre_steering_renormalizes_weights():
    """When enabled with weight_genre, the three edge weights renormalize to sum 1."""
    overrides = {
        "pier_bridge": {
            "genre_steering_enabled": True,
            "weight_genre_narrow": 0.20,
            "genre_arc_floor_narrow": 0.40,
            # leave bridge/transition at narrow defaults 0.7/0.3
        }
    }
    t, _ = resolve_pier_bridge_tuning(mode="narrow", similarity_floor=0.35, overrides=overrides)
    assert t.genre_steering_enabled is True
    assert abs(t.genre_arc_floor - 0.40) < 1e-6
    total = t.weight_bridge + t.weight_transition + t.weight_genre
    assert abs(total - 1.0) < 1e-6
    # genre share is 0.20 / 1.20
    assert abs(t.weight_genre - (0.20 / 1.20)) < 1e-6


def test_tuning_steering_on_zero_weight_genre_is_noop():
    overrides = {"pier_bridge": {"genre_steering_enabled": True}}  # no weight_genre
    t, _ = resolve_pier_bridge_tuning(mode="narrow", similarity_floor=0.35, overrides=overrides)
    assert t.genre_steering_enabled is True
    assert t.weight_genre == 0.0
    assert abs(t.weight_bridge - 0.7) < 1e-6
    assert abs(t.weight_transition - 0.3) < 1e-6


def test_beam_genreless_endpoint_skips_floor():
    """A candidate with a zero dense vector is genreless: the floor must NOT reject it
    and no genre weight is credited (the edge passes through cleanly)."""
    Xn, dense = _diag4()
    dense = dense.copy()
    dense[1] = 0.0  # candidate 1 is genreless (zero dense vector)
    # Only candidate 1 (genreless) offered; high absolute + percentile floor must
    # NOT reject it (genreless candidates are exempt from the on-arc floor).
    g_targets = [np.array([1.0, 0.0, 0.0])]
    cfg = PierBridgeConfig(
        bridge_floor=-1.0, transition_floor=-1.0, progress_enabled=False,
        genre_steering_enabled=True, weight_genre=0.2, genre_arc_floor=0.9,
        genre_arc_floor_percentile=0.9, weight_bridge=0.5, weight_transition=0.3,
    )
    path, _h, _e, err = _beam_search_segment(
        0, 3, 1, [1], Xn, Xn, None, None, None, None, cfg, 5,
        X_genre_dense=dense, g_targets_override=g_targets,
    )
    assert err is None
    assert path == [1]


def test_infeasible_handling_genre_floor_fields_default():
    from src.playlist.run_audit import InfeasibleHandlingConfig, parse_infeasible_handling_config
    cfg = InfeasibleHandlingConfig()
    assert cfg.genre_arc_relaxation_enabled is True
    assert cfg.min_genre_arc_percentile == 0.5
    parsed = parse_infeasible_handling_config({
        "enabled": True, "min_genre_arc_percentile": 0.15, "genre_arc_relaxation_enabled": False,
    })
    assert parsed.genre_arc_relaxation_enabled is False
    assert abs(parsed.min_genre_arc_percentile - 0.15) < 1e-6



def test_per_seed_admission_floor_adapts_to_density():
    # This guards the *helper contract* candidate_pool will use: floor derived
    # from THIS seed's dense-sim distribution at percentile P_admit.
    from src.playlist.pier_bridge.percentiles import floor_at_percentile
    seed = np.array([1.0, 0.0, 0.0])
    # sparse neighborhood: few aligned, many orthogonal
    D_sparse = np.vstack([np.tile([1, 0, 0], (50, 1)), np.tile([0, 1, 0], (950, 1))]).astype(float)
    D_dense = np.vstack([np.tile([1, 0, 0], (700, 1)), np.tile([0, 1, 0], (300, 1))]).astype(float)
    s_sparse = (D_sparse / np.linalg.norm(D_sparse, axis=1, keepdims=True)) @ seed
    s_dense = (D_dense / np.linalg.norm(D_dense, axis=1, keepdims=True)) @ seed
    f_sparse = floor_at_percentile(s_sparse, 0.90)
    f_dense = floor_at_percentile(s_dense, 0.90)
    # both admit ~top 10%, but the absolute floor differs by neighborhood density
    assert f_dense >= f_sparse


def test_arc_vote_is_first_class_and_uses_waypoint_target():
    # 4 tracks, sonic identical; candidate 1's dense vec matches g_target[0],
    # candidate 2 matches the previous track but NOT the target. Steering must
    # pick candidate 1 (on the arc), proving it scores vs g_target not prev-track.
    import numpy as np
    from src.playlist.pier_bridge.beam import _beam_search_segment
    from src.playlist.pier_bridge.config import PierBridgeConfig
    Xn = np.ones((4, 3))
    Xn = Xn / np.linalg.norm(Xn, axis=1, keepdims=True)
    dense = np.array([
        [1.0, 0.0, 0.0],   # pierA
        [0.0, 1.0, 0.0],   # cand 1: matches the step-0 target below
        [1.0, 0.0, 0.0],   # cand 2: matches pierA (prev track) but not target
        [0.0, 0.0, 1.0],   # pierB
    ])
    g_targets = [np.array([0.0, 1.0, 0.0])]  # step-0 target == cand 1's genre
    cfg = PierBridgeConfig(bridge_floor=-1.0, transition_floor=-1.0, progress_enabled=False,
                           genre_steering_enabled=True, weight_genre=0.4,
                           genre_arc_floor_percentile=0.0, weight_bridge=0.4, weight_transition=0.2)
    path, *_ = _beam_search_segment(0, 3, 1, [2, 1], Xn, Xn, None, None, None, None, cfg, 5,
                                    X_genre_dense=dense, g_targets_override=g_targets)
    assert path == [1]


def test_arc_floor_percentile_rejects_below_floor_candidate():
    # 3 candidates with arc-sims to target of 1.0, 0.5, 0.0. A 0.5 percentile
    # floor (median == 0.5) admits sims >= 0.5, dropping the 0.0 candidate.
    import numpy as np
    from src.playlist.pier_bridge.beam import _beam_search_segment
    from src.playlist.pier_bridge.config import PierBridgeConfig
    Xn = np.ones((5, 3))
    Xn = Xn / np.linalg.norm(Xn, axis=1, keepdims=True)
    dense = np.array([
        [1.0, 0.0, 0.0],   # pierA
        [0.0, 1.0, 0.0],   # cand 1: arc-sim 1.0 (on target)
        [0.70710678, 0.70710678, 0.0],  # cand 2: arc-sim ~0.707
        [1.0, 0.0, 0.0],   # cand 3: arc-sim 0.0 (off target) -> below median floor
        [0.0, 0.0, 1.0],   # pierB
    ])
    g_targets = [np.array([0.0, 1.0, 0.0])]
    cfg = PierBridgeConfig(bridge_floor=-1.0, transition_floor=-1.0, progress_enabled=False,
                           genre_steering_enabled=True, weight_genre=0.4,
                           genre_arc_floor_percentile=0.5, weight_bridge=0.4, weight_transition=0.2)
    path, *_ = _beam_search_segment(0, 3, 1, [1, 2, 3], Xn, Xn, None, None, None, None, cfg, 5,
                                    X_genre_dense=dense, g_targets_override=g_targets)
    # cand 3 (idx 3) must be rejected by the percentile floor; cand 1 wins on arc-vote.
    assert path == [1]


def test_arc_knobs_resolve():
    overrides = {"pier_bridge": {
        "genre_steering_enabled": True,
        "weight_genre_narrow": 0.20,
        "genre_arc_floor_percentile_narrow": 0.85,
        "genre_admission_percentile_narrow": 0.90,
        "dj_route_shape": "ladder",
    }}
    from src.playlist.config import resolve_pier_bridge_tuning
    t, _ = resolve_pier_bridge_tuning(mode="narrow", similarity_floor=0.35, overrides=overrides)
    assert t.genre_steering_enabled is True
    assert abs(t.genre_arc_floor_percentile - 0.85) < 1e-9
    assert abs(t.genre_admission_percentile - 0.90) < 1e-9


# --- pairwise genre-edge floor (deterministic bad-edge gate) -------------------
#
# The arc vote scores candidates against the smoothed WAYPOINT target, never
# against the track they're actually placed next to. A candidate can be on-arc
# yet genre-incompatible with its neighbor (Sharp Pins -> Springsteen: T=0.657
# approved, taxonomy pair sim ~0). genre_pair_floor gates the EDGE pairwise.


def _pair5_taxonomy():
    """5 tracks under taxonomy steering. Sonic identical (never decides).

    Genre (L2 rows over a 3-dim vocab):
      0 pierA  [1,0,0]
      1 candA  [1,0,0]        pair sim to pierA = 1.0
      2 candB  [0,1,0]        pair sim to pierA = 0.0 (genre-far)
      3 pierB  [1,0,0]
      4 candC  [.707,.707,0]  pair sim .707 to everything in-cluster
    """
    Xn = np.ones((5, 3), dtype=float)
    Xn = Xn / np.linalg.norm(Xn, axis=1, keepdims=True)
    g = np.array([
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.70710678, 0.70710678, 0.0],
    ], dtype=float)
    return Xn, g


def _tax_cfg(**kw):
    base = dict(
        bridge_floor=-1.0, transition_floor=-1.0, progress_enabled=False,
        genre_steering_enabled=True, genre_steering_source="taxonomy",
        weight_genre=0.4, genre_arc_floor=0.0, genre_arc_floor_percentile=0.0,
        weight_bridge=0.4, weight_transition=0.2,
    )
    base.update(kw)
    return PierBridgeConfig(**base)


def test_pair_floor_rejects_genre_far_interior_edge():
    """candB is arc-FAVORED but pairwise genre-far from pierA; the pair floor
    must reject the pierA->candB edge so candA wins."""
    Xn, g = _pair5_taxonomy()
    # Arc target favors candB (sim 1.0) over candA (sim 0.0): without the pair
    # floor candB wins on the arc vote. With it, candB's edge is illegal.
    g_targets = [np.array([0.0, 1.0, 0.0])]
    cfg = _tax_cfg(genre_pair_floor=0.5)
    path, _h, _e, err = _beam_search_segment(
        0, 3, 1, [1, 2], Xn, Xn, None, None, None, g, cfg, 5,
        g_targets_override=g_targets,
    )
    assert err is None
    assert path == [1], f"pair floor must reject genre-far candB; got {path}"


def test_pair_floor_zero_is_noop():
    """floor 0.0 (default): the genre-far candidate stays legal."""
    Xn, g = _pair5_taxonomy()
    g_targets = [np.array([0.0, 1.0, 0.0])]
    cfg = _tax_cfg(genre_pair_floor=0.0)
    path, _h, _e, err = _beam_search_segment(
        0, 3, 1, [2], Xn, Xn, None, None, None, g, cfg, 5,
        g_targets_override=g_targets,
    )
    assert err is None
    assert path == [2]


def test_pair_floor_gates_final_pier_connection():
    """candB passes the interior edge via candC-like geometry but its FINAL edge
    to pierB is genre-far: the final-connection gate must reject it.

    Here: pierA [0,1,0], candB [0,1,0] (interior edge sim 1.0), pierB [1,0,0]
    (final edge sim 0.0). candC [.707,.707,0] passes both (.707). Arc favors
    candB, so without the final gate candB wins."""
    Xn, _ = _pair5_taxonomy()
    g = np.array([
        [0.0, 1.0, 0.0],            # pierA
        [0.70710678, 0.70710678, 0.0],  # candC: interior .707, final .707
        [0.0, 1.0, 0.0],            # candB: interior 1.0, final 0.0
        [1.0, 0.0, 0.0],            # pierB
        [0.0, 0.0, 1.0],            # unused
    ], dtype=float)
    g_targets = [np.array([0.0, 1.0, 0.0])]  # arc favors candB
    cfg = _tax_cfg(genre_pair_floor=0.5)
    path, _h, _e, err = _beam_search_segment(
        0, 3, 1, [1, 2], Xn, Xn, None, None, None, g, cfg, 5,
        g_targets_override=g_targets,
    )
    assert err is None
    assert path == [1], f"final-connection pair floor must reject candB; got {path}"


def test_pair_floor_genreless_endpoint_exempt():
    """A genreless candidate (zero genre vector) is exempt from the pair floor
    on both its interior and final edges."""
    Xn, g = _pair5_taxonomy()
    g = g.copy()
    g[2] = 0.0  # candB genreless
    cfg = _tax_cfg(genre_pair_floor=0.9, weight_genre=0.0)
    path, _h, _e, err = _beam_search_segment(
        0, 3, 1, [2], Xn, Xn, None, None, None, g, cfg, 5,
    )
    assert err is None
    assert path == [2]


def test_pair_floor_infeasible_error_names_the_gate():
    """When the pair floor rejects every final connection, the error message
    must say so (loud failure, not a generic 'no valid continuations')."""
    Xn, _ = _pair5_taxonomy()
    g = np.array([
        [0.0, 1.0, 0.0],   # pierA
        [0.0, 1.0, 0.0],   # only cand: interior 1.0, final 0.0
        [0.0, 0.0, 1.0],   # unused
        [1.0, 0.0, 0.0],   # pierB genre-far from everything offered
        [0.0, 0.0, 1.0],   # unused
    ], dtype=float)
    cfg = _tax_cfg(genre_pair_floor=0.5, weight_genre=0.0)
    path, _h, _e, err = _beam_search_segment(
        0, 3, 1, [1], Xn, Xn, None, None, None, g, cfg, 5,
    )
    assert path is None
    assert err is not None and "genre_pair_floor" in err, f"err={err!r}"


def test_pair_floor_uses_injected_provider_over_vector_cosine():
    """When a pair_sim_provider is supplied (taxonomy tag-level scoring), the
    gate must score with IT, not the genre-vector cosine. Here the vectors say
    candB is genre-close (cosine 1.0 to pierA) but the provider says 0.0 —
    the provider verdict must win and candB must be rejected."""
    Xn, g = _pair5_taxonomy()
    g = g.copy()
    g[2] = [1.0, 0.0, 0.0]  # candB's VECTOR now matches pierA exactly

    class _Prov:
        def sim(self, a, b):
            # candB (idx 2) is genre-far from everything; all else compatible
            return 0.0 if 2 in (a, b) else 1.0

    g_targets = [np.array([1.0, 0.0, 0.0])]
    cfg = _tax_cfg(genre_pair_floor=0.5)
    path, _h, _e, err = _beam_search_segment(
        0, 3, 1, [1, 2], Xn, Xn, None, None, None, g, cfg, 5,
        g_targets_override=g_targets, pair_sim_provider=_Prov(),
    )
    assert err is None
    assert path == [1], f"provider verdict must override vector cosine; got {path}"


def test_pair_floor_provider_none_is_exempt():
    """Provider returning None (uncovered track) exempts the edge."""
    Xn, g = _pair5_taxonomy()

    class _Prov:
        def sim(self, a, b):
            return None

    cfg = _tax_cfg(genre_pair_floor=0.9, weight_genre=0.0)
    path, _h, _e, err = _beam_search_segment(
        0, 3, 1, [2], Xn, Xn, None, None, None, g, cfg, 5,
        pair_sim_provider=_Prov(),
    )
    assert err is None
    assert path == [2]


def test_pair_floor_knob_resolves_per_mode():
    overrides = {"pier_bridge": {
        "genre_steering_enabled": True,
        "genre_pair_floor_narrow": 0.15,
        "genre_pair_floor_discover": 0.05,
    }}
    t, _ = resolve_pier_bridge_tuning(mode="narrow", similarity_floor=0.35, overrides=overrides)
    assert abs(t.genre_pair_floor - 0.15) < 1e-9
    t2, _ = resolve_pier_bridge_tuning(mode="discover", similarity_floor=0.35, overrides=overrides)
    assert abs(t2.genre_pair_floor - 0.05) < 1e-9
    # default off
    t3, _ = resolve_pier_bridge_tuning(mode="narrow", similarity_floor=0.35, overrides=None)
    assert t3.genre_pair_floor == 0.0
