import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "cb_perturb", Path(__file__).parents[2] / "scripts" / "corridor_baseline" / "perturb.py")
perturb = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(perturb)


def test_flatten_leaves_nested():
    blob = {"candidate_pool": {"similarity_floor": 0.3, "broad_filters": {"enabled": True}},
            "playlist": {"weights": [1, 2]}}
    flat = perturb.flatten_leaves(blob)
    assert flat["candidate_pool.similarity_floor"] == 0.3
    assert flat["candidate_pool.broad_filters.enabled"] is True
    assert flat["playlist.weights"] == [1, 2]  # lists are leaves


def test_config_path_prefix_mapping():
    assert (perturb.config_path_for("candidate_pool.similarity_floor")
            == "playlists.ds_pipeline.candidate_pool.similarity_floor")
    assert (perturb.config_path_for("playlist.transition_floor")
            == "playlists.ds_pipeline.pier_bridge.transition_floor")
    assert perturb.config_path_for("sonic_variant.name") is None
    assert perturb.config_path_for("embedding.whatever") is None


def test_config_path_strips_pier_config_infix():
    # Step-5 real-blob fix: the effective blob nests PierBridgeConfig fields
    # under playlist.pier_config.<x>, but config.yaml stores them flat under
    # playlists.ds_pipeline.pier_bridge.<x> (no "pier_config" key). Without
    # stripping this infix, ~180 leaves round-tripped to a nonexistent path.
    assert (perturb.config_path_for("playlist.pier_config.weight_bridge")
            == "playlists.ds_pipeline.pier_bridge.weight_bridge")
    assert (perturb.config_path_for("playlist.pier_config.mini_pier_enabled")
            == "playlists.ds_pipeline.pier_bridge.mini_pier_enabled")


def test_perturb_bool_flips():
    assert perturb.perturb_value("some_enabled", True) is False
    assert perturb.perturb_value("some_enabled", False) is True


def test_perturb_int_increments():
    assert perturb.perturb_value("initial_beam_width", 24) == 25


def test_perturb_float_scales():
    # NOTE: "weight_bridge" is deliberately NOT used here — it's a contract
    # Category C11 field in C_TERM_FIELDS (see test_c_term_fields_zeroed) and
    # would be zeroed, not scaled. Use a non-C-term float field name instead.
    assert perturb.perturb_value("generic_scale_weight", 0.4) == 0.6000000000000001


def test_perturb_zero_float_gets_nonzero():
    assert perturb.perturb_value("some_weight", 0.0) == 0.3


def test_perturb_percentile_like_never_exceeds_one():
    v = perturb.perturb_value("sonic_admission_percentile", 0.9)
    assert v == 0.45  # x1.5 would exceed 1.0 -> halve instead


def test_c_term_fields_zeroed():
    assert "duration_penalty_weight" in perturb.C_TERM_FIELDS
    assert perturb.perturb_value("duration_penalty_weight", 0.6) == 0.0


def test_strings_lists_none_skip():
    assert perturb.perturb_value("dj_route_shape", "arc") is perturb.SKIP
    assert perturb.perturb_value("weights", [1, 2]) is perturb.SKIP
    assert perturb.perturb_value("thing", None) is perturb.SKIP


# ---- Fix-wave: nested pier_bridge sub-block mapping (config_path_for) --------
# config.yaml nests ~11 PierBridgeConfig sub-blocks (dj_bridging, edge_repair,
# edge_delete, progress, tail_dp, roam, ...) under a family key that does NOT
# match the flat playlist.pier_config.<leaf> blob path -- see
# src/playlist/pipeline/pier_bridge_overrides.py::apply_pier_bridge_overrides,
# which reads pb_overrides.get("<subblock>") for these fields. Before the fix,
# config_path_for mapped every leaf to the flat
# playlists.ds_pipeline.pier_bridge.<leaf> path regardless of family, which for
# these fields produces a yaml key production never reads (confirmed via a real
# generation: dj_genre_coverage_weight came back did_not_resolve).

def test_config_path_maps_dj_bridging_family_nested():
    # dj_bridging: Phase-2/3 sub-keys keep their "dj_" prefix even nested
    # (pier_bridge_overrides.py:735: dj_raw.get("dj_genre_coverage_weight")).
    assert (
        perturb.config_path_for("playlist.pier_config.dj_genre_coverage_weight")
        == "playlists.ds_pipeline.pier_bridge.dj_bridging.dj_genre_coverage_weight"
    )
    # dj_bridging: other sub-keys DROP the "dj_" prefix once nested under a
    # named sub-block (pier_bridge_overrides.py:594: pooling_raw.get("k_local")).
    assert (
        perturb.config_path_for("playlist.pier_config.dj_pooling_k_local")
        == "playlists.ds_pipeline.pier_bridge.dj_bridging.pooling.k_local"
    )
    assert (
        perturb.config_path_for("playlist.pier_config.dj_bridging_enabled")
        == "playlists.ds_pipeline.pier_bridge.dj_bridging.enabled"
    )


def test_config_path_maps_edge_repair_family_nested():
    assert (
        perturb.config_path_for("playlist.pier_config.edge_repair_enabled")
        == "playlists.ds_pipeline.pier_bridge.edge_repair.enabled"
    )
    assert (
        perturb.config_path_for("playlist.pier_config.edge_repair_variety_guard_threshold")
        == "playlists.ds_pipeline.pier_bridge.edge_repair.variety_guard.threshold"
    )


def test_config_path_maps_roam_family_nested():
    assert (
        perturb.config_path_for("playlist.pier_config.roam_width_sonic")
        == "playlists.ds_pipeline.pier_bridge.roam.width_sonic"
    )
    # worst_edge_minimax_enabled has NO flat alias -- roam_kwargs_from_dict is
    # the sole reader (src/playlist/pier_bridge/config.py:502).
    assert (
        perturb.config_path_for("playlist.pier_config.worst_edge_minimax_enabled")
        == "playlists.ds_pipeline.pier_bridge.roam.worst_edge_minimax"
    )


def test_config_path_maps_progress_and_tail_dp_families_nested():
    assert (
        perturb.config_path_for("playlist.pier_config.progress_penalty_weight")
        == "playlists.ds_pipeline.pier_bridge.progress.penalty_weight"
    )
    assert (
        perturb.config_path_for("playlist.pier_config.tail_dp_floor")
        == "playlists.ds_pipeline.pier_bridge.tail_dp.floor"
    )


def test_config_path_cross_family_redirects():
    # pipeline/core.py:865-888 unconditionally clobbers pb_cfg.pace_bridge_floor
    # from cfg.candidate.pace_bridge_floor (a CandidatePoolConfig field) after
    # apply_pier_bridge_overrides returns -- any pier_bridge.pace_bridge_floor
    # override is silently overwritten, so the real path is candidate_pool.*.
    assert (
        perturb.config_path_for("playlist.pier_config.pace_bridge_floor")
        == "playlists.ds_pipeline.candidate_pool.pace_bridge_floor"
    )
    # center_transitions/transition_floor are sourced from constraints.*
    # (src/playlist/config.py::default_ds_config), never from pier_bridge.*.
    assert (
        perturb.config_path_for("playlist.pier_config.center_transitions")
        == "playlists.ds_pipeline.constraints.center_transitions"
    )
    assert (
        perturb.config_path_for("playlist.pier_config.transition_floor")
        == "playlists.ds_pipeline.constraints.transition_floor"
    )


def test_config_path_base_name_mismatch_soft_genre_penalty():
    # resolve_pier_bridge_tuning reads "soft_genre_penalty_threshold/strength",
    # never the blob leaf names "genre_penalty_threshold/strength" at any
    # priority (src/playlist/config.py:328-336).
    assert (
        perturb.config_path_for("playlist.pier_config.genre_penalty_threshold")
        == "playlists.ds_pipeline.pier_bridge.soft_genre_penalty_threshold"
    )
    assert (
        perturb.config_path_for("playlist.pier_config.genre_penalty_strength")
        == "playlists.ds_pipeline.pier_bridge.soft_genre_penalty_strength"
    )


def test_config_path_genuine_dead_outlets_map_to_none():
    # Populated ONLY from mode_presets.PACE_MODE_PRESETS; resolve_pace_mode is
    # called without an overrides param in pipeline/core.py:362, so no yaml
    # path can reach these -- honestly mapped to None (unmapped), not a flat
    # path that will always read did_not_resolve.
    assert perturb.config_path_for("playlist.pier_config.bpm_bridge_max_log_distance") is None
    # Never assigned from any override dict anywhere in src/ (dataclass-default
    # only): weight_end_start / transition_calib_* / eta_destination_pull.
    assert perturb.config_path_for("playlist.pier_config.weight_end_start") is None
    assert perturb.config_path_for("playlist.pier_config.transition_calib_center") is None


def test_config_path_flat_family_a_fields_unaffected():
    # Fields whose flat dataclass-field-name IS the correct, working yaml path
    # (pb_overrides.get("<field>") direct) must still resolve via the plain
    # default -- the fix must not regress the ~32 already-correct flat fields.
    assert (
        perturb.config_path_for("playlist.pier_config.mini_pier_enabled")
        == "playlists.ds_pipeline.pier_bridge.mini_pier_enabled"
    )
    assert (
        perturb.config_path_for("playlist.pier_config.instrumental_penalty_weight")
        == "playlists.ds_pipeline.pier_bridge.instrumental_penalty_weight"
    )
