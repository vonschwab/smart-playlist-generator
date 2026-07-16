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
