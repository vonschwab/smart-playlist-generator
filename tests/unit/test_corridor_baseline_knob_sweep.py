"""Light unit coverage for the one pure piece of the knob-sweep orchestration:
the dead-outlets enabling-parent-flag annotation. Everything else in
capture_knob_sweep.py is engine-calling orchestration, exercised by the
smoke run (see task-5-report.md), not unit tests."""
import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "cb_knob_sweep", Path(__file__).parents[2] / "scripts" / "corridor_baseline" / "capture_knob_sweep.py")
knob_sweep = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(knob_sweep)


def test_finds_family_matched_disabled_flag():
    ref_flat = {
        "playlist.pier_config.dj_bridging_enabled": False,
        "playlist.pier_config.dj_genre_coverage_weight": 0.15,
        "playlist.pier_config.genre_steering_enabled": True,
    }
    assert (knob_sweep.find_enabling_parent("dj_genre_coverage_weight", ref_flat)
            == "playlist.pier_config.dj_bridging_enabled")


def test_prefers_longest_common_prefix_among_multiple_disabled_families():
    ref_flat = {
        "candidate_pool.genre_compatibility_enabled": False,
        "playlist.pier_config.genre_steering_enabled": False,
        "candidate_pool.genre_compatibility_penalty_strength": 0.5,
    }
    assert (knob_sweep.find_enabling_parent("genre_compatibility_penalty_strength", ref_flat)
            == "candidate_pool.genre_compatibility_enabled")


def test_no_match_when_no_disabled_sibling():
    ref_flat = {"playlist.pier_config.dj_bridging_enabled": True}
    assert knob_sweep.find_enabling_parent("dj_genre_coverage_weight", ref_flat) is None


def test_flag_itself_has_no_parent():
    ref_flat = {"playlist.pier_config.dj_bridging_enabled": False}
    assert knob_sweep.find_enabling_parent("dj_bridging_enabled", ref_flat) is None


# ---- Residue-fix: process_field's retry orchestration (mocked _attempt_config_path
# so this stays a fast unit test -- no real generation). Verifies the CONTROL FLOW
# (when a retry fires, when it doesn't, wall_s accumulation, config_path_used,
# did_not_resolve note attachment) independently of perturb.py's own path tables
# (already covered by test_corridor_baseline_perturb.py).

def test_process_field_retries_on_did_not_resolve_for_verified_field(monkeypatch):
    primary_record = {
        "field": "playlist.pier_config.weight_bridge", "config_path": "playlists.ds_pipeline.pier_bridge.weight_bridge",
        "baseline_value": 0.6, "perturbed_value": 0.0, "status": "did_not_resolve",
        "jaccard": None, "n_position_diffs": None, "delta_min_T": None, "delta_mean_T": None, "wall_s": 10.0,
    }
    retry_record = {
        "field": "playlist.pier_config.weight_bridge",
        "config_path": "playlists.ds_pipeline.artist_style.bridge_score_weights.dynamic.bridge",
        "baseline_value": 0.6, "perturbed_value": 0.0, "status": "inert",
        "jaccard": 1.0, "n_position_diffs": 0, "delta_min_T": 0.0, "delta_mean_T": 0.0, "wall_s": 12.0,
    }
    calls = []

    def fake_attempt(artist, detent, tag, field, baseline_value, ref, config_path, pv, log_level, attempt_suffix):
        calls.append((config_path, attempt_suffix))
        if attempt_suffix == "primary":
            return dict(primary_record), False
        return dict(retry_record), True

    monkeypatch.setattr(knob_sweep, "_attempt_config_path", fake_attempt)
    monkeypatch.setattr(knob_sweep, "config_path_for", lambda field: "playlists.ds_pipeline.pier_bridge.weight_bridge")
    monkeypatch.setattr(knob_sweep, "perturb_value", lambda field, value: 0.0)

    rec = knob_sweep.process_field("Bill Evans Trio", "open", "tag", "playlist.pier_config.weight_bridge", 0.6, {})

    assert [c[1] for c in calls] == ["primary", "retry"]
    assert rec["status"] == "inert"
    assert rec["config_path_used"] == "playlists.ds_pipeline.artist_style.bridge_score_weights.dynamic.bridge"
    assert rec["wall_s"] == 22.0  # 10.0 primary + 12.0 retry, accumulated


def test_process_field_no_retry_when_field_not_in_verified_set(monkeypatch):
    primary_record = {
        "field": "playlist.pier_config.mini_pier_enabled", "config_path": "playlists.ds_pipeline.pier_bridge.mini_pier_enabled",
        "baseline_value": True, "perturbed_value": False, "status": "did_not_resolve",
        "jaccard": None, "n_position_diffs": None, "delta_min_T": None, "delta_mean_T": None, "wall_s": 9.0,
    }
    calls = []

    def fake_attempt(artist, detent, tag, field, baseline_value, ref, config_path, pv, log_level, attempt_suffix):
        calls.append(attempt_suffix)
        return dict(primary_record), False

    monkeypatch.setattr(knob_sweep, "_attempt_config_path", fake_attempt)
    monkeypatch.setattr(knob_sweep, "config_path_for", lambda field: "playlists.ds_pipeline.pier_bridge.mini_pier_enabled")
    monkeypatch.setattr(knob_sweep, "perturb_value", lambda field, value: False)

    rec = knob_sweep.process_field("Bill Evans Trio", "open", "tag", "playlist.pier_config.mini_pier_enabled", True, {})

    assert calls == ["primary"]  # no retry -- mini_pier_enabled is not in the verified retry set
    assert rec["status"] == "did_not_resolve"
    assert "config_path_used" not in rec


def test_process_field_attaches_note_for_context_gated_survivor(monkeypatch):
    primary_record = {
        "field": "playlist.pier_config.experiment_bridge_scoring_enabled",
        "config_path": "playlists.ds_pipeline.pier_bridge.experiments.bridge_scoring.enabled",
        "baseline_value": False, "perturbed_value": True, "status": "did_not_resolve",
        "jaccard": None, "n_position_diffs": None, "delta_min_T": None, "delta_mean_T": None, "wall_s": 9.0,
    }

    def fake_attempt(artist, detent, tag, field, baseline_value, ref, config_path, pv, log_level, attempt_suffix):
        return dict(primary_record), False

    monkeypatch.setattr(knob_sweep, "_attempt_config_path", fake_attempt)
    monkeypatch.setattr(
        knob_sweep, "config_path_for",
        lambda field: "playlists.ds_pipeline.pier_bridge.experiments.bridge_scoring.enabled",
    )
    monkeypatch.setattr(knob_sweep, "perturb_value", lambda field, value: True)

    rec = knob_sweep.process_field(
        "Bill Evans Trio", "open", "tag", "playlist.pier_config.experiment_bridge_scoring_enabled", False, {},
    )

    assert rec["status"] == "did_not_resolve"
    assert "dry_run/audit" in rec["note"]
