from src.playlist.mode_presets import resolve_pace_mode


def test_narrow_has_onset_and_softpenalty_keys_and_zero_rhythm_floor():
    s = resolve_pace_mode("narrow")
    assert s["onset_admission_max_log_distance"] == 0.50
    assert s["onset_bridge_max_log_distance"] == 0.60
    assert s["rhythm_soft_penalty_threshold"] == 0.25
    assert s["rhythm_soft_penalty_strength"] == 0.15
    # rhythm-cosine hard floors are disabled (soft now)
    assert s["admission_floor"] == 0.0
    assert s["bridge_floor"] == 0.0


def test_off_disables_everything():
    s = resolve_pace_mode("off")
    assert s["onset_admission_max_log_distance"] == float("inf")
    assert s["rhythm_soft_penalty_strength"] == 0.0
