# tests/unit/test_pace_audition_analyze.py
from scripts.pace_audition_analyze import (
    distribution,
    join_scores,
    per_arm,
    discrimination_ok,
    confound_flag,
)


def test_distribution_reports_min_p10_p50_p90_n():
    d = distribution([1, 2, 3, 4, 5])
    assert d["n"] == 5
    assert d["min"] == 1
    assert d["p50"] == 3


def test_distribution_handles_empty():
    d = distribution([])
    assert d["n"] == 0
    assert d["p50"] is None


def test_join_scores_attaches_arm_and_regime():
    edge_data = {"e1": {"arm": "narrow", "seed": "s", "regime": "ambient"}}
    captures = [{"edge_id": "e1", "continuity": 4, "smoothness": 3}]
    joined = join_scores(captures, edge_data)
    assert joined[0]["arm"] == "narrow"
    assert joined[0]["regime"] == "ambient"
    assert joined[0]["continuity"] == 4


def test_per_arm_groups_continuity():
    joined = [
        {"arm": "narrow", "continuity": 5, "smoothness": 4, "regime": "ambient"},
        {"arm": "decoy", "continuity": 1, "smoothness": 1, "regime": "ambient"},
    ]
    out = per_arm(joined, "continuity")
    assert out["narrow"]["p50"] == 5
    assert out["decoy"]["p50"] == 1


def test_discrimination_ok_requires_decoy_lowest():
    good = {"narrow": {"p50": 4}, "dynamic": {"p50": 4}, "off": {"p50": 3}, "decoy": {"p50": 1}}
    bad = {"narrow": {"p50": 3}, "dynamic": {"p50": 3}, "off": {"p50": 3}, "decoy": {"p50": 3}}
    assert discrimination_ok(good) is True
    assert discrimination_ok(bad) is False


def test_confound_flag_true_when_continuity_gain_exceeds_smoothness_gain():
    # narrow beats dynamic more on continuity than on smoothness -> pace-specific
    joined = [
        {"arm": "narrow", "continuity": 5, "smoothness": 4},
        {"arm": "dynamic", "continuity": 2, "smoothness": 3.5},
    ]
    res = confound_flag(joined)
    assert res["pace_specific"] is True
    assert res["continuity_gain"] > res["smoothness_gain"]


from scripts.pace_audition_analyze import onset_variance_by_arm


def test_onset_variance_by_arm_lower_for_flatter_playlist():
    playlists = [
        {"seed": "s", "arm": "narrow", "onset_seq": [2.0, 2.0, 2.1, 2.0]},
        {"seed": "s", "arm": "off", "onset_seq": [1.0, 6.0, 0.5, 8.0]},
    ]
    out = onset_variance_by_arm(playlists)
    assert out["narrow"] < out["off"]
