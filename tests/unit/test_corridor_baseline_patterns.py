import importlib.util
import re
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "cb_patterns", Path(__file__).parents[2] / "scripts" / "corridor_baseline" / "patterns.py")
patterns = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(patterns)


def test_all_patterns_compile_and_declare_conditionality():
    assert len(patterns.F_PATTERNS) >= 15
    for name, spec in patterns.F_PATTERNS.items():
        re.compile(spec["regex"])
        assert "conditional_on" in spec, name


def test_segment_header_matches_real_format():
    line = "[Phase2] Segment 2→3: mode=dynamic, interior_length=4"
    assert re.search(patterns.F_PATTERNS["segment_header"]["regex"], line)


def test_segment_header_also_matches_unconditional_builder_line():
    # segment_header is intentionally an alternation: [Phase2] Segment ... only fires
    # under dj_bridging_enabled (default False, off in this corpus's config.yaml -- see
    # Layer 3 item 16 in CLAUDE.md, superseded by taxonomy steering), so the pattern
    # would be permanently absent if it only matched that clause. The unconditional
    # per-segment line (pier_bridge_builder.py:1795-1796, inside the bare
    # `for seg_idx in range(num_segments):` loop) is the one that's actually guaranteed
    # to appear on every pier-bridge generation.
    line = "Building segment 2: trk_a -> trk_b (interior=4)"
    assert re.search(patterns.F_PATTERNS["segment_header"]["regex"], line)


def test_ds_success_json_matches_real_payload():
    line = '{"pipeline": "ds", "mode": "dynamic", "metrics": {}, "effective": {}}'
    assert re.search(patterns.F_PATTERNS["ds_success_json"]["regex"], line)


def test_weakest_edges_matches_real_reporter_line():
    line = "Weakest transitions (bottom 3 by T):"
    assert re.search(patterns.F_PATTERNS["weakest_edges"]["regex"], line)


def test_solo_collab_split_matches_real_artist_style_line():
    line = "Artist style clustering scope: artist=Sade solo=12 collab=3 total=15"
    assert re.search(patterns.F_PATTERNS["solo_collab_split"]["regex"], line)


def test_recency_edge_diff_documents_dead_code():
    # log_recency_edge_diff (reporter.py:67) is defined but never called from
    # playlist_generator.py -- this pattern is expected to be permanently absent
    # regardless of config; conditional_on must say so rather than naming a knob.
    assert "DEAD CODE" in patterns.F_PATTERNS["recency_edge_diff"]["conditional_on"]


def test_extract_fingerprint_shape():
    log_text = (
        'noise\n'
        '{"pipeline": "ds", "mode": "dynamic", "metrics": {"min_transition": 0.5}, "effective": {}}\n'
        "Building segment 0: trkA -> trkB (interior=3)\n"
        "Building segment 1: trkB -> trkC (interior=2)\n"
        "Mini-piers: 1 waypoint(s) inserted (piers now 4)\n"
        "Tail-DP seg 0: window min 0.200 -> 0.300 (swapped [x] -> [y])\n"
        "Repair edge=3 pos=3 reason=below_floor old=1/a new=2/b old_worst_T=0.1 new_worst_T=0.3\n"
        "Artist style clustering scope: artist=Sade solo=10 collab=2 total=12\n"
        "Weakest transitions (bottom 3 by T):\n"
    )
    run = {
        "artist": "SADE", "detent": "home",
        "track_ids": ["1", "2", "3"], "err": None, "wall": 12.3,
        "min_transition": 0.5, "mean_transition": 0.7,
        "admitted": 40, "below_floor": 2, "distinct_artists": 5,
    }
    fp = patterns.extract_fingerprint(log_text, run)
    assert fp["artist"] == "SADE"
    assert fp["detent"] == "home"
    assert fp["n_tracks"] == 3
    assert fp["min_transition"] == 0.5
    assert fp["mean_transition"] == 0.7
    assert fp["below_floor"] == 2
    assert fp["distinct_artists"] == 5
    assert fp["admitted"] == 40
    assert fp["wall_s"] == 12.3
    assert fp["segments"] == 2
    assert fp["interior_lengths"] == [3, 2]
    assert fp["mini_pier_mentions"] == 1
    assert fp["tail_dp_fired"] == 1
    assert fp["edge_repair_fired"] == 1
    assert fp["solo_collab"] == "solo=10 collab=2"
    assert fp["reporting_presence"]["ds_success_json"] is True
    assert fp["reporting_presence"]["weakest_edges"] is True
    assert fp["reporting_presence"]["roam_segments"] is False
    assert set(fp["reporting_presence"]) == set(patterns.F_PATTERNS)


def test_extract_fingerprint_handles_absent_patterns():
    fp = patterns.extract_fingerprint("no matches at all\n", {"track_ids": []})
    assert fp["segments"] == 0
    assert fp["interior_lengths"] == []
    assert fp["mini_pier_mentions"] == 0
    assert fp["tail_dp_fired"] == 0
    assert fp["edge_repair_fired"] == 0
    assert fp["solo_collab"] is None
    assert all(v is False for v in fp["reporting_presence"].values())
