"""_build_edge_audit_rows must carry the builder's per-edge BPM diagnostics.

Regression: the audit row dict was built from an explicit key list that
omitted bpm_a/bpm_b/bpm_log_dist, so emit_selected_edge_audit rendered
"bpm=n/a" on every edge even when BPM data was fully loaded.
"""
from src.playlist_generator import _build_edge_audit_rows


def _edges_and_tracks():
    edges = [
        {
            "prev_idx": 1,
            "cur_idx": 2,
            "T": 0.55,
            "T_centered_cos": 0.1,
            "S": 0.2,
            "G": 0.8,
            "bpm_a": 92.0,
            "bpm_b": 118.0,
            "bpm_log_dist": 0.359,
        },
    ]
    tracks = [
        {"artist": "A", "title": "t1"},
        {"artist": "B", "title": "t2"},
    ]
    return edges, tracks


def test_bpm_fields_carried_into_audit_rows():
    edges, tracks = _edges_and_tracks()
    rows = _build_edge_audit_rows(edges, tracks)
    assert len(rows) == 1
    row = rows[0]
    assert row["bpm_a"] == 92.0
    assert row["bpm_b"] == 118.0
    assert row["bpm_log_dist"] == 0.359


def test_missing_bpm_stays_none():
    edges, tracks = _edges_and_tracks()
    for k in ("bpm_a", "bpm_b", "bpm_log_dist"):
        edges[0].pop(k)
    rows = _build_edge_audit_rows(edges, tracks)
    assert rows[0]["bpm_a"] is None
    assert rows[0]["bpm_b"] is None
    assert rows[0]["bpm_log_dist"] is None
