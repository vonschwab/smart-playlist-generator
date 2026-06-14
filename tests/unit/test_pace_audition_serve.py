# tests/unit/test_pace_audition_serve.py
from scripts.pace_audition_serve import blinded_manifest, upsert_capture_entry


def test_blinded_manifest_strips_all_server_side_keys():
    manifest = {
        "type": "pace_edges",
        "provenance": {"arms": {"narrow": {}}},
        "playlists": [{"seed": "s", "arm": "narrow"}],
        "edges": [{"edge_id": "e0001", "a": "ta", "b": "tb"}],
        "edge_data": {"e0001": {"arm": "narrow"}},
        "file_paths": {"ta": "x"},
    }
    served = blinded_manifest(manifest)
    assert set(served.keys()) == {"type", "edges"}
    assert "narrow" not in json_dumps(served)


def json_dumps(o):
    import json
    return json.dumps(o)


def test_upsert_capture_entry_dedupes_by_edge_id():
    entries = [{"edge_id": "e1", "continuity": 3, "smoothness": 3}]
    upsert_capture_entry(entries, {"edge_id": "e1", "continuity": 5, "smoothness": 4})
    upsert_capture_entry(entries, {"edge_id": "e2", "continuity": 2, "smoothness": 2})
    assert len(entries) == 2
    assert entries[0]["continuity"] == 5  # overwritten in place
