import yaml

from scripts.research.sonic_audition_serve import _parse_range_header, _append_capture_entry


def test_range_full():
    start, end = _parse_range_header("bytes=0-999", 1000)
    assert start == 0 and end == 999


def test_range_from_offset():
    start, end = _parse_range_header("bytes=500-", 1000)
    assert start == 500 and end == 999


def test_range_empty_header():
    start, end = _parse_range_header("", 1000)
    assert start == 0 and end == 999


def test_range_clamps_to_file_size():
    start, end = _parse_range_header("bytes=0-9999", 100)
    assert end == 99


def test_append_creates_file(tmp_path):
    p = tmp_path / "cap.yaml"
    _append_capture_entry(p, {"track_id": "t1", "verdict": "match", "notes": "great"})
    data = yaml.safe_load(p.read_text())
    assert len(data["entries"]) == 1
    assert data["entries"][0]["track_id"] == "t1"


def test_append_updates_existing(tmp_path):
    p = tmp_path / "cap.yaml"
    _append_capture_entry(p, {"track_id": "t1", "verdict": "match", "notes": "a"})
    _append_capture_entry(p, {"track_id": "t1", "verdict": "close", "notes": "b"})
    data = yaml.safe_load(p.read_text())
    assert len(data["entries"]) == 1
    assert data["entries"][0]["verdict"] == "close"


def test_append_adds_new(tmp_path):
    p = tmp_path / "cap.yaml"
    _append_capture_entry(p, {"track_id": "t1", "verdict": "match", "notes": ""})
    _append_capture_entry(p, {"track_id": "t2", "verdict": "off", "notes": "nope"})
    data = yaml.safe_load(p.read_text())
    assert len(data["entries"]) == 2
