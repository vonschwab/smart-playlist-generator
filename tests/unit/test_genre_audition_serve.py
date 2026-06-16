import yaml

from scripts.research.genre_audition_serve import _append_capture_entry, _blind_manifest


def test_append_creates_file(tmp_path):
    p = tmp_path / "cap.yaml"
    _append_capture_entry(p, {"name": "shoegaze", "verdict": "same", "notes": "x"})
    data = yaml.safe_load(p.read_text())
    assert len(data["entries"]) == 1
    assert data["entries"][0]["name"] == "shoegaze"


def test_append_updates_existing(tmp_path):
    p = tmp_path / "cap.yaml"
    _append_capture_entry(p, {"name": "shoegaze", "verdict": "same", "notes": "a"})
    _append_capture_entry(p, {"name": "shoegaze", "verdict": "loose", "notes": "b"})
    data = yaml.safe_load(p.read_text())
    assert len(data["entries"]) == 1
    assert data["entries"][0]["verdict"] == "loose"


def test_append_adds_new(tmp_path):
    p = tmp_path / "cap.yaml"
    _append_capture_entry(p, {"name": "a", "verdict": "same", "notes": ""})
    _append_capture_entry(p, {"name": "b", "verdict": "loose", "notes": ""})
    data = yaml.safe_load(p.read_text())
    assert len(data["entries"]) == 2


def test_blind_manifest_strips_hidden_fields():
    m = {
        "slug": "x", "seed": {"genre": "x", "artists": []},
        "cooc_token": "x raw", "neighbors": [{"name": "n", "artists": []}],
        "space_data": {"n": {"graph": {"rank": 1, "sim": 0.5}}},
    }
    blinded = _blind_manifest(m)
    assert "space_data" not in blinded
    assert "cooc_token" not in blinded
    assert blinded["neighbors"] == [{"name": "n", "artists": []}]
