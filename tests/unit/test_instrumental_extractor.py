import json
import pytest
from scripts.extract_instrumental_sidecar import voice_column_index


def _write_json(tmp_path, classes):
    p = tmp_path / "voice_instrumental.json"
    p.write_text(json.dumps({"classes": classes}), encoding="utf-8")
    return str(p)


def test_voice_column_index_voice_second(tmp_path):
    # Essentia zoo order is typically ["instrumental", "voice"]
    assert voice_column_index(_write_json(tmp_path, ["instrumental", "voice"])) == 1


def test_voice_column_index_voice_first(tmp_path):
    assert voice_column_index(_write_json(tmp_path, ["voice", "instrumental"])) == 0


def test_voice_column_index_ambiguous_raises(tmp_path):
    with pytest.raises(ValueError):
        voice_column_index(_write_json(tmp_path, ["classA", "classB"]))
