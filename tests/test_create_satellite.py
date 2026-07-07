"""Unit tests for the satellite bootstrap's pure functions (config rewrite, memory key)."""

import importlib.util
import pathlib

_MOD = pathlib.Path(__file__).resolve().parents[1] / "tools" / "create_satellite.py"
_spec = importlib.util.spec_from_file_location("create_satellite", _MOD)
assert _spec is not None and _spec.loader is not None
cs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cs)

CANON = pathlib.Path("C:/Users/Dylan/Desktop/PLAYLIST_GENERATOR_V3")
DB_LINE = "C:/Users/Dylan/Desktop/PLAYLIST_GENERATOR_V3/data/metadata.db"
ART_LINE = "C:/Users/Dylan/Desktop/PLAYLIST_GENERATOR_V3/data/artifacts/beat3tower_32k/data_matrices_step1.npz"


def test_db_path_rewritten_and_comments_preserved():
    text = (
        "library:\n"
        "  music_directory: E:\\MUSIC  # my library\n"
        "  database_path: data/metadata.db\n"
        "# a comment that must survive\n"
    )
    out = cs.rewrite_config_text(text, CANON)
    assert f"  database_path: {DB_LINE}\n" in out
    assert "# a comment that must survive" in out
    assert "music_directory: E:\\MUSIC  # my library" in out


def test_artifact_path_replaced_when_present():
    text = (
        "library:\n  database_path: data/metadata.db\n"
        "playlists:\n  ds_pipeline:\n    artifact_path: data/artifacts/beat3tower_32k/data_matrices_step1.npz\n"
    )
    out = cs.rewrite_config_text(text, CANON)
    assert f"    artifact_path: {ART_LINE}\n" in out


def test_artifact_key_inserted_under_existing_ds_pipeline():
    text = "playlists:\n  ds_pipeline:\n    enabled: true\nlibrary:\n  database_path: data/metadata.db\n"
    out = cs.rewrite_config_text(text, CANON)
    assert f"    artifact_path: {ART_LINE}\n" in out
    # inserted directly after the ds_pipeline: line
    assert out.index("ds_pipeline:") < out.index("artifact_path:") < out.index("enabled: true")


def test_playlists_block_appended_when_absent():
    text = "library:\n  database_path: data/metadata.db\n"
    out = cs.rewrite_config_text(text, CANON)
    assert "playlists:\n  ds_pipeline:\n    artifact_path: " + ART_LINE in out


def test_memory_project_key_matches_harness_munging():
    # Known ground truth: this project's own memory dir name.
    assert (
        cs.memory_project_key("C:\\Users\\Dylan\\Desktop\\PLAYLIST_GENERATOR_V3")
        == "C--Users-Dylan-Desktop-PLAYLIST-GENERATOR-V3"
    )
    assert (
        cs.memory_project_key("C:\\Users\\Dylan\\Desktop\\PG3_SAT1")
        == "C--Users-Dylan-Desktop-PG3-SAT1"
    )


def test_memory_pointer_names_canonical_index():
    text = cs.memory_pointer_text(CANON)
    assert "C--Users-Dylan-Desktop-PLAYLIST-GENERATOR-V3" in text
    assert "MEMORY.md" in text
