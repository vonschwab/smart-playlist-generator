# tests/unit/test_setup_state.py
"""Setup-state derivation: no config / config w/o music dir / empty DB / ready."""
import sqlite3

from src.mixarc.paths import MixarcHome
from src.playlist_web.setup_state import SetupState, derive_setup_state


def _home(tmp_path):
    return MixarcHome(config_path=tmp_path / "config.yaml", anchor_dir=tmp_path, source="env")


def test_no_config(tmp_path):
    st = derive_setup_state(_home(tmp_path))
    assert st.state == SetupState.NEEDS_SETUP
    assert st.config_exists is False


def test_config_without_music_dir(tmp_path):
    (tmp_path / "config.yaml").write_text("library: {}\n", encoding="utf-8")
    st = derive_setup_state(_home(tmp_path))
    assert st.state == SetupState.NEEDS_SETUP


def test_config_ok_but_no_db(tmp_path):
    music = tmp_path / "music"
    music.mkdir()
    (tmp_path / "config.yaml").write_text(
        f"library:\n  music_directory: {music.as_posix()}\n  database_path: data/metadata.db\n",
        encoding="utf-8")
    st = derive_setup_state(_home(tmp_path))
    assert st.state == SetupState.NEEDS_ANALYZE
    assert st.track_count in (None, 0)


def test_ready(tmp_path):
    music = tmp_path / "music"
    music.mkdir()
    dbdir = tmp_path / "data"
    dbdir.mkdir()
    db = dbdir / "metadata.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE tracks (track_id TEXT PRIMARY KEY)")
    conn.execute("INSERT INTO tracks VALUES ('t1')")
    conn.commit()
    conn.close()
    (tmp_path / "config.yaml").write_text(
        f"library:\n  music_directory: {music.as_posix()}\n  database_path: data/metadata.db\n",
        encoding="utf-8")
    st = derive_setup_state(_home(tmp_path))
    assert st.state == SetupState.READY
    assert st.track_count == 1
