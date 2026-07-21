import sqlite3
from pathlib import Path

import numpy as np
import pytest

# The re-bake resolves genres against the taxonomy/similarity data under data/,
# which is gitignored (not in the public repo / CI). Skip when it's absent.
_GENRE_DATA = Path(__file__).resolve().parents[2] / "data" / "genre_similarity.yaml"


@pytest.mark.slow
@pytest.mark.skipif(not _GENRE_DATA.exists(), reason="needs the genre taxonomy/similarity data (not in the public repo)")
def test_refresh_changes_genre_preserves_sonic(tmp_path, monkeypatch):
    """Re-bake updates X_genre for an edited album and leaves X_sonic intact."""
    import scripts.build_beat3tower_artifacts as bba

    art = tmp_path / "art.npz"
    sonic = np.random.RandomState(0).randn(2, 4).astype(np.float32)
    np.savez(
        art,
        X_sonic=sonic, X_sonic_raw=sonic,
        X_genre_raw=np.zeros((2, 1), dtype=np.float32),
        X_genre_smoothed=np.zeros((2, 1), dtype=np.float32),
        genre_vocab=np.array(["placeholder"], dtype=object),
        track_ids=np.array(["t1", "t2"], dtype=object),
        build_config=np.array({"genre_source": "graph"}, dtype=object),
    )

    db = tmp_path / "m.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        "CREATE TABLE tracks (track_id TEXT, artist TEXT, title TEXT, norm_artist TEXT, "
        " album TEXT, album_id TEXT, duration_ms INT);"
        "CREATE TABLE track_genres (track_id TEXT, genre TEXT);"
        "CREATE TABLE album_genres (album_id TEXT, genre TEXT);"
        "CREATE TABLE artist_genres (artist TEXT, genre TEXT);"
        "CREATE TABLE albums (album_id TEXT PRIMARY KEY, title TEXT, artist TEXT);"
        "CREATE TABLE release_effective_genres (album_id TEXT, release_key TEXT, genre_id TEXT, "
        " assignment_layer TEXT, confidence REAL, source TEXT);"
        "CREATE TABLE genre_graph_canonical_genres (genre_id TEXT PRIMARY KEY, name TEXT, kind TEXT, "
        " specificity_score REAL, status TEXT, taxonomy_version TEXT);"
    )
    conn.execute("INSERT INTO tracks VALUES ('t1','A','x','a','Alb','ALB1',1000)")
    conn.execute("INSERT INTO tracks VALUES ('t2','B','y','b','Alb2','ALB2',1000)")
    conn.execute("INSERT INTO release_effective_genres VALUES "
                 "('ALB1','k','dream_pop','observed_leaf',1.0,'user')")
    conn.execute("INSERT INTO genre_graph_canonical_genres VALUES "
                 "('dream_pop','Dream Pop','genre',0.8,'active','v1')")
    conn.commit()
    conn.close()

    rows = [
        {"track_id": "t1", "artist": "A", "title": "x", "norm_artist": "a",
         "album": "Alb", "album_id": "ALB1", "duration_ms": 1000},
        {"track_id": "t2", "artist": "B", "title": "y", "norm_artist": "b",
         "album": "Alb2", "album_id": "ALB2", "duration_ms": 1000},
    ]
    monkeypatch.setattr(
        bba, "load_tracks_with_beat3tower",
        lambda db_path, max_tracks=0: (rows, [{}, {}]),
    )

    out = bba.refresh_genre_matrices(
        str(art), str(db), genre_sim_path=None,
        sidecar_db=str(tmp_path / "s.db"), config_path="config.yaml",
    )
    data = np.load(art, allow_pickle=True)
    vocab_lower = [str(v).lower() for v in data["genre_vocab"].tolist()]
    assert "dream pop" in vocab_lower
    j = vocab_lower.index("dream pop")
    assert data["X_genre_raw"][0, j] > 0          # t1 (ALB1) has the genre
    assert np.array_equal(data["X_sonic_raw"], sonic)  # sonic preserved exactly
    assert out["n_tracks"] == 2
