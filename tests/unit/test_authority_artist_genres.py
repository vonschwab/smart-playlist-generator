"""Authority-side aggregation of an artist's published genres (tag-steering chips)."""
import sqlite3

import pytest

from src.genre.authority import ArtistGenreTag, resolved_genres_for_artist


@pytest.fixture()
def conn(tmp_path):
    db = sqlite3.connect(tmp_path / "meta.db")
    db.executescript(
        """
        CREATE TABLE tracks (track_id TEXT PRIMARY KEY, artist TEXT, album_id TEXT);
        CREATE TABLE release_effective_genres (
            album_id TEXT, release_key TEXT, genre_id TEXT,
            assignment_layer TEXT, confidence REAL, source TEXT
        );
        CREATE TABLE genre_graph_canonical_genres (genre_id TEXT, name TEXT);
        """
    )
    db.executemany(
        "INSERT INTO tracks VALUES (?, ?, ?)",
        [
            ("t1", "Herbie Hancock", "alb1"),
            ("t2", "Herbie Hancock", "alb1"),
            ("t3", "Herbie Hancock", "alb2"),
            ("t4", "Aretha Franklin", "alb3"),
            ("t5", "Herbie Hancock", None),  # albumless track must not crash
        ],
    )
    db.executemany(
        "INSERT INTO release_effective_genres VALUES (?, '', ?, ?, ?, 'graph')",
        [
            ("alb1", "g-jazzfunk", "observed_leaf", 0.9),
            ("alb2", "g-jazzfunk", "observed_leaf", 0.8),
            ("alb1", "g-postbop", "observed_leaf", 0.7),
            ("alb1", "g-jazz", "inferred_family", 0.95),   # hub family: excluded
            ("alb2", "g-fusion", "legacy", 0.6),            # legacy layer: included
            ("alb3", "g-soul", "observed_leaf", 0.9),       # other artist: excluded
        ],
    )
    db.executemany(
        "INSERT INTO genre_graph_canonical_genres VALUES (?, ?)",
        [("g-jazzfunk", "jazz-funk"), ("g-postbop", "post-bop"),
         ("g-jazz", "jazz"), ("g-fusion", "jazz fusion"), ("g-soul", "soul")],
    )
    db.commit()
    yield db
    db.close()


def test_aggregates_observed_leaf_and_legacy_across_releases(conn):
    tags = resolved_genres_for_artist(conn, "Herbie Hancock")
    names = [t.name for t in tags]
    assert names[0] == "jazz-funk"                       # 2 releases: strongest first
    assert set(names) == {"jazz-funk", "post-bop", "jazz fusion"}
    jf = tags[0]
    assert jf == ArtistGenreTag("g-jazzfunk", "jazz-funk", 2, 0.9)


def test_excludes_inferred_families_and_other_artists(conn):
    names = {t.name for t in resolved_genres_for_artist(conn, "Herbie Hancock")}
    assert "jazz" not in names   # inferred_family carries no steering signal
    assert "soul" not in names   # different artist


def test_artist_match_is_case_insensitive_exact(conn):
    assert resolved_genres_for_artist(conn, "  herbie hancock ")
    assert resolved_genres_for_artist(conn, "Herbie") == []  # no substring matching


def test_unknown_artist_and_missing_tables_return_empty(conn, tmp_path):
    assert resolved_genres_for_artist(conn, "Nobody") == []
    bare = sqlite3.connect(tmp_path / "bare.db")
    assert resolved_genres_for_artist(bare, "Herbie Hancock") == []
    bare.close()
