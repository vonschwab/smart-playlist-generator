import sqlite3

from src.genre.authority import on_tag_track_ids_for_artist


def _db():
    c = sqlite3.connect(":memory:")
    c.executescript(
        """
        CREATE TABLE tracks (track_id TEXT, artist TEXT, album_id TEXT);
        CREATE TABLE release_effective_genres (album_id TEXT, genre_id TEXT, assignment_layer TEXT);
        INSERT INTO tracks VALUES ('t1','Boards of Canada','alb_h'),
                                  ('t2','Boards of Canada','alb_h'),
                                  ('t3','Boards of Canada','alb_e'),
                                  ('t4','Aphex Twin','alb_a');
        INSERT INTO release_effective_genres VALUES
            ('alb_h','hauntology','observed_leaf'),
            ('alb_h','kosmische','observed_leaf'),
            ('alb_e','electronica','observed_leaf'),
            ('alb_e','hauntology','inferred_family'),  -- inferred: must NOT count
            ('alb_a','hauntology','observed_leaf');
        """
    )
    return c


def test_on_tag_union_and_hitcount():
    c = _db()
    m = on_tag_track_ids_for_artist(c, "boards of canada", {"hauntology", "kosmische"})
    assert m == {"t1": 2, "t2": 2}          # union; both tags hit; t3 inferred-only excluded


def test_on_tag_single_genre_and_case_insensitive_artist():
    c = _db()
    assert on_tag_track_ids_for_artist(c, "BOARDS OF CANADA", {"hauntology"}) == {"t1": 1, "t2": 1}


def test_on_tag_empty_inputs():
    c = _db()
    assert on_tag_track_ids_for_artist(c, "Boards of Canada", set()) == {}
    assert on_tag_track_ids_for_artist(c, "Nobody", {"hauntology"}) == {}
