import sqlite3
from scripts.research import pace_eval_corpus as c


def _db(rows):
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE tracks (track_id TEXT, artist TEXT, album TEXT, file_path TEXT)")
    conn.executemany("INSERT INTO tracks VALUES (?,?,?,?)", rows)
    return conn


def test_resolve_orders_by_filename_trackno_and_usable_range():
    conn = _db([
        ("t2", "Daft Punk", "Discovery", "/m/02 - Aerodynamic.flac"),
        ("t1", "Daft Punk", "Discovery", "/m/01 - One More Time.flac"),
        ("t3", "Daft Punk", "Discovery", "/m/03 - Digital Love.flac"),
    ])
    spec = c.AlbumSpec(key="disc", artist_like="%Daft Punk%", album_like="%Discovery%",
                       flow_type="gradient_flow", register="high", usable_first=1, usable_last=2)
    tracks, counts = c.resolve_corpus(conn, [spec])
    assert [t.track_id for t in tracks] == ["t1", "t2"]  # ordered, t3 outside usable range
    assert counts["disc"] == 2


def test_build_pairs_adjacent_nonadjacent_and_cross_register():
    tracks = [
        c.CorpusTrack("a1", "A", 1, "gradient_flow", "high"),
        c.CorpusTrack("a2", "A", 2, "gradient_flow", "high"),
        c.CorpusTrack("a3", "A", 3, "gradient_flow", "high"),
        c.CorpusTrack("b1", "B", 1, "tight_continuous", "low"),
        c.CorpusTrack("b2", "B", 2, "tight_continuous", "low"),
    ]
    pairs = c.build_pairs(tracks, seed=1, n_random=50)
    assert ("a1", "a2") in pairs["adjacent"] and ("a2", "a3") in pairs["adjacent"]
    assert ("b1", "b2") in pairs["adjacent"]
    # non-adjacent only from gradient album A (a1,a3); none from tight album B
    assert ("a1", "a3") in pairs["non_adjacent_same_album"]
    assert all(p not in pairs["non_adjacent_same_album"] for p in [("b1", "b2")])
    # adjacent_gradient: only adjacent pairs from gradient_flow albums (A only, not B)
    assert ("a1", "a2") in pairs["adjacent_gradient"]
    assert ("a2", "a3") in pairs["adjacent_gradient"]
    assert ("b1", "b2") not in pairs["adjacent_gradient"]
    # random_cross pairs are cross-register only
    reg = {t.track_id: t.register for t in tracks}
    assert all(reg[x] != reg[y] for x, y in pairs["random_cross"])
