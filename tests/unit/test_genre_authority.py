import sqlite3
from src.genre import genre_publish, authority


def _published_db(tmp_path):
    meta = tmp_path / "metadata.db"
    conn = sqlite3.connect(meta)
    conn.executescript(
        """
        CREATE TABLE albums (album_id TEXT PRIMARY KEY, title TEXT, artist TEXT);
        CREATE TABLE tracks (track_id TEXT PRIMARY KEY, artist TEXT, album TEXT,
                             album_id TEXT, norm_artist TEXT);
        CREATE TABLE track_genres (track_id TEXT, genre TEXT, source TEXT, weight REAL);
        CREATE TABLE album_genres (album_id TEXT, genre TEXT, source TEXT);
        CREATE TABLE artist_genres (artist TEXT, genre TEXT, source TEXT);
        INSERT INTO albums VALUES ('ALB1', 'York Blvd', 'Acetone');
        INSERT INTO albums VALUES ('ALB2', 'B', 'Y');
        INSERT INTO album_genres VALUES ('ALB2', 'Slowcore', 'discogs_release');
        INSERT INTO tracks VALUES ('T1', 'Acetone', 'York Blvd', 'ALB1', 'acetone');
        """
    )
    conn.commit()
    conn.close()
    from src.ai_genre_enrichment.storage import SidecarStore
    from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy
    side = tmp_path / "sidecar.db"
    store = SidecarStore(str(side))
    store.initialize()
    store.upsert_layered_taxonomy(load_default_layered_taxonomy())
    sconn = sqlite3.connect(side)
    sconn.execute(
        "INSERT INTO enriched_genre_signatures "
        "(release_key, normalized_artist, normalized_album, album_id, signature_json, updated_at) "
        "VALUES ('acetone::york blvd','acetone','york blvd','ALB1','{}','t')")
    sconn.execute(
        "INSERT INTO genre_graph_release_genre_assignments "
        "(release_id, artist, album, genre_id, assignment_layer, confidence, "
        " source_reliability, evidence_count, rejected_by_user, provenance_json, updated_at) "
        "VALUES ('acetone::york blvd','acetone','york blvd','alternative_rock',"
        " 'observed_leaf',0.9,0.7,2,0,'{}','t')")
    sconn.commit()
    sconn.close()
    genre_publish.publish(str(meta), str(side), dry_run=False)
    return meta


def test_resolved_genres_for_album(tmp_path):
    meta = _published_db(tmp_path)
    conn = sqlite3.connect(meta)
    conn.row_factory = sqlite3.Row
    rows = authority.resolved_genres_for_album(conn, "ALB1")
    assert any(r.genre_id == "alternative_rock" and r.source == "graph" for r in rows)


def test_resolved_genres_for_track(tmp_path):
    meta = _published_db(tmp_path)
    conn = sqlite3.connect(meta)
    conn.row_factory = sqlite3.Row
    rows = authority.resolved_genres_for_track(conn, "T1")
    assert any(r.genre_id == "alternative_rock" for r in rows)


def test_genre_source_for_album(tmp_path):
    meta = _published_db(tmp_path)
    conn = sqlite3.connect(meta)
    conn.row_factory = sqlite3.Row
    assert authority.genre_source_for_album(conn, "ALB1") == "graph"
    assert authority.genre_source_for_album(conn, "ALB2") == "legacy"
    assert authority.genre_source_for_album(conn, "NOPE") == "none"


def test_taxonomy_helpers(tmp_path):
    meta = _published_db(tmp_path)
    conn = sqlite3.connect(meta)
    conn.row_factory = sqlite3.Row
    fams = authority.families_for(conn, "alternative_rock")
    assert isinstance(fams, list)
