# tests/integration/test_web_track_genres_api.py
"""Integration tests for POST /api/tracks/genres (staged-seed canonical genres).

Uses the REAL production taxonomy for ordering (slowcore 0.88 > shoegaze 0.86
> rock 0.05; 'seen live' rejected) and tmp metadata/sidecar DBs.
"""
import json
import sqlite3
import sys

import pytest
from fastapi.testclient import TestClient

import src.playlist_web.app as app_module
from src.playlist_web.app import create_app

FAKE = [sys.executable, "tests/fixtures/fake_worker.py"]


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # --- tmp metadata.db ---
    db_path = tmp_path / "metadata.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE tracks (track_id TEXT PRIMARY KEY, artist TEXT, album TEXT,
                             album_id TEXT, duration_ms INTEGER, file_path TEXT,
                             title TEXT);
        CREATE TABLE track_effective_genres (track_id TEXT, genre TEXT, priority INTEGER);
        CREATE TABLE track_genres (track_id TEXT, genre TEXT, weight REAL);
        CREATE TABLE release_effective_genres (
            album_id TEXT, genre_id TEXT, assignment_layer TEXT,
            confidence REAL, source TEXT,
            PRIMARY KEY (album_id, genre_id, assignment_layer)
        );
        CREATE TABLE genre_graph_canonical_genres (
            genre_id TEXT PRIMARY KEY, name TEXT, kind TEXT,
            specificity_score REAL, status TEXT, taxonomy_version TEXT
        );

        INSERT INTO tracks VALUES ('t-enriched', 'Duster', 'Stratosphere', NULL, 0, '', 't-enriched title');
        INSERT INTO tracks VALUES ('t-metadata', 'Acetone', 'Cindy', NULL, 0, '', 't-metadata title');
        INSERT INTO tracks VALUES ('t-uncovered', 'Somebody', 'Something', NULL, 0, '', 't-uncovered title');
        INSERT INTO tracks VALUES ('t-published', 'Blood Orange', 'Angels Pulse', 'alb-pub', 0, '', 't-published title');

        -- t-published: rich published authority (the layer chips MUST read);
        -- its sidecar signature below is deliberately poor.
        INSERT INTO release_effective_genres VALUES
            ('alb-pub', 'slowcore', 'observed_leaf', 1.0, 'graph'),
            ('alb-pub', 'rock', 'inferred_family', 0.9, 'graph');
        INSERT INTO genre_graph_canonical_genres VALUES
            ('slowcore', 'slowcore', 'subgenre', 0.88, 'active', 'v-test'),
            ('rock', 'rock', 'family', 0.05, 'active', 'v-test');

        -- t-metadata: effective genres stored broad-first + one unknown
        INSERT INTO track_effective_genres VALUES ('t-metadata', 'rock', 1);
        INSERT INTO track_effective_genres VALUES ('t-metadata', 'shoegaze', 2);
        INSERT INTO track_effective_genres VALUES ('t-metadata', 'totally-not-a-genre', 3);

        -- t-uncovered: nothing canonicalizes
        INSERT INTO track_effective_genres VALUES ('t-uncovered', 'seen live', 1);
        INSERT INTO track_effective_genres VALUES ('t-uncovered', 'no-such-genre', 2);
        """
    )
    conn.commit()
    conn.close()

    # --- tmp enrichment sidecar: t-enriched's release, stored broad-first ---
    from src.ai_genre_enrichment.storage import SidecarStore

    sidecar_path = tmp_path / "sidecar.db"
    store = SidecarStore(str(sidecar_path))
    store.initialize()
    with store.connect() as sconn:
        sconn.execute(
            "INSERT INTO enriched_genre_signatures(release_key, normalized_artist, "
            "normalized_album, album_id, signature_json, updated_at) VALUES(?, ?, ?, ?, ?, ?)",
            (
                "duster::stratosphere",
                "duster",
                "stratosphere",
                None,
                json.dumps({"genres": ["rock", "slowcore", "seen live"], "sources": []}),
                "2026-06-11T00:00:00",
            ),
        )
        # t-published: poor bandcamp-era signature that must LOSE to the authority
        sconn.execute(
            "INSERT INTO enriched_genre_signatures(release_key, normalized_artist, "
            "normalized_album, album_id, signature_json, updated_at) VALUES(?, ?, ?, ?, ?, ?)",
            (
                "blood orange::angels pulse",
                "blood orange",
                "angels pulse",
                None,
                json.dumps({"genres": ["alternative"], "sources": []}),
                "2026-06-12T00:00:00",
            ),
        )
        sconn.commit()

    monkeypatch.setattr(app_module, "DB_PATH", db_path)
    monkeypatch.setattr(app_module, "SIDECAR_DB_PATH", sidecar_path)
    app = create_app(worker_cmd=FAKE)
    with TestClient(app) as c:
        yield c


def test_enriched_release_ordered_sub_to_broad(client):
    resp = client.post("/api/tracks/genres", json={"track_ids": ["t-enriched"]})
    assert resp.status_code == 200
    # slowcore (0.88) before rock (0.05); 'seen live' rejected by the graph.
    assert resp.json() == {"t-enriched": ["slowcore", "rock"]}


def test_metadata_fallback_ordered_and_denoised(client):
    resp = client.post("/api/tracks/genres", json={"track_ids": ["t-metadata"]})
    assert resp.status_code == 200
    assert resp.json() == {"t-metadata": ["shoegaze", "rock"]}


def test_uncovered_track_falls_back_to_raw(client):
    resp = client.post("/api/tracks/genres", json={"track_ids": ["t-uncovered"]})
    assert resp.status_code == 200
    # Nothing canonicalizes -> raw tags unordered (never blank).
    assert resp.json() == {"t-uncovered": ["seen live", "no-such-genre"]}


def test_unknown_ids_omitted_and_batch_works(client):
    resp = client.post(
        "/api/tracks/genres",
        json={"track_ids": ["t-enriched", "t-metadata", "nope"]},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"t-enriched", "t-metadata"}


def test_empty_request_returns_empty(client):
    resp = client.post("/api/tracks/genres", json={"track_ids": []})
    assert resp.status_code == 200
    assert resp.json() == {}


def test_published_authority_outranks_poor_signature(client):
    """release_effective_genres (rich) must beat the bandcamp-era signature
    (sparse) — the Blood Orange 'one chip' bug."""
    resp = client.post("/api/tracks/genres", json={"track_ids": ["t-published"]})
    assert resp.status_code == 200
    assert resp.json() == {"t-published": ["slowcore", "rock"]}


def test_search_results_use_published_authority(client):
    resp = client.get("/api/tracks/search", params={"q": "blood orange"})
    assert resp.status_code == 200
    items = resp.json()["items"]  # paginated shape: {items, has_more}
    assert len(items) == 1
    assert items[0]["genres"] == ["slowcore", "rock"]
