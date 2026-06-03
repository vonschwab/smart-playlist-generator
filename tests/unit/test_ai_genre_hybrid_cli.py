from __future__ import annotations

import json
import sqlite3
from pathlib import Path


def test_hybrid_enrich_one_dry_run_fuses_existing_sidecar_without_api(monkeypatch, tmp_path: Path, capsys):
    from scripts import ai_genre_enrich
    from src.ai_genre_enrichment.storage import SidecarStore

    metadata_db = tmp_path / "metadata.db"
    with sqlite3.connect(metadata_db) as conn:
        conn.execute("CREATE TABLE tracks(track_id TEXT, artist TEXT, album TEXT, album_id TEXT, title TEXT, year INTEGER)")
        conn.execute("CREATE TABLE artist_genres(artist TEXT, genre TEXT, source TEXT)")
        conn.execute("CREATE TABLE album_genres(album_id TEXT, genre TEXT, source TEXT)")
        conn.execute("CREATE TABLE track_genres(track_id TEXT, genre TEXT, source TEXT, weight REAL)")
        conn.execute("INSERT INTO tracks VALUES ('t1', 'Duster', 'Stratosphere', 'a1', 'Moon Age', 1998)")

    sidecar = tmp_path / "sidecar.db"
    store = SidecarStore(sidecar)
    store.initialize()
    page_id = store.upsert_source_page(
        release_key="duster::stratosphere",
        normalized_artist="duster",
        normalized_album="stratosphere",
        album_id="a1",
        source_url="https://example.bandcamp.com/album/stratosphere",
        source_type="bandcamp_release",
        identity_status="confirmed",
        identity_confidence=0.95,
        evidence_summary="Bandcamp release tags.",
    )
    store.replace_source_tags(page_id, ["slowcore"])
    store.classify_source_tags(page_id)

    monkeypatch.setattr(
        "src.ai_genre_enrichment.client.OpenAIEnrichmentClient._call_openai",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("OpenAI should not be called in dry-run")),
    )

    rc = ai_genre_enrich.main([
        "--metadata-db", str(metadata_db),
        "--sidecar-db", str(sidecar),
        "hybrid-enrich-one",
        "--artist", "Duster",
        "--album", "Stratosphere",
        "--dry-run",
    ])

    assert rc == 0
    output = json.loads(capsys.readouterr().out)
    assert output["release_key"] == "duster::stratosphere"
    assert output["dry_run"] is True
    assert output["accepted_genres"][0]["term"] == "slowcore"
    assert output["evidence_count"] == 1
