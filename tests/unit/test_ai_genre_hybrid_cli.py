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


def test_hybrid_enrich_one_with_model_prior_fuses_end_to_end(monkeypatch, tmp_path: Path, capsys):
    from scripts import ai_genre_enrich
    from src.ai_genre_enrichment.client import EnrichmentResult
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
    for source_type in ["local_metadata", "lastfm_tags"]:
        page_id = store.upsert_source_page(
            release_key="duster::stratosphere",
            normalized_artist="duster",
            normalized_album="stratosphere",
            album_id="a1",
            source_url=f"local://{source_type}/duster/stratosphere",
            source_type=source_type,
            identity_status="confirmed",
            identity_confidence=0.95,
            evidence_summary=f"{source_type} release tags.",
        )
        store.replace_source_tags(page_id, ["slowcore"])
        store.classify_source_tags(page_id)

    def fake_request_structured(self, **_kwargs):
        return EnrichmentResult(
            status="complete",
            response_json={
                "genres": [{
                    "term": "slowcore",
                    "confidence": 0.91,
                    "specificity": "subgenre",
                    "taxonomy_role": "core_style",
                    "notes": "Cautious hypothesis.",
                }],
                "warnings": [],
            },
            token_usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            estimated_cost_usd=0.00001,
        )

    monkeypatch.setattr(
        "src.ai_genre_enrichment.client.OpenAIEnrichmentClient.request_structured",
        fake_request_structured,
    )

    rc = ai_genre_enrich.main([
        "--metadata-db", str(metadata_db),
        "--sidecar-db", str(sidecar),
        "hybrid-enrich-one",
        "--artist", "Duster",
        "--album", "Stratosphere",
        "--dry-run",
        "--with-model-prior",
    ])

    assert rc == 0
    output = json.loads(capsys.readouterr().out)
    assert output["model_prior_status"] == "complete-transient"
    assert output["evidence_count"] == 3
    assert output["accepted_genres"][0]["term"] == "slowcore"
    assert output["accepted_genres"][0]["basis"] == "local_metadata+model_prior+lastfm_tags+taxonomy"


def test_hybrid_enrich_one_apply_persists_accepted_signature(tmp_path: Path, capsys):
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
    for source_type in ["local_metadata", "lastfm_tags"]:
        page_id = store.upsert_source_page(
            release_key="duster::stratosphere",
            normalized_artist="duster",
            normalized_album="stratosphere",
            album_id="a1",
            source_url=f"local://{source_type}/duster/stratosphere",
            source_type=source_type,
            identity_status="confirmed",
            identity_confidence=0.95,
            evidence_summary=f"{source_type} release tags.",
        )
        store.replace_source_tags(page_id, ["slowcore", "rock"])
        store.classify_source_tags(page_id)

    rc = ai_genre_enrich.main([
        "--metadata-db", str(metadata_db),
        "--sidecar-db", str(sidecar),
        "hybrid-enrich-one",
        "--artist", "Duster",
        "--album", "Stratosphere",
        "--apply",
    ])

    assert rc == 0
    output = json.loads(capsys.readouterr().out)
    assert output["applied"] is True
    assert output["applied_count"] == 1

    with sqlite3.connect(sidecar) as conn:
        conn.row_factory = sqlite3.Row
        genres = [
            dict(row)
            for row in conn.execute(
                "SELECT genre, basis, source_ref FROM enriched_genres WHERE release_key = ? ORDER BY genre",
                ("duster::stratosphere",),
            )
        ]
        signature_row = conn.execute(
            "SELECT signature_json FROM enriched_genre_signatures WHERE release_key = ?",
            ("duster::stratosphere",),
        ).fetchone()

    assert [row["genre"] for row in genres] == ["slowcore"]
    assert genres[0]["basis"] == "local_metadata+lastfm_tags+taxonomy"
    assert genres[0]["source_ref"].startswith("hybrid:")
    assert json.loads(signature_row["signature_json"])["genres"] == ["slowcore"]


def test_hybrid_enrich_one_rejects_dry_run_apply_combo(tmp_path: Path, capsys):
    from scripts import ai_genre_enrich

    rc = ai_genre_enrich.main([
        "--metadata-db", str(tmp_path / "missing.db"),
        "--sidecar-db", str(tmp_path / "sidecar.db"),
        "hybrid-enrich-one",
        "--artist", "Duster",
        "--album", "Stratosphere",
        "--dry-run",
        "--apply",
    ])

    assert rc == 2
    assert "cannot combine --dry-run and --apply" in capsys.readouterr().out
