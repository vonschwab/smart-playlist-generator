import json
import sqlite3


def _write_one_release_metadata(metadata_db):
    with sqlite3.connect(metadata_db) as conn:
        conn.execute("CREATE TABLE tracks(track_id TEXT, artist TEXT, album TEXT, album_id TEXT, title TEXT, year INTEGER)")
        conn.execute("CREATE TABLE artist_genres(artist TEXT, genre TEXT, source TEXT)")
        conn.execute("CREATE TABLE album_genres(album_id TEXT, genre TEXT, source TEXT)")
        conn.execute("CREATE TABLE track_genres(track_id TEXT, genre TEXT, source TEXT, weight REAL)")
        conn.execute("INSERT INTO tracks VALUES ('t1', 'The Clientele', 'Strange Geometry', 'a1', 'Since K Got Over Me', 2005)")


def _seed_source_tags(sidecar, tags):
    from src.ai_genre_enrichment.storage import SidecarStore

    store = SidecarStore(sidecar)
    store.initialize()
    for source_type in ["local_metadata", "lastfm_tags"]:
        page_id = store.upsert_source_page(
            release_key="the clientele::strange geometry",
            normalized_artist="the clientele",
            normalized_album="strange geometry",
            album_id="a1",
            source_url=f"local://{source_type}/the clientele/strange geometry",
            source_type=source_type,
            identity_status="confirmed",
            identity_confidence=0.95,
            evidence_summary=f"{source_type} release tags.",
        )
        store.replace_source_tags(page_id, tags)
        store.classify_source_tags(page_id)


def test_graph_init_upserts_seed_taxonomy_idempotently(tmp_path, capsys):
    from scripts import ai_genre_enrich

    sidecar = tmp_path / "sidecar.db"

    for _ in range(2):
        rc = ai_genre_enrich.main([
            "--sidecar-db", str(sidecar),
            "graph-init",
        ])
        assert rc == 0
        capsys.readouterr()

    with sqlite3.connect(sidecar) as conn:
        genre_count = conn.execute("SELECT COUNT(*) FROM genre_graph_canonical_genres").fetchone()[0]
        alias_count = conn.execute("SELECT COUNT(*) FROM genre_graph_aliases").fetchone()[0]
        facet_count = conn.execute("SELECT COUNT(*) FROM genre_graph_canonical_facets").fetchone()[0]
        bridge_count = conn.execute("SELECT COUNT(*) FROM genre_graph_bridge_rules").fetchone()[0]

    assert genre_count >= 30
    assert alias_count >= 5
    assert facet_count >= 10
    assert bridge_count >= 6


def test_graph_report_outputs_layered_taxonomy_counts(tmp_path, capsys):
    from scripts import ai_genre_enrich

    sidecar = tmp_path / "sidecar.db"
    assert ai_genre_enrich.main(["--sidecar-db", str(sidecar), "graph-init"]) == 0
    capsys.readouterr()

    rc = ai_genre_enrich.main(["--sidecar-db", str(sidecar), "graph-report"])

    assert rc == 0
    report = json.loads(capsys.readouterr().out)
    assert report["taxonomy_version"] == "layered-genre-graph-v0"
    assert report["genre_counts_by_kind"]["family"] == 14
    assert report["facet_counts_by_type"]["production"] >= 3
    assert report["alias_count"] >= 5
    assert report["bridge_rule_count"] >= 6
    assert report["rejected_term_count"] >= 5


def test_hybrid_enrich_one_apply_materializes_layered_assignments(tmp_path, capsys):
    from scripts import ai_genre_enrich

    metadata_db = tmp_path / "metadata.db"
    _write_one_release_metadata(metadata_db)
    sidecar = tmp_path / "sidecar.db"
    _seed_source_tags(sidecar, ["jangle pop"])

    rc = ai_genre_enrich.main([
        "--metadata-db", str(metadata_db),
        "--sidecar-db", str(sidecar),
        "hybrid-enrich-one",
        "--artist", "The Clientele",
        "--album", "Strange Geometry",
        "--apply",
    ])

    assert rc == 0
    output = json.loads(capsys.readouterr().out)
    assert output["applied"] is True
    assert output["layered_assignment_count"] == 4

    with sqlite3.connect(sidecar) as conn:
        rows = conn.execute(
            """
            SELECT genre_id, assignment_layer
            FROM genre_graph_release_genre_assignments
            WHERE release_id = ?
            ORDER BY assignment_layer, genre_id
            """,
            ("the clientele::strange geometry",),
        ).fetchall()

    assert rows == [
        ("pop", "inferred_family"),
        ("rock", "inferred_family"),
        ("indie_pop", "inferred_parent"),
        ("jangle_pop", "observed_leaf"),
    ]


def test_graph_build_assignments_and_show_release(tmp_path, capsys):
    from scripts import ai_genre_enrich

    metadata_db = tmp_path / "metadata.db"
    _write_one_release_metadata(metadata_db)
    sidecar = tmp_path / "sidecar.db"
    _seed_source_tags(sidecar, ["jangle pop"])

    rc = ai_genre_enrich.main([
        "--metadata-db", str(metadata_db),
        "--sidecar-db", str(sidecar),
        "graph-build-assignments",
        "--artist", "The Clientele",
        "--album", "Strange Geometry",
    ])

    assert rc == 0
    built = json.loads(capsys.readouterr().out)
    assert built["releases"] == [
        {
            "release_key": "the clientele::strange geometry",
            "dry_run": False,
            "genre_assignment_count": 4,
            "facet_assignment_count": 0,
            "rejected_term_count": 0,
            "review_term_count": 0,
        }
    ]

    rc = ai_genre_enrich.main([
        "--metadata-db", str(metadata_db),
        "--sidecar-db", str(sidecar),
        "graph-show-release",
        "--artist", "The Clientele",
        "--album", "Strange Geometry",
    ])

    assert rc == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["release_id"] == "the clientele::strange geometry"
    assert [row["name"] for row in shown["genres_by_layer"]["observed_leaf"]] == ["jangle pop"]
    assert [row["name"] for row in shown["genres_by_layer"]["inferred_parent"]] == ["indie pop"]
    assert [row["name"] for row in shown["genres_by_layer"]["inferred_family"]] == ["pop", "rock"]


def test_graph_build_assignments_dry_run_does_not_write(tmp_path, capsys):
    from scripts import ai_genre_enrich

    metadata_db = tmp_path / "metadata.db"
    _write_one_release_metadata(metadata_db)
    sidecar = tmp_path / "sidecar.db"
    _seed_source_tags(sidecar, ["jangle pop"])

    rc = ai_genre_enrich.main([
        "--metadata-db", str(metadata_db),
        "--sidecar-db", str(sidecar),
        "graph-build-assignments",
        "--artist", "The Clientele",
        "--album", "Strange Geometry",
        "--dry-run",
    ])

    assert rc == 0
    output = json.loads(capsys.readouterr().out)
    assert output["releases"][0]["accepted_genres"] == ["jangle pop"]
    with sqlite3.connect(sidecar) as conn:
        count = conn.execute("SELECT COUNT(*) FROM genre_graph_release_genre_assignments").fetchone()[0]
    assert count == 0
