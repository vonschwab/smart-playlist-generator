import json
import sqlite3
from types import SimpleNamespace


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
    from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy

    sidecar = tmp_path / "sidecar.db"
    assert ai_genre_enrich.main(["--sidecar-db", str(sidecar), "graph-init"]) == 0
    capsys.readouterr()

    rc = ai_genre_enrich.main(["--sidecar-db", str(sidecar), "graph-report"])

    assert rc == 0
    report = json.loads(capsys.readouterr().out)
    # Verify the report surfaces the loaded taxonomy version (propagation check),
    # without pinning a literal that breaks on every taxonomy growth pass.
    assert report["taxonomy_version"] == load_default_layered_taxonomy().version
    assert report["genre_counts_by_kind"]["family"] >= 14
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
    assert output["layered_assignment_count"] == 5

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
        ("indie_alternative", "inferred_family"),
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
            "genre_assignment_count": 5,
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
    assert [row["name"] for row in shown["genres_by_layer"]["inferred_family"]] == ["indie/alternative", "pop", "rock"]


def test_graph_show_release_accepts_exact_release_key_and_reports_evidence(tmp_path, capsys):
    from scripts import ai_genre_enrich

    metadata_db = tmp_path / "metadata.db"
    _write_one_release_metadata(metadata_db)
    sidecar = tmp_path / "sidecar.db"
    _seed_source_tags(sidecar, ["jangle pop", "pop/rock", "unknown scene"])

    assert ai_genre_enrich.main([
        "--metadata-db", str(metadata_db),
        "--sidecar-db", str(sidecar),
        "graph-build-assignments",
        "--artist", "The Clientele",
        "--album", "Strange Geometry",
    ]) == 0
    capsys.readouterr()

    rc = ai_genre_enrich.main([
        "--metadata-db", str(metadata_db),
        "--sidecar-db", str(sidecar),
        "graph-show-release",
        "--release-key", "the clientele::strange geometry",
    ])

    assert rc == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["release_id"] == "the clientele::strange geometry"
    assert shown["lookup"]["mode"] == "release_key"
    assert shown["evidence_status"] == "assignments_present"
    assert shown["model_prior_exists"] is False
    assert [row["term"] for row in shown["raw_evidence"]] == [
        "jangle pop",
        "pop/rock",
        "unknown scene",
        "jangle pop",
        "pop/rock",
        "unknown scene",
    ]
    assert [row["term"] for row in shown["accepted_leaf_terms"]] == ["jangle pop"]
    assert [row["term"] for row in shown["review_terms"]] == ["unknown scene"]
    assert shown["review_terms"][0]["reason"] == "Unknown layered taxonomy term."
    assert [row["term"] for row in shown["rejected_terms"]] == ["pop/rock"]
    assert "retail" in shown["rejected_terms"][0]["reason"]
    assert shown["inferred_terms"][0]["inference_edge"]["source_genre_id"] == "jangle_pop"
    assert shown["inferred_terms"][0]["inference_edge"]["target_genre_id"] == "indie_pop"


def test_graph_fixture_report_flags_zero_assignment_with_evidence(tmp_path, capsys):
    from scripts import ai_genre_enrich

    metadata_db = tmp_path / "metadata.db"
    _write_one_release_metadata(metadata_db)
    sidecar = tmp_path / "sidecar.db"
    _seed_source_tags(sidecar, ["unknown scene"])
    fixtures = tmp_path / "fixtures.yaml"
    fixtures.write_text(
        """
version: test-fixtures
fixtures:
  - id: clientele
    release_key: the clientele::strange geometry
    artist: The Clientele
    album: Strange Geometry
    fail_zero_assignments_when_evidence_exists: true
""".strip(),
        encoding="utf-8",
    )

    rc = ai_genre_enrich.main([
        "--metadata-db", str(metadata_db),
        "--sidecar-db", str(sidecar),
        "graph-fixture-report",
        "--fixtures", str(fixtures),
    ])

    assert rc == 1
    report = json.loads(capsys.readouterr().out)
    assert report["fixture_version"] == "test-fixtures"
    assert report["summary"] == {"fail": 1, "pass": 0}
    result = report["fixtures"][0]
    assert result["release_key"] == "the clientele::strange geometry"
    assert result["evidence_status"] == "evidence_present_no_assignments"
    assert "zero_assignments_with_evidence" in result["failures"]
    assert result["missing_taxonomy_terms"] == ["unknown scene"]


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


def test_graph_build_assignments_output_is_console_encoding_safe(tmp_path, capsys, monkeypatch):
    from scripts import ai_genre_enrich

    sidecar = tmp_path / "sidecar.db"

    monkeypatch.setattr(
        ai_genre_enrich,
        "_discover",
        lambda _args: [
            SimpleNamespace(
                release_key="unicode::release",
                normalized_artist="unicode",
                normalized_album="release",
                existing_genres_by_source={},
            )
        ],
    )
    monkeypatch.setattr(
        ai_genre_enrich,
        "_fuse_hybrid_for_release",
        lambda _store, _release: SimpleNamespace(
            accepted_genres=[SimpleNamespace(term="日本語")],
            rejected_noise=[],
        ),
    )

    rc = ai_genre_enrich.main([
        "--sidecar-db", str(sidecar),
        "graph-build-assignments",
        "--dry-run",
    ])

    assert rc == 0
    output = capsys.readouterr().out
    assert "\\u65e5\\u672c\\u8a9e" in output
    json.loads(output)
