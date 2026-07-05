import json
import sqlite3


def test_materialize_layered_assignments_observes_leaf_and_infers_parents(tmp_path):
    from src.ai_genre_enrichment.hybrid_evidence import FusedGenreDecision, HybridGenreReport
    from src.ai_genre_enrichment.layered_assignment import materialize_layered_assignments
    from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy
    from src.ai_genre_enrichment.storage import SidecarStore

    store = SidecarStore(tmp_path / "sidecar.db")
    store.initialize()
    taxonomy = load_default_layered_taxonomy()
    store.upsert_layered_taxonomy(taxonomy)
    report = HybridGenreReport(
        release_key="the clientele::strange geometry",
        accepted_genres=[
            FusedGenreDecision(
                term="jangle pop",
                confidence=0.91,
                basis="local_metadata+lastfm_tags+taxonomy",
                sources=["lastfm_tags", "local_metadata"],
                reason="Corroborated specific style.",
            )
        ],
        provisional_genres=[],
        rejected_noise=[],
    )

    summary = materialize_layered_assignments(
        store,
        release_id=report.release_key,
        artist="the clientele",
        album="strange geometry",
        report=report,
        taxonomy=taxonomy,
    )

    with sqlite3.connect(store.db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = [
            dict(row)
            for row in conn.execute(
                """
                SELECT genre_id, assignment_layer, confidence, source_reliability, evidence_count, provenance_json
                FROM genre_graph_release_genre_assignments
                WHERE release_id = ?
                ORDER BY assignment_layer, genre_id
                """,
                (report.release_key,),
            )
        ]

    assert summary.genre_assignment_count == 5
    assert {(row["genre_id"], row["assignment_layer"]) for row in rows} == {
        ("jangle_pop", "observed_leaf"),
        ("indie_pop", "inferred_parent"),
        ("indie_alternative", "inferred_family"),
        ("pop", "inferred_family"),
        ("rock", "inferred_family"),
    }
    observed = next(row for row in rows if row["assignment_layer"] == "observed_leaf")
    assert observed["confidence"] == 0.91
    assert observed["source_reliability"] == 0.70
    assert observed["evidence_count"] == 2
    assert json.loads(observed["provenance_json"])["basis"] == "local_metadata+lastfm_tags+taxonomy"


def test_materialize_layered_assignments_demotes_family_and_facets(tmp_path):
    from src.ai_genre_enrichment.hybrid_evidence import FusedGenreDecision, HybridGenreReport
    from src.ai_genre_enrichment.layered_assignment import materialize_layered_assignments
    from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy
    from src.ai_genre_enrichment.storage import SidecarStore

    store = SidecarStore(tmp_path / "sidecar.db")
    store.initialize()
    taxonomy = load_default_layered_taxonomy()
    store.upsert_layered_taxonomy(taxonomy)
    report = HybridGenreReport(
        release_key="test artist::test album",
        accepted_genres=[
            FusedGenreDecision(
                term="rock",
                confidence=0.95,
                basis="local_metadata+taxonomy",
                sources=["local_metadata"],
                reason="Broad metadata term.",
            ),
            FusedGenreDecision(
                term="lo-fi",
                confidence=0.86,
                basis="bandcamp_release+taxonomy",
                sources=["bandcamp_release"],
                reason="Descriptor term.",
            ),
        ],
        provisional_genres=[],
        rejected_noise=[],
    )

    summary = materialize_layered_assignments(
        store,
        release_id=report.release_key,
        artist="test artist",
        album="test album",
        report=report,
        taxonomy=taxonomy,
    )

    with sqlite3.connect(store.db_path) as conn:
        genre_rows = conn.execute(
            """
            SELECT genre_id, assignment_layer
            FROM genre_graph_release_genre_assignments
            WHERE release_id = ?
            """,
            (report.release_key,),
        ).fetchall()
        facet_rows = conn.execute(
            """
            SELECT facet_id, source
            FROM genre_graph_release_facet_assignments
            WHERE release_id = ?
            """,
            (report.release_key,),
        ).fetchall()

    assert summary.genre_assignment_count == 1
    assert summary.facet_assignment_count == 1
    assert genre_rows == [("rock", "inferred_family")]
    assert facet_rows == [("lo_fi", "bandcamp_release")]


def test_materialize_layered_assignments_reports_reject_and_review_terms(tmp_path):
    from src.ai_genre_enrichment.hybrid_evidence import FusedGenreDecision, HybridGenreReport
    from src.ai_genre_enrichment.layered_assignment import materialize_layered_assignments
    from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy
    from src.ai_genre_enrichment.storage import SidecarStore

    store = SidecarStore(tmp_path / "sidecar.db")
    store.initialize()
    taxonomy = load_default_layered_taxonomy()
    store.upsert_layered_taxonomy(taxonomy)
    # Zero-touch policy (2026-07-04): the fusion-layer needs_review bucket is
    # gone — "rare wistful kitchen pop" now arrives via provisional_genres
    # (published at evidence confidence), but it still gets classified as a
    # taxonomy "review" term by compute_layered_assignment_rows because it is
    # unknown to the layered taxonomy, so review_term_count is unaffected.
    report = HybridGenreReport(
        release_key="test artist::noise album",
        accepted_genres=[],
        provisional_genres=[
            FusedGenreDecision(
                term="rare wistful kitchen pop",
                confidence=0.60,
                basis="model_prior+taxonomy",
                sources=["model_prior"],
                reason="Evidence mapped but below the corroboration bar; published at evidence confidence.",
            )
        ],
        rejected_noise=[
            FusedGenreDecision(
                term="spotify",
                confidence=0.95,
                basis="lastfm_only",
                sources=["lastfm_tags"],
                reason="Known platform tag.",
            )
        ],
    )

    summary = materialize_layered_assignments(
        store,
        release_id=report.release_key,
        artist="test artist",
        album="noise album",
        report=report,
        taxonomy=taxonomy,
    )

    with sqlite3.connect(store.db_path) as conn:
        genre_count = conn.execute(
            "SELECT COUNT(*) FROM genre_graph_release_genre_assignments WHERE release_id = ?",
            (report.release_key,),
        ).fetchone()[0]
        facet_count = conn.execute(
            "SELECT COUNT(*) FROM genre_graph_release_facet_assignments WHERE release_id = ?",
            (report.release_key,),
        ).fetchone()[0]

    assert summary.rejected_term_count == 1
    assert summary.review_term_count == 1
    assert genre_count == 0
    assert facet_count == 0
