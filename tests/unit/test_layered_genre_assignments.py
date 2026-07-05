import json
import sqlite3


class _FakeDiagnosticsStore:
    """Minimal store for build_layered_release_diagnostics: the four hybrid
    evidence collector methods (mirrors _FakeFusionStore in
    test_publish_backfill.py, letting real fusion run without seeding the
    full sidecar schema) plus a stub layered_release_summary standing in for
    "nothing materialized yet" (the diagnostics call under test never writes
    or reads real assignment rows)."""

    def __init__(self, source_terms):
        self._source_terms = source_terms

    def hybrid_source_terms_for_release(self, release_key):
        return self._source_terms

    def accepted_enriched_genres_for_release(self, release_key):
        return []

    def latest_check_suggestions_for_release(self, release_key):
        return []

    def latest_model_prior_terms_for_release(self, release_key):
        return []

    def layered_release_summary(self, release_id):
        return {
            "release_id": release_id,
            "genres_by_layer": {},
            "facets": [],
            "genre_assignment_count": 0,
            "facet_assignment_count": 0,
        }


def test_build_layered_release_diagnostics_provisional_taxonomy_unknown_term_keeps_coverage_basis():
    """Regression (whole-branch review of the M1 always-publish policy flip).

    build_layered_release_diagnostics builds review_terms in two passes over
    the SAME dict: pass 1 seeds taxonomy-unknown evidence terms with
    source_basis="layered_taxonomy" (a coverage gap -- compute_layered_
    assignment_rows skips term_kind=="review", so it is never published);
    pass 2 folds every report.provisional_genres decision in via
    _merge_decision_row(..., "hybrid_provisional", ...).

    A term that is BOTH mapped in the genre vocabulary (so fusion routes it
    to provisional_genres -- here via the never-drop local_metadata lane) AND
    unknown to the layered taxonomy (so pass 1 classifies it "review") must
    keep the pass-1 "layered_taxonomy" basis. Before the fix, pass 2
    unconditionally clobbered it to "hybrid_provisional", so the queue-stats
    split (get_review_queue_page's pending_published_terms vs
    pending_coverage_terms, keyed off this exact basis string) miscounted a
    dropped coverage-gap term as a published one.
    """
    from src.ai_genre_enrichment.layered_assignment import build_layered_release_diagnostics
    from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy

    taxonomy = load_default_layered_taxonomy()
    # A single local_metadata source is fuse_hybrid_evidence's never-drop
    # lane: sources == ["local_metadata"] always routes to
    # provisional_genres, regardless of the term's taxonomy status.
    store = _FakeDiagnosticsStore(source_terms=[
        {
            "term": "rare wistful kitchen pop",
            "source_type": "local_metadata",
            "confidence": 0.55,
            "mapping_status": "mapped",
            "canonical_slug": "rare wistful kitchen pop",
        },
    ])

    diagnostics = build_layered_release_diagnostics(
        store,
        release_id="test artist::taxonomy gap album",
        taxonomy=taxonomy,
    )

    review_row = next(
        row for row in diagnostics["review_terms"] if row["term"] == "rare wistful kitchen pop"
    )
    assert review_row["term_kind"] == "review"
    assert review_row["reason"] == "Unknown layered taxonomy term."
    assert review_row["source_basis"] == "layered_taxonomy"  # NOT "hybrid_provisional"
    assert review_row["term"] in diagnostics["missing_taxonomy_terms"]


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
