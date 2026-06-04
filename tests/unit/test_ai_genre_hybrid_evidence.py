from __future__ import annotations

from pathlib import Path

from src.ai_genre_enrichment.hybrid_evidence import (
    EvidenceTerm,
    collect_hybrid_evidence,
    fuse_hybrid_evidence,
)


def test_bandcamp_and_model_accepts_specific_genre():
    report = fuse_hybrid_evidence(
        release_key="duster::stratosphere",
        evidence=[
            EvidenceTerm(term="slowcore", source_type="bandcamp_release", confidence=0.90),
            EvidenceTerm(term="slowcore", source_type="model_prior", confidence=0.88),
        ],
        sparse_release=False,
    )

    assert [item.term for item in report.accepted_genres] == ["slowcore"]
    assert report.accepted_genres[0].basis == "bandcamp_release+model_prior+taxonomy"
    assert report.accepted_genres[0].confidence >= 0.90


def test_lastfm_only_is_rejected_noise():
    report = fuse_hybrid_evidence(
        release_key="test::album",
        evidence=[EvidenceTerm(term="seen live", source_type="lastfm_tags", confidence=0.70)],
        sparse_release=True,
    )

    assert report.accepted_genres == []
    assert report.rejected_noise[0].term == "seen live"
    assert "Last.fm-only" in report.rejected_noise[0].reason


def test_model_only_high_confidence_sparse_release_is_provisional():
    report = fuse_hybrid_evidence(
        release_key="obscure::album",
        evidence=[EvidenceTerm(term="ambient americana", source_type="model_prior", confidence=0.86)],
        sparse_release=True,
    )

    assert report.accepted_genres == []
    assert report.provisional_genres[0].term == "ambient americana"
    assert report.provisional_genres[0].basis == "model_prior+taxonomy"


def test_model_only_high_confidence_with_release_evidence_is_provisional():
    report = fuse_hybrid_evidence(
        release_key="mount eerie::sauna",
        evidence=[
            EvidenceTerm(term="indie folk", source_type="local_metadata", confidence=0.95),
            EvidenceTerm(term="indie folk", source_type="lastfm_tags", confidence=0.95),
            EvidenceTerm(term="ambient", source_type="model_prior", confidence=0.84),
            EvidenceTerm(term="drone", source_type="model_prior", confidence=0.78),
        ],
        sparse_release=False,
    )

    assert [item.term for item in report.accepted_genres] == ["indie folk"]
    assert [item.term for item in report.provisional_genres] == ["ambient", "drone"]


def test_specific_lastfm_only_terms_are_provisional_when_release_evidence_exists():
    report = fuse_hybrid_evidence(
        release_key="mount eerie::sauna",
        evidence=[
            EvidenceTerm(term="indie folk", source_type="local_metadata", confidence=0.95),
            EvidenceTerm(term="avant-folk", source_type="lastfm_tags", confidence=0.95),
            EvidenceTerm(term="drone", source_type="lastfm_tags", confidence=0.95),
            EvidenceTerm(term="folk", source_type="lastfm_tags", confidence=0.95),
        ],
        sparse_release=False,
    )

    assert [item.term for item in report.accepted_genres] == []
    assert [item.term for item in report.provisional_genres] == ["avant-folk", "drone"]
    assert [item.term for item in report.rejected_noise] == ["folk"]


def test_ai_adjudicated_lastfm_only_terms_are_rejected_even_when_release_evidence_exists():
    report = fuse_hybrid_evidence(
        release_key="ada lea::one hand on the steering wheel the other sewing a garden",
        evidence=[
            EvidenceTerm(term="indie pop", source_type="local_metadata", confidence=0.95),
            EvidenceTerm(term="mixtaperoom", source_type="lastfm_tags", confidence=0.90, classifier="ai"),
            EvidenceTerm(term="rare sad girl", source_type="lastfm_tags", confidence=0.90, classifier="cached_ai"),
        ],
        sparse_release=False,
    )

    assert [item.term for item in report.provisional_genres] == []
    assert [item.term for item in report.rejected_noise] == ["mixtaperoom", "rare sad girl"]
    assert all("AI-adjudicated" in item.reason for item in report.rejected_noise)


def test_lastfm_lofi_alias_collapses_to_lo_fi():
    report = fuse_hybrid_evidence(
        release_key="mount eerie::sauna",
        evidence=[
            EvidenceTerm(term="indie folk", source_type="local_metadata", confidence=0.95),
            EvidenceTerm(term="lo-fi", source_type="lastfm_tags", confidence=0.95),
            EvidenceTerm(term="lofi", source_type="lastfm_tags", confidence=0.95),
        ],
        sparse_release=False,
    )

    assert [item.term for item in report.provisional_genres] == ["lo-fi"]


def test_local_and_model_can_accept_when_no_stronger_conflict():
    report = fuse_hybrid_evidence(
        release_key="test::album",
        evidence=[
            EvidenceTerm(term="dream pop", source_type="local_metadata", confidence=0.65),
            EvidenceTerm(term="dream pop", source_type="model_prior", confidence=0.82),
        ],
        sparse_release=False,
    )

    assert report.accepted_genres[0].term == "dream pop"
    assert "local_metadata" in report.accepted_genres[0].sources


def test_local_and_lastfm_accept_specific_corroborated_genre_without_bandcamp():
    report = fuse_hybrid_evidence(
        release_key="duster::stratosphere",
        evidence=[
            EvidenceTerm(term="slowcore", source_type="local_metadata", confidence=0.95),
            EvidenceTerm(term="slowcore", source_type="lastfm_tags", confidence=0.95),
        ],
        sparse_release=False,
    )

    assert [item.term for item in report.accepted_genres] == ["slowcore"]
    assert report.accepted_genres[0].basis == "local_metadata+lastfm_tags+taxonomy"
    assert "Local metadata and Last.fm corroborate" in report.accepted_genres[0].reason


def test_local_and_lastfm_reject_broad_parent_when_specific_terms_exist():
    report = fuse_hybrid_evidence(
        release_key="duster::stratosphere",
        evidence=[
            EvidenceTerm(term="rock", source_type="local_metadata", confidence=0.95),
            EvidenceTerm(term="rock", source_type="lastfm_tags", confidence=0.95),
        ],
        sparse_release=False,
    )

    assert report.accepted_genres == []
    assert report.rejected_noise[0].term == "rock"
    assert "Broad parent genre" in report.rejected_noise[0].reason


def test_local_only_broad_parent_is_rejected_not_reviewed():
    report = fuse_hybrid_evidence(
        release_key="duster::stratosphere",
        evidence=[
            EvidenceTerm(term="rock", source_type="local_metadata", confidence=0.95),
        ],
        sparse_release=False,
    )

    assert report.needs_review == []
    assert report.rejected_noise[0].term == "rock"
    assert "Broad parent genre" in report.rejected_noise[0].reason


def test_collect_hybrid_evidence_reads_sidecar_sources_and_prior(tmp_path: Path):
    from src.ai_genre_enrichment.storage import SidecarStore

    store = SidecarStore(tmp_path / "sidecar.db")
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

    store.record_model_prior(
        release_key="duster::stratosphere",
        normalized_artist="duster",
        normalized_album="stratosphere",
        album_id="a1",
        provider="openai",
        model="gpt-4o-mini",
        prompt_version="album-model-prior-v1",
        taxonomy_version="genre-vocabulary-v1",
        schema_version="album-model-prior-response-v1",
        enrichment_policy_version="genre-enrichment-v2",
        input_hash="hash-1",
        status="complete",
        response_json={"genres": [], "warnings": []},
        warnings=[],
        error_message=None,
        token_usage={},
        estimated_cost_usd=None,
        mapped_terms=[{
            "raw_term": "slowcore",
            "normalized_term": "slowcore",
            "canonical_slug": "slowcore",
            "confidence": 0.86,
            "specificity": "subgenre",
            "taxonomy_role": "core_style",
            "mapping_status": "mapped",
            "accepted_for_shadow": 1,
            "auto_apply_eligible": 0,
            "notes": "",
        }],
    )

    evidence = collect_hybrid_evidence(store, "duster::stratosphere")
    source_types = sorted({item.source_type for item in evidence if item.term == "slowcore"})

    assert source_types == ["bandcamp_release", "model_prior"]


def test_collect_hybrid_evidence_excludes_human_rejected_source_tags(tmp_path: Path):
    from src.ai_genre_enrichment.storage import SidecarStore

    store = SidecarStore(tmp_path / "sidecar.db")
    store.initialize()

    page_id = store.upsert_source_page(
        release_key="ada lea::one hand on the steering wheel the other sewing a garden",
        normalized_artist="ada lea",
        normalized_album="one hand on the steering wheel the other sewing a garden",
        album_id="a1",
        source_url="lastfm://artist/ada lea/album/one hand on the steering wheel the other sewing a garden",
        source_type="lastfm_tags",
        identity_status="confirmed",
        identity_confidence=0.9,
        evidence_summary="Last.fm top tags.",
    )
    store.replace_source_tags(page_id, ["mixtaperoom"])
    store.classify_source_tags(page_id, adjudicate=False)
    with store.connect() as conn:
        source_tag_id = int(conn.execute(
            "SELECT source_tag_id FROM ai_genre_source_tags WHERE normalized_tag = 'mixtaperoom'"
        ).fetchone()["source_tag_id"])
        conn.execute(
            """
            UPDATE ai_genre_tag_classifications
            SET classification = 'genre_style', confidence = 0.9, classifier = 'ai'
            WHERE source_tag_id = ?
            """,
            (source_tag_id,),
        )
    store.record_review_decision(
        source_tag_id=source_tag_id,
        release_key="ada lea::one hand on the steering wheel the other sewing a garden",
        raw_tag="mixtaperoom",
        normalized_tag="mixtaperoom",
        original_classification="genre_style",
        reviewed_classification="rejected",
    )

    evidence = collect_hybrid_evidence(store, "ada lea::one hand on the steering wheel the other sewing a garden")

    assert [item.term for item in evidence] == []
