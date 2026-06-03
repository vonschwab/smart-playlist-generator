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
