from __future__ import annotations

from src.ai_genre_enrichment.hybrid_evidence import EvidenceTerm, fuse_hybrid_evidence


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
