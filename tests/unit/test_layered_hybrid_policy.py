def test_hybrid_fusion_accepts_valid_broad_rock_when_sources_support_it():
    from src.ai_genre_enrichment.hybrid_evidence import EvidenceTerm, fuse_hybrid_evidence

    report = fuse_hybrid_evidence(
        release_key="artist::album",
        evidence=[
            EvidenceTerm(term="rock", source_type="local_metadata", confidence=0.95),
            EvidenceTerm(term="rock", source_type="lastfm_tags", confidence=0.95),
        ],
        sparse_release=False,
    )

    assert [decision.term for decision in report.accepted_genres] == ["rock"]
    assert report.rejected_noise == []


def test_hybrid_fusion_rejects_fake_pop_rock_bucket():
    from src.ai_genre_enrichment.hybrid_evidence import EvidenceTerm, fuse_hybrid_evidence

    report = fuse_hybrid_evidence(
        release_key="artist::album",
        evidence=[
            EvidenceTerm(term="pop/rock", source_type="local_metadata", confidence=0.95),
            EvidenceTerm(term="pop/rock", source_type="lastfm_tags", confidence=0.95),
        ],
        sparse_release=False,
    )

    assert report.accepted_genres == []
    assert [decision.term for decision in report.rejected_noise] == ["pop/rock"]
    assert "retail" in report.rejected_noise[0].reason


def test_hybrid_fusion_rejects_standalone_indie_without_rejecting_indie_rock():
    from src.ai_genre_enrichment.hybrid_evidence import EvidenceTerm, fuse_hybrid_evidence

    report = fuse_hybrid_evidence(
        release_key="artist::album",
        evidence=[
            EvidenceTerm(term="indie", source_type="local_metadata", confidence=0.95),
            EvidenceTerm(term="indie", source_type="lastfm_tags", confidence=0.95),
            EvidenceTerm(term="indie rock", source_type="local_metadata", confidence=0.95),
            EvidenceTerm(term="indie rock", source_type="lastfm_tags", confidence=0.95),
        ],
        sparse_release=False,
    )

    assert [decision.term for decision in report.accepted_genres] == ["indie rock"]
    assert [decision.term for decision in report.rejected_noise] == ["indie"]


def test_hybrid_fusion_routes_lastfm_only_mapped_terms_to_review():
    from src.ai_genre_enrichment.hybrid_evidence import EvidenceTerm, fuse_hybrid_evidence

    report = fuse_hybrid_evidence(
        release_key="artist::album",
        evidence=[
            EvidenceTerm(term="psychedelic folk", source_type="lastfm_tags", confidence=0.95),
        ],
        sparse_release=False,
    )

    assert report.accepted_genres == []
    assert report.rejected_noise == []
    assert [decision.term for decision in report.needs_review] == ["psychedelic folk"]
