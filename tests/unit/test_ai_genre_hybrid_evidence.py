from __future__ import annotations

from pathlib import Path

from src.ai_genre_enrichment.hybrid_evidence import (
    EvidenceTerm,
    collect_hybrid_evidence,
    fuse_hybrid_evidence,
)


def test_bandcamp_artist_and_model_accepts_specific_genre():
    # Artist-run bandcamp pages are self-reported — the strongest evidence we
    # have. (Unclassified legacy "bandcamp_release" no longer auto-accepts; see
    # the storefront tests below.)
    report = fuse_hybrid_evidence(
        release_key="duster::stratosphere",
        evidence=[
            EvidenceTerm(term="slowcore", source_type="bandcamp_artist", confidence=0.90),
            EvidenceTerm(term="slowcore", source_type="model_prior", confidence=0.88),
        ],
        sparse_release=False,
    )

    assert [item.term for item in report.accepted_genres] == ["slowcore"]
    assert report.accepted_genres[0].basis == "bandcamp_artist+model_prior+taxonomy"
    assert report.accepted_genres[0].confidence >= 0.90


def test_lastfm_only_publishes_provisionally_at_capped_confidence():
    # Zero-touch policy (2026-07-04): lastfm-only mapped terms publish at a
    # hard-capped confidence instead of blocking in the review queue. The cap
    # keeps artifact weight low (X_genre weight = confidence x layer weight);
    # the 'baroque on Debussy' incident came from this lane publishing at
    # >=0.90 full weight. Junk-tag rejection ("seen live") happens upstream at
    # classification time, and non-taxonomy terms are still dropped by the
    # materializer — this branch only lowers the stakes for real terms.
    report = fuse_hybrid_evidence(
        release_key="test::album",
        evidence=[EvidenceTerm(term="shoegaze", source_type="lastfm_tags", confidence=0.95)],
        sparse_release=True,
    )

    assert report.accepted_genres == []
    [decision] = [d for d in report.provisional_genres if d.term == "shoegaze"]
    assert decision.basis == "lastfm_only"
    assert decision.confidence <= 0.40
    assert "capped confidence" in decision.reason


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
    # High-confidence lastfm-only terms become provisional when release evidence
    # exists. Broad-parent suppression (e.g. demoting "folk" beneath "avant-folk")
    # is no longer a fusion-layer concern — it is a mode-aware taxonomy policy
    # (`broad_only: reject` in data/layered_genre_taxonomy.yaml, covered by the
    # layered-taxonomy/assignment tests), so fusion keeps "folk" provisional here.
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
    # Zero-touch policy (2026-07-04): lastfm-only terms publish provisionally at
    # capped confidence instead of waiting in review. "indie folk" (local-only,
    # never-drop) is provisional at full score; the lastfm-only terms join it
    # but capped at LASTFM_ONLY_CONFIDENCE_CAP.
    assert {item.term for item in report.provisional_genres} == {
        "indie folk", "avant-folk", "drone", "folk",
    }
    lastfm_only_terms = {"avant-folk", "drone", "folk"}
    assert all(
        item.confidence <= 0.40
        for item in report.provisional_genres
        if item.term in lastfm_only_terms
    )
    assert report.needs_review == []
    assert report.rejected_noise == []


def test_ai_adjudicated_lastfm_terms_are_handled_at_collection_not_fusion():
    # AI-adjudicated junk ("mixtaperoom", "rare sad girl") is rejected upstream:
    # classification + human review-decisions drop it before it reaches fusion
    # (see test_collect_hybrid_evidence_excludes_human_rejected_source_tags).
    # Fusion itself no longer special-cases classifier=ai/cached_ai — if such a
    # term survives collection, it now publishes provisionally at capped
    # confidence like any other lastfm-only term (zero-touch policy, 2026-07-04).
    report = fuse_hybrid_evidence(
        release_key="ada lea::one hand on the steering wheel the other sewing a garden",
        evidence=[
            EvidenceTerm(term="indie pop", source_type="local_metadata", confidence=0.95),
            EvidenceTerm(term="mixtaperoom", source_type="lastfm_tags", confidence=0.90, classifier="ai"),
            EvidenceTerm(term="rare sad girl", source_type="lastfm_tags", confidence=0.90, classifier="cached_ai"),
        ],
        sparse_release=False,
    )

    assert report.rejected_noise == []
    # local "indie pop" is provisional under never-drop; lastfm-only junk that
    # survives collection now also publishes provisionally, but capped.
    assert {item.term for item in report.provisional_genres} == {
        "indie pop", "mixtaperoom", "rare sad girl",
    }
    lastfm_only_terms = {"mixtaperoom", "rare sad girl"}
    assert all(
        item.confidence <= 0.40
        for item in report.provisional_genres
        if item.term in lastfm_only_terms
    )
    assert report.needs_review == []


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

    # lastfm-only "lo-fi" publishes provisionally at capped confidence
    # (zero-touch policy, 2026-07-04); the alias still collapses (one lo-fi
    # entry, not two). Local "indie folk" is provisional at full score.
    assert {item.term for item in report.provisional_genres} == {"indie folk", "lo-fi"}
    lofi_decision = next(d for d in report.provisional_genres if d.term == "lo-fi")
    assert lofi_decision.confidence <= 0.40
    assert lofi_decision.basis == "lastfm_only"
    assert report.needs_review == []


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


def test_local_and_lastfm_corroboration_accepts_even_broad_terms():
    # Local + Last.fm corroboration accepts the mapped term at the fusion layer,
    # including broad parents like "rock". Broad-parent suppression is enforced
    # downstream by the taxonomy policy (`broad_only: reject`), not here — so
    # fusion's job is just to record the corroborated signal.
    report = fuse_hybrid_evidence(
        release_key="duster::stratosphere",
        evidence=[
            EvidenceTerm(term="rock", source_type="local_metadata", confidence=0.95),
            EvidenceTerm(term="rock", source_type="lastfm_tags", confidence=0.95),
        ],
        sparse_release=False,
    )

    assert [item.term for item in report.accepted_genres] == ["rock"]
    assert report.rejected_noise == []
    assert "Local metadata and Last.fm corroborate" in report.accepted_genres[0].reason


def test_local_only_single_source_is_provisional_never_dropped():
    # NEVER-DROP invariant (2026-06-12 LPVV incident): the user's own file tags
    # are hand-curated and, for niche releases, often the only correct source.
    # Uncorroborated mapped local terms are retained as provisional (which the
    # materializer publishes to observed_leaf) — never silently queued away.
    report = fuse_hybrid_evidence(
        release_key="duster::stratosphere",
        evidence=[
            EvidenceTerm(term="rock", source_type="local_metadata", confidence=0.95),
        ],
        sparse_release=False,
    )

    assert report.accepted_genres == []
    assert [item.term for item in report.provisional_genres] == ["rock"]
    assert "local" in report.provisional_genres[0].reason.lower()


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

    # example.bandcamp.com does not match artist "duster" and hosts only one
    # artist in the store -> reclassified as bandcamp_unknown (no auto-accept).
    assert source_types == ["bandcamp_unknown", "model_prior"]


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


def test_fuse_release_evidence_injects_metadata_genres(tmp_path):
    """fuse_release_evidence pulls artist/album metadata.db genres in as evidence."""
    from types import SimpleNamespace
    from src.ai_genre_enrichment.hybrid_evidence import fuse_release_evidence
    from src.ai_genre_enrichment.storage import SidecarStore

    store = SidecarStore(str(tmp_path / "side.db"))
    store.initialize()
    release = SimpleNamespace(
        release_key="slowdive::souvlaki",
        normalized_artist="slowdive",
        normalized_album="souvlaki",
        album_id="alb1",
        existing_genres_by_source={"artist:musicbrainz_artist": ["shoegaze", "dream pop"]},
    )
    report = fuse_release_evidence(store, release)
    accepted = {d.term for d in report.accepted_genres}
    provisional = {d.term for d in report.provisional_genres}
    assert "shoegaze" in (accepted | provisional)


# ── Bandcamp source classification + storefront trust (2026-06-12) ──────────
# Root cause of the VV Torso/LPVV incident: a LABEL storefront page
# (jurassicpop.bandcamp.com) tagged a hardcore record "indie rock, pop, rock"
# and the fusion policy auto-accepted it at 0.95 while the user's correct file
# tags never entered the evidence stream. Bandcamp stays top-tier ONLY when
# the artist runs the page.


def test_classify_bandcamp_source_buckets():
    from src.ai_genre_enrichment.hybrid_evidence import classify_bandcamp_source

    # subdomain matches artist -> self-reported, artist-run
    assert classify_bandcamp_source(
        "billcallahan.bandcamp.com", "bill callahan", domain_artist_count=1
    ) == "bandcamp_artist"
    # domain hosts many artists in our store -> label storefront
    assert classify_bandcamp_source(
        "eccentricsoul.bandcamp.com", "tear drops", domain_artist_count=98
    ) == "bandcamp_label"
    # mismatched single-artist domain -> unknown operator (no auto-accept)
    assert classify_bandcamp_source(
        "jurassicpop.bandcamp.com", "vv torso", domain_artist_count=1
    ) == "bandcamp_unknown"


def test_storefront_bandcamp_does_not_auto_accept_even_with_recycled_acceptance():
    # The published LPVV row had evidence_count=2: the bandcamp page AND the
    # ai_enriched_accepted graduation OF that same page. Neither alone nor
    # together may they auto-accept.
    report = fuse_hybrid_evidence(
        release_key="vv torso::lpvv",
        evidence=[
            EvidenceTerm(term="pop", source_type="bandcamp_unknown", confidence=0.95),
            EvidenceTerm(term="pop", source_type="ai_enriched_accepted", confidence=0.88),
        ],
        sparse_release=False,
    )

    assert all(d.term != "pop" for d in report.accepted_genres)
    assert all(d.term != "pop" for d in report.provisional_genres)
    assert any(d.term == "pop" for d in report.needs_review)


def test_ai_enriched_accepted_alone_is_not_strong_evidence():
    report = fuse_hybrid_evidence(
        release_key="vv torso::lpvv",
        evidence=[
            EvidenceTerm(term="indie rock", source_type="ai_enriched_accepted", confidence=0.88),
        ],
        sparse_release=False,
    )

    assert report.accepted_genres == []
    assert report.provisional_genres == []


def test_lpvv_regression_local_tags_beat_storefront():
    # Full evidence mix as it exists for vv torso::lpvv, with file tags finally
    # in the stream: local tags publish, storefront genres wait for review,
    # standalone "indie" stays rejected noise.
    storefront = [
        EvidenceTerm(term=t, source_type="bandcamp_unknown", confidence=0.95)
        for t in ("indie", "indie rock", "pop", "rock")
    ] + [
        EvidenceTerm(term=t, source_type="ai_enriched_accepted", confidence=0.88)
        for t in ("indie", "indie rock", "pop", "rock")
    ]
    local = [
        EvidenceTerm(term=t, source_type="local_metadata", confidence=0.85)
        for t in ("hardcore", "post-punk", "punk")
    ]
    report = fuse_hybrid_evidence(
        release_key="vv torso::lpvv",
        evidence=storefront + local,
        sparse_release=False,
    )

    published = {d.term for d in report.accepted_genres} | {
        d.term for d in report.provisional_genres
    }
    assert {"hardcore", "post-punk", "punk"} <= published
    assert "pop" not in published
    assert "indie rock" not in published
    assert any(d.term == "indie" for d in report.rejected_noise)


def test_lastfm_only_terms_publish_capped_never_full_weight():
    # Publish dry-run regression (2026-06-12): 'baroque' on a Debussy record
    # and 'trip-hop' on an afrobeat record were lastfm-only terms that the old
    # rule promoted to provisional at FULL weight (score >= 0.90 with release
    # evidence present). The 2026-06-12 fix blocked lastfm-only in review
    # entirely; the zero-touch policy (2026-07-04) republishes it provisionally
    # but hard-capped at LASTFM_ONLY_CONFIDENCE_CAP, so 'baroque' can no longer
    # land at damaging full weight. Corroborated lastfm still counts through
    # the multi-source rules (unaffected here).
    report = fuse_hybrid_evidence(
        release_key="philharmonia::debussy nocturnes",
        evidence=[
            EvidenceTerm(term="contemporary classical", source_type="local_metadata", confidence=0.85),
            EvidenceTerm(term="baroque", source_type="lastfm_tags", confidence=0.99),
        ],
        sparse_release=False,
    )

    assert all(d.term != "baroque" for d in report.accepted_genres)
    [baroque] = [d for d in report.provisional_genres if d.term == "baroque"]
    assert baroque.basis == "lastfm_only"
    assert baroque.confidence <= 0.40
    assert all(d.term != "baroque" for d in report.needs_review)


def test_empty_sentinel_terms_are_dropped_from_fusion():
    # "__EMPTY__" is a data-pipeline sentinel, not a genre. Found leaking into
    # provisional via local_metadata source pages during the 2026-06-12 dry-run.
    report = fuse_hybrid_evidence(
        release_key="vv torso::lpvvii",
        evidence=[
            EvidenceTerm(term="__EMPTY__", source_type="local_metadata", confidence=0.85),
            EvidenceTerm(term="__empty__", source_type="musicbrainz", confidence=0.75),
            EvidenceTerm(term="hardcore", source_type="local_metadata", confidence=0.85),
        ],
        sparse_release=False,
    )

    all_terms = (
        {d.term for d in report.accepted_genres}
        | {d.term for d in report.provisional_genres}
        | {d.term for d in report.needs_review}
        | {d.term for d in report.rejected_noise}
    )
    assert "__empty__" not in all_terms
    assert "__EMPTY__" not in all_terms
    assert "hardcore" in {d.term for d in report.provisional_genres}


def test_local_corroborated_by_second_source_is_accepted():
    report = fuse_hybrid_evidence(
        release_key="vv torso::lpvv",
        evidence=[
            EvidenceTerm(term="hardcore", source_type="local_metadata", confidence=0.85),
            EvidenceTerm(term="hardcore", source_type="ai_check_metadata", confidence=0.70),
        ],
        sparse_release=False,
    )

    assert [d.term for d in report.accepted_genres] == ["hardcore"]
    assert "local_metadata" in report.accepted_genres[0].sources


def test_collect_reclassifies_artist_run_bandcamp_page(tmp_path: Path):
    from src.ai_genre_enrichment.storage import SidecarStore

    store = SidecarStore(tmp_path / "sidecar.db")
    store.initialize()
    page_id = store.upsert_source_page(
        release_key="duster::stratosphere",
        normalized_artist="duster",
        normalized_album="stratosphere",
        album_id="a1",
        source_url="https://duster.bandcamp.com/album/stratosphere",
        source_type="bandcamp_release",
        identity_status="confirmed",
        identity_confidence=0.95,
        evidence_summary="Bandcamp release tags.",
    )
    store.replace_source_tags(page_id, ["slowcore"])
    store.classify_source_tags(page_id)

    evidence = collect_hybrid_evidence(store, "duster::stratosphere")
    types = {item.source_type for item in evidence if item.term == "slowcore"}
    assert types == {"bandcamp_artist"}


def test_fuse_release_evidence_injects_local_file_tags(tmp_path):
    # track:file / album:file genres are the user's own curation — they MUST
    # enter fusion as local_metadata. (Before 2026-06-12 the track:* prefix was
    # skipped wholesale, which is how LPVV's hardcore/post-punk/punk vanished.)
    from types import SimpleNamespace
    from src.ai_genre_enrichment.hybrid_evidence import fuse_release_evidence
    from src.ai_genre_enrichment.storage import SidecarStore

    store = SidecarStore(str(tmp_path / "side.db"))
    store.initialize()
    release = SimpleNamespace(
        release_key="vv torso::lpvv",
        normalized_artist="vv torso",
        normalized_album="lpvv",
        album_id="alb1",
        existing_genres_by_source={
            "track:file": ["hardcore", "post-punk", "punk"],
            "album:file": ["rock"],
            "artist:lastfm": ["seen live"],
            "track:lastfm": ["noise"],
        },
    )
    report = fuse_release_evidence(store, release)
    published = {d.term for d in report.accepted_genres} | {
        d.term for d in report.provisional_genres
    }
    assert {"hardcore", "post-punk", "punk", "rock"} <= published
    assert "seen live" not in published
    assert "noise" not in published
