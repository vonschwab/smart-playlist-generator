def test_layered_taxonomy_classifies_family_leaf_facet_alias_reject_and_review():
    from src.ai_genre_enrichment.layered_assignment import classify_layered_term
    from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy

    taxonomy = load_default_layered_taxonomy()

    assert classify_layered_term(taxonomy, "rock").term_kind == "family"
    assert classify_layered_term(taxonomy, "jangle pop").term_kind == "leaf"
    assert classify_layered_term(taxonomy, "lo-fi").term_kind == "facet"

    alias = classify_layered_term(taxonomy, "jangle-pop")
    assert alias.term_kind == "leaf"
    assert alias.canonical_id == "jangle_pop"

    alias = classify_layered_term(taxonomy, "singer songwriter")
    assert alias.term_kind == "alias"
    assert alias.canonical_id == "singer_songwriter"
    assert alias.canonical_kind == "family"

    alias = classify_layered_term(taxonomy, "dreampop")
    assert alias.term_kind == "alias"
    assert alias.canonical_id == "dream_pop"
    assert alias.canonical_kind == "leaf"

    reject = classify_layered_term(taxonomy, "spotify")
    assert reject.term_kind == "reject"
    assert reject.reason == "source_noise"

    review = classify_layered_term(taxonomy, "rare wistful kitchen pop")
    assert review.term_kind == "review"
    assert review.reason == "Unknown layered taxonomy term."


def test_layered_taxonomy_rejects_standalone_indie_but_accepts_indie_rock():
    from src.ai_genre_enrichment.layered_assignment import classify_layered_term
    from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy

    taxonomy = load_default_layered_taxonomy()

    indie = classify_layered_term(taxonomy, "indie")
    indie_rock = classify_layered_term(taxonomy, "indie rock")
    indie_pop = classify_layered_term(taxonomy, "indie pop")

    assert indie.term_kind == "reject"
    assert indie.reason == "source_noise"
    assert indie_rock.term_kind == "leaf"
    assert indie_rock.canonical_id == "indie_rock"
    assert indie_pop.term_kind == "leaf"
    assert indie_pop.canonical_id == "indie_pop"


def test_layered_taxonomy_rejects_fake_pop_rock_bucket():
    from src.ai_genre_enrichment.layered_assignment import classify_layered_term
    from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy

    taxonomy = load_default_layered_taxonomy()

    rejected = classify_layered_term(taxonomy, "pop/rock")

    assert rejected.term_kind == "reject"
    assert rejected.reason == "retail_bucket"


def test_dream_pop_does_not_infer_pop_without_explicit_edge():
    from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy

    taxonomy = load_default_layered_taxonomy()

    families = {genre.genre_id for genre in taxonomy.families_for_genre("dream_pop")}

    assert "pop" not in families


def test_reviewed_taxonomy_facets_are_not_leaf_genres():
    from src.ai_genre_enrichment.layered_assignment import classify_layered_term
    from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy

    taxonomy = load_default_layered_taxonomy()

    assert classify_layered_term(taxonomy, "lo-fi").term_kind == "facet"
    assert classify_layered_term(taxonomy, "lofi").term_kind == "alias"
    assert classify_layered_term(taxonomy, "lofi").canonical_kind == "production"
    assert classify_layered_term(taxonomy, "instrumental").term_kind == "facet"


def test_graph_report_counts_rejected_terms(tmp_path):
    from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy
    from src.ai_genre_enrichment.storage import SidecarStore

    store = SidecarStore(tmp_path / "sidecar.db")
    store.initialize()
    store.upsert_layered_taxonomy(load_default_layered_taxonomy())

    report = store.layered_taxonomy_report()

    assert report["rejected_term_count"] >= 5
