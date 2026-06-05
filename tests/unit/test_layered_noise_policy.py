def test_layered_taxonomy_classifies_family_leaf_facet_alias_reject_and_review():
    from src.ai_genre_enrichment.layered_assignment import classify_layered_term
    from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy

    taxonomy = load_default_layered_taxonomy()

    assert classify_layered_term(taxonomy, "rock").term_kind == "family"
    assert classify_layered_term(taxonomy, "jangle pop").term_kind == "leaf"
    assert classify_layered_term(taxonomy, "lo-fi").term_kind == "facet"

    alias = classify_layered_term(taxonomy, "jangle-pop")
    assert alias.term_kind == "alias"
    assert alias.canonical_id == "jangle_pop"
    assert alias.canonical_kind == "leaf"

    reject = classify_layered_term(taxonomy, "seen live")
    assert reject.term_kind == "reject"
    assert reject.reason == "Known non-genre/noise term."

    review = classify_layered_term(taxonomy, "rare wistful kitchen pop")
    assert review.term_kind == "review"
    assert review.reason == "Unknown layered taxonomy term."


def test_layered_taxonomy_treats_indie_as_broad_context_not_leaf():
    from src.ai_genre_enrichment.layered_assignment import classify_layered_term
    from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy

    taxonomy = load_default_layered_taxonomy()

    indie = classify_layered_term(taxonomy, "indie")
    indie_pop = classify_layered_term(taxonomy, "indie pop")

    assert indie.term_kind == "family"
    assert indie.canonical_id == "indie_context"
    assert indie_pop.term_kind == "leaf"
    assert indie_pop.canonical_id == "indie_pop"


def test_graph_report_counts_rejected_terms(tmp_path):
    from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy
    from src.ai_genre_enrichment.storage import SidecarStore

    store = SidecarStore(tmp_path / "sidecar.db")
    store.initialize()
    store.upsert_layered_taxonomy(load_default_layered_taxonomy())

    report = store.layered_taxonomy_report()

    assert report["rejected_term_count"] >= 5

