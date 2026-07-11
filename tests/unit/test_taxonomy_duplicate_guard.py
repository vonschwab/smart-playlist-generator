"""Regression guard: the taxonomy editor must not re-offer or duplicate a term
that is already recorded.

Root cause (2026-07-11): "does this term already exist?" was asked with
context-free lookups that structurally cannot see a *conditional* alias — a
conditional alias resolves to None without context, so a term already recorded
as one leaked into the untriaged queue (`gather_growth_candidates`) AND passed
the apply-time collision guard (`validate_proposal` / `_name_exists`), appending
a duplicate that silently shadowed the conditional record. `_name_exists` also
never consulted the reject registry. See `LayeredTaxonomy.name_is_recorded`.
"""
from src.ai_genre_enrichment import graph_growth
from src.ai_genre_enrichment.graph_growth import GrowthProposal, validate_proposal
from src.ai_genre_enrichment.layered_taxonomy import (
    CanonicalFacet, CanonicalGenre, GenreAlias, LayeredTaxonomy, RejectedTerm,
)
from src.ai_genre_enrichment.storage import SidecarStore


def _tax() -> LayeredTaxonomy:
    """A tiny taxonomy carrying one of each record kind, including the load-bearing
    case: a *conditional* alias ('twee' -> twee pop, requires pop/indie context)."""
    genres = (
        CanonicalGenre(genre_id="twee_pop", name="twee pop", kind="subgenre",
                       specificity_score=0.8, status="active",
                       taxonomy_version="test", role="leaf"),
    )
    aliases = (
        GenreAlias(alias="twee", canonical_genre_id="twee_pop", source="curated",
                   confidence=1.0,
                   alias_policy={"type": "conditional",
                                 "requires_any_context": ["pop", "indie"]}),
        GenreAlias(alias="tweepop", canonical_genre_id="twee_pop", source="curated",
                   confidence=1.0, alias_policy={"type": "plain"}),
    )
    facets = (
        CanonicalFacet(facet_id="lo_fi", name="lo-fi", facet_type="production",
                       status="active"),
    )
    rejected = (RejectedTerm(term="great lyrics", reason="source_noise", notes="x"),)
    return LayeredTaxonomy(version="test", genres=genres, aliases=aliases,
                           edges=(), facets=facets, bridge_rules=(),
                           rejected_terms=rejected)


def test_name_is_recorded_sees_every_record_kind_context_free():
    tax = _tax()
    assert tax.name_is_recorded("twee pop")       # canonical genre
    assert tax.name_is_recorded("twee")           # CONDITIONAL alias (the bug)
    assert tax.name_is_recorded("tweepop")        # plain alias
    assert tax.name_is_recorded("lo-fi")          # facet
    assert tax.name_is_recorded("great lyrics")   # reject
    assert not tax.name_is_recorded("shoegaze")   # genuinely unknown


def test_validate_proposal_blocks_alias_that_shadows_a_conditional_alias():
    proposal = GrowthProposal(
        name="twee", kind="alias", status="alias_only", specificity_score=0.0,
        canonical_target="twee pop", term_kind_confirm="genre",
    )
    errors = validate_proposal(_tax(), proposal)
    assert any("already exists" in e for e in errors), errors


def test_validate_proposal_blocks_add_that_shadows_a_reject():
    proposal = GrowthProposal(
        name="great lyrics", kind="genre", status="active", specificity_score=0.5,
        parent_edges=[{"target": "twee pop", "edge_type": "is_a",
                       "weight": 0.8, "confidence": 0.8}],
        term_kind_confirm="genre",
    )
    errors = validate_proposal(_tax(), proposal)
    assert any("already exists" in e for e in errors), errors


def _page_with_tags(store, release_key, artist, album, tags):
    page_id = store.upsert_source_page(
        release_key=release_key, normalized_artist=artist, normalized_album=album,
        album_id=None, source_url=f"lastfm://{release_key}/{album}",
        source_type="lastfm_tags", identity_status="confirmed",
        identity_confidence=0.9, evidence_summary="t",
    )
    store.replace_source_tags(page_id, tags)


def test_gather_candidates_excludes_already_recorded_conditional_alias(tmp_path):
    store = SidecarStore(tmp_path / "sidecar.db")
    store.initialize()
    tax = _tax()
    # 'twee' is already a conditional alias -> must NOT be offered for adjudication.
    # 'shoegaze' is genuinely unknown -> still a candidate.
    for i in range(3):
        _page_with_tags(store, f"a{i}::alb{i}", f"a{i}", f"alb{i}",
                        ["twee", "shoegaze"])
    cands = graph_growth.gather_growth_candidates(store, tax, min_album_freq=3)
    terms = {c.term for c in cands}
    assert "twee" not in terms
    assert "shoegaze" in terms
