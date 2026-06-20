from __future__ import annotations

from src.ai_genre_enrichment.adjudication_materializer import (
    ADJUDICATOR_SOURCE,
    compute_adjudication_rows,
)
from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy

TAX = load_default_layered_taxonomy()


def _resp(genres, facets=None):
    return {
        "genres": [{"term": t, "confidence": 0.8, "layer": "core"} for t in genres],
        "facets": [{"term": t, "facet_type": ft} for t, ft in (facets or [])],
        "escalate": False, "overall_confidence": 0.8,
    }


def test_leaf_expands_to_observed_parent_and_family():
    # shoegaze is a stable leaf with parent/family edges in the taxonomy
    genre_rows, facet_rows, skipped = compute_adjudication_rows(
        _resp(["shoegaze"]), TAX, prompt_version="pv", model="haiku")
    layers = {(r["genre_id"], r["assignment_layer"]) for r in genre_rows}
    sg = TAX.genre_by_name("shoegaze")
    assert (sg.genre_id, "observed_leaf") in layers
    assert any(layer == "inferred_family" for _, layer in layers)
    assert skipped == []


def test_facet_term_routes_to_facets_not_genres():
    # 'lo-fi' is a facet; it must never land in genre_rows
    genre_rows, facet_rows, skipped = compute_adjudication_rows(
        _resp(["shoegaze", "lo-fi"]), TAX, prompt_version="pv", model="haiku")
    genre_names = {TAX.genre_by_id(r["genre_id"]).name for r in genre_rows}
    assert "lo-fi" not in genre_names
    assert any(r["source"] == ADJUDICATOR_SOURCE for r in facet_rows)


def test_unknown_term_is_skipped_not_invented():
    genre_rows, facet_rows, skipped = compute_adjudication_rows(
        _resp(["shoegaze", "xyzzy not a real genre"]), TAX,
        prompt_version="pv", model="haiku")
    assert "xyzzy not a real genre" in skipped
    # only the real leaf produced an observed_leaf row
    obs = [r for r in genre_rows if r["assignment_layer"] == "observed_leaf"]
    assert len(obs) == 1


def test_provenance_and_reliability_stamped():
    genre_rows, _, _ = compute_adjudication_rows(
        _resp(["shoegaze"]), TAX, prompt_version="pv-X", model="sonnet")
    row = next(r for r in genre_rows if r["assignment_layer"] == "observed_leaf")
    assert row["source_reliability"] == 0.85
    assert row["provenance"]["source"] == ADJUDICATOR_SOURCE
    assert row["provenance"]["prompt_version"] == "pv-X"
    assert row["provenance"]["model"] == "sonnet"


def test_compound_facet_string_is_split_into_atomic_terms():
    # Use facet atoms known to exist in the taxonomy facet vocabulary.
    response = {
        "genres": [{"term": "shoegaze", "confidence": 0.9}],
        "facets": [{"term": "instrumental, lo-fi"}],
        "overall_confidence": 0.8,
    }
    _, facet_rows, _ = compute_adjudication_rows(
        response, TAX, prompt_version="pv", model="sonnet")
    facet_ids = {r["facet_id"] for r in facet_rows}
    # both atoms resolve to facets and are present; the compound string is NOT a single row
    assert "instrumental" in {TAX.facet_by_id(fid).name for fid in facet_ids}
    assert "lo-fi" in {TAX.facet_by_id(fid).name for fid in facet_ids}
