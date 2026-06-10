"""Tests for src/genre/graph_adapter.py — read-only SP3a taxonomy adapter.

Stage 1 of the graph-taxonomy integration: these tests pin the adapter's
canonicalization semantics before any generation-path wiring exists.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.genre.graph_adapter import (
    CanonicalizationResult,
    GenreGraphAdapter,
    load_graph_adapter,
)

FIXTURE = {
    "taxonomy_version": "test-0.1",
    "records": [
        {"name": "rock", "kind": "family", "status": "active", "specificity_score": 0.2},
        {"name": "dance", "kind": "umbrella", "status": "active", "specificity_score": 0.3},
        {
            "name": "shoegaze",
            "kind": "genre",
            "status": "active",
            "specificity_score": 0.62,
            "parent_edges": [
                {"target": "rock", "edge_type": "family_context", "weight": 0.5, "confidence": 0.8},
            ],
        },
        {
            "name": "dream pop",
            "kind": "genre",
            "status": "active",
            "specificity_score": 0.6,
            "parent_edges": [
                {"target": "shoegaze", "edge_type": "bridge_to", "weight": 0.4, "confidence": 0.6},
            ],
        },
        {"name": "chillwave", "kind": "genre", "status": "review", "specificity_score": 0.55},
        {"name": "nu gaze", "kind": "alias", "status": "alias_only", "canonical_target": "shoegaze"},
        # Alias whose name collides with a canonical genre — canonical must win.
        {"name": "dream pop", "kind": "alias", "status": "alias_only", "canonical_target": "shoegaze"},
        {"name": "lo-fi", "kind": "facet", "facet_type": "production", "status": "active"},
        {"name": "lofi", "kind": "alias", "status": "alias_only", "canonical_target": "lo-fi"},
        {"name": "seen live", "kind": "reject", "status": "rejected", "reject_reason": "source_noise"},
        {
            "name": "garage",
            "kind": "alias",
            "status": "alias_only",
            "canonical_target": "shoegaze",
            "alias_policy": {"type": "conditional", "requires_any_context": ["rock"]},
        },
    ],
}


@pytest.fixture()
def adapter(tmp_path: Path) -> GenreGraphAdapter:
    path = tmp_path / "taxonomy.yaml"
    path.write_text(yaml.safe_dump(FIXTURE, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return load_graph_adapter(path)


# --- canonicalization -------------------------------------------------------


def test_canonical_term_resolves_with_node_metadata(adapter: GenreGraphAdapter):
    result = adapter.canonicalize_tag("Shoegaze")
    assert isinstance(result, CanonicalizationResult)
    assert result.resolution == "canonical"
    assert result.canonical == "shoegaze"
    assert result.node is not None
    assert result.node.kind == "genre"
    assert result.node.status == "active"
    assert result.node.specificity_score == pytest.approx(0.62)
    assert result.node.is_broad is False


def test_normalization_variants_resolve_to_same_canonical(adapter: GenreGraphAdapter):
    for raw in ("dream pop", "Dream-Pop", "dream_pop", "  DREAM  POP  "):
        result = adapter.canonicalize_tag(raw)
        assert result.resolution == "canonical", raw
        assert result.canonical == "dream pop", raw


def test_alias_resolves_to_canonical_name_never_alias(adapter: GenreGraphAdapter):
    result = adapter.canonicalize_tag("nu gaze")
    assert result.resolution == "alias"
    assert result.canonical == "shoegaze"
    assert result.node is not None and result.node.name == "shoegaze"


def test_canonical_name_shadows_colliding_alias(adapter: GenreGraphAdapter):
    # "dream pop" exists both as a canonical genre and as an alias to shoegaze;
    # the canonical record must win.
    result = adapter.canonicalize_tag("dream pop")
    assert result.resolution == "canonical"
    assert result.canonical == "dream pop"


def test_rejected_term_is_flagged_and_never_canonical(adapter: GenreGraphAdapter):
    result = adapter.canonicalize_tag("Seen Live")
    assert result.resolution == "rejected"
    assert result.canonical is None
    assert result.reject_reason == "source_noise"
    assert adapter.is_active_genre("seen live") is False
    assert "seen live" not in adapter.active_genre_vocabulary()


def test_facet_term_does_not_become_genre(adapter: GenreGraphAdapter):
    result = adapter.canonicalize_tag("lo-fi")
    assert result.resolution == "facet"
    assert result.canonical is None
    assert result.facet_type == "production"
    assert "lo-fi" not in adapter.active_genre_vocabulary()
    assert adapter.is_active_genre("lo-fi") is False


def test_facet_targeted_alias_resolves_to_facet(adapter: GenreGraphAdapter):
    result = adapter.canonicalize_tag("lofi")
    assert result.resolution == "facet"
    assert result.canonical is None
    assert result.facet_type == "production"


def test_conditional_alias_requires_context(adapter: GenreGraphAdapter):
    without = adapter.canonicalize_tag("garage")
    assert without.resolution == "unknown"
    with_ctx = adapter.canonicalize_tag("garage", context_terms=["rock"])
    assert with_ctx.resolution == "alias"
    assert with_ctx.canonical == "shoegaze"


def test_unknown_term_passes_through_as_unknown(adapter: GenreGraphAdapter):
    result = adapter.canonicalize_tag("xyzzy unknown genre")
    assert result.resolution == "unknown"
    assert result.canonical is None
    assert result.node is None


# --- vocabulary and node metadata ------------------------------------------


def test_active_vocabulary_excludes_review_facets_rejects(adapter: GenreGraphAdapter):
    vocab = adapter.active_genre_vocabulary()
    assert vocab == sorted(vocab)
    assert "shoegaze" in vocab and "dream pop" in vocab
    assert "chillwave" not in vocab  # status: review
    assert "lo-fi" not in vocab  # facet
    assert "seen live" not in vocab  # reject
    assert "nu gaze" not in vocab  # alias

    with_review = adapter.active_genre_vocabulary(include_review=True)
    assert "chillwave" in with_review


def test_broad_nodes_are_marked(adapter: GenreGraphAdapter):
    assert adapter.node("rock").is_broad is True  # family
    assert adapter.node("dance").is_broad is True  # umbrella
    assert adapter.node("shoegaze").is_broad is False
    assert adapter.node("nonexistent") is None


def test_alias_map_targets_are_canonical_genre_names(adapter: GenreGraphAdapter):
    alias_map = adapter.alias_map()
    assert alias_map["nu gaze"] == "shoegaze"
    assert "lofi" not in alias_map  # facet-target alias is not a genre mapping
    assert "garage" not in alias_map  # conditional alias needs context, not static
    assert set(alias_map.values()) <= set(adapter.active_genre_vocabulary(include_review=True))


def test_rejected_terms_set(adapter: GenreGraphAdapter):
    assert adapter.rejected_terms() == {"seen live"}


# --- edges ------------------------------------------------------------------


def test_edges_emit_canonical_names(adapter: GenreGraphAdapter):
    edges = adapter.edges()
    by_pair = {(e.source, e.target): e for e in edges}
    edge = by_pair[("shoegaze", "rock")]
    assert edge.edge_type == "family_context"
    assert edge.weight == pytest.approx(0.5)
    assert edge.confidence == pytest.approx(0.8)
    assert ("dream pop", "shoegaze") in by_pair


def test_missing_edge_target_fails_load(tmp_path: Path):
    broken = {
        "taxonomy_version": "test-broken",
        "records": [
            {
                "name": "shoegaze",
                "kind": "genre",
                "status": "active",
                "specificity_score": 0.6,
                "parent_edges": [
                    {"target": "does not exist", "edge_type": "is_a", "weight": 0.7, "confidence": 0.9},
                ],
            },
        ],
    }
    path = tmp_path / "broken.yaml"
    path.write_text(yaml.safe_dump(broken, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError):
        load_graph_adapter(path)


# --- live taxonomy smoke -----------------------------------------------------


def test_live_taxonomy_loads_and_is_internally_consistent():
    adapter = load_graph_adapter()
    assert adapter.taxonomy_version
    vocab = adapter.active_genre_vocabulary()
    assert len(vocab) >= 400
    assert "shoegaze" in vocab

    known = set(adapter.active_genre_vocabulary(include_review=True))
    for edge in adapter.edges():
        assert edge.source in known, edge
        assert edge.target in known, edge

    for alias, target in adapter.alias_map().items():
        assert target in known, (alias, target)
        assert alias != target
