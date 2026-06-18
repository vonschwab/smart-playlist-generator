import sqlite3


def test_sidecar_initializes_layered_genre_graph_tables(tmp_path):
    from src.ai_genre_enrichment.storage import SidecarStore

    store = SidecarStore(tmp_path / "sidecar.db")
    store.initialize()

    expected_tables = {
        "genre_graph_canonical_genres",
        "genre_graph_aliases",
        "genre_graph_edges",
        "genre_graph_canonical_facets",
        "genre_graph_bridge_rules",
        "genre_graph_release_genre_assignments",
        "genre_graph_release_facet_assignments",
    }
    with sqlite3.connect(store.db_path) as conn:
        actual_tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name LIKE 'genre_graph_%'"
            )
        }

    assert expected_tables <= actual_tables


def test_layered_taxonomy_loads_seed_aliases_parents_and_bridge_rules():
    from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy

    taxonomy = load_default_layered_taxonomy()

    # Version is a moving target (grows with each SP3a pass); assert it loaded a
    # well-formed version rather than pinning a literal. Structural assertions
    # below are the real verification that the default taxonomy loaded.
    assert isinstance(taxonomy.version, str) and taxonomy.version
    assert taxonomy.genre_by_name("jangle pop").genre_id == "jangle_pop"
    assert taxonomy.facet_by_name("lo-fi").facet_type == "production"

    parent_names = {genre.name for genre in taxonomy.parents_for_genre("jangle_pop")}
    family_names = {genre.name for genre in taxonomy.families_for_genre("jangle_pop")}
    bridge = taxonomy.bridge_rule_for("jangle_pop", "twee_pop")

    assert parent_names == {"indie pop"}
    assert family_names == {"indie/alternative", "pop", "rock"}
    assert bridge is not None
    assert bridge.mode_allowed == ("strict", "narrow", "dynamic", "discover")


def test_layered_taxonomy_rejects_unknown_aliases():
    from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy

    taxonomy = load_default_layered_taxonomy()

    assert taxonomy.genre_by_name("rare sad girl") is None
    assert taxonomy.facet_by_name("spotify") is None


def test_reviewed_structured_taxonomy_validation_and_policy_fields():
    from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy

    taxonomy = load_default_layered_taxonomy()

    assert taxonomy.genre_by_name("art pop").status == "active"
    assert taxonomy.genre_by_name("space rock").status == "active"
    assert taxonomy.genre_by_name("slacker rock").status == "active"
    assert taxonomy.genre_by_name("pop rock").status == "active"
    assert taxonomy.facet_by_name("lo-fi").facet_type == "production"
    assert taxonomy.facet_by_name("instrumental").facet_type == "function"
    assert taxonomy.rejected_term_by_name("pop/rock").reason == "retail_bucket"
    assert taxonomy.rejected_term_by_name("indie").reason == "source_noise"


def test_reviewed_taxonomy_conditional_aliases_require_context():
    from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy

    taxonomy = load_default_layered_taxonomy()

    assert taxonomy.alias_target_for_name("modal") is None
    assert taxonomy.alias_target_for_name("modal", context_terms=["jazz"]) == taxonomy.genre_by_name("modal jazz")
    assert taxonomy.alias_target_for_name("twee") is None
    assert taxonomy.alias_target_for_name("twee", context_terms=["indie pop"]) == taxonomy.genre_by_name("twee pop")


def test_reviewed_taxonomy_rejects_bad_structured_references(tmp_path):
    from src.ai_genre_enrichment.layered_taxonomy import load_layered_taxonomy

    bad = tmp_path / "bad.yaml"
    bad.write_text(
        """
schema_version: 1
taxonomy_version: bad
records:
  - name: bad alias
    kind: alias
    role: alias
    status: alias_only
    canonical_target: missing target
    parent_edges: []
    secondary_roles: []
    alias_policy:
      type: plain
bridge_rules: []
""".strip(),
        encoding="utf-8",
    )

    try:
        load_layered_taxonomy(bad)
    except ValueError as exc:
        assert "Alias points to unknown genre" in str(exc)
    else:
        raise AssertionError("Expected structured taxonomy validation failure")
