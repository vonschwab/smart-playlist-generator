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

    assert taxonomy.version == "layered-genre-graph-v0"
    assert taxonomy.genre_by_name("jangle pop").genre_id == "jangle_pop"
    assert taxonomy.genre_by_name("jangle-pop").genre_id == "jangle_pop"
    assert taxonomy.facet_by_name("lo-fi").facet_type == "production"

    parent_names = {genre.name for genre in taxonomy.parents_for_genre("jangle_pop")}
    family_names = {genre.name for genre in taxonomy.families_for_genre("jangle_pop")}
    bridge = taxonomy.bridge_rule_for("jangle_pop", "twee_pop")

    assert parent_names == {"indie pop"}
    assert family_names == {"pop", "rock"}
    assert bridge is not None
    assert bridge.mode_allowed == ("narrow", "dynamic", "discover")


def test_layered_taxonomy_rejects_unknown_aliases():
    from src.ai_genre_enrichment.layered_taxonomy import load_default_layered_taxonomy

    taxonomy = load_default_layered_taxonomy()

    assert taxonomy.genre_by_name("rare sad girl") is None
    assert taxonomy.facet_by_name("spotify") is None
