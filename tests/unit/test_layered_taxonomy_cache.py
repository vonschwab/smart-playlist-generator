from pathlib import Path

import pytest
import yaml

from src.ai_genre_enrichment.layered_taxonomy import load_layered_taxonomy


def _seed_taxonomy(tmp_path: Path, filename: str = "tax.yaml", version: str = "0.8.0-test") -> Path:
    data = {
        "taxonomy_version": version,
        "enums": {"reject_reason": ["user_list", "source_noise"],
                  "facet_type": ["mood"]},
        "records": [
            {"name": "rock", "kind": "family", "status": "active",
             "specificity_score": 0.05, "parent_edges": []},
            {"name": "shoegaze", "kind": "genre", "status": "active",
             "specificity_score": 0.6,
             "parent_edges": [{"target": "rock", "edge_type": "family_context",
                               "weight": 0.5, "confidence": 0.8}]},
        ],
    }
    p = tmp_path / filename
    p.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return p


def test_unchanged_file_returns_same_cached_object(tmp_path):
    path = _seed_taxonomy(tmp_path)
    first = load_layered_taxonomy(path)
    second = load_layered_taxonomy(path)
    assert first is second


def test_rewrite_with_different_content_invalidates_cache(tmp_path):
    path = _seed_taxonomy(tmp_path)
    first = load_layered_taxonomy(path)
    assert first.genre_by_name("dreampop") is None

    # Rewrite the same path with genuinely different valid content: bump the
    # version and add a new record.
    data = {
        "taxonomy_version": "0.9.0-test-grown",
        "enums": {"reject_reason": ["user_list", "source_noise"],
                  "facet_type": ["mood"]},
        "records": [
            {"name": "rock", "kind": "family", "status": "active",
             "specificity_score": 0.05, "parent_edges": []},
            {"name": "shoegaze", "kind": "genre", "status": "active",
             "specificity_score": 0.6,
             "parent_edges": [{"target": "rock", "edge_type": "family_context",
                               "weight": 0.5, "confidence": 0.8}]},
            {"name": "dreampop", "kind": "genre", "status": "active",
             "specificity_score": 0.6,
             "parent_edges": [{"target": "rock", "edge_type": "family_context",
                               "weight": 0.5, "confidence": 0.8}]},
        ],
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    second = load_layered_taxonomy(path)
    assert second is not first
    assert second.version == "0.9.0-test-grown"
    assert second.genre_by_name("dreampop") is not None


def test_two_different_paths_cache_independently(tmp_path):
    path_a = _seed_taxonomy(tmp_path, filename="tax_a.yaml", version="0.8.0-a")
    path_b = _seed_taxonomy(tmp_path, filename="tax_b.yaml", version="0.8.0-b")

    tax_a = load_layered_taxonomy(path_a)
    tax_b = load_layered_taxonomy(path_b)

    assert tax_a is not tax_b
    assert tax_a.version == "0.8.0-a"
    assert tax_b.version == "0.8.0-b"

    # Reloading A again still returns A's cached object, unaffected by B.
    assert load_layered_taxonomy(path_a) is tax_a


def test_missing_version_still_raises_value_error(tmp_path):
    data = {
        "records": [
            {"name": "rock", "kind": "family", "status": "active",
             "specificity_score": 0.05, "parent_edges": []},
        ],
    }
    path = tmp_path / "no_version.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="version"):
        load_layered_taxonomy(path)
