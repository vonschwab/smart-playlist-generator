from pathlib import Path

import yaml

from src.ai_genre_enrichment.layered_taxonomy import load_layered_taxonomy
from src.ai_genre_enrichment.taxonomy_decision_store import TaxonomyDecisionStore
from src.ai_genre_enrichment import taxonomy_review_queue as trq


def _tiny_taxonomy(tmp_path: Path) -> Path:
    """A minimal records-based taxonomy with one family so loads validate."""
    data = {
        "taxonomy_version": "0.0.1-test",
        "enums": {
            "reject_reason": ["source_noise", "malformed", "label"],
            "facet_type": ["mood", "instrumentation"],
        },
        "records": [
            {"name": "rock", "kind": "family", "status": "active",
             "specificity_score": 0.05, "parent_edges": []},
        ],
    }
    p = tmp_path / "tax.yaml"
    p.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return p


class _FakeStore:
    """Stands in for SidecarStore.all_collected_tags()."""

    def __init__(self, rows):
        self._rows = rows

    def all_collected_tags(self):
        return self._rows


def _row(tag, rk, artist="A", album="B"):
    return {"normalized_tag": tag, "release_key": rk,
            "normalized_artist": artist, "normalized_album": album}


def test_build_candidate_index_finds_unmapped_terms(tmp_path):
    tax = load_layered_taxonomy(_tiny_taxonomy(tmp_path))
    # "vaporwave" appears on 3 distinct releases -> a candidate; "rock" is mapped.
    rows = [
        _row("vaporwave", "r1"), _row("vaporwave", "r2"), _row("vaporwave", "r3"),
        _row("rock", "r1"),
    ]
    index = trq.build_candidate_index(_FakeStore(rows), tax, min_album_freq=3)
    assert "vaporwave" in index
    assert index["vaporwave"].album_frequency == 3
    assert index["vaporwave"].source == "growth"
    assert "rock" not in index  # mapped -> excluded


def test_build_candidate_index_drops_deterministic_noise(tmp_path):
    # The deterministic noise policy (classify_source_tag, backed by the real
    # genre_vocabulary.yaml) already knows these are non-genres/facets — they must
    # NOT flood the graph-adjudication queue. Only genuine unknowns survive.
    tax = load_layered_taxonomy(_tiny_taxonomy(tmp_path))
    rows = []
    for tag in ["new york", "2016", "piano", "vinyl", "chill", "instrumental"]:
        rows += [_row(tag, f"{tag}-r{i}") for i in range(3)]  # place/year/instrument/format/mood/descriptor
    rows += [_row("vaporwave", f"vw-r{i}") for i in range(3)]  # review_only -> genuine candidate
    rows += [_row("slowcore", f"sc-r{i}") for i in range(3)]   # genre_style -> known genre, keep
    index = trq.build_candidate_index(_FakeStore(rows), tax, min_album_freq=3)
    assert "vaporwave" in index
    assert "slowcore" in index
    for noise in ["new york", "2016", "piano", "vinyl", "chill", "instrumental"]:
        assert noise not in index, f"deterministic non-genre leaked into queue: {noise}"


def test_build_candidate_index_collapses_spacing_variants(tmp_path):
    tax = load_layered_taxonomy(_tiny_taxonomy(tmp_path))
    rows = [
        _row("vapor wave", "r1"), _row("vapor wave", "r2"), _row("vapor wave", "r3"),
        _row("vaporwave", "r4"), _row("vaporwave", "r5"),
    ]
    index = trq.build_candidate_index(_FakeStore(rows), tax, min_album_freq=3)
    # collapsed into one representative with summed frequency
    assert len(index) == 1
    rep = next(iter(index.values()))
    assert rep.album_frequency == 5
    assert "vaporwave" in rep.variants or "vapor wave" in rep.variants


def test_list_page_joins_decisions_and_filters_status(tmp_path, monkeypatch):
    tax_path = _tiny_taxonomy(tmp_path)
    db = tmp_path / "ai_genre_enrichment.db"
    rows = [_row("vaporwave", "r1"), _row("vaporwave", "r2"), _row("vaporwave", "r3"),
            _row("slowcore", "r1"), _row("slowcore", "r2"), _row("slowcore", "r3")]

    # Patch SidecarStore so list_page reads our fake collected tags.
    monkeypatch.setattr(trq, "_open_store_readonly", lambda p: _FakeStore(rows))

    store = TaxonomyDecisionStore(db)
    store.record_decision(term="slowcore", raw_term="slowcore", verdict="add",
                          proposal_json="{}", claude_json="{}", human_edited=0)
    store.close()

    untriaged = trq.list_page(db, tax_path, status="untriaged")
    decided = trq.list_page(db, tax_path, status="decided")
    assert [t["term"] for t in untriaged["terms"]] == ["vaporwave"]
    assert untriaged["untriaged_terms"] == 1
    assert untriaged["decided_terms"] == 1
    assert [t["term"] for t in decided["terms"]] == ["slowcore"]
    assert decided["terms"][0]["decision"]["verdict"] == "add"
