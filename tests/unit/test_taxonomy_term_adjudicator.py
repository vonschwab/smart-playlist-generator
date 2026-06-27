import pytest
import yaml

from src.ai_genre_enrichment.graph_growth import GrowthProposal, validate_proposal
from src.ai_genre_enrichment.layered_taxonomy import load_layered_taxonomy
from src.ai_genre_enrichment import taxonomy_term_adjudicator as tta


def _taxonomy(tmp_path):
    data = {
        "taxonomy_version": "0.0.1-test",
        "enums": {"reject_reason": ["source_noise", "malformed", "label", "user_list"],
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
    p = tmp_path / "tax.yaml"
    p.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return load_layered_taxonomy(p)


def test_validate_add_maps_to_growthproposal(tmp_path):
    tax = _taxonomy(tmp_path)
    data = {
        "verdict": "add", "name": "dreampop", "kind": "genre", "status": "active",
        "specificity_score": 0.62,
        "parent_edges": [{"target": "rock", "edge_type": "family_context",
                          "weight": 0.5, "confidence": 0.8}],
        "similar_to": ["shoegaze"], "alias_variants": ["dream pop"],
        "reject_reason": "", "canonical_target": "", "rationale": "ok",
    }
    out = tta.validate_response(data, term="dreampop", taxonomy=tax)
    assert isinstance(out, GrowthProposal)
    assert out.kind == "genre"
    assert out.term_kind_confirm == "genre"
    # The mapped proposal must pass the real growth validator against this taxonomy.
    assert validate_proposal(tax, out) == []


def test_validate_alias_maps_to_alias_proposal(tmp_path):
    tax = _taxonomy(tmp_path)
    data = {"verdict": "alias", "name": "shoe gaze", "kind": "alias",
            "canonical_target": "shoegaze", "rationale": "spelling",
            "parent_edges": [], "similar_to": [], "alias_variants": [],
            "specificity_score": 0.0, "status": "alias_only", "reject_reason": ""}
    out = tta.validate_response(data, term="shoe gaze", taxonomy=tax)
    assert isinstance(out, GrowthProposal)
    assert out.kind == "alias"
    assert out.canonical_target == "shoegaze"
    assert validate_proposal(tax, out) == []


def test_validate_reject_maps_to_rejectverdict(tmp_path):
    tax = _taxonomy(tmp_path)
    data = {"verdict": "reject", "name": "my favorite albums", "kind": "reject",
            "reject_reason": "user_list", "rationale": "a personal list, not a genre",
            "parent_edges": [], "similar_to": [], "alias_variants": [],
            "specificity_score": 0.0, "status": "rejected", "canonical_target": ""}
    out = tta.validate_response(data, term="my favorite albums", taxonomy=tax)
    assert isinstance(out, tta.RejectVerdict)
    assert out.reject_reason == "user_list"


def test_validate_reject_rejects_bad_reason(tmp_path):
    tax = _taxonomy(tmp_path)
    data = {"verdict": "reject", "name": "x", "reject_reason": "not_an_enum",
            "rationale": "", "parent_edges": [], "similar_to": [],
            "alias_variants": [], "specificity_score": 0.0}
    with pytest.raises(ValueError, match="reject_reason"):
        tta.validate_response(data, term="x", taxonomy=tax)


def test_validate_alias_unknown_target_raises(tmp_path):
    tax = _taxonomy(tmp_path)
    data = {"verdict": "alias", "name": "x", "canonical_target": "no_such_genre",
            "rationale": "", "parent_edges": [], "similar_to": [],
            "alias_variants": [], "specificity_score": 0.0}
    with pytest.raises(ValueError):
        tta.validate_response(data, term="x", taxonomy=tax)


def test_validate_add_leaf_without_parent_raises(tmp_path):
    tax = _taxonomy(tmp_path)
    data = {"verdict": "add", "name": "y", "kind": "genre", "status": "active",
            "specificity_score": 0.6, "parent_edges": [], "similar_to": [],
            "alias_variants": [], "reject_reason": "", "canonical_target": "",
            "rationale": ""}
    with pytest.raises(ValueError, match="parent edge"):
        tta.validate_response(data, term="y", taxonomy=tax)
