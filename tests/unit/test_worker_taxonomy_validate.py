# tests/unit/test_worker_taxonomy_validate.py
"""Worker round-trip for the manual ADD-wizard's validation command."""
import json

from src.playlist_gui.worker import handle_validate_taxonomy_proposal


def _events(capsys):
    return [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]


def _validate(capsys, proposal):
    handle_validate_taxonomy_proposal({
        "cmd": "validate_taxonomy_proposal", "request_id": "r1", "proposal": proposal})
    events = _events(capsys)
    result = next(e for e in events if e["type"] == "result")
    done = next(e for e in events if e["type"] == "done")
    assert result["result_type"] == "taxonomy_validate"
    return result["errors"], done


def _genre(**over):
    p = {
        "name": "xyzzy wizard genre", "kind": "genre", "status": "active",
        "specificity_score": 0.55,
        "parent_edges": [{"target": "indie rock", "edge_type": "is_a",
                          "weight": 0.75, "confidence": 0.85}],
        "similar_to": [], "alias_variants": [], "term_kind_confirm": "genre",
        "facet_type": None, "canonical_target": None, "rationale": "",
    }
    p.update(over)
    return p


def test_valid_genre_proposal_returns_no_errors(capsys):
    errors, done = _validate(capsys, _genre())
    assert errors == []
    assert done["ok"] is True


def test_leaf_without_parent_is_rejected(capsys):
    errors, _ = _validate(capsys, _genre(parent_edges=[]))
    assert any("parent edge" in e for e in errors)


def test_nonexistent_parent_target_is_rejected(capsys):
    errors, _ = _validate(capsys, _genre(
        parent_edges=[{"target": "xyzzy nonexistent parent", "edge_type": "is_a",
                       "weight": 0.75, "confidence": 0.85}]))
    assert any("does not exist" in e for e in errors)


def test_duplicate_name_is_rejected(capsys):
    errors, _ = _validate(capsys, _genre(name="shoegaze"))
    assert any("already exists" in e for e in errors)


def test_facet_with_parent_edges_is_rejected(capsys):
    errors, _ = _validate(capsys, _genre(
        kind="facet", term_kind_confirm="facet", facet_type="instrumentation"))
    assert any("Facet proposals cannot have parent_edges" in e for e in errors)


def test_valid_facet_proposal_returns_no_errors(capsys):
    errors, done = _validate(capsys, _genre(
        name="xyzzy wizard facet", kind="facet", term_kind_confirm="facet",
        facet_type="instrumentation", parent_edges=[]))
    assert errors == []
    assert done["ok"] is True


def test_malformed_proposal_reports_error_not_crash(capsys):
    handle_validate_taxonomy_proposal({
        "cmd": "validate_taxonomy_proposal", "request_id": "r1",
        "proposal": {"specificity_score": "not-a-number"}})
    events = _events(capsys)
    done = next(e for e in events if e["type"] == "done")
    assert done["ok"] is False
