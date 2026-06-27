from pathlib import Path

import yaml

from src.ai_genre_enrichment.graph_growth import GrowthProposal
from src.ai_genre_enrichment.layered_taxonomy import load_layered_taxonomy
from src.ai_genre_enrichment import taxonomy_apply as ta


def _seed_taxonomy(tmp_path: Path) -> Path:
    data = {
        "taxonomy_version": "0.8.0-test",
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
    p = tmp_path / "tax.yaml"
    p.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return p


def _add(name, parent="rock"):
    return ta.Decision(
        term=name, verdict="add",
        proposal=GrowthProposal(
            name=name, kind="genre", status="active", specificity_score=0.6,
            parent_edges=[{"target": parent, "edge_type": "family_context",
                           "weight": 0.5, "confidence": 0.8}],
            term_kind_confirm="genre"),
        reject_reason=None, rationale="")


def test_apply_add_writes_record_and_bumps_version(tmp_path):
    path = _seed_taxonomy(tmp_path)
    result = ta.apply_decisions(
        path, [_add("dreampop")],
        new_version="0.9.0-gui-20260626-grown", backup_dir=tmp_path / "bak")
    assert result.added == 1
    assert result.validation_failures == []
    tax = load_layered_taxonomy(path)
    assert tax.genre_by_name("dreampop") is not None
    assert tax.version == "0.9.0-gui-20260626-grown"
    assert Path(result.backup_path).exists()


def test_apply_alias_writes_alias(tmp_path):
    path = _seed_taxonomy(tmp_path)
    d = ta.Decision(term="shoe gaze", verdict="alias",
                    proposal=GrowthProposal(
                        name="shoe gaze", kind="alias", status="alias_only",
                        specificity_score=0.0, canonical_target="shoegaze",
                        term_kind_confirm="genre"),
                    reject_reason=None, rationale="")
    result = ta.apply_decisions(path, [d], new_version="0.9.0-x", backup_dir=tmp_path / "b")
    assert result.aliased == 1
    tax = load_layered_taxonomy(path)
    # alias resolves to the canonical genre
    assert tax.genre_by_name("shoe gaze").name == "shoegaze"


def test_apply_reject_writes_reject_record(tmp_path):
    path = _seed_taxonomy(tmp_path)
    d = ta.Decision(term="my list", verdict="reject", proposal=None,
                    reject_reason="user_list", rationale="not a genre")
    result = ta.apply_decisions(path, [d], new_version="0.9.0-x", backup_dir=tmp_path / "b")
    assert result.rejected == 1
    tax = load_layered_taxonomy(path)  # loads -> reject_reason enum-validated
    assert tax.rejected_term_by_name("my list") is not None


def test_apply_aborts_on_validation_failure_without_writing(tmp_path):
    path = _seed_taxonomy(tmp_path)
    before = path.read_text(encoding="utf-8")
    bad = _add("bad", parent="no_such_parent")  # parent doesn't exist -> fail
    result = ta.apply_decisions(path, [bad], new_version="0.9.0-x", backup_dir=tmp_path / "b")
    assert result.validation_failures  # non-empty
    assert result.added == 0
    assert path.read_text(encoding="utf-8") == before  # untouched


def test_apply_orders_same_batch_forward_reference(tmp_path):
    # "childnew" parents on "parentnew", which is ALSO new in this batch.
    path = _seed_taxonomy(tmp_path)
    parentnew = _add("parentnew")  # parents on rock (exists)
    child = _add("childnew", parent="parentnew")  # parents on a same-batch new term
    result = ta.apply_decisions(
        path, [child, parentnew],  # deliberately out of order
        new_version="0.9.0-x", backup_dir=tmp_path / "b")
    assert result.validation_failures == []
    assert result.added == 2
    tax = load_layered_taxonomy(path)
    assert tax.genre_by_name("childnew") is not None
    assert tax.genre_by_name("parentnew") is not None
