# tests/unit/test_taxonomy_wizard_apply_e2e.py
"""A wizard-shaped ADD decision flows through record -> apply into the YAML.

The real taxonomy is copied to tmp and DEFAULT_TAXONOMY_PATH monkeypatched —
the canonical data/layered_genre_taxonomy.yaml is never written.
"""
import json
import shutil

import yaml

import src.ai_genre_enrichment.layered_taxonomy as lt
from src.playlist_gui.worker import (
    handle_apply_taxonomy_decisions,
    handle_record_taxonomy_decision,
)

WIZARD_PROPOSAL = {  # exactly what TaxonomyAddWizard.buildProposal() stages
    "name": "xyzzy wizard genre", "kind": "genre", "status": "active",
    "specificity_score": 0.55,
    "parent_edges": [{"target": "indie rock", "edge_type": "is_a",
                      "weight": 0.75, "confidence": 0.85}],
    "similar_to": [], "alias_variants": ["xyzzy wizard variant"],
    "term_kind_confirm": "genre", "facet_type": None, "canonical_target": None,
    "rationale": "e2e",
}


def test_wizard_add_lands_in_isolated_taxonomy(tmp_path, monkeypatch, capsys):
    tmp_yaml = tmp_path / "taxonomy.yaml"
    shutil.copyfile(lt.DEFAULT_TAXONOMY_PATH, tmp_yaml)
    original_version = (yaml.safe_load(tmp_yaml.read_text(encoding="utf-8")) or {}).get("taxonomy_version")
    monkeypatch.setattr(lt, "DEFAULT_TAXONOMY_PATH", str(tmp_yaml))
    monkeypatch.setattr("src.playlist_gui.worker.SIDECAR_DB_PATH", str(tmp_path / "sidecar.db"))

    handle_record_taxonomy_decision({
        "cmd": "record_taxonomy_decision", "request_id": "r1",
        "term": "xyzzy wizard genre", "raw_term": "Xyzzy Wizard Genre",
        "verdict": "add", "proposal": WIZARD_PROPOSAL, "claude": None,
        "human_edited": True})
    handle_apply_taxonomy_decisions({
        "cmd": "apply_taxonomy_decisions", "request_id": "r2", "job_id": "j1"})

    events = [json.loads(line) for line in capsys.readouterr().out.strip().splitlines()]
    dones = [e for e in events if e["type"] == "done" and e.get("cmd") == "apply_taxonomy_decisions"]
    assert dones and dones[-1]["ok"] is True

    data = yaml.safe_load(tmp_yaml.read_text(encoding="utf-8"))
    assert data.get("taxonomy_version") != original_version
    records = data.get("records") or data.get("genres") or []
    names = {str(r.get("name", "")).lower(): r for r in records if isinstance(r, dict)}
    rec = names.get("xyzzy wizard genre")
    assert rec is not None, "wizard record not appended"
    edge_targets = {(e.get("target") or "").lower() for e in (rec.get("parent_edges") or [])}
    assert "indie rock" in edge_targets
    alias = names.get("xyzzy wizard variant")
    assert alias is not None and alias.get("kind") == "alias"
