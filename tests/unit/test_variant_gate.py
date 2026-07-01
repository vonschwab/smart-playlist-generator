import yaml
from scripts.analyze_library import _variant_gate

def _cfg(tmp_path, variant):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"artifacts": {"sonic_variant_override": variant}}), encoding="utf-8")
    return str(p)

def test_active_variant_runs_inactive_skips(tmp_path):
    muq_cfg = _cfg(tmp_path, "muq")
    assert _variant_gate(muq_cfg, "muq") is None                # muq active -> muq runs
    assert _variant_gate(muq_cfg, "mert") is not None           # muq active -> mert skips
    mert_cfg = _cfg(tmp_path, "mert")
    assert _variant_gate(mert_cfg, "mert") is None
    assert _variant_gate(mert_cfg, "muq") is not None

def test_default_variant_is_mert(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("{}", encoding="utf-8")   # no override -> 'mert'
    assert _variant_gate(str(p), "mert") is None
    assert _variant_gate(str(p), "muq") is not None
