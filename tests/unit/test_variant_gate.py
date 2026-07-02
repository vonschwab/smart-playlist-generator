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


def test_unknown_variant_warns_loudly(tmp_path, caplog):
    import logging
    with caplog.at_level(logging.WARNING):
        _variant_gate(_cfg(tmp_path, "muq2"), "muq")   # a typo
    assert any("not a recognized sonic variant" in r.getMessage() for r in caplog.records)


def test_known_variant_does_not_warn(tmp_path, caplog):
    import logging
    with caplog.at_level(logging.WARNING):
        _variant_gate(_cfg(tmp_path, "muq"), "muq")
    assert not any("not a recognized" in r.getMessage() for r in caplog.records)


def test_tower_weighted_override_now_warns(tmp_path, caplog):
    import logging
    with caplog.at_level(logging.WARNING):
        _variant_gate(_cfg(tmp_path, "tower_weighted"), "muq")
    assert any("not a recognized sonic variant" in r.getMessage() for r in caplog.records)
