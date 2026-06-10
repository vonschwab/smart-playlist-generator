from scripts.genre_audition_analyze import (
    aggregate_by_provenance,
    mean_score_by_provenance,
    sim_verdict_rows,
)


def _entry(verdict, spaces):
    return {"verdict": verdict, "notes": "", "spaces": spaces}


def test_aggregate_expands_by_provenance():
    entries = [
        _entry("same", {"graph": {"rank": 1, "sim": 0.8}, "cooccurrence": {"rank": 2, "sim": 0.4}}),
        _entry("unrelated", {"decoy": {}}),
    ]
    agg = aggregate_by_provenance(entries)
    assert agg["graph"]["same"] == 1
    assert agg["cooccurrence"]["same"] == 1   # same entry counts for both sources
    assert agg["decoy"]["unrelated"] == 1


def test_aggregate_skips_empty_verdict():
    agg = aggregate_by_provenance([_entry("", {"graph": {"rank": 1, "sim": 0.5}})])
    assert agg.get("graph", {}) == {}


def test_mean_score_by_provenance():
    entries = [
        _entry("same", {"graph": {"rank": 1, "sim": 0.8}}),       # 3
        _entry("loose", {"graph": {"rank": 2, "sim": 0.3}}),      # 1
        _entry("unrelated", {"decoy": {}}),                       # 0
    ]
    means = mean_score_by_provenance(entries)
    assert means["graph"] == 2.0
    assert means["decoy"] == 0.0


def test_sim_verdict_rows_skips_decoy_and_missing_sim():
    entries = [
        _entry("same", {"graph": {"rank": 1, "sim": 0.8}}),
        _entry("unrelated", {"decoy": {}}),
    ]
    rows = sim_verdict_rows(entries)
    assert rows == [{"provenance": "graph", "sim": 0.8, "score": 3}]
