from scripts.sonic_audition_analyze import aggregate_by_space, cosine_verdict_correlation

VERDICT_SCORE = {"match": 3, "close": 2, "off": 1, "wrong": 0}


def _entry(verdict, spaces):
    return {
        "verdict": verdict, "notes": "",
        "spaces": {s: {"rank": 1, "cosine": 0.4} for s in spaces},
    }


def test_aggregate_counts_by_space():
    entries = [
        _entry("match", ["timbre", "full_track"]),
        _entry("off", ["timbre"]),
        _entry("close", ["rhythm"]),
    ]
    result = aggregate_by_space(entries)
    assert result["timbre"]["match"] == 1
    assert result["timbre"]["off"] == 1
    assert result["full_track"]["match"] == 1
    assert result["rhythm"]["close"] == 1


def test_aggregate_ignores_empty_verdict():
    entries = [_entry("", ["timbre"])]
    result = aggregate_by_space(entries)
    assert "timbre" not in result or all(v == 0 for v in result.get("timbre", {}).values())


def test_cosine_verdict_correlation_pairs():
    entries = [
        {"verdict": "match", "spaces": {"timbre": {"cosine": 0.8, "rank": 1}}},
        {"verdict": "wrong", "spaces": {"timbre": {"cosine": 0.1, "rank": 10}}},
    ]
    rows = cosine_verdict_correlation(entries)
    assert len(rows) == 2
    assert any(r["cosine"] == 0.8 and r["score"] == 3 for r in rows)
    assert any(r["cosine"] == 0.1 and r["score"] == 0 for r in rows)


def test_cosine_verdict_skips_missing_spaces():
    entries = [{"verdict": "match", "spaces": None}]
    rows = cosine_verdict_correlation(entries)
    assert rows == []
