# tests/unit/test_receipt_compose.py
"""Receipt composer: numbers come only from the run's own stats; confessions
(notes) fire only when true. Dial vocabulary only — no engine terms."""
from src.playlist_gui.receipt import compose_receipt


def _stats(**over):
    base = {
        "min_transition": 0.58,
        "mean_transition": 0.81,
        "bpm_summary": {"min": 88.0, "mean": 120.0, "max": 150.0, "std": 9.0, "n": 28, "total": 30},
        "warnings": [],
    }
    base.update(over)
    return base


def test_happy_path_numbers_and_no_notes():
    r = compose_receipt(_stats(), {"admitted": 445, "considered": 1700, "genre_rescued": 0})
    assert r["range"] == {"pool": 445, "considered": 1700}
    assert r["flow"] == {"worst": 0.58, "mean": 0.81}
    assert r["pace"]["bpm_std"] == 9.0 and r["pace"]["n"] == 28
    assert r["notes"] == []


def test_confessions_fire_only_when_true():
    stats = _stats(warnings=[{"type": "relaxation", "message": "bridge floor relaxed to 0.01"}])
    r = compose_receipt(stats, {"admitted": 378, "considered": 1700, "genre_rescued": 40})
    assert any("relaxed" in n for n in r["notes"])
    assert any("40" in n and "connector" in n for n in r["notes"])


def test_sparse_bpm_confessed():
    stats = _stats(bpm_summary={"min": 60.0, "mean": 61.0, "max": 62.0, "std": 1.0, "n": 4, "total": 30})
    r = compose_receipt(stats, {"admitted": 100, "considered": 200, "genre_rescued": 0})
    assert any("tempo data" in n for n in r["notes"])


def test_missing_stats_degrade_to_none_not_crash():
    r = compose_receipt({}, {})
    assert r["range"]["pool"] is None and r["flow"]["worst"] is None
    assert isinstance(r["notes"], list)
