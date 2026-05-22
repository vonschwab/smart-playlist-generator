import logging

from src.playlist.reporter import diagnose_t_mismatch


def test_diagnose_t_mismatch_flags_disagreement(caplog):
    caplog.set_level(logging.WARNING)
    edges = [
        {"from_idx": 14, "to_idx": 15,
         "T": 0.092, "trans_score_in_beam": 0.25,
         "below_transition_floor": True},
        {"from_idx": 0, "to_idx": 1,
         "T": 0.95, "trans_score_in_beam": 0.94,
         "below_transition_floor": False},
    ]
    issues = diagnose_t_mismatch(edges, transition_floor=0.20, tolerance=0.05)
    assert len(issues) == 1
    assert issues[0]["from_idx"] == 14
    assert "beam_trans=0.250" in caplog.text
    assert "final_T=0.092" in caplog.text


def test_diagnose_t_mismatch_quiet_on_agreement(caplog):
    caplog.set_level(logging.WARNING)
    edges = [
        {"from_idx": 0, "to_idx": 1,
         "T": 0.95, "trans_score_in_beam": 0.94,
         "below_transition_floor": False},
    ]
    issues = diagnose_t_mismatch(edges, transition_floor=0.20, tolerance=0.05)
    assert issues == []
    # Confirm no warning emitted
    assert "beam_trans" not in caplog.text
