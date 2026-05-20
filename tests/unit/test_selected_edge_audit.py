import logging

from src.playlist.reporter import emit_selected_edge_audit


def test_emit_selected_edge_audit_writes_one_row_per_edge(caplog):
    caplog.set_level(logging.INFO)
    edges = [
        {
            "from_idx": 1, "to_idx": 2,
            "from_artist": "Geese", "from_title": "Cowboy Nudes",
            "to_artist": "Arctic Monkeys", "to_title": "Library Pictures",
            "T": 0.989, "T_centered_cos": 0.978, "S": 0.491, "G": 0.890,
            "bridge_score": 0.61, "trans_score_in_beam": 0.95,
            "progress_t": 0.14, "progress_jump": 0.14,
            "local_sonic_raw_cos": 0.42, "local_sonic_penalty_applied": 0.0,
            "genre_penalty_applied": 0.0,
            "below_transition_floor": False,
        },
        {
            "from_idx": 14, "to_idx": 15,
            "from_artist": "Hideous Sun Demon", "from_title": "Gimmicks",
            "to_artist": "Stove", "to_title": "Nightwalk",
            "T": 0.092, "T_centered_cos": -0.817, "S": 0.306, "G": 1.000,
            "bridge_score": 0.55, "trans_score_in_beam": 0.25,
            "progress_t": 0.85, "progress_jump": 0.10,
            "local_sonic_raw_cos": 0.03, "local_sonic_penalty_applied": 0.021,
            "genre_penalty_applied": 0.0,
            "below_transition_floor": True,
        },
    ]
    emit_selected_edge_audit(edges)
    text = caplog.text
    assert "Selected-edge audit" in text
    assert "Stove" in text and "Nightwalk" in text
    assert "T=0.092" in text
    assert "T_centered_cos=-0.817" in text
    assert "below_floor=True" in text


def test_emit_selected_edge_audit_handles_missing_fields(caplog):
    caplog.set_level(logging.INFO)
    edges = [{"from_idx": 0, "to_idx": 1, "T": 0.5}]
    emit_selected_edge_audit(edges)
    assert "Selected-edge audit" in caplog.text
