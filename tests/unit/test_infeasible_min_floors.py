from src.playlist.run_audit import InfeasibleHandlingConfig

def test_min_floors_default_to_zero():
    cfg = InfeasibleHandlingConfig()
    # Defaults must let the transition tier relax BELOW a 0.20 transition_floor,
    # and the genre-arc tier reach percentile 0. (Old defaults: 0.20 / 0.5 — inert.)
    assert cfg.min_transition_floor == 0.0
    assert cfg.min_genre_arc_percentile == 0.0
