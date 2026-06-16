from src.playlist.run_audit import InfeasibleHandlingConfig, parse_infeasible_handling_config

def test_min_floors_default_to_zero():
    cfg = InfeasibleHandlingConfig()
    # Defaults must let the transition tier relax BELOW a 0.20 transition_floor,
    # and the genre-arc tier reach percentile 0. (Old defaults: 0.20 / 0.5 — inert.)
    assert cfg.min_transition_floor == 0.0
    assert cfg.min_genre_arc_percentile == 0.0


def test_guarantee_feasible_defaults_true_and_parses():
    assert InfeasibleHandlingConfig().guarantee_feasible is True
    assert parse_infeasible_handling_config({"guarantee_feasible": False}).guarantee_feasible is False


def test_parse_fallbacks_match_dataclass_defaults():
    # A config that OMITS the min-floor keys (like the live config.yaml) must still
    # get 0.0 via the parse path — not the old 0.20/0.5 — so the relax tiers aren't
    # silently re-inerted by an out-of-sync .get() fallback.
    cfg = parse_infeasible_handling_config({"enabled": True})
    assert cfg.min_transition_floor == 0.0
    assert cfg.min_genre_arc_percentile == 0.0
