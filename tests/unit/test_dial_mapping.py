"""Dial -> axis mapping (GUI dials spec 2026-07-04).

The DIAL_TO_AXES table is the single source of truth for what each detent
does. Every detent maps to a grid-verified axis state
(docs/run_audits/slider_differentiation_2026-07-04/).
"""
from src.playlist_gui.policy import DIAL_TO_AXES, resolve_dial_axes


def test_every_detent_maps_to_exact_axis_values():
    assert DIAL_TO_AXES["range"]["home"] == {"sonic_mode": "strict", "genre_mode": "strict"}
    assert DIAL_TO_AXES["range"]["close"] == {"sonic_mode": "narrow", "genre_mode": "narrow"}
    assert DIAL_TO_AXES["range"]["open"] == {"sonic_mode": "dynamic", "genre_mode": "dynamic"}
    assert DIAL_TO_AXES["range"]["wander"] == {"sonic_mode": "discover", "genre_mode": "discover"}
    assert DIAL_TO_AXES["flow"]["drift"] == {"cohesion_mode": "discover"}
    assert DIAL_TO_AXES["flow"]["balanced"] == {"cohesion_mode": "dynamic"}
    assert DIAL_TO_AXES["flow"]["journey"] == {"cohesion_mode": "strict"}
    assert DIAL_TO_AXES["pace"]["steady"] == {"pace_mode": "narrow"}
    assert DIAL_TO_AXES["pace"]["natural"] == {"pace_mode": "dynamic"}
    assert DIAL_TO_AXES["pace"]["free"] == {"pace_mode": "off"}


def test_resolve_defaults_equal_todays_all_dynamic():
    axes = resolve_dial_axes(None, None, None)
    assert axes == {
        "cohesion_mode": "dynamic",
        "genre_mode": "dynamic",
        "sonic_mode": "dynamic",
        "pace_mode": "dynamic",
    }


def test_resolve_combines_all_three_dials():
    axes = resolve_dial_axes("wander", "journey", "free")
    assert axes == {
        "cohesion_mode": "strict",
        "genre_mode": "discover",
        "sonic_mode": "discover",
        "pace_mode": "off",
    }


def test_unknown_detent_falls_back_to_default_loudly(caplog):
    import logging
    with caplog.at_level(logging.WARNING):
        axes = resolve_dial_axes("bogus", None, None)
    assert axes["sonic_mode"] == "dynamic" and axes["genre_mode"] == "dynamic"
    assert any("bogus" in r.message for r in caplog.records)
