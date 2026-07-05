# tests/unit/test_web_dial_translation.py
"""The web layer translates dials -> axes exactly once, via policy's resolver."""
from src.playlist_gui.policy import resolve_dial_axes
from src.playlist_web.schemas import GenerateRequestBody


def test_request_body_carries_dials_not_axes():
    body = GenerateRequestBody(mode="artist", artist="Codeine",
                               range_dial="close", flow_dial="drift", pace_dial="steady")
    assert not hasattr(body, "cohesion_mode")
    assert not hasattr(body, "genre_mode")
    axes = resolve_dial_axes(body.range_dial, body.flow_dial, body.pace_dial)
    req = body.to_request(axes)
    assert req.genre_mode == "narrow"
    assert req.sonic_mode == "narrow"
    assert req.pace_mode == "narrow"


def test_default_body_resolves_to_all_dynamic():
    body = GenerateRequestBody(mode="artist", artist="Codeine")
    axes = resolve_dial_axes(body.range_dial, body.flow_dial, body.pace_dial)
    assert axes == {"cohesion_mode": "dynamic", "genre_mode": "dynamic",
                    "sonic_mode": "dynamic", "pace_mode": "dynamic"}
