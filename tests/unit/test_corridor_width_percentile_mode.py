"""Unit tests for the sonic_mode -> corridor width percentile mapping (Phase 1
per-mode corridor width, spec section 4, pulled forward from Phase 2 by
Dylan's 2026-07-18 decision).

``resolve_corridor_width_percentile`` is a pure function: given the run's
``sonic_mode`` string plus the per-mode config fields (+ the escape-hatch
override), it returns the concrete percentile to build the corridor at.
"""
from src.playlist.pier_bridge.corridor import resolve_corridor_width_percentile


KW = dict(strict=0.99, narrow=0.965, dynamic=0.95, discover=0.93)


def test_strict_mode_resolves_to_strict_field():
    assert resolve_corridor_width_percentile("strict", override=None, **KW) == 0.99


def test_narrow_mode_resolves_to_narrow_field():
    assert resolve_corridor_width_percentile("narrow", override=None, **KW) == 0.965


def test_dynamic_mode_resolves_to_dynamic_field():
    assert resolve_corridor_width_percentile("dynamic", override=None, **KW) == 0.95


def test_discover_mode_resolves_to_discover_field():
    assert resolve_corridor_width_percentile("discover", override=None, **KW) == 0.93


def test_off_mode_resolves_to_zero_hardcoded_no_field_needed():
    """'off' means percentile 0.0 -> the whole eligible universe qualifies as
    a corridor member -- hardcoded per spec section 4, not a config field."""
    assert resolve_corridor_width_percentile("off", override=None, **KW) == 0.0


def test_unspecified_mode_falls_back_to_dynamic():
    """None (mode never threaded / not set) must still resolve to a concrete,
    usable float -- unlike the genre relevance mask, a corridor can never be
    legitimately 'off' just because the mode wasn't specified. Mirrors the
    project's established sonic_mode fallback (policy.py:316)."""
    assert resolve_corridor_width_percentile(None, override=None, **KW) == 0.95


def test_unrecognized_mode_falls_back_to_dynamic():
    assert resolve_corridor_width_percentile("bogus_mode", override=None, **KW) == 0.95


def test_case_and_whitespace_insensitive():
    assert resolve_corridor_width_percentile("  STRICT ", override=None, **KW) == 0.99


def test_escape_hatch_override_wins_over_every_mode():
    """The plain corridor_width_percentile override, when explicitly set,
    wins unconditionally -- even over 'off' and 'strict'."""
    for mode in ("strict", "narrow", "dynamic", "discover", "off", None, "bogus"):
        assert resolve_corridor_width_percentile(mode, override=0.777, **KW) == 0.777


def test_escape_hatch_override_of_zero_is_respected_not_treated_as_unset():
    """override=0.0 is a legitimate explicit value (not a falsy 'unset'
    sentinel) -- must be distinguished from override=None via `is not None`,
    not truthiness."""
    assert resolve_corridor_width_percentile("strict", override=0.0, **KW) == 0.0
