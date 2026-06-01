"""Fast guards on the GUI-fidelity harness itself.

These pin the property that prevented a whole class of wasted debugging this
session: the harness must surface PRODUCTION config defaults (from config.yaml /
config.example.yaml), not the dataclass defaults that a hand-built overrides dict
silently falls back to. If build_ds_overrides or the policy chain ever stops
carrying these keys, these tests fail loudly.

No live artifact needed — config resolution only, so these run in the fast suite.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.support.gui_fidelity import gui_ui_state, resolve_gui_overrides

# config.yaml is gitignored; config.example.yaml is committed and carries the same
# production-shaped defaults, so use it for deterministic CI-stable assertions.
EXAMPLE_CONFIG = "config.example.yaml"
_requires_example = pytest.mark.skipif(
    not Path(EXAMPLE_CONFIG).exists(), reason="config.example.yaml not found"
)


@_requires_example
def test_harness_surfaces_production_config_defaults():
    """The two keys that hand-built overrides kept dropping must resolve True."""
    ui = gui_ui_state(cohesion_mode="narrow", genre_mode="narrow", sonic_mode="narrow")
    ov = resolve_gui_overrides(ui, config_path=EXAMPLE_CONFIG)

    pier = ov.get("pier_bridge", {})
    constraints = ov.get("constraints", {})

    assert pier.get("disallow_pier_artists_in_interiors") is True, (
        "harness lost config.yaml's disallow_pier_artists_in_interiors=true — "
        "tests built on it would silently allow pier-artist repeats"
    )
    assert constraints.get("artist_identity", {}).get("enabled") is True, (
        "harness lost config.yaml's artist_identity.enabled=true — "
        "tests built on it would mis-resolve same-artist identity"
    )
    assert "candidate_pool" in ov, "ds overrides must carry candidate_pool config"


@_requires_example
@pytest.mark.parametrize(
    "spacing,expected_min_gap",
    [("loose", 3), ("normal", 6), ("strong", 9), ("very_strong", 12)],
)
def test_artist_spacing_slider_maps_to_min_gap(spacing, expected_min_gap):
    """The 'Artist Gap' slider must reach constraints.min_gap (this was the bug:
    min_gap=9 was resolved but never carried into the pier-bridge)."""
    ui = gui_ui_state(cohesion_mode="narrow", artist_spacing=spacing)
    ov = resolve_gui_overrides(ui, config_path=EXAMPLE_CONFIG)
    assert ov.get("constraints", {}).get("min_gap") == expected_min_gap
