"""End-to-end regression catalog, driven through the GUI-fidelity harness.

Each test here reproduces a real bug fixed during the 2026-06 genre/diversity work
by generating EXACTLY as the GUI would (production config resolved from config.yaml).
Add a case here whenever a generation bug is fixed — that is the "update regularly"
half of the playlist-testing skill.

Marked @integration @slow: they need the live artifact bundle + sidecar.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.support.gui_fidelity import (
    generate_like_gui,
    gui_ui_state,
    artist_at_positions,
    find_min_gap_violations,
)
from src.features.artifacts import load_artifact_bundle

ART = Path("data/artifacts/beat3tower_32k/data_matrices_step1.npz")
_requires_artifact = pytest.mark.skipif(not ART.exists(), reason="live artifact required")

# Five seeds from the seeds-mode run where Smog clustered at 14/17 under min_gap=9.
SEEDS = [
    "f28fd5cebac845cf64fee59d5ac3b3aa",  # William Tyler - Howling at the Second Moon
    "b8f8aa0e86f977f9fcb26f615e130ac9",  # Hayden Pedigo - Nearer, Nearer
    "42473b911cef5674e56b8e2ce87df7cb",  # Steve Hiett - Are These My Memories?
    "49f8bba75408d4e0e0e000d1dc708add",  # Songs: Ohia - Hold On Magnolia
    "b587eb56fa1e173138152bf09565eb80",  # Bill Callahan - Let's Move to the Country
]


@pytest.mark.integration
@pytest.mark.slow
@_requires_artifact
def test_strong_artist_gap_enforced_across_segments():
    """Regression for commit 8101ac1: cross-segment min_gap was hardcoded to 1, so
    'Artist Gap: strong' (min_gap=9) still produced same-artist pairs 3-5 apart that
    straddled segment boundaries (Smog 14/17, Hayden Pedigo 6/11, William Ackerman 21/24).
    """
    load_artifact_bundle.cache_clear()
    bundle = load_artifact_bundle(str(ART))
    ui = gui_ui_state(
        cohesion_mode="narrow", genre_mode="narrow", sonic_mode="narrow",
        pace_mode="narrow", artist_spacing="strong",
    )
    res = generate_like_gui(seeds=SEEDS, ui=ui, length=30, random_seed=0)

    assert len(res.track_ids) == 30, f"expected 30 tracks, got {len(res.track_ids)}"
    artists = artist_at_positions(bundle, res.track_ids)
    violations = find_min_gap_violations(artists, min_gap=9)
    assert not violations, f"cross-segment min_gap=9 violated: {violations}"
