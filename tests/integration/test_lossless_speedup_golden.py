"""Bit-identical replay gate for the lossless-speedup work.

Loads each frozen golden fixture, replays generate_playlist_ds, and asserts
the ordered track_ids are IDENTICAL and min/mean transition match exactly.
This is THE regression gate for every optimization in the plan
(docs/superpowers/plans/2026-07-03-lossless-generation-speedup.md).
"""
import glob
import os

import pytest

from tests.support.lossless_golden import load_golden, replay_golden

FIXTURE_DIR = os.path.join("tests", "fixtures", "lossless_speedup")
FIXTURES = sorted(glob.glob(os.path.join(FIXTURE_DIR, "*.json")))


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.skipif(not FIXTURES, reason="no golden fixtures captured yet")
@pytest.mark.parametrize("fixture_path", FIXTURES, ids=lambda p: os.path.basename(p))
def test_golden_bit_identical(fixture_path):
    golden = load_golden(fixture_path)
    artifact = golden["kwargs"]["artifact_path"]
    if not os.path.exists(artifact):
        pytest.skip(f"artifact missing: {artifact}")

    result = replay_golden(golden)

    assert list(result.track_ids) == list(golden["track_ids"]), (
        f"track_ids diverged for {os.path.basename(fixture_path)} — NOT lossless"
    )
    for key in ("min_transition", "mean_transition"):
        got, want = result.metrics.get(key), golden[key]
        if want is not None:
            assert got == want, f"{key} drift: {got!r} != {want!r} (delta-T != 0)"
