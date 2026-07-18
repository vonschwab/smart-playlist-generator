"""Generation-fidelity test: the ``instrumental`` lean lowers mean voice_prob.

Runs through generate_like_gui (the production config chain, seeds-mode,
multi-pier) against the live artifact + instrumental sidecar. Per the
playlist-testing skill, this is a robust mean-shift assertion (mirrors
test_tag_steering_behavioral.py's base-vs-steered pattern) rather than a
single-track presence/absence check, so it tolerates pool nondeterminism.

Depends on Checkpoint B (the instrumental sidecar extraction — canonical-only,
a data write blocked in satellite workspaces). Skips cleanly until then.
"""
from __future__ import annotations


import numpy as np
import pytest

from tests.support.gui_fidelity import generate_like_gui
from tests.integration.test_gui_fidelity_regressions import ART, SEEDS
from src.features.artifacts import load_artifact_bundle
from src.playlist.instrumental_loader import load_voice_prob

SIDECAR = ART.parent / "instrumental" / "instrumental_sidecar.npz"

_requires_artifact = pytest.mark.skipif(
    not ART.exists() or not SIDECAR.exists(),
    reason="live artifact + instrumental sidecar required (Checkpoint B)",
)


def _mean_voice_prob(bundle, track_ids) -> float:
    vp = load_voice_prob(bundle.track_ids, sidecar_path=str(SIDECAR))
    idx = bundle.track_id_to_index
    vals = [vp[idx[str(t)]] for t in track_ids if str(t) in idx]
    finite = [v for v in vals if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


@pytest.mark.integration
@pytest.mark.slow
@_requires_artifact
def test_instrumental_flag_lowers_mean_voice_prob():
    """instrumental=True must lower the playlist's mean voice_prob vs instrumental=False,
    through the real GUI config chain, on the validated multi-pier SEEDS set.
    """
    load_artifact_bundle.cache_clear()
    # sonic_variant_override passed explicitly (established precedent:
    # test_dense_genre_integration.py's live_bundle fixture) rather than relied on as
    # a generate_like_gui side effect: tests/conftest.py's autouse
    # _reset_sonic_variant_override resets the process-wide override to None before
    # every test, so loading directly here (before any generate_like_gui call in THIS
    # test) would otherwise fail with "Artifact missing required keys: ['X_sonic']" --
    # pre-existing bug found running -m slow in isolation for the first time (Phase 1
    # Task 9, 2026-07-18).
    bundle = load_artifact_bundle(str(ART), sonic_variant_override="muq")

    off = generate_like_gui(
        seeds=SEEDS, cohesion_mode="narrow", genre_mode="narrow",
        sonic_mode="narrow", pace_mode="narrow",
        instrumental=False, length=30, random_seed=0,
    )
    on = generate_like_gui(
        seeds=SEEDS, cohesion_mode="narrow", genre_mode="narrow",
        sonic_mode="narrow", pace_mode="narrow",
        instrumental=True, length=30, random_seed=0,
    )

    mv_off = _mean_voice_prob(bundle, off.track_ids)
    mv_on = _mean_voice_prob(bundle, on.track_ids)
    assert mv_on < mv_off, (
        f"instrumental lean should lower mean voice_prob: on={mv_on} off={mv_off}"
    )
