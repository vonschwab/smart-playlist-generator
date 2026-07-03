"""Behavioral: steering tags shift the generated playlist's genre affinity.

Runs through generate_like_gui (the production config chain) against the live
artifact. Read the log lines, not just the metric (playlist-testing skill).
"""
import logging
from pathlib import Path

import numpy as np
import pytest

import sys
sys.path.insert(0, "tests")
from support.gui_fidelity import (  # noqa: E402
    generate_like_gui,
    gui_ui_state,
    resolved_artifact_path,
)

from src.features.artifacts import load_artifact_bundle  # noqa: E402
from src.playlist.tag_steering import resolve_tag_steering_target  # noqa: E402

ARTIFACT = Path("data/artifacts/beat3tower_32k/data_matrices_step1.npz")

pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.fixture(scope="module")
def bundle():
    if not ARTIFACT.exists():
        pytest.skip("live artifact not present")
    # resolved_artifact_path() runs the real config chain (load_config_with_overrides),
    # which publishes config.yaml's artifacts.sonic_variant_override (muq) as a
    # process-wide side effect. The live artifact was rebuilt muq-native (SP-B,
    # 2026-07-01/02) and no longer carries a plain X_sonic key, so a bare
    # load_artifact_bundle() call here — outside the harness's config chain —
    # would raise "Artifact missing required keys: ['X_sonic']". generate_like_gui
    # triggers the same side effect internally on every call; this just does it
    # once up front so the fixture's own direct load succeeds too.
    resolved_artifact_path()
    return load_artifact_bundle(str(ARTIFACT))


def _seeds_and_tag(bundle):
    """Two same-artist seeds + the library's most common vocab tag (always mappable)."""
    if getattr(bundle, "genre_emb", None) is None:
        pytest.skip("dense genre sidecar absent")
    col_mass = np.asarray(bundle.X_genre_raw.sum(axis=0)).ravel()
    tag = str(bundle.genre_vocab[int(np.argmax(col_mass))])
    keys, counts = np.unique(bundle.artist_keys, return_counts=True)
    artist = keys[np.argmax(counts)]  # best-represented artist -> feasible piers
    idx = np.nonzero(bundle.artist_keys == artist)[0][:2]
    if len(idx) < 2:
        pytest.skip("no artist with 2+ tracks in artifact")
    return [str(bundle.track_ids[i]) for i in idx], tag


def _mean_affinity(bundle, track_ids, target):
    rows = [bundle.track_id_to_index[t] for t in track_ids if t in bundle.track_id_to_index]
    return float(np.mean(bundle.X_genre_dense[rows] @ target))


def test_steering_shifts_playlist_affinity_and_logs(bundle, caplog):
    seeds, tag = _seeds_and_tag(bundle)
    target, _, _ = resolve_tag_steering_target(
        [tag], genre_vocab=[str(v) for v in bundle.genre_vocab],
        genre_emb=bundle.genre_emb,
    )
    assert target is not None

    base = generate_like_gui(seeds=seeds, length=15, random_seed=0)
    with caplog.at_level(logging.INFO):
        steered = generate_like_gui(
            seeds=seeds, length=15, random_seed=0, steering_tags=[tag]
        )

    # 1) The lever FIRED (log evidence, not just a metric).
    assert any("Tag steering pool lever" in r.message for r in caplog.records)
    assert any("Tag steering target" in r.message for r in caplog.records)

    # 2) The playlist's mean tag affinity moved toward the target.
    aff_base = _mean_affinity(bundle, base.track_ids, target)
    aff_steered = _mean_affinity(bundle, steered.track_ids, target)
    assert aff_steered >= aff_base, (
        f"steered affinity {aff_steered:.3f} < baseline {aff_base:.3f} — "
        "read the gate tally + pool lines before concluding the lever is weak"
    )
