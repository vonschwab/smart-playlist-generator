import os
import logging

import numpy as np
import pytest
from types import SimpleNamespace

from src.playlist.pipeline import generate_playlist_ds


def test_pool_logging_after_exclusion(monkeypatch, caplog):
    # Fake artifact bundle
    bundle = SimpleNamespace(
        track_ids=np.array(["seed", "a", "b", "c", "d", "e"]),
        artist_keys=np.array(["s", "A", "B", "C", "D", "E"]),
        track_artists=np.array(["s", "A", "B", "C", "D", "E"]),
        track_titles=np.array(["s", "A", "B", "C", "D", "E"]),
        X_sonic=np.zeros((6, 2)),
        X_sonic_start=np.zeros((6, 2)),
        X_sonic_mid=np.zeros((6, 2)),
        X_sonic_end=np.zeros((6, 2)),
        X_genre_raw=np.zeros((6, 2)),
        X_genre_smoothed=np.zeros((6, 2)),
        genre_vocab=np.array(["g1", "g2"]),
        track_id_to_index={"seed": 0, "a": 1, "b": 2, "c": 3, "d": 4, "e": 5},
        artifact_path="dummy.npz",
    )
    monkeypatch.setattr("src.playlist.pipeline.load_artifact_bundle", lambda path: bundle)
    # Enable diagnostic env var
    monkeypatch.setenv("PLAYLIST_DIAG_RECENCY", "1")
    caplog.set_level(logging.INFO)
    # Exclude one real id and one missing id
    generate_playlist_ds(
        artifact_path="dummy.npz",
        seed_track_id="seed",
        num_tracks=3,
        mode="dynamic",
        random_seed=0,
        excluded_track_ids={"a", "missing"},
    )
    text = "\n".join(caplog.messages)
    assert "DS candidate pool after exclusions" in text
    assert "total=6" in text
    assert "requested_excluded=2" in text
    assert "applied_excluded=1" in text
    assert "final_pool=5" in text  # seed preserved; one exclusion applied
