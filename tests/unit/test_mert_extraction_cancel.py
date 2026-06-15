"""Cancellation safety for MERT extraction.

The data-safety contract: cancellation is honored only at the between-track
boundary, and completed embeddings are flushed (atomically) before the
extraction unwinds — so a cancelled run leaves no partial/garbage data and is
fully resumable from the manifest.
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts.extract_mert_sidecar import ShardStore, run_extraction

EMB_DIM = 768


class _CancelAfter:
    """cancellation_check stand-in that raises once it has been polled N times."""

    def __init__(self, raise_on_call: int):
        self.calls = 0
        self.raise_on_call = raise_on_call

    def __call__(self) -> None:
        self.calls += 1
        if self.calls >= self.raise_on_call:
            raise KeyboardInterrupt("cancelled")


def _embedder(y: np.ndarray) -> np.ndarray:
    return np.full(EMB_DIM, 0.5, dtype=np.float32)


def _items(tmp_path, n: int):
    # Real (empty) files so the existence check passes; the stubbed loader below
    # means file contents are never decoded.
    items = []
    for i in range(1, n + 1):
        fp = tmp_path / f"t{i}.wav"
        fp.write_bytes(b"")
        items.append((f"t{i}", str(fp)))
    return items


def _stub_io(monkeypatch):
    # Deterministic prober/loader so no real audio is touched.
    return dict(
        prober=lambda fp: 120.0,
        loader=lambda fp, off, dur: np.zeros(int(24000 * dur), np.float32) + 0.01,
    )


def test_cancel_flushes_completed_tracks_and_propagates(tmp_path, monkeypatch):
    store = ShardStore(tmp_path / "shards", shard_size=100)
    # Check is polled at the TOP of each iteration; raising on the 3rd poll means
    # tracks 1 and 2 fully embedded before the cancel lands.
    cancel = _CancelAfter(raise_on_call=3)

    with pytest.raises(KeyboardInterrupt):
        run_extraction(
            _items(tmp_path, 5), _embedder, store,
            cancellation_check=cancel,
            **_stub_io(monkeypatch),
        )

    # Completed work was flushed atomically before unwinding (finally-flush).
    import json
    manifest = json.loads((tmp_path / "shards" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["done"] == ["t1", "t2"]
    # No track left half-written: every shard row is a complete 768-d triple.
    shards = list((tmp_path / "shards").glob("shard_*.npz"))
    assert shards, "expected the completed tracks to be persisted in a shard"
    z = np.load(shards[0], allow_pickle=True)
    assert z["emb_start"].shape == (2, EMB_DIM)
    assert z["emb_mid"].shape == (2, EMB_DIM)
    assert z["emb_end"].shape == (2, EMB_DIM)


def test_no_cancellation_check_runs_to_completion(tmp_path, monkeypatch):
    store = ShardStore(tmp_path / "shards", shard_size=100)
    result = run_extraction(
        _items(tmp_path, 3), _embedder, store,
        cancellation_check=None,
        **_stub_io(monkeypatch),
    )
    assert result == {"ok": 3, "failed": 0}
    import json
    manifest = json.loads((tmp_path / "shards" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["done"] == ["t1", "t2", "t3"]
