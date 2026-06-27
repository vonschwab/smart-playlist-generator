"""Unit tests for scripts/extract_mert_sidecar.py (MERT Phase 1 extraction infra).

No real audio, no real model: the embedder is a deterministic fake injected as a
callable, audio probing/loading are injected fakes, and all IO goes to tmp_path.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from scripts.extract_mert_sidecar import (
    CLIP_S,
    EMB_DIM,
    ShardStore,
    clip_windows,
    merge_shards,
    run_extraction,
)


def _fake_emb(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=EMB_DIM).astype(np.float32)


def fake_embedder(y: np.ndarray) -> np.ndarray:
    """Deterministic waveform -> 768-d embedding (hash of the samples)."""
    h = int.from_bytes(hashlib.md5(np.asarray(y, np.float32).tobytes()).digest()[:4], "little")
    return _fake_emb(h)


def _store(tmp_path, **kw):
    kw.setdefault("shard_size", 500)
    return ShardStore(tmp_path / "shards", model_name="fake-model", model_revision="deadbeef", **kw)


class TestClipWindows:
    def test_normal_track_three_windows(self):
        wins = clip_windows(240.0)
        assert len(wins) == 3
        start, mid, end = wins
        assert start == (0.0, CLIP_S)
        # mid is centered at duration/2
        assert mid[1] == CLIP_S
        assert abs((mid[0] + CLIP_S / 2) - 120.0) < 1e-9
        # end is the final CLIP_S seconds
        assert end == (240.0 - CLIP_S, CLIP_S)

    def test_short_track_single_window_replicated(self):
        # 28s: shorter than ~30s -> one window used for all three slots
        wins = clip_windows(28.0)
        assert len(wins) == 3
        assert wins[0] == wins[1] == wins[2]
        off, dur = wins[0]
        assert dur == CLIP_S
        assert off >= 0.0
        assert off + dur <= 28.0

    def test_very_short_track_uses_whole_track(self):
        # 15s: shorter than the clip itself -> whole track, replicated
        wins = clip_windows(15.0)
        assert wins == [(0.0, 15.0)] * 3

    def test_windows_stay_in_bounds(self):
        for dur_s in (30.0, 31.0, 48.0, 60.0, 7200.0):
            for off, dur in clip_windows(dur_s):
                assert off >= 0.0
                assert off + dur <= dur_s + 1e-9

    def test_nonpositive_duration_raises(self):
        with pytest.raises(ValueError):
            clip_windows(0.0)
        with pytest.raises(ValueError):
            clip_windows(-3.0)


class TestShardStore:
    def test_flush_writes_shard_and_manifest(self, tmp_path):
        store = _store(tmp_path)
        store.add("t1", _fake_emb(1), _fake_emb(2), _fake_emb(3))
        store.add("t2", _fake_emb(4), _fake_emb(5), _fake_emb(6))
        store.flush()

        shards = sorted((tmp_path / "shards").glob("shard_*.npz"))
        assert len(shards) == 1
        z = np.load(shards[0], allow_pickle=True)
        assert [str(t) for t in z["track_ids"]] == ["t1", "t2"]
        assert z["emb_start"].shape == (2, EMB_DIM)
        assert z["emb_start"].dtype == np.float32
        np.testing.assert_array_equal(z["emb_mid"][0], _fake_emb(2))

        manifest = json.loads((tmp_path / "shards" / "manifest.json").read_text())
        assert manifest["done"] == ["t1", "t2"]
        assert manifest["failed"] == {}
        assert manifest["model_name"] == "fake-model"
        assert manifest["model_revision"] == "deadbeef"
        assert manifest["emb_dim"] == EMB_DIM

    def test_auto_flush_at_shard_size(self, tmp_path):
        store = _store(tmp_path, shard_size=2)
        for i in range(5):
            store.add(f"t{i}", _fake_emb(i), _fake_emb(i), _fake_emb(i))
        store.flush()
        shards = sorted((tmp_path / "shards").glob("shard_*.npz"))
        assert len(shards) == 3  # 2 + 2 + 1

    def test_resume_skips_done_ids(self, tmp_path):
        store = _store(tmp_path)
        store.add("t1", _fake_emb(1), _fake_emb(1), _fake_emb(1))
        store.record_failure("t2", "FileNotFoundError: gone")
        store.flush()

        # New store instance over the same dir (fresh process) sees both.
        store2 = _store(tmp_path)
        assert store2.skip_ids() == {"t1", "t2"}
        pending = [t for t in ["t1", "t2", "t3"] if t not in store2.skip_ids()]
        assert pending == ["t3"]

    def test_failure_reason_persisted(self, tmp_path):
        store = _store(tmp_path)
        store.record_failure("bad", "ValueError: empty window")
        store2 = _store(tmp_path)
        manifest = json.loads((tmp_path / "shards" / "manifest.json").read_text())
        assert manifest["failed"]["bad"] == "ValueError: empty window"
        assert "bad" in store2.skip_ids()

    def test_model_mismatch_raises(self, tmp_path):
        store = _store(tmp_path)
        store.flush()
        with pytest.raises(ValueError, match="manifest"):
            ShardStore(tmp_path / "shards", model_name="fake-model", model_revision="0ther",
                       shard_size=500)


class TestMergeShards:
    def test_merge_exactly_once_across_shards(self, tmp_path):
        store = _store(tmp_path, shard_size=2)
        store.add("t1", _fake_emb(1), _fake_emb(1), _fake_emb(1))
        store.add("t2", _fake_emb(2), _fake_emb(2), _fake_emb(2))
        store.add("t3", _fake_emb(3), _fake_emb(3), _fake_emb(3))
        # duplicate id in a later shard (e.g. a --no-resume rerun): newest wins
        store.add("t1", _fake_emb(9), _fake_emb(9), _fake_emb(9))
        store.flush()

        out = tmp_path / "merged.npz"
        merge_shards(tmp_path / "shards", out)
        z = np.load(out, allow_pickle=True)
        ids = [str(t) for t in z["track_ids"]]
        assert sorted(ids) == ["t1", "t2", "t3"]  # exactly once each
        np.testing.assert_array_equal(z["emb_start"][ids.index("t1")], _fake_emb(9))
        assert z["emb_end"].dtype == np.float32
        assert str(z["model_name"]) == "fake-model"
        assert str(z["model_revision"]) == "deadbeef"

    def test_failed_entries_absent_from_merge(self, tmp_path):
        store = _store(tmp_path)
        store.add("ok", _fake_emb(1), _fake_emb(1), _fake_emb(1))
        store.record_failure("broken", "RuntimeError: corrupt")
        store.flush()

        out = tmp_path / "merged.npz"
        merge_shards(tmp_path / "shards", out)
        z = np.load(out, allow_pickle=True)
        assert [str(t) for t in z["track_ids"]] == ["ok"]

    def test_merge_empty_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            merge_shards(tmp_path / "nothing", tmp_path / "merged.npz")

    def test_merge_backs_up_existing_sidecar(self, tmp_path):
        # The sidecar is irreplaceable; re-merging over an existing one must
        # leave a timestamped backup of the prior contents before overwriting.
        store = _store(tmp_path, shard_size=10)
        store.add("t1", _fake_emb(1), _fake_emb(1), _fake_emb(1))
        store.flush()
        out = tmp_path / "mert_sidecar.npz"
        merge_shards(tmp_path / "shards", out)  # first write: no prior file, no backup
        assert not list(tmp_path.glob("mert_sidecar.npz.bak.*"))

        first_bytes = out.read_bytes()
        store.add("t2", _fake_emb(2), _fake_emb(2), _fake_emb(2))
        store.flush()
        merge_shards(tmp_path / "shards", out)  # second write: prior file backed up

        backups = list(tmp_path.glob("mert_sidecar.npz.bak.*"))
        assert len(backups) == 1
        assert backups[0].read_bytes() == first_bytes  # backup holds the pre-overwrite copy
        z = np.load(out, allow_pickle=True)
        assert sorted(str(t) for t in z["track_ids"]) == ["t1", "t2"]  # new sidecar is current


class TestRunExtraction:
    @staticmethod
    def _touch(tmp_path, name):
        p = tmp_path / name
        p.write_bytes(b"\x00")
        return str(p)

    def test_pipeline_with_fake_embedder(self, tmp_path):
        long_fp = self._touch(tmp_path, "long.flac")
        short_fp = self._touch(tmp_path, "short.flac")
        durations = {long_fp: 240.0, short_fp: 15.0}

        def prober(fp):
            return durations[fp]

        def loader(fp, offset, duration):
            # deterministic per (file, window) waveform
            seed = int.from_bytes(hashlib.md5(f"{fp}|{offset}|{duration}".encode()).digest()[:4], "little")
            return np.random.default_rng(seed).normal(size=64).astype(np.float32)

        store = _store(tmp_path)
        summary = run_extraction(
            [("long", long_fp), ("short", short_fp)],
            fake_embedder, store, prober=prober, loader=loader,
        )
        assert summary == {"ok": 2, "failed": 0}

        out = tmp_path / "merged.npz"
        merge_shards(tmp_path / "shards", out)
        z = np.load(out, allow_pickle=True)
        ids = [str(t) for t in z["track_ids"]]
        assert sorted(ids) == ["long", "short"]
        i_long, i_short = ids.index("long"), ids.index("short")
        # long track: distinct windows -> distinct embeddings
        assert not np.array_equal(z["emb_start"][i_long], z["emb_end"][i_long])
        # short track: single window replicated -> identical embeddings in all slots
        np.testing.assert_array_equal(z["emb_start"][i_short], z["emb_mid"][i_short])
        np.testing.assert_array_equal(z["emb_mid"][i_short], z["emb_end"][i_short])
        # embeddings are exactly what the fake embedder says for the loaded windows
        np.testing.assert_array_equal(z["emb_start"][i_long], fake_embedder(loader(long_fp, 0.0, CLIP_S)))

    def test_missing_file_recorded_and_run_continues(self, tmp_path):
        ok_fp = self._touch(tmp_path, "ok.flac")

        def prober(fp):
            return 100.0

        def loader(fp, offset, duration):
            return np.ones(32, np.float32)

        store = _store(tmp_path)
        summary = run_extraction(
            [
                ("gone", str(tmp_path / "does_not_exist.flac")),
                ("nopath", None),
                ("ok", ok_fp),
            ],
            fake_embedder, store, prober=prober, loader=loader,
        )
        assert summary == {"ok": 1, "failed": 2}
        manifest = json.loads((tmp_path / "shards" / "manifest.json").read_text())
        assert set(manifest["failed"]) == {"gone", "nopath"}
        assert manifest["done"] == ["ok"]

    def test_heartbeat_logs_between_every_n_boundaries(self, tmp_path, capsys):
        # On CPU each MERT track takes longer than the heartbeat window, so the
        # run must emit a progress line per track (not stay silent until the
        # every-N=10 boundary, which made it "look hung").
        ok_fp = self._touch(tmp_path, "ok.flac")

        def prober(fp):
            return 100.0

        def loader(fp, offset, duration):
            return np.ones(32, np.float32)

        # Fake clock advancing 20s/call -> every track exceeds the 15s heartbeat.
        ticks = iter(range(0, 100_000, 20))
        clock = lambda: float(next(ticks))  # noqa: E731

        store = _store(tmp_path)
        items = [(f"t{i}", ok_fp) for i in range(5)]
        run_extraction(
            items, fake_embedder, store, prober=prober, loader=loader,
            clock=clock, heartbeat_s=15.0,  # log_every defaults to 10 -> never hits for 5 items
        )
        out = capsys.readouterr().out
        # An intermediate track logged via the heartbeat (every-N alone would not).
        assert "2/5" in out
        # First and last track always announced.
        assert "1/5" in out
        assert "5/5" in out

    def test_decode_error_recorded_and_run_continues(self, tmp_path):
        bad_fp = self._touch(tmp_path, "corrupt.flac")
        ok_fp = self._touch(tmp_path, "ok.flac")

        def prober(fp):
            if fp == bad_fp:
                raise RuntimeError("corrupt header")
            return 100.0

        def loader(fp, offset, duration):
            return np.ones(32, np.float32)

        store = _store(tmp_path)
        summary = run_extraction(
            [("bad", bad_fp), ("ok", ok_fp)],
            fake_embedder, store, prober=prober, loader=loader,
        )
        assert summary == {"ok": 1, "failed": 1}
        manifest = json.loads((tmp_path / "shards" / "manifest.json").read_text())
        assert "RuntimeError" in manifest["failed"]["bad"]
