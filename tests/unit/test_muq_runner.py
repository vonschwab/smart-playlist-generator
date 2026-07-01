import numpy as np
from src.analyze import muq_runner
from src.analyze.muq_runner import (
    MODEL_NAME, sidecar_ids, pending_muq, run_muq_extraction,
)

def _write_sidecar(p, ids):
    np.savez(str(p), track_ids=np.array(ids, dtype=object),
             embeddings=np.zeros((len(ids), 4), np.float32), model=MODEL_NAME)

def test_pending_is_universe_minus_sidecar(tmp_path):
    sc = tmp_path / "muq_sidecar.npz"
    _write_sidecar(sc, ["a", "b"])
    assert sidecar_ids(sc) == {"a", "b"}
    pend, done = pending_muq(sc, ["a", "b", "c", "d"])
    assert pend == ["c", "d"] and done == 2

def test_pending_all_when_no_sidecar(tmp_path):
    pend, done = pending_muq(tmp_path / "absent.npz", ["a", "b"])
    assert pend == ["a", "b"] and done == 0

def test_run_extraction_appends_backs_up_and_is_resumable(tmp_path):
    sc = tmp_path / "muq_sidecar.npz"
    _write_sidecar(sc, ["a"])                      # pre-existing vector
    stub = lambda path: np.ones(4, np.float32)    # deterministic, no model
    res = run_muq_extraction([("b", "/x.flac"), ("c", None)], stub, sc, backup_stamp="20260701_000000")
    assert res["ok"] == 1 and res["failed"] == 1  # b embedded, c skipped (no path)
    assert res["fails"] == [("c", "no_path")]
    assert sidecar_ids(sc) == {"a", "b"}           # additive — 'a' preserved
    assert (tmp_path / "muq_sidecar.bak_20260701_000000.npz").exists()  # backup written

def test_run_extraction_survives_a_bad_file(tmp_path):
    sc = tmp_path / "muq_sidecar.npz"
    def stub(path):
        if path == "/bad":
            raise RuntimeError("decode fail")
        return np.ones(4, np.float32)
    res = run_muq_extraction([("g", "/good"), ("b", "/bad")], stub, sc)
    assert res["ok"] == 1 and res["failed"] == 1 and sidecar_ids(sc) == {"g"}

def test_run_extraction_no_crash_when_no_items_embed_successfully(tmp_path):
    sc = tmp_path / "absent.npz"
    res = run_muq_extraction([("x", None)], lambda p: np.ones(4), sc)
    assert res["failed"] == 1 and res["ok"] == 0

def test_run_extraction_no_crash_on_empty_items(tmp_path):
    sc = tmp_path / "absent.npz"
    res = run_muq_extraction([], lambda p: None, sc)
    assert res["ok"] == 0 and res["failed"] == 0

def test_periodic_checkpoint_fires_mid_loop(tmp_path, monkeypatch):
    sc = tmp_path / "muq_sidecar.npz"
    calls = []
    real_atomic_save = muq_runner._atomic_save

    def counting_atomic_save(sidecar_path, done):
        calls.append(1)
        return real_atomic_save(sidecar_path, done)

    monkeypatch.setattr(muq_runner, "_atomic_save", counting_atomic_save)
    stub = lambda path: np.ones(4, np.float32)
    items = [("a", "/a"), ("b", "/b"), ("c", "/c")]
    res = run_muq_extraction(items, stub, sc, save_every=1)
    assert res["ok"] == 3
    assert len(calls) > 1


def test_failed_ids_persisted_excluded_from_pending_and_cleared_on_success(tmp_path):
    from src.analyze.muq_runner import failed_ids
    sc = tmp_path / "muq_sidecar.npz"
    _write_sidecar(sc, ["a"])

    def stub(path):
        if path == "/bad":
            raise RuntimeError("decode fail")
        return np.ones(4, np.float32)

    # 'b' fails, 'c' succeeds -> b recorded in muq_failed.json, c added to the sidecar
    run_muq_extraction([("b", "/bad"), ("c", "/good")], stub, sc)
    assert failed_ids(sc) == {"b"}
    pend, done = pending_muq(sc, ["a", "b", "c", "d"])
    assert pend == ["d"] and done == 2          # done{a,c} + failed{b} both skipped; only d pending
    # a later successful embed of 'b' clears it from the failed set
    run_muq_extraction([("b", "/good")], stub, sc)
    assert failed_ids(sc) == set() and "b" in sidecar_ids(sc)
