import numpy as np
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
        if path == "/bad": raise RuntimeError("decode fail")
        return np.ones(4, np.float32)
    res = run_muq_extraction([("g", "/good"), ("b", "/bad")], stub, sc)
    assert res["ok"] == 1 and res["failed"] == 1 and sidecar_ids(sc) == {"g"}
