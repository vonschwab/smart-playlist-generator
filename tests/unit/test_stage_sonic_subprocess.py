"""stage_sonic runs the sonic pool via the import-light entry subprocess.

This is the prevention half of the spawn-deadlock fix: the pool's __main__ must
be numpy-free, which means running it out-of-process via scripts/run_sonic_pool.py
rather than in-process under the heavy analyze_library/worker __main__.

Tests target the extracted helper ``_run_sonic_analysis`` so no real process,
DB, or audio is needed.
"""
import types

import pytest

import scripts.analyze_library as al


def _ctx(cancel=None):
    args = types.SimpleNamespace(progress_interval=15.0, progress_every=500, verbose=False)
    return {"args": args, "db_path": "DB.db", "cancellation_check": cancel}


def test_invokes_entry_subprocess_with_expected_argv(monkeypatch):
    recorded = {}

    def fake_stream(argv, *, on_line, cancellation_check=None, **kwargs):
        recorded["argv"] = list(argv)
        on_line("Using 8 parallel workers")  # exercise output forwarding
        return 0

    monkeypatch.setattr(al, "run_streaming_subprocess", fake_stream)
    fell_back = {"v": False}
    monkeypatch.setattr(al, "_run_sonic_in_process", lambda *a, **k: fell_back.__setitem__("v", True))

    al._run_sonic_analysis(_ctx(), workers=8, force=True, limit=None)

    argv = recorded["argv"]
    assert any("run_sonic_pool.py" in str(a) for a in argv)
    assert "--db-path" in argv and "DB.db" in argv
    assert "--force" in argv
    assert "--workers" in argv and "8" in argv
    assert fell_back["v"] is False


def test_falls_back_to_in_process_on_nonzero_exit(monkeypatch):
    monkeypatch.setattr(al, "run_streaming_subprocess", lambda *a, **k: 1)
    fell_back = {"v": False}
    monkeypatch.setattr(al, "_run_sonic_in_process", lambda *a, **k: fell_back.__setitem__("v", True))

    al._run_sonic_analysis(_ctx(), workers=4, force=False, limit=10)

    assert fell_back["v"] is True


def test_cancellation_propagates_without_fallback(monkeypatch):
    class _Cancel(Exception):
        pass

    def fake_stream(argv, *, on_line, cancellation_check=None, **kwargs):
        raise _Cancel()

    monkeypatch.setattr(al, "run_streaming_subprocess", fake_stream)
    fell_back = {"v": False}
    monkeypatch.setattr(al, "_run_sonic_in_process", lambda *a, **k: fell_back.__setitem__("v", True))

    with pytest.raises(_Cancel):
        al._run_sonic_analysis(_ctx(), workers=2, force=False, limit=None)

    assert fell_back["v"] is False
