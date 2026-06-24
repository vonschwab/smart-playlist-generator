"""The import-light entry point that runs the sonic pool in its own process.

Regression guard for the Analyze Library spawn deadlock (root-caused
2026-06-23): on Windows, multiprocessing `spawn` re-imports the __main__ module
in every pool child during prepare(). If __main__ imports numpy at module scope,
N children load numpy's C extension simultaneously and can deadlock. This entry
module exists so the pool's __main__ is numpy-free; the heavy import happens
inside main(), in the parent only.
"""
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_entry_module_imports_no_numpy_at_module_scope():
    # Importing the entry must NOT pull numpy into sys.modules. Checked in a
    # fresh interpreter so a pre-imported numpy (from pytest) can't mask it.
    code = (
        "import sys; import scripts.run_sonic_pool; "
        "print('NUMPY_PRESENT' if 'numpy' in sys.modules else 'NUMPY_ABSENT')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    assert proc.returncode == 0, proc.stderr
    assert "NUMPY_ABSENT" in proc.stdout, proc.stdout


def test_main_runs_pipeline_with_resolved_args(monkeypatch):
    import scripts.update_sonic as update_sonic
    from scripts import run_sonic_pool

    calls = {}

    class _FakePipeline:
        def __init__(self, **kwargs):
            calls["init"] = kwargs

        def run(self, **kwargs):
            calls["run"] = kwargs

        def close(self):
            calls["closed"] = True

    monkeypatch.setattr(update_sonic, "SonicFeaturePipeline", _FakePipeline)

    run_sonic_pool.main(["--db-path", "X.db", "--workers", "3", "--limit", "7", "--force"])

    assert calls["init"]["db_path"] == "X.db"
    assert calls["init"]["use_beat3tower"] is True
    assert calls["run"]["workers"] == 3
    assert calls["run"]["limit"] == 7
    assert calls["run"]["force"] is True
    assert calls["closed"] is True
