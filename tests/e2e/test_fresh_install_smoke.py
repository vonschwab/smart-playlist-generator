"""The 'works installed, not just in-repo' gate: wheel -> clean venv -> boot -> needs_setup.

Requires: dist/mixarc-6.0.0-py3-none-any.whl (python scripts/build_wheel.py) and network
for pip. Skips (loudly) otherwise. ~2-4 min.
"""
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
WHEEL = next(iter((ROOT / "dist").glob("mixarc-*.whl")), None)

pytestmark = [pytest.mark.slow, pytest.mark.integration]


@pytest.mark.skipif(WHEEL is None, reason="no wheel in dist/ — run scripts/build_wheel.py first")
def test_fresh_install_boots_to_needs_setup(tmp_path):
    venv = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    vpy = venv / ("Scripts" if os.name == "nt" else "bin") / "python"
    subprocess.run([str(vpy), "-m", "pip", "install", "--quiet", f"{WHEEL}[web]"], check=True)

    env = {**os.environ, "MIXARC_HOME": str(tmp_path / "home")}  # empty home -> needs_setup
    proc = subprocess.Popen([str(vpy), "-m", "src.mixarc.cli", "--no-browser", "--port", "8975"],
                            env=env, cwd=str(tmp_path))  # cwd OUTSIDE the repo — the whole point
    try:
        deadline = time.time() + 60
        last_err = None
        while time.time() < deadline:
            try:
                with urllib.request.urlopen("http://127.0.0.1:8975/api/setup/status", timeout=2) as r:
                    body = json.load(r)
                    assert body["state"] == "needs_setup"
                    break
            except AssertionError:
                raise
            except Exception as exc:
                last_err = exc
                time.sleep(1.0)
        else:
            pytest.fail(f"server never answered /api/setup/status: {last_err}")
    finally:
        proc.terminate()
        proc.wait(timeout=10)

    # The installed console script also boots — exercise that entry point too.
    mixarc_script = venv / ("Scripts" if os.name == "nt" else "bin") / "mixarc"
    subprocess.run([str(mixarc_script), "--help"], check=True)
