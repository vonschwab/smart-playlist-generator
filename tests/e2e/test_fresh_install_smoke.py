"""The 'works installed, not just in-repo' gate: wheel -> clean venv -> boot -> needs_setup.

Requires: a dist/mixarc-*.whl (python scripts/build_wheel.py) and network for pip.
Skips (loudly) otherwise. ~2-4 min.
"""
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
WHEEL = next(iter((ROOT / "dist").glob("mixarc-*.whl")), None)

pytestmark = [pytest.mark.slow, pytest.mark.integration]


def _free_port() -> int:
    """Bind to port 0 to get an OS-assigned free port, then release it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _isolated_env(tmp_path: Path) -> dict:
    """Env for the wheel-under-test subprocess: no PYTHONPATH leak from the parent.

    A dev/CI environment that exports PYTHONPATH pointing at the repo root could
    otherwise let `python -m src.mixarc.cli` resolve repo source ahead of the venv's
    installed wheel, silently defeating the whole point of this test.
    """
    env = {**os.environ, "MIXARC_HOME": str(tmp_path / "home")}  # empty home -> needs_setup
    env.pop("PYTHONPATH", None)
    return env


@pytest.mark.skipif(WHEEL is None, reason="no wheel in dist/ — run scripts/build_wheel.py first")
def test_fresh_install_boots_to_needs_setup(tmp_path):
    venv = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    vpy = venv / ("Scripts" if os.name == "nt" else "bin") / "python"
    subprocess.run([str(vpy), "-m", "pip", "install", "--quiet", f"{WHEEL}[web]"], check=True)

    port = _free_port()
    env = _isolated_env(tmp_path)
    proc = subprocess.Popen([str(vpy), "-m", "src.mixarc.cli", "--no-browser", "--port", str(port)],
                            env=env, cwd=str(tmp_path))  # cwd OUTSIDE the repo — the whole point
    try:
        deadline = time.time() + 60
        last_err = None
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/setup/status", timeout=2) as r:
                    body = json.load(r)
                    assert body["state"] == "needs_setup"

                # Finding 2.2: also confirm the bundled UI itself is served, not
                # just the setup-status API — a wheel built without static_dist/
                # (e.g. a package-data glob miss) would still pass the check
                # above while mounting no front-end at all.
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2) as r:
                    assert r.status == 200
                    html = r.read().decode("utf-8", errors="replace")
                    lowered = html.lower()
                    assert (
                        "<!doctype html" in lowered
                        or 'id="root"' in lowered
                        or "<title" in lowered
                    ), f"'/' did not look like the app's HTML shell: {html[:200]!r}"
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
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)

    # The installed console script also boots — exercise that entry point too, with the
    # same PYTHONPATH-stripped env and cwd OUTSIDE the repo as the primary launch above.
    mixarc_script = venv / ("Scripts" if os.name == "nt" else "bin") / "mixarc"
    subprocess.run([str(mixarc_script), "--help"], env=env, cwd=str(tmp_path), check=True)
