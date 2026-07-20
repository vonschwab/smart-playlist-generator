"""Launch the browser playlist GUI: start FastAPI (which owns the worker) and open the browser."""
from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = ROOT / "web"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def ensure_frontend_built() -> None:
    """Build web/dist from source so the server never serves a stale bundle.

    The browser is served the prebuilt static bundle in web/dist; editing
    web/src/ does nothing until it is rebuilt. To make a stale dist impossible,
    the launcher rebuilds on every start. Installs node deps when node_modules is
    absent, and on a build failure (commonly: package.json gained a dep that
    isn't installed yet) runs `npm install` and retries once. A missing npm is a
    loud warning rather than a silent stale serve. Bypass with --no-build.
    """
    npm = shutil.which("npm")
    if npm is None:
        print(
            "WARNING: npm not found on PATH — skipping frontend build; "
            "web/dist may be STALE.",
            file=sys.stderr,
        )
        return

    if not (WEB_DIR / "node_modules").exists():
        print("Installing web dependencies (first run)…", flush=True)
        subprocess.run([npm, "install"], cwd=WEB_DIR, check=True)

    print("Building frontend (web/dist)…", flush=True)
    if subprocess.run([npm, "run", "build"], cwd=WEB_DIR).returncode != 0:
        # Most common cause: package.json gained deps not yet installed.
        print("Build failed — running `npm install` and retrying once…", flush=True)
        subprocess.run([npm, "install"], cwd=WEB_DIR, check=True)
        if subprocess.run([npm, "run", "build"], cwd=WEB_DIR).returncode != 0:
            sys.exit(
                "Frontend build failed; refusing to serve a stale dist. "
                "Fix the build above, or start with --no-build to serve the "
                "existing web/dist as-is."
            )


def main() -> None:
    from src.mixarc.cli import run_server

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--no-build", action="store_true",
                    help="Skip the frontend build and serve the existing web/dist as-is.")
    args = ap.parse_args()

    if not args.no_build:
        ensure_frontend_built()

    worker_cmd_env = os.environ.get("PG_WEB_WORKER_CMD", "").strip()
    worker_cmd = shlex.split(worker_cmd_env) if worker_cmd_env else None
    run_server(host=args.host, port=args.port, open_browser=not args.no_browser,
               worker_cmd=worker_cmd, config_path=None)


if __name__ == "__main__":
    main()
