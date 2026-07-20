#!/usr/bin/env python
"""Build the distributable MixArc wheel: npm build -> bundle dist -> python -m build."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "src" / "playlist_web" / "static_dist"


def main() -> None:
    npm = shutil.which("npm")
    if npm is None:
        sys.exit("npm is required to build the frontend for the wheel")
    subprocess.run([npm, "run", "build"], cwd=ROOT / "web", check=True)
    if STATIC.exists():
        shutil.rmtree(STATIC)
    shutil.copytree(ROOT / "web" / "dist", STATIC)
    subprocess.run([sys.executable, "-m", "build", "--wheel"], cwd=ROOT, check=True)
    print("wheel in dist/ — static bundle included")


if __name__ == "__main__":
    main()
