#!/usr/bin/env python3
"""Bootstrap a standing satellite clone for simultaneous Claude sessions.

Spec: docs/superpowers/specs/2026-07-06-simultaneous-sessions-design.md (§7).

    python tools/create_satellite.py --name PG3_SAT1 --port 8771

Steps: git clone -> config.yaml copy with absolute canonical data paths ->
copy untracked local config (.claude/settings.local.json, .mcp.json) ->
npm install + build -> auto-memory pointer -> doctor gate (fails loudly).
Satellites NEVER get links/junctions of any kind, and data-writing pipeline
stages stay canonical-only (satellite_data_write_guard enforces this).
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

CANONICAL_ROOT = Path(__file__).resolve().parent.parent


def rewrite_config_text(text: str, canonical_root: Path) -> str:
    """Point database_path + ds_pipeline.artifact_path at canonical, by absolute path.

    Targeted line edits only — the rest of the file (comments included) is
    preserved byte-for-byte. Handles: artifact_path present; ds_pipeline
    present without artifact_path; playlists absent entirely.
    """
    db_abs = (canonical_root / "data" / "metadata.db").as_posix()
    art_abs = (
        canonical_root / "data" / "artifacts" / "beat3tower_32k" / "data_matrices_step1.npz"
    ).as_posix()

    out = re.sub(
        r"(?m)^(\s*)database_path:.*$",
        rf"\g<1>database_path: {db_abs}",
        text,
        count=1,
    )

    if re.search(r"(?m)^\s*artifact_path:", out):
        out = re.sub(
            r"(?m)^(\s*)artifact_path:.*$",
            rf"\g<1>artifact_path: {art_abs}",
            out,
            count=1,
        )
    elif (m := re.search(r"(?m)^(\s*)ds_pipeline:\s*$", out)):
        indent = m.group(1) + "  "
        insert_at = m.end()
        out = out[:insert_at] + f"\n{indent}artifact_path: {art_abs}" + out[insert_at:]
    else:
        block = f"playlists:\n  ds_pipeline:\n    artifact_path: {art_abs}\n"
        out = out.rstrip("\n") + "\n" + block
    return out


def memory_project_key(path_str: str) -> str:
    """The harness's project-dir munging: [:\\/_] -> '-' (verified against this repo's key)."""
    return re.sub(r"[:\\/_]", "-", path_str)


def memory_pointer_text(canonical_root: Path) -> str:
    canon_key = memory_project_key(str(canonical_root))
    canon_memory = Path.home() / ".claude" / "projects" / canon_key / "memory"
    return (
        "<!-- Satellite clone pointer: this workspace shares the canonical project's memory. -->\n"
        f"- [CANONICAL MEMORY — read this first](file://{(canon_memory / 'MEMORY.md').as_posix()}) — "
        "this satellite has no memory of its own. Read the canonical MEMORY.md index at "
        f"`{canon_memory / 'MEMORY.md'}` at session start, and write any new memories into "
        f"`{canon_memory}` (absolute path), not this directory.\n"
    )


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    if result.returncode != 0:
        sys.exit(f"FAILED ({result.returncode}): {' '.join(cmd)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True, help="satellite dir name, e.g. PG3_SAT1")
    ap.add_argument("--port", type=int, required=True, help="GUI port, e.g. 8771")
    ap.add_argument("--dest-root", default=str(CANONICAL_ROOT.parent),
                    help="parent dir for the clone (default: canonical's parent)")
    args = ap.parse_args()

    sat = Path(args.dest_root) / args.name
    if sat.exists():
        sys.exit(f"Refusing: {sat} already exists (satellites are standing; delete manually first).")

    print(f"[1/6] Cloning canonical -> {sat}")
    _run(["git", "clone", str(CANONICAL_ROOT), str(sat)])

    print("[2/6] Writing satellite config.yaml (absolute canonical data paths)")
    canon_cfg = CANONICAL_ROOT / "config.yaml"
    if not canon_cfg.exists():
        sys.exit("Canonical config.yaml missing — cannot derive satellite config.")
    (sat / "config.yaml").write_text(
        rewrite_config_text(canon_cfg.read_text(encoding="utf-8"), CANONICAL_ROOT),
        encoding="utf-8",
    )

    print("[3/6] Copying untracked local config (settings.local.json, .mcp.json)")
    for rel in (Path(".claude") / "settings.local.json", Path(".mcp.json")):
        src = CANONICAL_ROOT / rel
        if src.exists():
            dest = sat / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)

    print("[4/6] npm install + build (one-time; a few minutes)")
    npm = shutil.which("npm")
    if npm is None:
        print(
            "  ! npm not found on PATH — skipping web build. Install Node.js, then run "
            f"`npm install` and `npm run build` in {sat / 'web'} before using the satellite GUI."
        )
    else:
        # Run npm FROM the web dir: `npm install` reads package.json from cwd, not
        # from --prefix (an absolute --prefix here made npm look at the canonical root).
        web_dir = sat / "web"
        _run([npm, "install"], cwd=web_dir)
        _run([npm, "run", "build"], cwd=web_dir)

    print("[5/6] Auto-memory pointer")
    mem_dir = Path.home() / ".claude" / "projects" / memory_project_key(str(sat)) / "memory"
    mem_dir.mkdir(parents=True, exist_ok=True)
    pointer = mem_dir / "MEMORY.md"
    if not pointer.exists():
        pointer.write_text(memory_pointer_text(CANONICAL_ROOT), encoding="utf-8")

    print("[6/6] Doctor gate (satellite data-path checks)")
    _run([sys.executable, "tools/doctor.py"], cwd=sat)

    print(
        f"\nSatellite ready: {sat}\n"
        f"  Launch Claude Code with cwd = {sat}  (never switch into it mid-session)\n"
        f"  GUI: python tools/serve_web.py --port {args.port}\n"
        f"  Work on feature branches; land via: git push origin <branch>, then merge in canonical.\n"
        f"  Data writes (analyze/publish/folds) stay in canonical — the guard will remind you."
    )


if __name__ == "__main__":
    main()
