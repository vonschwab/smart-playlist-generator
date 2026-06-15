"""Static reachability audit: find src/ modules unreachable from production entrypoints.

Read-only maintenance tool. Builds an AST import graph over src/ + scripts/ +
tools/ + main_app.py, BFS from every production entrypoint (every script and tool
is its own entrypoint), and reports internal `src/` modules nothing reaches, each
with its reverse-importers. A module being *unreachable with zero importers* is
strong evidence it is dead; verify with the test suite before deleting.

This exists because dead modules have repeatedly been mistaken for live wiring.
Re-run after any structural change:  python tools/dead_code_audit.py
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PKG_ROOTS = ["src", "scripts"]


def module_name_for(path: Path) -> str:
    rel = path.relative_to(ROOT).with_suffix("")
    parts = list(rel.parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def build_module_map() -> tuple[dict[str, Path], set[str]]:
    mods: dict[str, Path] = {}
    packages: set[str] = set()
    for r in PKG_ROOTS:
        for p in (ROOT / r).rglob("*.py"):
            name = module_name_for(p)
            mods[name] = p
            if p.name == "__init__.py":
                packages.add(name)
    for p in [ROOT / "main_app.py", *(ROOT / "tools").glob("*.py")]:
        if p.exists():
            mods[module_name_for(p)] = p
    return mods, packages


def resolve_relative(cur_mod: str, is_pkg: bool, node: ast.ImportFrom) -> str | None:
    if node.level == 0:
        return node.module
    parts = cur_mod.split(".")
    drop = node.level - (1 if is_pkg else 0)  # `.` from a package == the package itself
    base = parts[: len(parts) - drop] if drop >= 0 else parts
    if node.module:
        base = base + node.module.split(".")
    return ".".join(base) if base else None


def imports_of(path: Path, cur_mod: str, is_pkg: bool) -> set[str]:
    out: set[str] = set()
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return out
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                out.add(a.name)
        elif isinstance(node, ast.ImportFrom):
            base = resolve_relative(cur_mod, is_pkg, node)
            if base:
                out.add(base)
                for a in node.names:
                    out.add(f"{base}.{a.name}")
    return out


def main() -> None:
    mods, packages = build_module_map()
    fwd: dict[str, set[str]] = {}
    for name, path in mods.items():
        targets = set()
        for imp in imports_of(path, name, name in packages):
            if imp in mods:
                targets.add(imp)
            else:
                parent = imp.rsplit(".", 1)[0] if "." in imp else ""
                if parent in mods:
                    targets.add(parent)
        fwd[name] = targets

    rev: dict[str, set[str]] = {m: set() for m in mods}
    for src_mod, tgts in fwd.items():
        for t in tgts:
            rev[t].add(src_mod)

    # Entrypoints: top-level scripts + tools + main_app + worker/web + analyze.
    entrypoints = {
        m for m, p in mods.items()
        if m == "main_app"
        or m.startswith("tools.")
        or (m.startswith("scripts.") and p.parent == ROOT / "scripts")
    }
    entrypoints |= {"src.playlist_gui.worker", "src.playlist_web.app",
                    "src.playlist_web.worker_bridge"}
    entrypoints &= set(mods)

    seen: set[str] = set()
    stack = list(entrypoints)
    while stack:
        m = stack.pop()
        if m in seen:
            continue
        seen.add(m)
        stack.extend(t for t in fwd.get(m, ()) if t not in seen)

    # Exclude package __init__ markers: they show "unreached" when consumers import
    # submodules (from pkg.sub import X) rather than the package itself. A genuinely
    # dead package surfaces via its submodules instead.
    src_mods = {m for m in mods if m.startswith("src.") and m not in packages}
    unreached = sorted(src_mods - seen)

    print("# Dead-code reachability audit (src/ only; scripts+tools are entrypoints)")
    print(f"modules: {len(mods)} | reached: {len(seen)} | src unreached (non-pkg): {len(unreached)}\n")
    print("## Unreached src/ modules — module | #importers | importer files")
    for m in unreached:
        importers = sorted(rev.get(m, set()))
        names = ", ".join(Path(mods[i]).name for i in importers) or "(none)"
        flag = "ORPHAN" if not importers else "cluster"
        print(f"  [{flag}] {m}  ({len(importers)})  <- {names}")


if __name__ == "__main__":
    main()
