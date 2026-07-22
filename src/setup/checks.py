# src/setup/checks.py
"""Pure health-check functions extracted from tools/doctor.py.

Each `check_*` here mirrors one `DoctorChecks.check_*` method (see the cited
`tools/doctor.py` line range on each function below), but RETURNS structured
CheckResult(s) instead of printing. The pass/warn/fail conditions and
messages are transcribed faithfully from doctor's current behavior; only the
output channel changes (print -> return).

`run_all_checks(home)` runs every check in doctor's documented section order
(doctor.py L371-389) and returns a flat list of CheckResult.
"""
from __future__ import annotations

import importlib
import importlib.util as _ilu
import sqlite3
import sys
from pathlib import Path

import yaml

from src.config_loader import resolve_database_path
from src.mixarc.paths import MixarcHome
from src.setup.result import CheckResult

# This package's own repo root (src/setup/checks.py -> src -> repo root).
# Used to locate repo-shipped templates/helpers (config.example.yaml, the
# workspace_identity hook) that live with the source tree itself, not with
# wherever a given `home` points its data — mirrors doctor.py's ROOT_DIR.
_THIS_REPO_ROOT = Path(__file__).resolve().parents[2]

# doctor.py L106-115
_CORE_DEPENDENCIES = [
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("yaml", "pyyaml"),
    ("librosa", "librosa"),
    ("mutagen", "mutagen"),
    ("sklearn", "scikit-learn"),
    ("tqdm", "tqdm"),
    ("requests", "requests"),
]

_STATUS_RANK = {"pass": 0, "warn": 1, "fail": 2}


def _worst(statuses: list[str]) -> str:
    """Aggregate several sub-check statuses into one — fail beats warn beats pass."""
    return max(statuses, key=lambda s: _STATUS_RANK[s])


def _load_config_dict(config_path: Path) -> dict:
    """Parse a config.yaml path into a dict; empty dict if absent or malformed.

    Mirrors doctor.py's tolerant `except Exception: pass` behavior — a bad
    config (missing file, YAML syntax error, non-mapping top level, ...) must
    never crash a check. Every config-parse site in this module routes
    through here so the malformed-config path degrades the same way
    everywhere instead of raising out of an individual check. Takes a path
    (not a `home`) because call sites don't always mean `home.config_path` —
    `check_satellite_data_paths` deliberately reads `anchor_dir/"config.yaml"`
    regardless of where `home.config_path` points.
    """
    try:
        if not config_path.exists():
            return {}
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        return {}


def check_python_version() -> CheckResult:
    """doctor.py L75-88."""
    version = sys.version_info
    if version >= (3, 11):
        return CheckResult(
            "python_version", "pass",
            f"Python {version.major}.{version.minor}.{version.micro}",
        )
    return CheckResult(
        "python_version", "fail",
        f"Python {version.major}.{version.minor} (need 3.11+)",
        fix_hint="Install Python 3.11 or newer",
    )


def check_core_dependencies() -> list[CheckResult]:
    """doctor.py L90 (check_dependency) + L104-120 (check_core_dependencies)."""
    results: list[CheckResult] = []
    for module, pip_name in _CORE_DEPENDENCIES:
        try:
            importlib.import_module(module)
            results.append(CheckResult(f"dep_{module}", "pass", f"{module} available"))
        except ImportError:
            results.append(CheckResult(
                f"dep_{module}", "fail", f"{module} not found",
                fix_hint=f"pip install {pip_name}",
            ))
    return results


def check_playlist_module_imports() -> CheckResult:
    """doctor.py L336-350."""
    try:
        # All three imports are availability checks; the noqa marks satisfy
        # ruff F401 since the names aren't referenced after the import.
        from src.config_loader import Config  # noqa: F401
        from src.local_library_client import LocalLibraryClient  # noqa: F401
        from src.playlist.pipeline import DSPipelineResult  # noqa: F401
        return CheckResult("module_imports", "pass", "Core modules importable")
    except ImportError as e:
        return CheckResult("module_imports", "fail", f"Import error: {e}")


def check_config_file(home: MixarcHome) -> CheckResult:
    """doctor.py L122-139."""
    if home.config_path.exists():
        return CheckResult("config_file", "pass", f"config.yaml found ({home.config_path})")

    example_path = _THIS_REPO_ROOT / "config.example.yaml"
    if example_path.exists():
        fix = "cp config.example.yaml config.yaml && edit config.yaml"
    else:
        fix = "Create config.yaml from template"
    return CheckResult("config_file", "fail", "config.yaml not found", fix_hint=fix)


def check_satellite_data_paths(home: MixarcHome) -> CheckResult:
    """doctor.py L249-321.

    ONLY meaningful for a repo-rooted checkout that turns out to be a
    satellite clone. For `home.source == "platformdirs"` (public/wheel
    install) this is a no-op pass — there is no satellite concept there.
    Sub-checks (database_path / artifact_path) are aggregated into one
    CheckResult (worst status wins) since the contract exposes a single
    `satellite_paths` id.
    """
    if home.source == "platformdirs":
        return CheckResult(
            "satellite_paths", "pass",
            "No-op: public install has no satellite concept.",
        )

    root = home.anchor_dir

    # Load the helper from THIS repo (doctor's own location), not from
    # `root`: in tests `root` is a bare fake satellite dir that has no
    # .claude/hooks.
    wsi_path = _THIS_REPO_ROOT / ".claude" / "hooks" / "workspace_identity.py"
    if not wsi_path.exists():
        # canonical fallback for odd layouts: nothing to enforce
        return CheckResult(
            "satellite_paths", "pass",
            "Workspace: canonical (no workspace_identity helper)",
        )

    spec = _ilu.spec_from_file_location("workspace_identity", wsi_path)
    assert spec is not None and spec.loader is not None
    wsi = _ilu.module_from_spec(spec)
    spec.loader.exec_module(wsi)

    if not wsi.is_satellite(root):
        return CheckResult(
            "satellite_paths", "pass",
            "Workspace: canonical checkout (satellite checks n/a)",
        )

    cfg_path = root / "config.yaml"
    if not cfg_path.exists():
        return CheckResult(
            "satellite_paths", "fail", "Satellite has no config.yaml",
            fix_hint="python tools/create_satellite.py rewrites one from canonical",
        )
    cfg = _load_config_dict(cfg_path)

    floors = {"database_path": 1 * 1024 * 1024, "artifact_path": 10 * 1024 * 1024}
    raw_db = str(((cfg.get("library") or {}).get("database_path")) or "")
    raw_art = str((((cfg.get("playlists") or {}).get("ds_pipeline") or {})
                   .get("artifact_path")) or "")

    statuses: list[str] = []
    messages: list[str] = []
    fix_hint: str | None = None
    for label, raw, floor in (
        ("database_path", raw_db, floors["database_path"]),
        ("artifact_path", raw_art, floors["artifact_path"]),
    ):
        p = Path(raw) if raw else None
        if p is None or not p.is_absolute():
            statuses.append("fail")
            messages.append(f"Satellite {label} must be an ABSOLUTE canonical path (got {raw!r})")
            fix_hint = fix_hint or "Point it at the canonical checkout's data/ (create_satellite.py does this)"
            continue
        resolved = p.resolve()
        if resolved.is_relative_to(root.resolve()):
            statuses.append("fail")
            messages.append(f"Satellite {label} resolves INSIDE this clone ({resolved}) — that's the stub")
            continue
        if not resolved.exists():
            statuses.append("fail")
            messages.append(f"Satellite {label} target missing: {resolved}")
            continue
        size = resolved.stat().st_size
        if size < floor:
            statuses.append("fail")
            messages.append(f"Satellite {label} target suspiciously small ({size} bytes < {floor}) — stub?")
            continue
        statuses.append("pass")
        messages.append(f"Satellite {label}: {resolved} ({size / (1024 * 1024):.0f} MB)")

    return CheckResult("satellite_paths", _worst(statuses), "; ".join(messages), fix_hint=fix_hint)


def check_database(home: MixarcHome) -> CheckResult:
    """doctor.py L141-216.

    Note: the sonic_features count query (L182) assumes that column exists,
    which is true of the real metadata.db schema but not of a minimal
    ad-hoc `tracks` table (e.g. in tests) — that sub-query is wrapped so a
    missing column degrades to omitting the sonic-feature count rather than
    failing the whole check.
    """
    raw = _load_config_dict(home.config_path)
    db_path = Path(resolve_database_path(raw, anchor=home.anchor_dir))

    if not db_path.exists():
        return CheckResult(
            "database", "warn",
            f"Database not found: {db_path}. Run: python scripts/scan_library.py",
        )

    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='tracks'"
        )
        if not cursor.fetchone():
            conn.close()
            return CheckResult(
                "database", "fail", "Database missing 'tracks' table",
                fix_hint="Run: python scripts/scan_library.py",
            )

        cursor.execute("SELECT COUNT(*) FROM tracks")
        count = cursor.fetchone()[0]

        try:
            cursor.execute("SELECT COUNT(*) FROM tracks WHERE sonic_features IS NOT NULL")
            sonic_count = cursor.fetchone()[0]
        except sqlite3.Error:
            sonic_count = None

        conn.close()

        statuses: list[str] = []
        messages: list[str] = []

        if count == 0:
            statuses.append("warn")
            messages.append("Database has no tracks. Run: python scripts/scan_library.py")
        else:
            statuses.append("pass")
            if sonic_count is None:
                messages.append(f"Database: {count:,} tracks")
            else:
                messages.append(f"Database: {count:,} tracks ({sonic_count:,} with sonic features)")

        # Integrity check — surfaces the index/table corruption class that a raw
        # file-copy backup of an open DB can propagate (2026-07 idx_tracks_file_path
        # incident). Not fatal (data is still readable), but the fix is a REINDEX.
        try:
            from src.db_backup import check_integrity
            integ_ok, integ_detail = check_integrity(db_path)
            if integ_ok:
                statuses.append("pass")
                messages.append("Database integrity: ok")
            else:
                statuses.append("warn")
                messages.append(
                    f"Database integrity issue — run: sqlite3 metadata.db 'REINDEX;'  ({integ_detail})"
                )
        except Exception as e:  # never let the health check itself break doctor
            statuses.append("warn")
            messages.append(f"Could not run integrity_check: {e}")

        return CheckResult("database", _worst(statuses), "; ".join(messages))

    except sqlite3.Error as e:
        return CheckResult("database", "fail", f"Database error: {e}")


def check_artifacts(home: MixarcHome) -> CheckResult:
    """doctor.py L218-247."""
    artifacts_dir = home.anchor_dir / "data" / "artifacts"

    if not artifacts_dir.exists():
        return CheckResult(
            "artifacts", "warn",
            "Artifacts directory not found. Run: python scripts/build_beat3tower_artifacts.py ...",
        )

    npz_files = list(artifacts_dir.rglob("*.npz"))
    if not npz_files:
        return CheckResult(
            "artifacts", "warn",
            "No artifact files found (.npz). Run: python scripts/build_beat3tower_artifacts.py ...",
        )

    main_artifact = artifacts_dir / "beat3tower_32k" / "data_matrices_step1.npz"
    if main_artifact.exists():
        size_mb = main_artifact.stat().st_size / (1024 * 1024)
        return CheckResult(
            "artifacts", "pass", f"Main artifact: {main_artifact.name} ({size_mb:.1f} MB)",
        )
    return CheckResult(
        "artifacts", "warn",
        f"Main artifact not at expected path: {main_artifact}. "
        f"Found {len(npz_files)} artifact(s) in {artifacts_dir}",
    )


def run_all_checks(home: MixarcHome) -> list[CheckResult]:
    """All checks in doctor's documented section order (doctor.py L371-389)."""
    results: list[CheckResult] = [check_python_version()]
    results.extend(check_core_dependencies())
    results.append(check_playlist_module_imports())
    results.append(check_config_file(home))
    results.append(check_satellite_data_paths(home))
    results.append(check_database(home))
    results.append(check_artifacts(home))
    return results
