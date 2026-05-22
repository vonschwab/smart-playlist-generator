"""
Lightweight diagnostics checks for readiness.
"""
from __future__ import annotations

import sqlite3
import yaml
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str = ""


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def run_checks(base_config_path: str, config_model: Optional[Any] = None) -> List[CheckResult]:
    """
    Run quick readiness checks. Best-effort and non-destructive.
    """
    results: List[CheckResult] = []
    cfg_path = Path(base_config_path)

    # Config exists
    if not cfg_path.exists():
        results.append(CheckResult("config", False, f"Config missing: {cfg_path}"))
        return results

    try:
        cfg = config_model.get_merged_config() if config_model else _load_config(cfg_path)
        results.append(CheckResult("config", True, "Config loaded"))
    except Exception as e:
        results.append(CheckResult("config", False, f"Config load failed: {e}"))
        return results

    # DB check
    db_path = Path(cfg.get("library", {}).get("database_path", "data/metadata.db"))
    if not db_path.is_absolute():
        db_path = cfg_path.parent / db_path
    if not db_path.exists():
        results.append(CheckResult("database", False, f"Database not found: {db_path}"))
    else:
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM tracks")
            count = cursor.fetchone()[0]
            conn.close()
            results.append(CheckResult("database", count > 0, f"{count} tracks indexed"))
        except Exception as e:
            results.append(CheckResult("database", False, f"DB query failed: {e}"))

    # Artifacts check
    artifact_path = Path(cfg.get("playlists", {}).get("ds_pipeline", {}).get("artifact_path", "data/artifacts/beat3tower_32k/data_matrices_step1.npz"))
    if not artifact_path.is_absolute():
        artifact_path = cfg_path.parent / artifact_path
    results.append(CheckResult("artifacts", artifact_path.exists(), f"Artifacts {'found' if artifact_path.exists() else 'missing'} at {artifact_path}"))

    return results
