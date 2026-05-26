"""
Presets Manager - Handles saving/loading UIState preset configurations.

Presets are stored under %APPDATA%\\PlaylistGenerator\\presets\\ on Windows.
Each preset is a YAML file containing the full UIStateModel.
"""
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import asdict, fields

import yaml

from ..ui_state import UIStateModel

try:
    import platformdirs
except ImportError:
    platformdirs = None

logger = logging.getLogger(__name__)

PRESET_VERSION = 1


def get_app_data_dir() -> Path:
    """Get the application data directory for storing user files."""
    if platformdirs:
        base = platformdirs.user_data_dir("PlaylistGenerator", "PlaylistGenerator")
    else:
        base = os.environ.get("APPDATA", os.path.expanduser("~"))
        base = os.path.join(base, "PlaylistGenerator")
    return Path(base)


def get_presets_dir() -> Path:
    """Get the presets directory."""
    return get_app_data_dir() / "presets"


def get_logs_dir() -> Path:
    """Get the logs directory."""
    return get_app_data_dir() / "logs"


def serialize_ui_state(state: UIStateModel) -> dict:
    """Convert UIStateModel to a plain dict for YAML/JSON storage."""
    return asdict(state)


def deserialize_ui_state(data: dict) -> UIStateModel:
    """Construct UIStateModel from a dict, ignoring unknown keys and filling missing with defaults."""
    valid_fields = {f.name for f in fields(UIStateModel)}
    filtered = {k: v for k, v in data.items() if k in valid_fields}
    return UIStateModel(**filtered)


class PresetManager:
    """
    Manages UIState preset storage and retrieval.

    Usage:
        manager = PresetManager()
        presets = manager.list_presets()
        manager.save_preset("My Preset", ui_state)
        state = manager.load_preset("My Preset")
        manager.delete_preset("My Preset")
    """

    def __init__(self):
        self.presets_dir = get_presets_dir()
        self._ensure_dir_exists()

    def _ensure_dir_exists(self) -> None:
        """Create presets directory if it doesn't exist."""
        self.presets_dir.mkdir(parents=True, exist_ok=True)

    def _get_preset_path(self, name: str) -> Path:
        """Get the file path for a preset by name."""
        safe_name = "".join(c for c in name if c.isalnum() or c in " -_").strip()
        if not safe_name:
            safe_name = "preset"
        return self.presets_dir / f"{safe_name}.yaml"

    def list_presets(self) -> List[Dict[str, Any]]:
        """
        List all available presets (excludes session file).

        Returns:
            List of dicts with 'name', 'path', and 'modified' keys
        """
        presets = []
        if not self.presets_dir.exists():
            return presets

        for path in sorted(self.presets_dir.glob("*.yaml")):
            if path.stem.startswith("_"):
                continue
            presets.append({
                "name": path.stem,
                "path": str(path),
                "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat()
            })

        return presets

    def save_preset(
        self,
        name: str,
        state: UIStateModel,
        description: str = "",
    ) -> Path:
        """
        Save UIState as a named preset.

        Args:
            name: Preset name (used as filename)
            state: UIStateModel to persist
            description: Optional description

        Returns:
            Path to the saved preset file
        """
        self._ensure_dir_exists()
        path = self._get_preset_path(name)

        data = {
            "name": name,
            "description": description,
            "version": PRESET_VERSION,
            "state": serialize_ui_state(state),
        }

        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

        return path

    def load_preset(self, name: str) -> Optional[UIStateModel]:
        """
        Load a preset as UIStateModel.

        Args:
            name: Preset name

        Returns:
            UIStateModel, or None if not found or corrupt
        """
        path = self._get_preset_path(name)
        if not path.exists():
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except Exception:
            logger.warning("Failed to parse preset file: %s", path)
            return None

        if not data or "state" not in data:
            logger.warning("Preset file missing 'state' key: %s", path)
            return None

        try:
            return deserialize_ui_state(data["state"])
        except Exception:
            logger.warning("Failed to deserialize state in: %s", path)
            return None

    def load_preset_full(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Load full preset data including metadata.

        Returns:
            Full preset dict with name, description, version, and state
        """
        path = self._get_preset_path(name)
        if not path.exists():
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception:
            logger.warning("Failed to parse preset file: %s", path)
            return None

    def delete_preset(self, name: str) -> bool:
        """Delete a preset. Returns True if deleted."""
        path = self._get_preset_path(name)
        if path.exists():
            path.unlink()
            return True
        return False

    def preset_exists(self, name: str) -> bool:
        """Check if a preset with the given name exists."""
        return self._get_preset_path(name).exists()

    def save_session(self, state: UIStateModel) -> Path:
        """Save current UIState as session file for restore on next launch."""
        self._ensure_dir_exists()
        path = self.presets_dir / "_session.json"
        data = {
            "version": PRESET_VERSION,
            "state": serialize_ui_state(state),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return path

    def load_session(self) -> Optional[UIStateModel]:
        """Load session UIState. Returns None if missing or corrupt."""
        path = self.presets_dir / "_session.json"
        if not path.exists():
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            logger.warning("Failed to load session file: %s", path)
            return None

        if not data or "state" not in data:
            return None

        try:
            return deserialize_ui_state(data["state"])
        except Exception:
            logger.warning("Failed to deserialize session state in: %s", path)
            return None

    def export_preset(self, name: str, export_path: str) -> bool:
        """Export a preset to a specific path."""
        full = self.load_preset_full(name)
        if full is None:
            return False

        with open(export_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(full, f, default_flow_style=False, sort_keys=False)

        return True

    def import_preset(self, import_path: str, name: Optional[str] = None) -> Optional[str]:
        """
        Import a preset from a file.

        Returns:
            The imported preset name, or None if failed
        """
        path = Path(import_path)
        if not path.exists():
            return None

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except Exception:
            return None

        if not data or "state" not in data:
            return None

        preset_name = name or data.get("name") or path.stem
        try:
            state = deserialize_ui_state(data["state"])
        except Exception:
            return None
        self.save_preset(preset_name, state, data.get("description", ""))
        return preset_name
