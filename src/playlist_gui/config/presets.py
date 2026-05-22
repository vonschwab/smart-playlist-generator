"""
Presets Manager - Handles saving/loading preset configurations.

Presets are stored under %APPDATA%\\PlaylistGenerator\\presets\\ on Windows.
Each preset is a YAML file containing only the overrides (not the full config).
"""
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

try:
    import platformdirs
except ImportError:
    platformdirs = None


def get_app_data_dir() -> Path:
    """Get the application data directory for storing user files."""
    if platformdirs:
        base = platformdirs.user_data_dir("PlaylistGenerator", "PlaylistGenerator")
    else:
        # Fallback for Windows without platformdirs
        base = os.environ.get("APPDATA", os.path.expanduser("~"))
        base = os.path.join(base, "PlaylistGenerator")
    return Path(base)


def get_presets_dir() -> Path:
    """Get the presets directory."""
    return get_app_data_dir() / "presets"


def get_logs_dir() -> Path:
    """Get the logs directory."""
    return get_app_data_dir() / "logs"


class PresetManager:
    """
    Manages preset storage and retrieval.

    Usage:
        manager = PresetManager()
        presets = manager.list_presets()
        manager.save_preset("My Preset", overrides_dict)
        overrides = manager.load_preset("My Preset")
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
        # Sanitize name for filesystem
        safe_name = "".join(c for c in name if c.isalnum() or c in " -_").strip()
        if not safe_name:
            safe_name = "preset"
        return self.presets_dir / f"{safe_name}.yaml"

    def list_presets(self) -> List[Dict[str, Any]]:
        """
        List all available presets.

        Returns:
            List of dicts with 'name', 'path', and 'modified' keys
        """
        presets = []
        if not self.presets_dir.exists():
            return presets

        for path in sorted(self.presets_dir.glob("*.yaml")):
            presets.append({
                "name": path.stem,
                "path": str(path),
                "modified": datetime.fromtimestamp(path.stat().st_mtime).isoformat()
            })

        return presets

    def save_preset(
        self,
        name: str,
        overrides: Dict[str, Any],
        description: Optional[str] = None
    ) -> Path:
        """
        Save overrides as a preset.

        Args:
            name: Preset name (used as filename)
            overrides: Dictionary of override values
            description: Optional description for the preset

        Returns:
            Path to the saved preset file
        """
        self._ensure_dir_exists()
        path = self._get_preset_path(name)

        data = {
            "name": name,
            "description": description or "",
            "created": datetime.now().isoformat(),
            "overrides": overrides
        }

        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

        return path

    def load_preset(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Load a preset's overrides.

        Args:
            name: Preset name

        Returns:
            Dictionary of overrides, or None if not found
        """
        path = self._get_preset_path(name)
        if not path.exists():
            return None

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if data and "overrides" in data:
            return data["overrides"]

        return None

    def load_preset_full(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Load full preset data including metadata.

        Args:
            name: Preset name

        Returns:
            Full preset dict with name, description, created, and overrides
        """
        path = self._get_preset_path(name)
        if not path.exists():
            return None

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def delete_preset(self, name: str) -> bool:
        """
        Delete a preset.

        Args:
            name: Preset name

        Returns:
            True if deleted, False if not found
        """
        path = self._get_preset_path(name)
        if path.exists():
            path.unlink()
            return True
        return False

    def preset_exists(self, name: str) -> bool:
        """Check if a preset with the given name exists."""
        return self._get_preset_path(name).exists()

    def export_preset(self, name: str, export_path: str) -> bool:
        """
        Export a preset to a specific path.

        Args:
            name: Preset name
            export_path: Destination path

        Returns:
            True if exported successfully
        """
        data = self.load_preset_full(name)
        if data is None:
            return False

        with open(export_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)

        return True

    def import_preset(self, import_path: str, name: Optional[str] = None) -> Optional[str]:
        """
        Import a preset from a file.

        Args:
            import_path: Source file path
            name: Override the preset name (uses file's name if None)

        Returns:
            The imported preset name, or None if failed
        """
        path = Path(import_path)
        if not path.exists():
            return None

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data or "overrides" not in data:
            return None

        preset_name = name or data.get("name") or path.stem
        self.save_preset(
            preset_name,
            data["overrides"],
            data.get("description")
        )

        return preset_name


# Built-in presets that ship with the application
BUILTIN_PRESETS = {
    "Focused": {
        "description": "Narrow, cohesive playlists similar to seed",
        "overrides": {
            "playlists": {
                "ds_pipeline": {
                    "mode": "narrow",
                    "scoring": {
                        "alpha": 0.70,
                        "gamma": 0.01
                    },
                    "candidate_pool": {
                        "similarity_floor": 0.35,
                        "max_pool_size": 500
                    }
                }
            }
        }
    },
    "Discovery": {
        "description": "Explore new music with varied selections",
        "overrides": {
            "playlists": {
                "ds_pipeline": {
                    "mode": "discover",
                    "scoring": {
                        "alpha": 0.40,
                        "gamma": 0.10
                    },
                    "candidate_pool": {
                        "similarity_floor": 0.10,
                        "max_pool_size": 2000
                    }
                }
            }
        }
    },
    "Smooth Transitions": {
        "description": "Prioritize smooth track-to-track flow",
        "overrides": {
            "playlists": {
                "ds_pipeline": {
                    "scoring": {
                        "beta": 0.70
                    },
                    "constraints": {
                        "transition_floor": 0.30
                    },
                    "repair": {
                        "enabled": True
                    }
                }
            }
        }
    },
    "Artist Variety": {
        "description": "Maximize artist diversity",
        "overrides": {
            "playlists": {
                "ds_pipeline": {
                    "scoring": {
                        "gamma": 0.08
                    },
                    "constraints": {
                        "min_gap": 8
                    },
                    "candidate_pool": {
                        "max_artist_fraction": 0.08
                    }
                }
            }
        }
    },
    "Pure Sonic": {
        "description": "Audio similarity only, ignore genres",
        "overrides": {
            "playlists": {
                "ds_pipeline": {
                    "mode": "sonic_only",
                    "embedding": {
                        "sonic_weight": 1.0,
                        "genre_weight": 0.0
                    }
                },
                "genre_similarity": {
                    "enabled": False
                }
            }
        }
    }
}


def install_builtin_presets(manager: PresetManager) -> int:
    """
    Install built-in presets if they don't exist.

    Args:
        manager: PresetManager instance

    Returns:
        Number of presets installed
    """
    installed = 0
    for name, data in BUILTIN_PRESETS.items():
        if not manager.preset_exists(name):
            manager.save_preset(name, data["overrides"], data["description"])
            installed += 1
    return installed
