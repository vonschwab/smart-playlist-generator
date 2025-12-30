"""
Config Model - Handles loading, merging, and overriding configuration

Key responsibilities:
1. Load base config from YAML file
2. Merge overrides from GUI controls
3. Get/set values by dot-path (e.g., "playlists.ds_pipeline.tower_weights.rhythm")
4. Validate types based on schema
5. Handle normalization groups (weights that must sum to 1.0)
6. Track modified state and provide diff utilities

Security: Never log or expose values from keys containing secret-like names.
"""
import copy
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .settings_schema import (
    SETTINGS_SCHEMA,
    SettingSpec,
    SettingType,
    get_normalize_groups,
    get_setting_by_key,
)


# Keys that should never be logged or displayed
SECRET_PATTERNS = re.compile(
    r"(api_key|token|secret|password|credential|plex|discogs|lastfm|bearer)",
    re.IGNORECASE
)


def is_secret_key(key: str) -> bool:
    """Check if a key path contains secret-like names."""
    return bool(SECRET_PATTERNS.search(key))


def redact_secrets(data: Any, path: str = "") -> Any:
    """Recursively redact secret values from a data structure."""
    if isinstance(data, dict):
        result = {}
        for k, v in data.items():
            new_path = f"{path}.{k}" if path else k
            if isinstance(v, (dict, list)):
                result[k] = redact_secrets(v, new_path)
            else:
                result[k] = "***REDACTED***" if is_secret_key(k) else v
        return result
    elif isinstance(data, list):
        return [redact_secrets(item, path) for item in data]
    else:
        return data


class ConfigModel:
    """
    Configuration model that handles base config + overrides.

    Usage:
        model = ConfigModel("config.yaml")
        value = model.get("playlists.ds_pipeline.tower_weights.rhythm")
        model.set("playlists.ds_pipeline.tower_weights.rhythm", 0.25)
        overrides = model.get_overrides()  # Only changed values
        merged = model.get_merged_config()  # Full config with overrides applied
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self._base_config: Dict[str, Any] = {}
        self._overrides: Dict[str, Any] = {}

        if config_path:
            self.load(config_path)

    def load(self, config_path: str) -> None:
        """Load base configuration from YAML file."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(path, "r", encoding="utf-8") as f:
            self._base_config = yaml.safe_load(f) or {}

        self.config_path = config_path
        self._overrides = {}  # Reset overrides on new load

    def reload(self) -> None:
        """Reload configuration from the current path."""
        if self.config_path:
            self.load(self.config_path)

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a value by dot-path, checking overrides first, then base config.

        Args:
            key_path: Dot-separated path (e.g., "playlists.ds_pipeline.mode")
            default: Value to return if key not found

        Returns:
            The value at the path, or default if not found
        """
        # Check overrides first
        override_value = self._get_nested(self._overrides, key_path)
        if override_value is not None:
            return override_value

        # Fall back to base config
        base_value = self._get_nested(self._base_config, key_path)
        if base_value is not None:
            return base_value

        # Fall back to schema default
        spec = get_setting_by_key(key_path)
        if spec and spec.default is not None:
            return spec.default

        return default

    def set(self, key_path: str, value: Any) -> None:
        """
        Set an override value by dot-path.

        Args:
            key_path: Dot-separated path
            value: New value to set

        The value is stored in overrides, not modifying the base config.
        """
        # Validate type if we have a schema spec
        spec = get_setting_by_key(key_path)
        if spec:
            value = self._coerce_type(value, spec)

        self._set_nested(self._overrides, key_path, value)

    def reset(self, key_path: Optional[str] = None) -> None:
        """
        Reset overrides.

        Args:
            key_path: If provided, reset only this key. Otherwise reset all.
        """
        if key_path:
            self._delete_nested(self._overrides, key_path)
        else:
            self._overrides = {}

    def get_base_value(self, key_path: str, default: Any = None) -> Any:
        """Get value from base config only (ignoring overrides)."""
        value = self._get_nested(self._base_config, key_path)
        if value is not None:
            return value

        # Fall back to schema default
        spec = get_setting_by_key(key_path)
        if spec and spec.default is not None:
            return spec.default

        return default

    def get_overrides(self) -> Dict[str, Any]:
        """Get the current overrides dictionary (for sending to worker)."""
        return copy.deepcopy(self._overrides)

    def get_merged_config(self) -> Dict[str, Any]:
        """Get full config with overrides merged in."""
        return self._deep_merge(
            copy.deepcopy(self._base_config),
            copy.deepcopy(self._overrides)
        )

    def get_merged_config_redacted(self) -> Dict[str, Any]:
        """Get merged config with secrets redacted (safe for logging)."""
        return redact_secrets(self.get_merged_config())

    def set_overrides(self, overrides: Dict[str, Any]) -> None:
        """Set overrides from a dictionary (e.g., loading a preset)."""
        self._overrides = copy.deepcopy(overrides)

    def is_modified(self, key_path: str) -> bool:
        """Check if a key has been modified from the base value."""
        return self._get_nested(self._overrides, key_path) is not None

    def has_override(self, key_path: str) -> bool:
        """Check if a key path has an override set. Alias for is_modified()."""
        return self.is_modified(key_path)

    def clear_override(self, key_path: str) -> None:
        """
        Clear a single override, reverting to base config value.

        Args:
            key_path: Dot-separated path to clear
        """
        self._delete_nested(self._overrides, key_path)
        self._cleanup_empty_dicts(self._overrides)

    def list_overrides(self) -> Dict[str, Any]:
        """
        Get a flat dictionary of all overrides with key_path -> value.

        Returns:
            Dict mapping key_path strings to their override values
        """
        result = {}
        self._flatten_dict(self._overrides, "", result)
        return result

    def override_count(self) -> int:
        """Get the number of individual overrides set."""
        return len(self.list_overrides())

    def diff_summary(self, include_secrets: bool = False) -> List[Tuple[str, Any, Any]]:
        """
        Get a summary of all differences between base config and current overrides.

        Args:
            include_secrets: If False (default), secret key_paths are excluded

        Returns:
            List of (key_path, base_value, override_value) tuples
        """
        diffs = []
        flat_overrides = self.list_overrides()

        for key_path, override_value in flat_overrides.items():
            # Skip secrets unless explicitly requested
            if not include_secrets and is_secret_key(key_path):
                continue

            base_value = self.get_base_value(key_path)
            diffs.append((key_path, base_value, override_value))

        return diffs

    def clear_group_overrides(self, group_name: str) -> int:
        """
        Clear all overrides for settings in a specific UI group.

        Args:
            group_name: The group name (e.g., "Scoring", "Tower Weights")

        Returns:
            Number of overrides cleared
        """
        from .settings_schema import get_settings_by_group

        groups = get_settings_by_group()
        if group_name not in groups:
            return 0

        cleared = 0
        for spec in groups[group_name]:
            if self.has_override(spec.key_path):
                self.clear_override(spec.key_path)
                cleared += 1

        return cleared

    def get_effective_value(self, key_path: str, default: Any = None) -> Any:
        """
        Get the effective value (override if exists, else base config).

        This is an alias for get() but named for clarity in override contexts.
        """
        return self.get(key_path, default)

    @staticmethod
    def _flatten_dict(
        data: Dict[str, Any],
        prefix: str,
        result: Dict[str, Any]
    ) -> None:
        """Recursively flatten a nested dict into key_path -> value mapping."""
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                ConfigModel._flatten_dict(value, full_key, result)
            else:
                result[full_key] = value

    @staticmethod
    def _cleanup_empty_dicts(data: Dict[str, Any]) -> None:
        """Remove empty nested dictionaries after clearing overrides."""
        keys_to_remove = []
        for key, value in data.items():
            if isinstance(value, dict):
                ConfigModel._cleanup_empty_dicts(value)
                if not value:  # Now empty
                    keys_to_remove.append(key)
        for key in keys_to_remove:
            del data[key]

    # ─────────────────────────────────────────────────────────────────────────
    # Normalization Support
    # ─────────────────────────────────────────────────────────────────────────

    def normalize_group(self, group_name: str, changed_key: Optional[str] = None) -> None:
        """
        Normalize a group of weights to sum to 1.0.

        Args:
            group_name: Name of the normalization group
            changed_key: The key that was just changed (will be preserved)
        """
        groups = get_normalize_groups()
        if group_name not in groups:
            return

        specs = groups[group_name]
        if len(specs) < 2:
            return

        # Get current values
        values = {}
        for spec in specs:
            values[spec.key_path] = self.get(spec.key_path, spec.default or 0.0)

        total = sum(values.values())
        if abs(total - 1.0) < 0.0001:
            return  # Already normalized

        if changed_key and changed_key in values:
            # Preserve the changed key, adjust others proportionally
            changed_value = values[changed_key]
            remaining_keys = [k for k in values.keys() if k != changed_key]
            remaining_sum = sum(values[k] for k in remaining_keys)
            target_remaining = 1.0 - changed_value

            if remaining_sum > 0:
                scale = target_remaining / remaining_sum
                for key in remaining_keys:
                    new_value = round(values[key] * scale, 3)
                    self.set(key, max(0.0, min(1.0, new_value)))
            else:
                # Distribute equally
                equal_share = target_remaining / len(remaining_keys) if remaining_keys else 0
                for key in remaining_keys:
                    self.set(key, round(equal_share, 3))
        else:
            # No specific key changed, just scale all proportionally
            if total > 0:
                for key in values.keys():
                    new_value = round(values[key] / total, 3)
                    self.set(key, new_value)

    def get_group_sum(self, group_name: str) -> float:
        """Get the sum of all values in a normalization group."""
        groups = get_normalize_groups()
        if group_name not in groups:
            return 0.0

        total = 0.0
        for spec in groups[group_name]:
            total += self.get(spec.key_path, spec.default or 0.0)
        return total

    # ─────────────────────────────────────────────────────────────────────────
    # Helper Methods
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _get_nested(data: Dict[str, Any], key_path: str) -> Any:
        """Get a value from a nested dict by dot-path."""
        if not data or not key_path:
            return None

        keys = key_path.split(".")
        current = data

        for key in keys:
            if not isinstance(current, dict):
                return None
            if key not in current:
                return None
            current = current[key]

        return current

    @staticmethod
    def _set_nested(data: Dict[str, Any], key_path: str, value: Any) -> None:
        """Set a value in a nested dict by dot-path, creating intermediate dicts."""
        keys = key_path.split(".")
        current = data

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    @staticmethod
    def _delete_nested(data: Dict[str, Any], key_path: str) -> None:
        """Delete a value from a nested dict by dot-path."""
        keys = key_path.split(".")
        current = data

        for key in keys[:-1]:
            if not isinstance(current, dict) or key not in current:
                return
            current = current[key]

        if isinstance(current, dict) and keys[-1] in current:
            del current[keys[-1]]

    @staticmethod
    def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge override into base, returning new dict."""
        result = copy.deepcopy(base)

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigModel._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)

        return result

    @staticmethod
    def _coerce_type(value: Any, spec: SettingSpec) -> Any:
        """Coerce a value to the type specified in the schema."""
        if value is None:
            return spec.default

        if spec.setting_type == SettingType.INT:
            return int(float(value))
        elif spec.setting_type == SettingType.FLOAT:
            return float(value)
        elif spec.setting_type == SettingType.BOOL:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.lower() in ("true", "1", "yes", "on")
            return bool(value)
        elif spec.setting_type == SettingType.STRING:
            return str(value)
        elif spec.setting_type == SettingType.CHOICE:
            str_value = str(value)
            if spec.choices and str_value in spec.choices:
                return str_value
            return spec.default

        return value


def merge_config_with_overrides(base_path: str, overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    Utility function to merge a base config file with overrides dict.

    Args:
        base_path: Path to the base YAML config
        overrides: Dictionary of overrides (can be nested or flat key_paths)

    Returns:
        Merged configuration dictionary
    """
    model = ConfigModel(base_path)
    model.set_overrides(overrides)
    return model.get_merged_config()
