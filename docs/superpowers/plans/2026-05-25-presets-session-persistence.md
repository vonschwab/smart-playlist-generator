# Presets & Session Persistence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the preset system to save/load full UIState and replace incomplete session persistence with a single-file UIState snapshot.

**Architecture:** `PresetManager` changes from storing config-override dicts to serialized `UIStateModel`. `GeneratePanel` gains `apply_ui_state()` as the inverse of `build_ui_state()`. Session state auto-saves on close and auto-restores on startup via `_session.json`.

**Tech Stack:** Python 3.11+, PySide6, PyYAML, dataclasses

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `src/playlist_gui/config/presets.py` | Modify | Storage format → UIState; add serialize/deserialize helpers; remove built-ins |
| `src/playlist_gui/widgets/generate_panel.py` | Modify | Add `apply_ui_state()`; add `_set_menu_button_value()` helper; remove `apply_saved_state()` |
| `src/playlist_gui/widgets/mode_panels.py` | Modify | Add `set_presence()`, `set_variety()`, `set_auto_order()` setters |
| `src/playlist_gui/main_window.py` | Modify | Wire preset save/load to UIState; replace session persistence; simplify dirty tracking |
| `tests/unit/test_preset_serialization.py` | Create | Round-trip, forward-compat, edge-case tests for serialize/deserialize |
| `tests/unit/test_preset_manager.py` | Create | PresetManager save/load/list/delete/session with UIState |
| `tests/unit/test_apply_ui_state.py` | Create | GeneratePanel.apply_ui_state round-trip tests |

---

### Task 1: UIState Serialization Helpers

**Files:**
- Modify: `src/playlist_gui/config/presets.py`
- Create: `tests/unit/test_preset_serialization.py`

- [ ] **Step 1: Write failing tests for round-trip serialization**

```python
"""Tests for UIState serialization/deserialization."""

from dataclasses import asdict

from src.playlist_gui.config.presets import deserialize_ui_state, serialize_ui_state
from src.playlist_gui.ui_state import UIStateModel


def test_round_trip_default_state():
    state = UIStateModel()
    data = serialize_ui_state(state)
    restored = deserialize_ui_state(data)
    assert restored == state


def test_round_trip_custom_state():
    state = UIStateModel(
        mode="genre",
        cohesion_mode="strict",
        genre_mode="dynamic",
        sonic_mode="off",
        pace_mode="strict",
        track_count=50,
        diversity_gamma=0.08,
        artist_diversity_mode="one_per_artist",
        recency_enabled=False,
        recency_days=30,
        recency_plays_threshold=3,
        artist_spacing="very_strong",
        artist_queries=["Slowdive", "Cocteau Twins"],
        artist_presence="high",
        artist_variety="sprawling",
        include_collaborations=True,
        genre_query="ambient",
        seed_track_ids=["track_001", "track_002"],
        seed_auto_order=False,
    )
    data = serialize_ui_state(state)
    restored = deserialize_ui_state(data)
    assert restored == state


def test_deserialize_ignores_unknown_fields():
    data = asdict(UIStateModel())
    data["future_field"] = "some_value"
    data["another_unknown"] = 42
    restored = deserialize_ui_state(data)
    assert restored == UIStateModel()


def test_deserialize_fills_missing_fields_with_defaults():
    data = {"mode": "genre", "genre_query": "shoegaze"}
    restored = deserialize_ui_state(data)
    assert restored.mode == "genre"
    assert restored.genre_query == "shoegaze"
    assert restored.cohesion_mode == "dynamic"
    assert restored.track_count == 30
    assert restored.artist_spacing == "normal"


def test_serialize_produces_plain_dict():
    state = UIStateModel(artist_queries=["Miles Davis"])
    data = serialize_ui_state(state)
    assert isinstance(data, dict)
    assert data["artist_queries"] == ["Miles Davis"]
    assert "mode" in data
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_preset_serialization.py -v`
Expected: FAIL with `ImportError: cannot import name 'deserialize_ui_state'`

- [ ] **Step 3: Implement serialization helpers**

Add to `src/playlist_gui/config/presets.py` (at module level, before the `PresetManager` class):

```python
from dataclasses import asdict, fields
from src.playlist_gui.ui_state import UIStateModel


def serialize_ui_state(state: UIStateModel) -> dict:
    """Convert UIStateModel to a plain dict for YAML/JSON storage."""
    return asdict(state)


def deserialize_ui_state(data: dict) -> UIStateModel:
    """Construct UIStateModel from a dict, ignoring unknown keys and filling missing with defaults."""
    valid_fields = {f.name for f in fields(UIStateModel)}
    filtered = {k: v for k, v in data.items() if k in valid_fields}
    return UIStateModel(**filtered)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_preset_serialization.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add tests/unit/test_preset_serialization.py src/playlist_gui/config/presets.py
git commit -m "feat: add UIState serialize/deserialize helpers"
```

---

### Task 2: Refactor PresetManager to Store UIState

**Files:**
- Modify: `src/playlist_gui/config/presets.py`
- Create: `tests/unit/test_preset_manager.py`

- [ ] **Step 1: Write failing tests for UIState-based PresetManager**

```python
"""Tests for PresetManager with UIState storage."""

import json
from pathlib import Path

import pytest

from src.playlist_gui.config.presets import PresetManager
from src.playlist_gui.ui_state import UIStateModel


@pytest.fixture
def manager(tmp_path, monkeypatch):
    """PresetManager with isolated temp directory."""
    monkeypatch.setattr(
        "src.playlist_gui.config.presets.get_presets_dir",
        lambda: tmp_path,
    )
    return PresetManager()


def test_save_and_load_preset(manager):
    state = UIStateModel(mode="genre", genre_query="ambient", cohesion_mode="strict")
    manager.save_preset("My Ambient", state)
    loaded = manager.load_preset("My Ambient")
    assert loaded is not None
    assert loaded == state


def test_load_nonexistent_returns_none(manager):
    assert manager.load_preset("does not exist") is None


def test_list_presets_excludes_session_file(manager):
    state = UIStateModel()
    manager.save_preset("Preset One", state)
    manager.save_session(state)
    presets = manager.list_presets()
    names = [p["name"] for p in presets]
    assert "Preset One" in names
    assert "_session" not in names


def test_delete_preset(manager):
    state = UIStateModel(mode="seeds")
    manager.save_preset("Temporary", state)
    assert manager.preset_exists("Temporary")
    manager.delete_preset("Temporary")
    assert not manager.preset_exists("Temporary")


def test_save_session_and_load_session(manager):
    state = UIStateModel(
        mode="artist",
        cohesion_mode="discover",
        track_count=40,
        artist_queries=["Boards of Canada"],
    )
    manager.save_session(state)
    loaded = manager.load_session()
    assert loaded == state


def test_load_session_returns_none_when_missing(manager):
    assert manager.load_session() is None


def test_load_session_returns_none_on_corrupt_file(manager):
    session_path = manager.presets_dir / "_session.json"
    session_path.write_text("not valid json {{{", encoding="utf-8")
    assert manager.load_session() is None


def test_load_preset_handles_missing_fields(manager):
    """Old preset file missing new fields gets defaults."""
    import yaml

    path = manager._get_preset_path("Old Preset")
    data = {
        "name": "Old Preset",
        "version": 1,
        "state": {"mode": "artist", "genre_mode": "narrow"},
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)

    loaded = manager.load_preset("Old Preset")
    assert loaded is not None
    assert loaded.mode == "artist"
    assert loaded.genre_mode == "narrow"
    assert loaded.cohesion_mode == "dynamic"  # default
    assert loaded.track_count == 30  # default


def test_load_preset_full_returns_metadata(manager):
    state = UIStateModel(cohesion_mode="narrow")
    manager.save_preset("With Meta", state, description="A test preset")
    full = manager.load_preset_full("With Meta")
    assert full is not None
    assert full["name"] == "With Meta"
    assert full["description"] == "A test preset"
    assert full["version"] == 1
    assert full["state"]["cohesion_mode"] == "narrow"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_preset_manager.py -v`
Expected: FAIL — `save_preset` signature mismatch (expects `overrides: dict`, not `UIStateModel`)

- [ ] **Step 3: Rewrite PresetManager to use UIState**

Replace the body of `src/playlist_gui/config/presets.py` with:

```python
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

import yaml

from ..ui_state import UIStateModel
from dataclasses import asdict, fields

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
            return None

        return deserialize_ui_state(data["state"])

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

        return deserialize_ui_state(data["state"])

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
        state = deserialize_ui_state(data["state"])
        self.save_preset(preset_name, state, data.get("description", ""))
        return preset_name
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_preset_manager.py tests/unit/test_preset_serialization.py -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/playlist_gui/config/presets.py tests/unit/test_preset_manager.py
git commit -m "feat: refactor PresetManager to store UIState instead of config overrides"
```

---

### Task 3: Add Setters to Mode Panels

**Files:**
- Modify: `src/playlist_gui/widgets/mode_panels.py`

The `ArtistModePanel` has getters (`get_presence()`, `get_variety()`, `get_include_collaborations()`) and some setters (`set_primary_artist()`, `set_include_collaborations()`), but is missing `set_presence()` and `set_variety()`. The `SeedsModePanel` is missing `set_auto_order()`.

- [ ] **Step 1: Write failing tests**

Add to `tests/unit/test_apply_ui_state.py` (this file will grow in Task 4):

```python
"""Tests for programmatic UI state restoration."""

import pytest

from src.playlist_gui.widgets.generate_panel import GeneratePanel


def test_artist_panel_set_presence(qtbot):
    panel = GeneratePanel()
    qtbot.addWidget(panel)

    panel._artist_panel.set_presence("very_high")
    assert panel._artist_panel.get_presence() == "very_high"

    panel._artist_panel.set_presence("very_low")
    assert panel._artist_panel.get_presence() == "very_low"


def test_artist_panel_set_variety(qtbot):
    panel = GeneratePanel()
    qtbot.addWidget(panel)

    panel._artist_panel.set_variety("sprawling")
    assert panel._artist_panel.get_variety() == "sprawling"

    panel._artist_panel.set_variety("focused")
    assert panel._artist_panel.get_variety() == "focused"


def test_seeds_panel_set_auto_order(qtbot):
    panel = GeneratePanel()
    qtbot.addWidget(panel)

    panel._seeds_panel.set_auto_order(False)
    assert panel._seeds_panel.get_auto_order() is False

    panel._seeds_panel.set_auto_order(True)
    assert panel._seeds_panel.get_auto_order() is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_apply_ui_state.py::test_artist_panel_set_presence tests/unit/test_apply_ui_state.py::test_artist_panel_set_variety tests/unit/test_apply_ui_state.py::test_seeds_panel_set_auto_order -v`
Expected: FAIL with `AttributeError: 'ArtistModePanel' object has no attribute 'set_presence'`

- [ ] **Step 3: Add setters to ArtistModePanel**

In `src/playlist_gui/widgets/mode_panels.py`, add after the `set_primary_artist` method (around line 283):

```python
    def set_presence(self, level: str) -> None:
        """Set presence level programmatically."""
        levels = ["very_low", "low", "medium", "high", "very_high"]
        if level in levels:
            self._presence_combo.setCurrentIndex(levels.index(level))

    def set_variety(self, level: str) -> None:
        """Set variety level programmatically."""
        levels = ["focused", "balanced", "sprawling"]
        if level in levels:
            self._variety_slider.setValue(levels.index(level))
```

- [ ] **Step 4: Add setter to SeedsModePanel**

In `src/playlist_gui/widgets/mode_panels.py`, add after the `get_auto_order` method (around line 501):

```python
    def set_auto_order(self, enabled: bool) -> None:
        """Set auto-order checkbox programmatically."""
        self._auto_order_check.setChecked(enabled)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/unit/test_apply_ui_state.py -v`
Expected: All 3 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/playlist_gui/widgets/mode_panels.py tests/unit/test_apply_ui_state.py
git commit -m "feat: add set_presence, set_variety, set_auto_order setters to mode panels"
```

---

### Task 4: GeneratePanel.apply_ui_state()

**Files:**
- Modify: `src/playlist_gui/widgets/generate_panel.py`
- Modify: `tests/unit/test_apply_ui_state.py`

- [ ] **Step 1: Write failing tests for apply_ui_state round-trip**

Append to `tests/unit/test_apply_ui_state.py`:

```python
from src.playlist_gui.ui_state import UIStateModel


def test_apply_ui_state_round_trip_defaults(qtbot):
    panel = GeneratePanel()
    qtbot.addWidget(panel)

    original = UIStateModel()
    panel.apply_ui_state(original)
    restored = panel.build_ui_state()

    assert restored.mode == original.mode
    assert restored.cohesion_mode == original.cohesion_mode
    assert restored.genre_mode == original.genre_mode
    assert restored.sonic_mode == original.sonic_mode
    assert restored.pace_mode == original.pace_mode
    assert restored.track_count == original.track_count
    assert restored.diversity_gamma == original.diversity_gamma
    assert restored.artist_diversity_mode == original.artist_diversity_mode
    assert restored.recency_enabled == original.recency_enabled
    assert restored.recency_days == original.recency_days
    assert restored.recency_plays_threshold == original.recency_plays_threshold
    assert restored.artist_spacing == original.artist_spacing


def test_apply_ui_state_round_trip_custom_artist_mode(qtbot):
    panel = GeneratePanel()
    qtbot.addWidget(panel)

    state = UIStateModel(
        mode="artist",
        cohesion_mode="strict",
        genre_mode="dynamic",
        sonic_mode="off",
        pace_mode="strict",
        track_count=50,
        diversity_gamma=0.08,
        artist_diversity_mode="one_per_artist",
        recency_enabled=False,
        recency_days=30,
        recency_plays_threshold=3,
        artist_spacing="very_strong",
        artist_queries=["Slowdive"],
        artist_presence="high",
        artist_variety="sprawling",
        include_collaborations=True,
    )
    panel.apply_ui_state(state)
    restored = panel.build_ui_state()

    assert restored.mode == "artist"
    assert restored.cohesion_mode == "strict"
    assert restored.genre_mode == "dynamic"
    assert restored.sonic_mode == "off"
    assert restored.pace_mode == "strict"
    assert restored.track_count == 50
    assert restored.diversity_gamma == 0.08
    assert restored.artist_diversity_mode == "one_per_artist"
    assert restored.recency_enabled is False
    assert restored.recency_days == 30
    assert restored.recency_plays_threshold == 3
    assert restored.artist_spacing == "very_strong"
    assert restored.artist_queries == ["Slowdive"]
    assert restored.artist_presence == "high"
    assert restored.artist_variety == "sprawling"
    assert restored.include_collaborations is True


def test_apply_ui_state_round_trip_genre_mode(qtbot):
    panel = GeneratePanel()
    qtbot.addWidget(panel)

    state = UIStateModel(mode="genre", genre_query="shoegaze")
    panel.apply_ui_state(state)
    restored = panel.build_ui_state()

    assert restored.mode == "genre"
    assert restored.genre_query == "shoegaze"


def test_apply_ui_state_round_trip_seeds_mode(qtbot):
    panel = GeneratePanel()
    qtbot.addWidget(panel)

    state = UIStateModel(
        mode="seeds",
        seed_auto_order=False,
    )
    panel.apply_ui_state(state)
    restored = panel.build_ui_state()

    assert restored.mode == "seeds"
    assert restored.seed_auto_order is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_apply_ui_state.py::test_apply_ui_state_round_trip_defaults -v`
Expected: FAIL with `AttributeError: 'GeneratePanel' object has no attribute 'apply_ui_state'`

- [ ] **Step 3: Add `_set_menu_button_value` helper to GeneratePanel**

In `src/playlist_gui/widgets/generate_panel.py`, add after `_create_menu_button` (around line 466):

```python
    @staticmethod
    def _set_menu_button_value(button: QToolButton, value: object) -> None:
        """Set a menu button's selection to the action matching value."""
        for action in button.menu().actions():
            if action.data() == value:
                action.trigger()
                return
```

- [ ] **Step 4: Implement `apply_ui_state`**

In `src/playlist_gui/widgets/generate_panel.py`, replace the `apply_saved_state` method with:

```python
    def apply_ui_state(self, state: UIStateModel) -> None:
        """Restore all controls from a UIStateModel (inverse of build_ui_state)."""
        self.set_current_mode(state.mode)

        self._cohesion_slider.set_cohesion_mode(state.cohesion_mode)
        self._mode_sliders.set_genre_mode(state.genre_mode)
        self._mode_sliders.set_sonic_mode(state.sonic_mode)
        self._mode_sliders.set_pace_mode(state.pace_mode)

        self._set_menu_button_value(self._length_combo, state.track_count)

        # Diversity: find the slider position matching gamma + mode
        if state.artist_diversity_mode == "one_per_artist":
            self._diversity_slider.setValue(len(self._diversity_levels) - 1)
        else:
            closest = min(
                range(len(self._diversity_values) - 1),
                key=lambda i: abs(self._diversity_values[i] - state.diversity_gamma),
            )
            self._diversity_slider.setValue(closest)

        # Spacing
        if state.artist_spacing in self._spacing_levels:
            self._spacing_slider.setValue(self._spacing_levels.index(state.artist_spacing))

        # Recency
        self._recency_check.setChecked(state.recency_enabled)
        self._set_menu_button_value(self._recency_days, state.recency_days)
        self._set_menu_button_value(self._recency_plays, state.recency_plays_threshold)

        # Mode-specific controls
        if state.mode == "artist":
            if state.artist_queries:
                self._artist_panel.set_primary_artist(state.artist_queries[0])
            self._artist_panel.set_presence(state.artist_presence)
            self._artist_panel.set_variety(state.artist_variety)
            self._artist_panel.set_include_collaborations(state.include_collaborations)
        elif state.mode == "genre":
            self._genre_panel.set_genre(state.genre_query)
        elif state.mode == "seeds":
            self._seeds_panel.set_auto_order(state.seed_auto_order)
```

- [ ] **Step 5: Remove old `apply_saved_state` method**

Delete the `apply_saved_state` method entirely from `generate_panel.py`.

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/unit/test_apply_ui_state.py -v`
Expected: All 7 tests PASS

- [ ] **Step 7: Run full test suite to check for breakage**

Run: `pytest tests/unit/test_generate_panel.py -v`
Expected: `test_generate_panel_restores_saved_artist_state` FAILS (uses removed `apply_saved_state`)

- [ ] **Step 8: Update test_generate_panel.py**

Replace `test_generate_panel_restores_saved_artist_state` in `tests/unit/test_generate_panel.py`:

```python
def test_generate_panel_restores_saved_artist_state(qtbot):
    panel = GeneratePanel()
    qtbot.addWidget(panel)

    from src.playlist_gui.ui_state import UIStateModel

    state = UIStateModel(mode="artist", artist_queries=["Slowdive"])
    panel.apply_ui_state(state)

    assert panel.get_current_mode() == "artist"
    assert panel.get_primary_artist() == "Slowdive"
    assert panel.build_ui_state().artist_queries == ["Slowdive"]
```

- [ ] **Step 9: Run tests to verify all pass**

Run: `pytest tests/unit/test_generate_panel.py tests/unit/test_apply_ui_state.py -v`
Expected: All PASS

- [ ] **Step 10: Commit**

```bash
git add src/playlist_gui/widgets/generate_panel.py tests/unit/test_apply_ui_state.py tests/unit/test_generate_panel.py
git commit -m "feat: add GeneratePanel.apply_ui_state(), remove apply_saved_state()"
```

---

### Task 5: Wire MainWindow Preset Save/Load to UIState

**Files:**
- Modify: `src/playlist_gui/main_window.py`

- [ ] **Step 1: Update imports**

In `src/playlist_gui/main_window.py`, change the presets import from:

```python
from .config.presets import PresetManager, install_builtin_presets
```

to:

```python
from .config.presets import PresetManager
```

- [ ] **Step 2: Remove `install_builtin_presets` call**

Delete the line (around line 176):

```python
        install_builtin_presets(self._preset_manager)
```

- [ ] **Step 3: Rewrite `_on_save_preset`**

Replace the method body:

```python
    @Slot()
    def _on_save_preset(self) -> None:
        """Save current UIState as a preset."""
        if not self._generate_panel:
            return

        from PySide6.QtWidgets import QInputDialog

        default_name = self._active_preset_name or ""
        name, ok = QInputDialog.getText(
            self, "Save Preset", "Preset name:",
            text=default_name
        )

        if ok and name:
            state = self._generate_panel.build_ui_state()
            self._preset_manager.save_preset(name, state)
            self._log_panel.append_log("INFO", f"Saved preset: {name}")

            self._active_preset_name = name
            self._preset_ui_state_snapshot = state
            self._update_override_status()
```

- [ ] **Step 4: Rewrite `_on_load_preset`**

Replace the method body:

```python
    @Slot()
    def _on_load_preset(self) -> None:
        """Load a preset and apply its UIState."""
        if not self._generate_panel:
            return

        presets = self._preset_manager.list_presets()
        if not presets:
            QMessageBox.information(self, "No Presets", "No presets available.")
            return

        from PySide6.QtWidgets import QInputDialog
        names = [p["name"] for p in presets]
        name, ok = QInputDialog.getItem(self, "Load Preset", "Select preset:", names, 0, False)

        if ok and name:
            state = self._preset_manager.load_preset(name)
            if state:
                self._generate_panel.apply_ui_state(state)
                self._log_panel.append_log("INFO", f"Loaded preset: {name}")

                self._active_preset_name = name
                self._preset_ui_state_snapshot = state
                self._update_override_status()
```

- [ ] **Step 5: Rewrite `_on_reset_overrides`**

Replace the method body:

```python
    @Slot()
    def _on_reset_overrides(self) -> None:
        """Reset all controls to default UIState."""
        if self._generate_panel:
            from .ui_state import UIStateModel

            self._generate_panel.apply_ui_state(UIStateModel())
            self._log_panel.append_log("INFO", "Reset to default settings")

            self._active_preset_name = None
            self._preset_ui_state_snapshot = None
            self._update_override_status()
```

- [ ] **Step 6: Replace dirty tracking state**

In `__init__`, replace:

```python
        self._active_preset_name: Optional[str] = None
        self._preset_overrides_snapshot: dict = {}
        self._dirty_overrides = False
        self._pending_preset_name: Optional[str] = None
```

with:

```python
        self._active_preset_name: Optional[str] = None
        self._preset_ui_state_snapshot: Optional[UIStateModel] = None
```

Add the import at the top of the file (with other local imports):

```python
from .ui_state import UIStateModel
```

- [ ] **Step 7: Update `_update_override_status`**

Find the `_update_override_status` method and replace the dirty-check logic. The status label should show:
- `"Preset: {name}"` when a preset is loaded and current state matches
- `"Preset: {name} (modified)"` when a preset is loaded but state has drifted
- `""` (or "Base config") when no preset is active

Replace the method body:

```python
    def _update_override_status(self) -> None:
        """Update the override status indicator."""
        if self._active_preset_name:
            if self._preset_ui_state_snapshot and self._generate_panel:
                current = self._generate_panel.build_ui_state()
                if current == self._preset_ui_state_snapshot:
                    text = f"Preset: {self._active_preset_name}"
                    state = "preset"
                else:
                    text = f"Preset: {self._active_preset_name} (modified)"
                    state = "preset_modified"
            else:
                text = f"Preset: {self._active_preset_name}"
                state = "preset"
        else:
            text = ""
            state = "none"

        self._override_status.setText(text)
        self._override_status.setProperty("state", state)
        self._override_status.style().unpolish(self._override_status)
        self._override_status.style().polish(self._override_status)
```

- [ ] **Step 8: Remove `_check_override_drift` method**

Delete the entire `_check_override_drift` method (it compared override dicts).

- [ ] **Step 9: Remove `_pending_preset_name` usage in `_restore_settings`**

Delete the block (around line 1910-1912):

```python
            preset = self._settings.value("state/preset")
            if preset:
                self._pending_preset_name = str(preset)
```

And delete the block that restores it after config load (around line 749-762):

```python
            if self._pending_preset_name:
                presets = [p["name"] for p in self._preset_manager.list_presets()]
                if self._pending_preset_name in presets:
                    overrides = self._preset_manager.load_preset(self._pending_preset_name)
                    ...
                self._pending_preset_name = None
```

- [ ] **Step 10: Run full test suite**

Run: `pytest tests/ -x -q`
Expected: PASS (any test referencing `install_builtin_presets` or `apply_saved_state` should have been updated in earlier tasks; if not, fix forward)

- [ ] **Step 11: Commit**

```bash
git add src/playlist_gui/main_window.py
git commit -m "feat: wire preset save/load to UIState, simplify dirty tracking"
```

---

### Task 6: Replace Session Persistence with UIState

**Files:**
- Modify: `src/playlist_gui/main_window.py`

- [ ] **Step 1: Rewrite `_save_settings` state portion**

Replace the state-saving block in `_save_settings` (the lines saving `state/mode`, `state/artist`, `state/genre`, `state/genre_mode`, `state/sonic_mode`, `state/pace_mode`, `state/preset`):

```python
    def _save_settings(self) -> None:
        """Persist window/layout and form state."""
        try:
            self._settings.setValue("ui/geometry", self.saveGeometry())
            self._settings.setValue("ui/state", self.saveState())
            self._settings.setValue("ui/content_splitter", self._content_splitter.saveState())
            self._settings.setValue("ui/main_splitter", self._main_splitter.saveState())
            self._settings.setValue("state/config_path", self._config_path)
            self._settings.setValue("state/filter", self._track_table.get_filter_text())
            self._settings.setValue("ui/track_table/header_state", self._track_table.get_header_state())
            self._settings.setValue(
                "ui/track_table/visibility",
                json.dumps(self._track_table.get_column_visibility_state()),
            )

            # Save full UIState as session file
            if self._generate_panel:
                ui_state = self._generate_panel.build_ui_state()
                self._preset_manager.save_session(ui_state)

            # Persist active preset name
            if self._active_preset_name:
                self._settings.setValue("state/preset", self._active_preset_name)
            else:
                self._settings.remove("state/preset")
        except Exception as e:
            self._logger.warning("Failed to save settings: %s", e)
```

- [ ] **Step 2: Rewrite `_restore_settings` state portion**

Replace the state-restoring block. Remove the `_restore_generation_state(...)` call and the individual QSettings reads for mode/artist/genre/genre_mode/sonic_mode/pace_mode. Replace with session load:

```python
            # Restore UIState from session file
            if self._generate_panel:
                session_state = self._preset_manager.load_session()
                if session_state:
                    self._generate_panel.apply_ui_state(session_state)

            # Restore active preset name
            preset = self._settings.value("state/preset")
            if preset:
                self._active_preset_name = str(preset)
                loaded = self._preset_manager.load_preset(self._active_preset_name)
                if loaded:
                    self._preset_ui_state_snapshot = loaded
                self._update_override_status()
```

- [ ] **Step 3: Remove `_restore_generation_state` and `_normalize_saved_generation_mode`**

Delete both methods entirely (around lines 776-807).

- [ ] **Step 4: Clean up stale QSettings keys**

Remove the individual state key reads from `_restore_settings` that are now handled by session file:
- `state/mode`
- `state/artist`
- `state/genre`
- `state/genre_mode`
- `state/sonic_mode`
- `state/pace_mode`

These are no longer written in `_save_settings`, so they just need to be removed from the read side.

- [ ] **Step 5: Run full test suite**

Run: `pytest tests/ -x -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/playlist_gui/main_window.py
git commit -m "feat: replace scattered QSettings session state with UIState session file"
```

---

### Task 7: Remove Stale Built-in Presets and Dead Code

**Files:**
- Modify: `src/playlist_gui/config/presets.py` (already done in Task 2, verify clean)
- Modify: any file referencing `BUILTIN_PRESETS` or `install_builtin_presets`

- [ ] **Step 1: Verify no remaining references to removed code**

Run: `grep -r "BUILTIN_PRESETS\|install_builtin_presets\|apply_saved_state" src/ tests/ --include="*.py"`

Expected: no hits. If any remain, fix them.

- [ ] **Step 2: Verify no references to `_pending_preset_name`**

Run: `grep -r "_pending_preset_name" src/ tests/ --include="*.py"`

Expected: no hits.

- [ ] **Step 3: Verify no references to `_preset_overrides_snapshot` or `_dirty_overrides`**

Run: `grep -r "_preset_overrides_snapshot\|_dirty_overrides\|_check_override_drift" src/ tests/ --include="*.py"`

Expected: no hits.

- [ ] **Step 4: Run full test suite**

Run: `pytest tests/ -q`
Expected: all pass, no warnings about removed imports

- [ ] **Step 5: Commit (if any cleanup was needed)**

```bash
git add -u
git commit -m "chore: remove stale preset references and dead code"
```

---

### Task 8: Integration Smoke Test

**Files:**
- Modify: `tests/unit/test_apply_ui_state.py`

- [ ] **Step 1: Write integration test for full preset workflow**

Append to `tests/unit/test_apply_ui_state.py`:

```python
from src.playlist_gui.config.presets import PresetManager


def test_preset_save_load_round_trip_through_panel(qtbot, tmp_path, monkeypatch):
    """Full workflow: build state from panel → save preset → load preset → apply → verify."""
    monkeypatch.setattr(
        "src.playlist_gui.config.presets.get_presets_dir",
        lambda: tmp_path,
    )
    manager = PresetManager()
    panel = GeneratePanel()
    qtbot.addWidget(panel)

    # Set up non-default state via controls
    panel._cohesion_slider.set_cohesion_mode("strict")
    panel._mode_sliders.set_genre_mode("dynamic")
    panel._spacing_slider.setValue(3)  # very_strong
    panel._diversity_slider.setValue(5)  # One Each

    # Save preset from panel state
    original_state = panel.build_ui_state()
    manager.save_preset("Test Preset", original_state)

    # Reset panel to defaults
    panel.apply_ui_state(UIStateModel())
    assert panel.build_ui_state().cohesion_mode == "dynamic"

    # Load preset and apply
    loaded_state = manager.load_preset("Test Preset")
    assert loaded_state is not None
    panel.apply_ui_state(loaded_state)

    # Verify round-trip
    final_state = panel.build_ui_state()
    assert final_state.cohesion_mode == "strict"
    assert final_state.genre_mode == "dynamic"
    assert final_state.artist_spacing == "very_strong"
    assert final_state.artist_diversity_mode == "one_per_artist"


def test_session_save_load_round_trip_through_panel(qtbot, tmp_path, monkeypatch):
    """Full workflow: build state → save session → load session → apply → verify."""
    monkeypatch.setattr(
        "src.playlist_gui.config.presets.get_presets_dir",
        lambda: tmp_path,
    )
    manager = PresetManager()
    panel = GeneratePanel()
    qtbot.addWidget(panel)

    # Set up state
    panel._mode_sliders.set_pace_mode("strict")
    panel._recency_check.setChecked(False)

    # Save session
    state = panel.build_ui_state()
    manager.save_session(state)

    # Reset
    panel.apply_ui_state(UIStateModel())
    assert panel.build_ui_state().pace_mode == "dynamic"
    assert panel.build_ui_state().recency_enabled is True

    # Restore session
    session = manager.load_session()
    assert session is not None
    panel.apply_ui_state(session)

    final = panel.build_ui_state()
    assert final.pace_mode == "strict"
    assert final.recency_enabled is False
```

- [ ] **Step 2: Run integration tests**

Run: `pytest tests/unit/test_apply_ui_state.py -v`
Expected: All tests PASS

- [ ] **Step 3: Run full suite as final verification**

Run: `pytest tests/ -q`
Expected: All pass

- [ ] **Step 4: Commit**

```bash
git add tests/unit/test_apply_ui_state.py
git commit -m "test: add integration smoke tests for preset and session round-trip"
```
