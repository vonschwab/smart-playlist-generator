# Sub-project C: Presets & Session Persistence

## Goal

Refactor the preset system to save/load full UIState (slider positions, dropdown values, checkboxes) instead of raw config overrides, and replace the incomplete session persistence with a single-file UIState snapshot that restores all generation controls on restart.

## Architecture

Three changes to the existing system:

1. **PresetManager** changes storage format from config-override dicts to serialized `UIStateModel` — same file location, same YAML format, new payload shape.
2. **GeneratePanel** gains `apply_ui_state(state: UIStateModel)` — the inverse of `build_ui_state()`, setting every control from a state object.
3. **MainWindow** replaces scattered `QSettings` state keys and override-dict dirty tracking with UIState-based equivalents.

## Scope

- Named presets: save/load full UIState via Presets menu (existing menu bar location)
- Session persistence: auto-save last UIState on close, auto-restore on startup
- Remove 5 stale built-in presets (`BUILTIN_PRESETS` dict and `install_builtin_presets()`)
- Simplify dirty tracking to UIState equality comparison

## Out of scope

- New built-in presets (separate brainstorm)
- Preset categories, tags, or search
- Preset sharing/sync
- Advanced panel / config-model override interaction (presets drive UIState only; policy derives config from UIState)

---

## Storage Format

### Named presets

Location: `%APPDATA%\PlaylistGenerator\presets\<name>.yaml`

```yaml
name: My Preset
version: 1
state:
  mode: artist
  cohesion_mode: strict
  genre_mode: narrow
  sonic_mode: narrow
  pace_mode: dynamic
  track_count: 30
  diversity_gamma: 0.04
  artist_diversity_mode: weighted
  recency_enabled: true
  recency_days: 14
  recency_plays_threshold: 1
  history_window_days: 30
  genre_query: ""
  artist_spacing: normal
  artist_queries: []
  artist_presence: medium
  artist_variety: balanced
  include_collaborations: false
  seed_track_ids: []
  seed_auto_order: true
```

### Session file

Location: `%APPDATA%\PlaylistGenerator\presets\_session.json`

Same `state:` payload as presets, JSON format. Written on `closeEvent`, read on startup. Not shown in the Load Preset menu (prefixed with `_`).

### Forward compatibility

- **Unknown fields on load:** silently ignored (future version added a field this version doesn't know about)
- **Missing fields on load:** filled with `UIStateModel` defaults (old preset predates a new field)
- **`version` key:** integer, currently `1`. Bump when a breaking schema change requires migration logic.

---

## Component Changes

### PresetManager (`src/playlist_gui/config/presets.py`)

**Removed:**
- `BUILTIN_PRESETS` dict
- `install_builtin_presets()` function

**Changed signatures:**
- `save_preset(name: str, state: UIStateModel, description: str = "")` — serializes UIStateModel to YAML with `version: 1` envelope
- `load_preset(name: str) → Optional[UIStateModel]` — deserializes YAML, applies defaults for missing fields, returns `UIStateModel`
- `load_preset_full(name: str) → Optional[Dict[str, Any]]` — returns full YAML dict (name, version, state) for metadata display
- `import_preset(path: str, name: Optional[str] = None) → Optional[str]` — validates imported file has `state` key (not `overrides`), loads into presets dir
- `export_preset(name: str, export_path: str) → bool` — unchanged (writes full YAML to path)

**New methods:**
- `save_session(state: UIStateModel) → Path` — writes `_session.json`
- `load_session() → Optional[UIStateModel]` — reads `_session.json`, returns None if missing/corrupt
- `list_presets()` — unchanged behavior, but skips files starting with `_` (excludes session file)

**Serialization helpers (module-level or static):**
- `serialize_ui_state(state: UIStateModel) → dict` — converts UIStateModel to plain dict via `dataclasses.asdict()`
- `deserialize_ui_state(data: dict) → UIStateModel` — constructs UIStateModel from dict, ignoring unknown keys, filling missing with defaults

### GeneratePanel (`src/playlist_gui/widgets/generate_panel.py`)

**New method:**
- `apply_ui_state(state: UIStateModel) → None` — sets every control:
  - `set_current_mode(state.mode)`
  - `_cohesion_slider.set_cohesion_mode(state.cohesion_mode)`
  - `_mode_sliders.set_genre_mode(state.genre_mode)`
  - `_mode_sliders.set_sonic_mode(state.sonic_mode)`
  - `_mode_sliders.set_pace_mode(state.pace_mode)`
  - `_length_combo` set to value matching `state.track_count`
  - `_diversity_slider` to position matching `state.diversity_gamma` / `state.artist_diversity_mode`
  - `_spacing_slider` to position matching `state.artist_spacing`
  - Recency controls: checkbox + spinners
  - Artist panel: artist queries, presence, variety, collaborations
  - Genre panel: genre_query
  - Seeds panel: seed_track_ids, seed_auto_order

**Existing `apply_saved_state()`:** removed (replaced by `apply_ui_state`)

### MainWindow (`src/playlist_gui/main_window.py`)

**Session persistence:**
- `_save_settings()`: replaces 6 individual `state/*` QSettings writes with `self._preset_manager.save_session(self._current_generation_ui_state())`
- `_restore_settings()`: replaces `_restore_generation_state(...)` call with `state = self._preset_manager.load_session()` → `self._generate_panel.apply_ui_state(state)`
- Remove `_restore_generation_state()` method and `_normalize_saved_generation_mode()` helper

**Preset menu actions:**
- `_on_save_preset()`: calls `self._generate_panel.build_ui_state()` → `self._preset_manager.save_preset(name, state)`
- `_on_load_preset()`: calls `self._preset_manager.load_preset(name)` → `self._generate_panel.apply_ui_state(state)`
- `_on_reset_overrides()`: calls `self._generate_panel.apply_ui_state(UIStateModel())` (resets to defaults)

**Dirty tracking:**
- Remove: `_preset_overrides_snapshot`, `_dirty_overrides`, `_check_override_drift()`
- Replace with: `_preset_ui_state_snapshot: Optional[UIStateModel] = None`
- Dirty detection: `self._generate_panel.build_ui_state() != self._preset_ui_state_snapshot`
- `_update_override_status()` checks equality using dataclass `__eq__`

**Removed:**
- `_pending_preset_name` logic (preset is restored immediately via UIState, no need to defer)
- `install_builtin_presets(self._preset_manager)` call

---

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Preset file missing | `load_preset()` returns None; MainWindow shows "Preset not found" |
| Corrupt YAML | `load_preset()` returns None; log warning |
| Unknown fields in file | Silently ignored during deserialization |
| Missing fields in file | Filled with UIStateModel defaults |
| `_session.json` missing | Normal first-launch; all controls start at defaults |
| `_session.json` corrupt | Log warning, start at defaults |

---

## Testing Strategy

- **Unit tests for serialization:** round-trip `UIStateModel → serialize → deserialize → UIStateModel` equals original
- **Unit tests for forward-compat:** extra keys in dict are ignored; missing keys get defaults
- **Unit tests for PresetManager:** save/load/list/delete with UIState payloads
- **Unit tests for GeneratePanel.apply_ui_state():** set non-default state, apply, verify `build_ui_state()` matches
- **Unit tests for session persistence:** save session, load session, verify round-trip
- **Integration test:** save preset, restart (simulated), load preset, verify controls match
