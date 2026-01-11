# GUI "Just Works" Implementation Plan

**Version:** 1.0
**Date:** 2026-01-10
**Goal:** Redesign the GUI to work out-of-the-box with minimal controls and a policy layer that derives runtime generator settings from UI state.

---

## 1. Current State Audit

### 1.1 UI Components and Files/Modules

| File | Size | Purpose |
|------|------|---------|
| `src/playlist_gui/main_window.py` | 73KB | Main window; ~1600 lines; layout, signals, state |
| `src/playlist_gui/worker.py` | 44KB | Worker subprocess; command execution |
| `src/playlist_gui/worker_client.py` | 24KB | QProcess wrapper; NDJSON protocol |
| `src/playlist_gui/widgets/advanced_panel.py` | 36KB | Schema-driven settings UI |
| `src/playlist_gui/widgets/seed_tracks_input.py` | 5KB | Multi-row track input |
| `src/playlist_gui/widgets/track_table.py` | 26KB | Results table with sorting/filtering |
| `src/playlist_gui/widgets/log_panel.py` | 8KB | Log display with level filters |
| `src/playlist_gui/config/config_model.py` | 17KB | Config load/merge/override |
| `src/playlist_gui/config/settings_schema.py` | 26KB | Schema for all settings |

**Current Window Layout:**
```
┌─────────────────────────────────────────────────────────────────┐
│ Mode: [Artist ▼] | Artist: [___________] | [Generate] [Cancel] │
│ Track Seeds: [___________] [Add] [Remove]                       │
│ Genre: [___________] (visible in Genre mode)                    │
├─────────────────────────────────────────────────────────────────┤
│ Similarity Tuning:                                              │
│   Genre: ○ Strict ○ Narrow ● Dynamic ○ Discover ○ Off          │
│   Sonic: ○ Strict ○ Narrow ● Dynamic ○ Discover ○ Off          │
├─────────────────────────────────────────────────────────────────┤
│ [===========  70%  ===========] Stage: Filtering candidates...  │
├─────────────────────────────────────────────────────────────────┤
│                     TRACK TABLE                                  │
│  # | Artist | Title | Album | Duration | Sonic | Genre | Path  │
├─────────────────────────────────────────────────────────────────┤
│ [Export M3U8] [Export to Plex]                                  │
└─────────────────────────────────────────────────────────────────┘
┌─────────────┐  ┌──────────────────┐  ┌─────────────┐
│ RIGHT DOCK  │  │   BOTTOM DOCK    │  │ (Optional)  │
│ Advanced    │  │   Log Panel      │  │ Jobs Panel  │
│ Settings    │  │                  │  │             │
└─────────────┘  └──────────────────┘  └─────────────┘
```

### 1.2 Current Mode Selection

**Location:** `main_window.py:221-224`

```python
self._mode_combo = QComboBox()
self._mode_combo.addItems(["Artist", "Genre", "History"])  # Artist default
```

**Mode Handling:** `_on_generate()` at line 986:
```python
mode = "artist" if mode_text == "Artist" else "genre" if mode_text == "Genre" else "history"
```

**Current Modes:**
- **Artist**: Artist input + optional seed tracks input
- **Genre**: Genre input with similarity autocomplete
- **History**: No input required (Last.fm history)

**What's Missing:**
- No dedicated "Seed(s)" mode for multi-seed DJ bridging
- Seed tracks in Artist mode are optional filters, not DJ routing anchors

### 1.3 Current Config Wiring

**Flow:**
```
main_window._on_generate()
  └─ self._config_model.get_overrides()     # Dict of user-changed values
  └─ Add genre_mode, sonic_mode overrides
  └─ worker_client.generate_playlist(config_path, overrides, mode, artist, ...)
      └─ Worker receives JSON command
          └─ load_and_merge_config(config_path, overrides)
          └─ Passes merged config to PlaylistGenerator
```

**Key Override Paths Used:**
- `playlists.genre_mode` → strict/narrow/dynamic/discover/off
- `playlists.sonic_mode` → strict/narrow/dynamic/discover/off
- `playlists.tracks_per_playlist` → 30 (from Advanced Panel)
- `playlists.recently_played_filter.enabled` → true/false
- `playlists.recently_played_filter.lookback_days` → 14/30/etc

### 1.4 Current Generation Trigger and Output Handling

**Trigger:** `_generate_btn.clicked.connect(self._on_generate)` at line 256

**Validation (lines 1001-1007):**
```python
if mode == "artist" and not artist and not seed_tracks:
    QMessageBox.warning(self, "Missing Artist", "Please enter an artist name.")
if mode == "genre" and not genre:
    QMessageBox.warning(self, "Missing Genre", "Please enter a genre name.")
```

**Result Handling:** `_on_worker_result()` at line 1345:
- Extracts tracks from result
- Calls `_track_table.set_tracks(tracks, playlist_name=name)`
- Enables export buttons

**Error Handling:** `_on_worker_error()` at line 1387:
- Logs error to log panel
- No popup dialogs for generation failures
- No relaxation notifications

### 1.5 Current Feature Toggle Locations

#### DJ Bridging
- **Config Key:** `playlists.ds_pipeline.pier_bridge.dj_bridging.enabled`
- **Data Class:** `PierBridgeConfig.dj_bridging_enabled` (pier_bridge_builder.py:132)
- **Parser:** `pipeline.py:887-889`
- **Current Default:** `false`
- **Touch Points:** 20+ locations in pier_bridge_builder.py

#### Genre Pool
- **Config Key:** `playlists.ds_pipeline.pier_bridge.dj_bridging.pooling.strategy`
- **Values:** `"baseline"` | `"dj_union"`
- **Genre Pool Size:** `pooling.k_genre` (default: 80 when enabled)
- **Parser:** `pipeline.py:967-980`
- **How Enabled:** `strategy == "dj_union"` AND `k_genre > 0`

#### Recency Enforcement
- **Config Keys:**
  - `playlists.recently_played_filter.enabled` (boolean)
  - `playlists.recently_played_filter.lookback_days` (int)
  - `playlists.recently_played_filter.min_playcount_threshold` (int, default 0)
- **Data Class:** `FilterConfig` (filtering.py:32-39)
- **Applied:** Pre-order only in candidate pool construction (pipeline.py:195-204)

#### Artist Spacing
- **Config Key:** `playlists.ds_pipeline.constraints.min_gap`
- **Mode Defaults:** narrow=3, dynamic=6, discover=9
- **Enforced:** pier_bridge_builder.py lines 3365-3400, `_enforce_min_gap_global()`

#### Seed Ordering
- **Config Key:** `playlists.ds_pipeline.pier_bridge.dj_bridging.seed_ordering`
- **Values:** `"auto"` (preserve order) | `"fixed"` (optimize)
- **Applied:** pier_bridge_builder.py lines 3720-3741

#### Failure Handling
- **Config Keys:**
  - `pier_bridge.audit_run.enabled` → write markdown reports
  - `pier_bridge.infeasible_handling.enabled` → backoff retries
  - `pier_bridge.infeasible_handling.strategy` → "backoff"
  - `pier_bridge.infeasible_handling.backoff_steps` → [0.08, 0.06, ..., 0.00]
- **Result:** `PierBridgeResult.failure_reason` (pier_bridge_builder.py:216)
- **Current Behavior:** Failures logged but no UI dialogs

---

## 2. Gap Analysis

### 2.1 What Prevents the Target UX Today

| Gap | Description | Impact |
|-----|-------------|--------|
| **No Seed(s) Mode** | Current modes are Artist/Genre/History; seed tracks are sub-feature of Artist mode | Cannot independently use DJ bridging with multi-seed routing |
| **No Cohesion Dial** | Uses 5 radio buttons for genre mode + 5 for sonic mode (10 controls) | Too granular; users must understand internal mechanics |
| **No Policy Layer** | Config toggles set directly; no automated derivation from UI state | DJ bridging/genre pool require manual config tweaking |
| **No Relaxation Popups** | Failures logged silently; no user notification of constraint loosening | Users don't know why a playlist succeeded with different parameters |
| **No Failure Dialogs** | Generation failures show in log panel only | Users may not notice failure; no actionable suggestions |
| **Recency Always On/Off** | Single toggle; no days/plays threshold UI | Cannot fine-tune recency exclusion |
| **No Artist Spacing Control** | Fixed per mode (narrow=3, dynamic=6) | Users cannot adjust artist repeat frequency |
| **No Seed Reordering UI** | When `seed_ordering=fixed`, no visual feedback | Users don't see how seeds were reordered |
| **Export on Every Generate** | Not decoupled | Cannot regenerate without exporting |

### 2.2 What Must Change

**UI Changes:**
1. Replace mode selector: `["Artist", "Genre", "History"]` → `["Artist(s)", "History", "Seed(s)"]`
2. Replace genre/sonic radio buttons with single "Cohesion" dial (4 notches)
3. Add length dropdown (20/30/40/50)
4. Add recency controls (ON/OFF + Days + Plays threshold)
5. Add artist spacing dropdown (Normal/Strong)
6. Create dedicated Seed(s) panel with drag-reorder and auto-order toggle
7. Add mode-specific Artist panel (presence, variety sliders)
8. Add relaxation popup component
9. Add failure dialog component

**Backend Wiring Changes:**
1. Create `UIStateModel` dataclass capturing all UI state
2. Create `PolicyLayer` module that derives runtime config from UI state
3. Modify `_on_generate()` to use policy-derived overrides
4. Add relaxation event type to worker protocol
5. Surface failure reasons from worker as structured events

---

## 3. Implementation Plan

### Phase 1: UIStateModel and PolicyLayer Foundation

**Files to Create:**
- `src/playlist_gui/ui_state.py` — UIStateModel dataclass
- `src/playlist_gui/policy.py` — PolicyLayer derivation logic

**UIStateModel Fields:**
```python
@dataclass
class UIStateModel:
    # Top-level mode
    mode: Literal["artist", "history", "seeds"] = "artist"

    # Cohesion (replaces genre_mode + sonic_mode)
    cohesion: Literal["tight", "balanced", "wide", "discover"] = "balanced"

    # Length
    track_count: int = 30

    # Recency
    recency_enabled: bool = True
    recency_days: int = 14
    recency_plays_threshold: int = 1

    # Artist spacing
    artist_spacing: Literal["normal", "strong"] = "normal"

    # Artist(s) mode specific
    artist_query: str = ""
    artist_presence: Literal["low", "medium", "high", "max"] = "medium"
    artist_variety: Literal["focused", "balanced", "sprawling"] = "balanced"

    # History mode specific
    history_window_days: int = 30

    # Seed(s) mode specific
    seed_tracks: List[str] = field(default_factory=list)
    seed_auto_order: bool = True
```

**PolicyLayer API:**
```python
@dataclass
class PolicyDecisions:
    # Derived config overrides
    overrides: Dict[str, Any]

    # Decisions for logging/display
    dj_bridging_enabled: bool
    genre_pool_enabled: bool
    seed_ordering: str
    min_artist_gap: int
    bridge_floor: float

    # Relaxation limits
    can_relax_pool: bool
    can_relax_bridge_floor: bool


def derive_runtime_config(ui_state: UIStateModel) -> PolicyDecisions:
    """
    Derive all runtime configuration from UI state.

    Rules:
    - DJ bridging ON only when mode == "seeds" AND len(seed_tracks) >= 2
      AND unique_seed_artists >= 2
    - Genre pool ON only when cohesion == "discover"
    - Discover relaxations: pool expansion + bridge floor only
    - Never relax recency or artist spacing
    """
```

**Commit:** "feat(gui): Add UIStateModel and PolicyLayer foundation"

---

### Phase 2: Simplified Top Controls

**Files to Modify:**
- `src/playlist_gui/main_window.py` — Replace controls
- `src/playlist_gui/widgets/cohesion_dial.py` — New widget (optional, could use QSlider)

**Changes:**

1. **Replace Mode Combo:**
```python
# Old: ["Artist", "Genre", "History"]
# New: ["Artist(s)", "History", "Seed(s)"]
self._mode_combo.addItems(["Artist(s)", "History", "Seed(s)"])
```

2. **Replace Genre/Sonic Radio Buttons with Cohesion Dial:**
```python
# New: Single QSlider with 4 positions
self._cohesion_slider = QSlider(Qt.Horizontal)
self._cohesion_slider.setMinimum(0)
self._cohesion_slider.setMaximum(3)
self._cohesion_slider.setTickPosition(QSlider.TicksBelow)
self._cohesion_slider.setTickInterval(1)

# Labels: Tight (0) | Balanced (1) | Wide (2) | Discover (3)
self._cohesion_labels = ["Tight", "Balanced", "Wide", "Discover"]
```

3. **Add Length Dropdown:**
```python
self._length_combo = QComboBox()
self._length_combo.addItems(["20", "30", "40", "50"])
self._length_combo.setCurrentText("30")
```

4. **Add Recency Controls:**
```python
self._recency_checkbox = QCheckBox("Recency Filter")
self._recency_days_spin = QSpinBox()
self._recency_days_spin.setRange(1, 365)
self._recency_days_spin.setValue(14)
self._recency_plays_spin = QSpinBox()
self._recency_plays_spin.setRange(1, 10)
self._recency_plays_spin.setValue(1)
# Label: "Exclude tracks played >= N times in last D days"
```

5. **Add Artist Spacing Dropdown:**
```python
self._spacing_combo = QComboBox()
self._spacing_combo.addItems(["Normal", "Strong"])
# Normal → min_gap=6, Strong → min_gap=9
```

**Commit:** "feat(gui): Replace granular controls with simplified top bar"

---

### Phase 3: Mode-Specific Panels

**Files to Modify/Create:**
- `src/playlist_gui/main_window.py` — Panel switching
- `src/playlist_gui/widgets/artist_panel.py` — New: Artist(s) mode controls
- `src/playlist_gui/widgets/seeds_panel.py` — New: Seed(s) mode controls

**Artist(s) Panel:**
```python
class ArtistPanel(QWidget):
    """Controls specific to Artist(s) mode."""

    def __init__(self, parent=None):
        # Artist autocomplete (existing _artist_edit)
        # Multi-artist note (future feature callout)
        # Presence slider: Low (10%) / Medium (25%) / High (40%) / Max (60%)
        # Variety slider: Focused <──────> Sprawling
```

**Seed(s) Panel:**
```python
class SeedsPanel(QWidget):
    """Controls specific to Seed(s) mode."""

    def __init__(self, parent=None):
        # Seed list with drag handles (QListWidget with drag-drop)
        # Add track button with autocomplete popup
        # Auto-order toggle checkbox
        # When auto-order ON and seeds reordered: show visual feedback
```

**History Panel:**
```python
class HistoryPanel(QWidget):
    """Controls specific to History mode."""

    def __init__(self, parent=None):
        # Time window dropdown: 7/14/30/90 days
        # (Future: source selector if multiple sources supported)
```

**Panel Switching Logic:**
```python
def _on_mode_changed(self, mode_text: str):
    self._artist_panel.setVisible(mode_text == "Artist(s)")
    self._history_panel.setVisible(mode_text == "History")
    self._seeds_panel.setVisible(mode_text == "Seed(s)")
```

**Commit:** "feat(gui): Add mode-specific panels for Artist(s), History, Seed(s)"

---

### Phase 4: Policy Integration in Generate Flow

**Files to Modify:**
- `src/playlist_gui/main_window.py` — Use PolicyLayer in `_on_generate()`
- `src/playlist_gui/worker.py` — Accept policy-derived overrides

**New Generate Flow:**
```python
def _on_generate(self):
    # 1. Build UI state model
    ui_state = self._build_ui_state()

    # 2. Derive runtime config via policy layer
    policy = derive_runtime_config(ui_state)

    # 3. Log decisions
    logger.info(f"Policy: DJ bridging={policy.dj_bridging_enabled}, "
                f"Genre pool={policy.genre_pool_enabled}")

    # 4. Pass overrides to worker
    self._worker_client.generate_playlist(
        config_path=self._config_path,
        overrides=policy.overrides,
        mode=ui_state.mode,
        artist=ui_state.artist_query if ui_state.mode == "artist" else None,
        seed_tracks=ui_state.seed_tracks if ui_state.mode == "seeds" else None,
        tracks=ui_state.track_count,
    )
```

**Policy Override Mappings:**
```python
# Cohesion → genre_mode + sonic_mode
COHESION_MAP = {
    "tight": ("strict", "strict"),
    "balanced": ("dynamic", "dynamic"),
    "wide": ("discover", "narrow"),
    "discover": ("discover", "discover"),
}

# Artist spacing → min_gap
SPACING_MAP = {
    "normal": 6,
    "strong": 9,
}

# DJ bridging rule
def should_enable_dj_bridging(ui_state):
    if ui_state.mode != "seeds":
        return False
    if len(ui_state.seed_tracks) < 2:
        return False
    # Check unique artists
    unique_artists = len(set(extract_artist(t) for t in ui_state.seed_tracks))
    return unique_artists >= 2

# Genre pool rule
def should_enable_genre_pool(ui_state):
    return ui_state.cohesion == "discover"
```

**Commit:** "feat(gui): Integrate PolicyLayer into generate flow"

---

### Phase 5: Relaxation Popups

**Files to Create:**
- `src/playlist_gui/widgets/relaxation_popup.py` — Toast-style popup

**Worker Protocol Extension:**
Add new event type:
```json
{"type": "relaxation", "request_id": "<uuid>", "parameter": "bridge_floor", "old_value": 0.08, "new_value": 0.04}
```

**Popup Component:**
```python
class RelaxationPopup(QFrame):
    """Non-blocking toast showing constraint relaxation."""

    def __init__(self, parent=None):
        # Appears in bottom-right corner
        # Auto-dismisses after 5 seconds
        # Shows: "Relaxed bridge_floor: 0.08 → 0.04"

    def show_relaxation(self, parameter: str, old_val: Any, new_val: Any):
        self._label.setText(f"Relaxed {parameter}: {old_val} → {new_val}")
        self.show()
        QTimer.singleShot(5000, self.hide)
```

**Worker Emission Point:**
In `pier_bridge_builder.py` during backoff:
```python
if bridge_floor != original_bridge_floor:
    emit_relaxation("bridge_floor", original_bridge_floor, bridge_floor)
```

**Commit:** "feat(gui): Add relaxation popup for constraint loosening"

---

### Phase 6: Failure Dialogs

**Files to Create:**
- `src/playlist_gui/widgets/failure_dialog.py` — Modal dialog

**Dialog Content:**
```python
class FailureDialog(QDialog):
    """Modal dialog when playlist generation fails."""

    def __init__(self, failure_reason: str, suggestions: List[str], parent=None):
        # Title: "Playlist Generation Failed"
        # Body: failure_reason
        # Suggestions list:
        #   - "Reduce track count (currently 50)"
        #   - "Disable or adjust recency filter"
        #   - "Move cohesion toward Discover"
        #   - "Try different seeds"
        # Buttons: [Try Again] [Cancel]
```

**Failure Reason Parsing:**
```python
FAILURE_SUGGESTIONS = {
    "pool_exhausted": [
        "Reduce track count",
        "Move cohesion toward Discover",
        "Disable recency filter",
    ],
    "infeasible_segment": [
        "Move cohesion toward Discover",
        "Try seeds from closer genres",
    ],
    "no_candidates": [
        "Disable recency filter",
        "Increase recency window",
    ],
}
```

**Integration in `_on_worker_done()`:**
```python
def _on_worker_done(self, cmd, ok, detail, cancelled, job_id, summary):
    if not ok and not cancelled:
        if "failed" in detail.lower():
            dialog = FailureDialog(
                failure_reason=detail,
                suggestions=self._parse_failure_suggestions(detail),
                parent=self
            )
            dialog.exec()
```

**Commit:** "feat(gui): Add failure dialog with actionable suggestions"

---

### Phase 7: Log Panel Integration

**Files to Modify:**
- `src/playlist_gui/widgets/log_panel.py` — Add minimal messages
- `src/playlist_gui/main_window.py` — Connect relaxation/failure signals

**Minimal Messages:**
```python
# On relaxation (in addition to popup):
logger.info(f"[RELAXATION] {parameter}: {old_val} → {new_val}")

# On success:
logger.info(f"[SUCCESS] Generated {track_count} tracks in {elapsed:.1f}s")

# On failure:
logger.error(f"[FAILED] {failure_reason}")
```

**Commit:** "feat(gui): Integrate relaxation/failure messages into log panel"

---

### Phase 8: Advanced Panel Cleanup

**Files to Modify:**
- `src/playlist_gui/config/settings_schema.py` — Mark items as "advanced only"
- `src/playlist_gui/widgets/advanced_panel.py` — Filter visible items

**Strategy:**
- Keep Advanced Panel for power users
- Hide items now controlled by simplified UI:
  - `genre_mode` / `sonic_mode` → controlled by Cohesion
  - `tracks_per_playlist` → controlled by Length dropdown
  - `recently_played_filter.*` → controlled by Recency controls
  - `constraints.min_gap` → controlled by Artist Spacing
- Show remaining items as "Advanced Settings"

**Commit:** "refactor(gui): Hide policy-controlled settings from Advanced Panel"

---

### Phase 9: Export Decoupling

**Files to Modify:**
- `src/playlist_gui/main_window.py` — Add regenerate button

**Changes:**
1. Add "Regenerate" button (same seeds, new generation, no export)
2. Rename "Generate" to just run generation (no auto-export)
3. Keep export buttons separate and explicit

**Button Layout:**
```
[Generate] [Regenerate] | [Export M3U8] [Export to Plex]
```

**Regenerate Logic:**
```python
def _on_regenerate(self):
    # Use same UI state but trigger new generation
    # Does NOT auto-export
    self._on_generate(export=False)
```

**Commit:** "feat(gui): Decouple regeneration from export"

---

## 4. Proposed File Structure

```
src/playlist_gui/
├── main_window.py           # Modified: simplified controls, panel switching
├── ui_state.py              # NEW: UIStateModel dataclass
├── policy.py                # NEW: PolicyLayer derivation
├── worker.py                # Modified: accept relaxation events
├── worker_client.py         # Modified: handle relaxation events
├── widgets/
│   ├── artist_panel.py      # NEW: Artist(s) mode controls
│   ├── history_panel.py     # NEW: History mode controls
│   ├── seeds_panel.py       # NEW: Seed(s) mode controls
│   ├── cohesion_dial.py     # NEW (optional): Custom dial widget
│   ├── relaxation_popup.py  # NEW: Toast-style relaxation notification
│   ├── failure_dialog.py    # NEW: Modal failure dialog
│   ├── advanced_panel.py    # Modified: filter hidden items
│   ├── seed_tracks_input.py # Modified or replaced by seeds_panel.py
│   └── ...
├── config/
│   ├── settings_schema.py   # Modified: mark items as advanced-only
│   └── ...
```

---

## 5. Acceptance Checklist

### Mode Gating
- [ ] DJ bridging ON only when Mode == Seed(s) AND seed_count >= 2 AND unique_seed_artists >= 2
- [ ] DJ bridging OFF in Artist(s) mode (even with seed tracks)
- [ ] Genre pool ON only when Cohesion == Discover
- [ ] Genre pool OFF for Tight/Balanced/Wide

### Relaxation Rules
- [ ] Discover allows: candidate pool expansion
- [ ] Discover allows: bridge floor relaxation
- [ ] Never relax recency filter
- [ ] Never relax artist spacing (min_gap)
- [ ] Relaxation shows popup with old → new values

### Failure Handling
- [ ] Never fail silently
- [ ] Failure shows blocking dialog
- [ ] Dialog shows failure reason
- [ ] Dialog shows actionable suggestions
- [ ] Suggestions based on failure type

### Export Behavior
- [ ] Generate does NOT auto-export
- [ ] Regenerate does NOT auto-export
- [ ] Export only when export buttons pressed

### UI Simplification
- [ ] Single Cohesion dial (4 notches) replaces 10 radio buttons
- [ ] Length dropdown visible in top controls
- [ ] Recency controls visible (ON/OFF + Days + Plays)
- [ ] Artist spacing dropdown visible
- [ ] Mode-specific panels appear/hide correctly

### Seed(s) Mode Specific
- [ ] Seed list supports drag-to-reorder
- [ ] Auto-order toggle visible
- [ ] When auto-order ON, seeds visually reorder after optimization
- [ ] Multi-artist note hidden (or shows "coming soon") in Artist(s) mode

### Logs Integration
- [ ] Relaxation events appear in log panel
- [ ] Success message appears in log panel
- [ ] Failure message appears in log panel

---

## 6. Recommended Approach

**Justification:**

Given the existing architecture:
1. **UIStateModel as single source of truth** — Centralizes all UI state, making it easy to serialize/restore and pass to policy layer.

2. **PolicyLayer as pure function** — Takes UIStateModel, returns PolicyDecisions. No side effects. Easy to test and reason about.

3. **Minimal worker changes** — Worker already accepts overrides; policy layer just populates them. Only change: add relaxation event emission.

4. **Progressive enhancement** — Each phase can be reviewed and merged independently. UI remains functional between phases.

5. **Keep Advanced Panel** — Power users can still access raw settings. Policy layer provides sensible defaults; advanced overrides still work.

**Alternative Considered:**

Direct config binding (each UI control directly sets one override) was considered but rejected because:
- Creates tight coupling between UI and config schema
- Policy rules (e.g., "DJ bridging requires 2+ seeds from 2+ artists") would be scattered
- No central place to document or test policy logic

---

**Document Ready for Review**

Path: `docs/ui_ux/plan_gui_just_works.md`
