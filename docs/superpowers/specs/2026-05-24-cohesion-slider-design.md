# Cohesion Slider Design (Sub-Project B)

**Date:** 2026-05-24
**Branch:** `feature/track-replacement`
**Predecessor:** Sub-project A (cohesion mode refactor, completed 2026-05-24) — added `playlists.cohesion_mode` backend axis.

## Goal

Expose `playlists.cohesion_mode` as a first-class slider control in the main generate panel, alongside the existing Genre/Sonic/Pace sliders. Remove the obsolete `CohesionDial` widget and the policy backward-compat code that mapped a single "cohesion" level to (genre_mode, sonic_mode) pairs.

After this sub-project, the GUI exposes all four mode axes as independent sliders:
- **OVERALL COHESION** (new card) — drives pier-bridge beam tuning
- **MATCHING** (existing card) — Genre, Sonic, Pace sliders driving candidate pool composition

## Non-goals

- The Advanced Panel's `cohesion_mode` CHOICE dropdown (`settings_schema.py:398-409`) stays as-is. Once policy owns the key, the dropdown becomes redundant (policy always wins), but removing it requires deciding what to do with the broader Advanced Panel — that belongs in sub-project C.
- No changes to the four backend axes (`cohesion_mode` / `genre_mode` / `sonic_mode` / `pace_mode`) themselves.
- No presets / state persistence (sub-project C).

## Architecture

A new `CohesionSlider` widget is a single-axis variant of the existing `ModeSliders` pattern. It lives in its own header "card" (a control group, in `generate_panel.py` parlance) titled **OVERALL COHESION**, positioned between the "Mode" card (playlist type) and the "Matching" card (Genre/Sonic/Pace).

The value flows the same way as the other three mode axes:

```
CohesionSlider value
  → GeneratePanel.build_ui_state()
  → UIStateModel.cohesion_mode
  → policy.derive_runtime_config()
  → playlists.cohesion_mode in merged config
  → worker reads via resolve_cohesion_mode()
  → playlist_generator cohesion_mode_override
```

After this change, `policy.py` is the **single writer** for `playlists.cohesion_mode` — it joins `genre_mode`, `sonic_mode`, and `pace_mode` in `POLICY_OWNED_KEYS`.

## Components

### 1. `CohesionSlider` widget — new file

**File:** `src/playlist_gui/widgets/cohesion_slider.py`

- Type alias: `CohesionModeLevel = Literal["strict", "narrow", "dynamic", "discover"]`
- Module-level constants:
  - `COHESION_MODE_LEVELS = ["strict", "narrow", "dynamic", "discover"]` (left-to-right slider order: tightest → loosest)
  - `COHESION_MODE_LABELS = {"strict": "Strict", "narrow": "Narrow", "dynamic": "Dynamic", "discover": "Discover"}`
  - `COHESION_MODE_TOOLTIPS` keyed by level:
    - `strict` — "Ultra-cohesive transitions; tightest beam, narrowest bridges"
    - `narrow` — "Cohesive transitions; tight beam"
    - `dynamic` — "Balanced beam (default)"
    - `discover` — "Loosest transitions; widest exploration in bridging"
- Class `CohesionSlider(QWidget)`:
  - `__init__` constructs a single horizontal row: `[QSlider with 4 ticks] [value label]`
  - No row-level prefix label ("Genre:", "Sonic:", "Pace:"-style) — the card title supplies the label
  - QSlider styling matches `ModeSliders` exactly: `objectName="modeSlider"`, same min/max widths (90/130), `TicksBelow`, `tickInterval=1`, `pageStep=1`, `singleStep=1`
  - Value label uses `objectName="modeValue"`, same min width (68), `MinimumExpanding` size policy
  - Default value: `"dynamic"` (matches `config.yaml`)
  - Signal: `cohesion_mode_changed = Signal(str)`
  - Methods: `get_cohesion_mode() → CohesionModeLevel`, `set_cohesion_mode(value: CohesionModeLevel) → None`

### 2. `GeneratePanel` integration

**File:** `src/playlist_gui/widgets/generate_panel.py`

Three edits inside `_setup_ui()`:

1. Insert `"cohesion"` into `_header_group_order` (line ~95) immediately after `"mode"` and before `"matching"`:
   ```python
   self._header_group_order = [
       "mode",
       "cohesion",   # new
       "matching",
       "length",
       "freshness",
       "spacing",
       "diversity",
       "actions",
   ]
   ```

2. After the existing `_create_control_group("mode", ...)` call (around line 124), instantiate and register the cohesion card:
   ```python
   self._cohesion_slider = CohesionSlider()
   self._create_control_group("cohesion", "OVERALL COHESION", self._cohesion_slider)
   ```

3. In `build_ui_state()` (around line 560), include:
   ```python
   cohesion_mode=self._cohesion_slider.get_cohesion_mode(),
   ```

**Layout note (narrow-mode reflow):** The two-row breakpoint splits at `_header_group_order[:4]` (`generate_panel.py:375`). Before this change, the top row in narrow mode was `[mode, matching, length, freshness]`. After, it becomes `[mode, cohesion, matching, length]` — `freshness` shifts to the bottom row. Acceptable; the cohesion+matching grouping reads more naturally on a narrow display anyway.

### 3. `UIStateModel` update

**File:** `src/playlist_gui/ui_state.py`

- Add field: `cohesion_mode: CohesionModeLevel = "dynamic"`
- Remove field: `cohesion: Literal["tight", "balanced", "wide", "discover"] = "balanced"` (line 37 — obsolete)
- Import the new type from `widgets/cohesion_slider.py`

### 4. `policy.py` rewiring

**File:** `src/playlist_gui/policy.py`

- Add `"playlists.cohesion_mode"` to `POLICY_OWNED_KEYS` (line ~26)
- Add a validation set: `VALID_COHESION_MODES = {"strict", "narrow", "dynamic", "discover"}`
- In `derive_runtime_config()` (around line 260), alongside genre/sonic/pace derivation:
  ```python
  cohesion_mode = getattr(ui, "cohesion_mode", "dynamic")
  if cohesion_mode not in VALID_COHESION_MODES:
      cohesion_mode = "dynamic"
  overrides["playlists.cohesion_mode"] = cohesion_mode
  ```
- **Remove** the backward-compat block (lines ~260-265) that does `if getattr(ui, "cohesion", None) in COHESION_MAP: genre_mode, sonic_mode = COHESION_MAP[ui.cohesion]`. This was the old single-knob-to-genre+sonic mapping and is obsolete.
- **Remove** the `COHESION_MAP` import from this file
- Update the docstring at line 71 referencing "the original cohesion dial"

### 5. Cleanup of obsolete CohesionDial machinery

**Files to delete:**
- `src/playlist_gui/widgets/cohesion_dial.py` — never integrated into the GUI; the old design was for a single knob mapping to genre+sonic pairs

**Files to update (remove references):**
- `tests/unit/test_gui_policy.py`:
  - Drop the `COHESION_MAP` import (line 14)
  - Drop the entire `TestCohesionMapping` class (lines ~82-150)
  - Update integration tests that pass `cohesion="..."` to use `cohesion_mode="..."` instead (lines 44, 302-310, 321, 338, 579, 626 — verify with `grep -n 'cohesion=' tests/unit/test_gui_policy.py` before editing)
  - Drop assertions that depend on `COHESION_MAP` semantics (the "genre pool disabled for tight/balanced/wide" test at lines ~302-310 — that logic is gone)
- `tests/unit/test_generate_panel.py`:
  - Add an assertion confirming `build_ui_state()` includes `cohesion_mode`
- Docs in `docs/ui_ux/`:
  - Three markdown files reference `CohesionDial`/`COHESION_MAP` (`plan_gui_just_works.md`, `phase_notes/phase_ui_polish_plan.md`, `phase_notes/phase2_ui_wiring.md`). These are historical phase-notes / plans, not user-facing docs. Add a short "superseded by 2026-05-24 sub-project B" footnote in each; do not rewrite them.

### 6. `worker.py` — no change

Already reads `cohesion_mode` via `resolve_cohesion_mode(config.get('playlists', {}))` (worker.py line ~1138). Once policy writes `playlists.cohesion_mode`, the worker picks it up without any change.

## Data flow (end to end)

```
User moves Cohesion slider to "Strict"
  → CohesionSlider._on_changed emits cohesion_mode_changed("strict")
User clicks Generate
  → GeneratePanel.build_ui_state() captures cohesion_mode="strict"
  → MainWindow._on_generate_requested(ui_state_dict)
  → policy.derive_runtime_config(ui_state)
  → overrides["playlists.cohesion_mode"] = "strict"
  → merge_overrides(user, policy) — policy wins
  → GeneratePlaylistRequest sent to worker
  → worker reads cohesion_mode = resolve_cohesion_mode(merged_config["playlists"])  # = "strict"
  → generator.create_playlist_for_artist(..., cohesion_mode_override="strict", ...)
  → pier-bridge beam uses bridge_floor_strict=0.10, etc.
```

## Error handling

- `CohesionSlider.set_cohesion_mode()` validates against `COHESION_MODE_LEVELS` and silently clamps to `"dynamic"` if invalid (matches existing `ModeSliders` behavior)
- `policy.derive_runtime_config()` validates `ui.cohesion_mode` against `VALID_COHESION_MODES` and falls back to `"dynamic"` if invalid
- `resolve_cohesion_mode()` already handles the read side (warns on stale `ds_pipeline.mode`, validates value)
- No new failure modes introduced

## Testing

### New unit tests

**`tests/unit/test_cohesion_slider.py`:**
- `test_default_value` — `CohesionSlider().get_cohesion_mode() == "dynamic"`
- `test_set_and_get_roundtrip` — for each of 4 values: `set_cohesion_mode(v); get_cohesion_mode() == v`
- `test_invalid_value_clamps_to_dynamic` — `set_cohesion_mode("invalid")` results in `get_cohesion_mode() == "dynamic"`
- `test_signal_emitted_on_change` — connect to `cohesion_mode_changed`, verify it fires with the right string when the slider value changes (drive via `set_cohesion_mode` since `qtbot` is a no-op per CLAUDE.md)
- `test_signal_not_emitted_when_value_unchanged` — setting the same value twice fires the signal only once
- `test_slider_position_maps_to_level` — slider value 0 = strict, 3 = discover (positions follow `COHESION_MODE_LEVELS` order)

### Policy tests (extended)

In `tests/unit/test_gui_policy.py`, add a `TestCohesionModeDerivation` class:
- For each of 4 valid values, `derive_runtime_config(UIStateModel(cohesion_mode=v))` writes `overrides["playlists.cohesion_mode"] == v`
- Invalid value falls back to `"dynamic"` with no exception
- `playlists.cohesion_mode` appears in `POLICY_OWNED_KEYS` (lock the contract)
- User-supplied override `{"playlists.cohesion_mode": "discover"}` is overridden by policy's `"strict"` when ui_state requests strict (POLICY_OWNED_KEYS precedence)

### Generate panel test (extended)

In `tests/unit/test_generate_panel.py`:
- `test_build_ui_state_includes_cohesion_mode` — instantiate `GeneratePanel`, set cohesion slider to "strict" via `set_cohesion_mode`, call `build_ui_state()`, assert `cohesion_mode == "strict"`

### Removed tests

- `TestCohesionMapping` (entire class) — the tight/balanced/wide → genre+sonic mapping no longer exists
- Any test that depends on `COHESION_MAP` semantics (e.g., the "genre pool disabled for tight cohesion" assertion)
- Any test that passes `cohesion="..."` to `UIStateModel` constructor — rename to `cohesion_mode="..."` and update expected mode values

### `qtbot` reality check

Per CLAUDE.md, `tests/conftest.py:60-70` provides a `qtbot` no-op stub. Slider interaction tests (`qtbot.mouseClick(...)`) silently pass without actually clicking anything. All new tests drive state via direct `set_cohesion_mode()` calls and inspect the resulting state/signals — no `qtbot` interaction.

## Manual verification checklist

Before declaring done:
1. `python -m playlist_gui.app` launches without exceptions
2. New "OVERALL COHESION" card appears between Mode and Matching cards on the header row
3. Card contains a single 4-tick horizontal slider with value label, default position = Dynamic
4. Moving the slider through all 4 positions updates the value label correctly
5. With Cohesion = Strict, generate a 10-track Sundays playlist (`Artist` mode), log shows `Cohesion mode: strict` and `bridge_floor=0.10` in the pier-bridge tuning line
6. Set Cohesion back to Dynamic, regenerate, log shows `Cohesion mode: dynamic` and `bridge_floor=0.02`
7. Narrow the window below the header reflow breakpoint — the top row should show `[Mode, OVERALL COHESION, Matching, Length]` and the bottom row `[Freshness, Spacing, Diversity, Actions]`
8. The Advanced Panel still has its Pipeline → Cohesion mode dropdown; changing it has no effect on a generated playlist (policy wins) — confirm and accept this as known sub-project C scope

## File summary

| File | Change |
|---|---|
| `src/playlist_gui/widgets/cohesion_slider.py` | NEW: widget |
| `src/playlist_gui/widgets/generate_panel.py` | Add header card + build_ui_state |
| `src/playlist_gui/ui_state.py` | Add `cohesion_mode`, remove `cohesion` |
| `src/playlist_gui/policy.py` | Add policy ownership, remove COHESION_MAP fallback |
| `src/playlist_gui/widgets/cohesion_dial.py` | DELETE |
| `tests/unit/test_cohesion_slider.py` | NEW |
| `tests/unit/test_gui_policy.py` | Remove `TestCohesionMapping`, add `TestCohesionModeDerivation`, rename `cohesion=` → `cohesion_mode=` |
| `tests/unit/test_generate_panel.py` | Add `cohesion_mode` assertion |
| `docs/ui_ux/*.md` | Add "superseded" footnote (3 files) |

No backend changes. No `worker.py` changes. No `settings_schema.py` changes.
