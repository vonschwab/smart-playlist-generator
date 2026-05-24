# Cohesion Slider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an "OVERALL COHESION" slider card to the main generate panel that drives `playlists.cohesion_mode`, alongside the existing Genre/Sonic/Pace sliders, and remove the obsolete `CohesionDial` machinery.

**Architecture:** New `CohesionSlider` widget follows the existing `ModeSliders` pattern (single-axis variant). Wired through `UIStateModel.cohesion_mode → policy.derive_runtime_config() → playlists.cohesion_mode override`, becoming the **single writer** for that key in `POLICY_OWNED_KEYS`. The old `CohesionDial`/`COHESION_MAP` mapping (cohesion-level → genre+sonic pairs) is deleted because cohesion is now an independent backend axis (per sub-project A).

**Tech Stack:** Python 3.11+, PySide6, pytest. No backend changes.

**Spec:** `docs/superpowers/specs/2026-05-24-cohesion-slider-design.md`

---

## File Structure

**New files:**
- `src/playlist_gui/widgets/cohesion_slider.py` — single-slider widget with 4 positions
- `tests/unit/test_cohesion_slider.py` — widget unit tests

**Substantively modified:**
- `src/playlist_gui/widgets/generate_panel.py` — instantiate widget + card + extend `build_ui_state`
- `src/playlist_gui/ui_state.py` — add `cohesion_mode` field, drop obsolete `cohesion` field
- `src/playlist_gui/policy.py` — add `playlists.cohesion_mode` to `POLICY_OWNED_KEYS`, derive from `ui_state.cohesion_mode`, remove `COHESION_MAP` fallback block

**Test updates:**
- `tests/unit/test_gui_policy.py` — drop `COHESION_MAP` import + `TestCohesionMapping` class, rename `cohesion=` → `cohesion_mode=` in fixtures, add `TestCohesionModeDerivation`
- `tests/unit/test_generate_panel.py` — assert `cohesion_mode` appears in `build_ui_state()`

**Deletions:**
- `src/playlist_gui/widgets/cohesion_dial.py` — never integrated; obsolete

**Doc footnotes (one line each):**
- `docs/ui_ux/plan_gui_just_works.md`
- `docs/ui_ux/phase_notes/phase_ui_polish_plan.md`
- `docs/ui_ux/phase_notes/phase2_ui_wiring.md`

---

## Task 1: Create `CohesionSlider` widget with tests (TDD)

The widget is a single-slider variant of `ModeSliders`. Independent of all other tasks.

**Files:**
- Create: `src/playlist_gui/widgets/cohesion_slider.py`
- Create: `tests/unit/test_cohesion_slider.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_cohesion_slider.py`:

```python
"""Tests for CohesionSlider widget."""
from __future__ import annotations

import pytest
from PySide6.QtWidgets import QApplication

from src.playlist_gui.widgets.cohesion_slider import (
    COHESION_MODE_LEVELS,
    CohesionSlider,
)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


class TestCohesionSlider:
    def test_default_value_is_dynamic(self, qapp):
        slider = CohesionSlider()
        assert slider.get_cohesion_mode() == "dynamic"

    @pytest.mark.parametrize("mode", ["strict", "narrow", "dynamic", "discover"])
    def test_set_and_get_roundtrip(self, qapp, mode):
        slider = CohesionSlider()
        slider.set_cohesion_mode(mode)
        assert slider.get_cohesion_mode() == mode

    def test_invalid_value_is_ignored(self, qapp):
        slider = CohesionSlider()
        slider.set_cohesion_mode("dynamic")
        slider.set_cohesion_mode("not_a_mode")  # should silently ignore
        assert slider.get_cohesion_mode() == "dynamic"

    def test_signal_emitted_on_change(self, qapp):
        slider = CohesionSlider()
        received: list[str] = []
        slider.cohesion_mode_changed.connect(received.append)
        slider.set_cohesion_mode("strict")
        assert received == ["strict"]

    def test_signal_not_emitted_when_value_unchanged(self, qapp):
        slider = CohesionSlider()
        received: list[str] = []
        slider.cohesion_mode_changed.connect(received.append)
        slider.set_cohesion_mode("dynamic")  # already dynamic by default
        assert received == []

    def test_levels_ordered_strict_to_discover(self):
        # Slider position 0 = strict (leftmost, tightest)
        # Slider position 3 = discover (rightmost, loosest)
        assert COHESION_MODE_LEVELS == ["strict", "narrow", "dynamic", "discover"]
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/unit/test_cohesion_slider.py -v
```

Expected: ImportError — `cohesion_slider` module does not exist.

- [ ] **Step 3: Create the widget file**

Create `src/playlist_gui/widgets/cohesion_slider.py`:

```python
"""
Cohesion Slider — single-axis control for playlists.cohesion_mode.

Mirrors the per-row layout of ModeSliders but as a standalone single-slider
widget for use in its own header card ("OVERALL COHESION"). Drives the
pier-bridge beam tuning axis independently of Genre/Sonic/Pace.
"""
from __future__ import annotations

from typing import Literal

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSlider,
    QWidget,
)

CohesionModeLevel = Literal["strict", "narrow", "dynamic", "discover"]

COHESION_MODE_LEVELS: list[CohesionModeLevel] = [
    "strict",
    "narrow",
    "dynamic",
    "discover",
]
COHESION_MODE_LABELS = {
    "strict": "Strict",
    "narrow": "Narrow",
    "dynamic": "Dynamic",
    "discover": "Discover",
}
COHESION_MODE_TOOLTIPS = {
    "strict": "Ultra-cohesive transitions; tightest beam, narrowest bridges",
    "narrow": "Cohesive transitions; tight beam",
    "dynamic": "Balanced beam (default)",
    "discover": "Loosest transitions; widest exploration in bridging",
}


class CohesionSlider(QWidget):
    """
    Single-slider widget for selecting overall cohesion mode.

    Emits:
        cohesion_mode_changed: New cohesion mode value
    """

    cohesion_mode_changed = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._value: CohesionModeLevel = "dynamic"
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        self._setup_ui()

    def _setup_ui(self) -> None:
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(len(COHESION_MODE_LEVELS) - 1)
        self._slider.setValue(COHESION_MODE_LEVELS.index(self._value))
        self._slider.setTickPosition(QSlider.TicksBelow)
        self._slider.setTickInterval(1)
        self._slider.setPageStep(1)
        self._slider.setSingleStep(1)
        self._slider.setMinimumWidth(90)
        self._slider.setMaximumWidth(130)
        self._slider.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self._slider.setObjectName("modeSlider")
        self._slider.setToolTip(
            "Overall cohesion (pier-bridge beam tightness)"
        )
        self._slider.valueChanged.connect(self._on_changed)
        row.addWidget(self._slider)

        self._value_label = QLabel(COHESION_MODE_LABELS[self._value])
        self._value_label.setObjectName("modeValue")
        self._value_label.setMinimumWidth(68)
        self._value_label.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        self._value_label.setToolTip(COHESION_MODE_TOOLTIPS[self._value])
        row.addWidget(self._value_label)

    def _on_changed(self, slider_value: int) -> None:
        if 0 <= slider_value < len(COHESION_MODE_LEVELS):
            new_value = COHESION_MODE_LEVELS[slider_value]
            if new_value != self._value:
                self._value = new_value
                self._value_label.setText(COHESION_MODE_LABELS[new_value])
                self._value_label.setToolTip(COHESION_MODE_TOOLTIPS[new_value])
                self.cohesion_mode_changed.emit(new_value)

    def get_cohesion_mode(self) -> CohesionModeLevel:
        return self._value

    def set_cohesion_mode(self, mode: CohesionModeLevel) -> None:
        if mode in COHESION_MODE_LEVELS:
            self._slider.setValue(COHESION_MODE_LEVELS.index(mode))

    def reset(self) -> None:
        self.set_cohesion_mode("dynamic")
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/unit/test_cohesion_slider.py -v
```

Expected: 9 tests pass (1 default + 4 parametrized roundtrip + 1 invalid + 1 signal emit + 1 signal not emit + 1 ordering).

- [ ] **Step 5: Commit**

```
git add src/playlist_gui/widgets/cohesion_slider.py tests/unit/test_cohesion_slider.py
git commit -m "feat(gui): add CohesionSlider widget

Single-slider variant of ModeSliders with 4 positions (Strict/Narrow/
Dynamic/Discover), default Dynamic. Designed to live in its own header
card. Part of sub-project B."
```

---

## Task 2: Add `cohesion_mode` field to `UIStateModel`

Add the new field. Keep the legacy `cohesion` field for now — Task 7 removes it after the policy + tests are migrated.

**Files:**
- Modify: `src/playlist_gui/ui_state.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_cohesion_slider.py`:

```python
class TestUIStateModelCohesionMode:
    def test_ui_state_has_cohesion_mode_field_default_dynamic(self):
        from src.playlist_gui.ui_state import UIStateModel

        state = UIStateModel()
        assert state.cohesion_mode == "dynamic"

    def test_ui_state_cohesion_mode_settable(self):
        from src.playlist_gui.ui_state import UIStateModel

        state = UIStateModel(cohesion_mode="strict")
        assert state.cohesion_mode == "strict"
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/unit/test_cohesion_slider.py::TestUIStateModelCohesionMode -v
```

Expected: AttributeError or TypeError — `cohesion_mode` is not a field on `UIStateModel`.

- [ ] **Step 3: Add the field**

In `src/playlist_gui/ui_state.py`, find the existing block defining mode fields (around line 37):

```python
    cohesion: Literal["tight", "balanced", "wide", "discover"] = "balanced"
    """
    Backward-compatible aggregate control for genre_mode and sonic_mode.
    Newer UI paths may set genre_mode/sonic_mode directly.
    """

    genre_mode: ModeValue = "narrow"
    sonic_mode: ModeValue = "narrow"
    pace_mode: PaceModeValue = "dynamic"
```

Add a new field directly above `genre_mode` (and leave the `cohesion` field in place for now — Task 7 removes it):

```python
    cohesion: Literal["tight", "balanced", "wide", "discover"] = "balanced"
    """
    Backward-compatible aggregate control for genre_mode and sonic_mode.
    Newer UI paths may set genre_mode/sonic_mode directly.
    """

    cohesion_mode: Literal["strict", "narrow", "dynamic", "discover"] = "dynamic"
    """
    Overall cohesion (pier-bridge beam tightness). Independent of
    genre_mode/sonic_mode/pace_mode (which control pool composition).
    Single writer: policy.derive_runtime_config() → playlists.cohesion_mode.
    """

    genre_mode: ModeValue = "narrow"
    sonic_mode: ModeValue = "narrow"
    pace_mode: PaceModeValue = "dynamic"
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/unit/test_cohesion_slider.py::TestUIStateModelCohesionMode -v
```

Expected: 2 tests pass.

Run the broader suite to ensure no regressions:
```
pytest tests/unit/test_gui_policy.py tests/unit/test_generate_panel.py -q
```

Expected: all pass (we haven't removed anything yet).

- [ ] **Step 5: Commit**

```
git add src/playlist_gui/ui_state.py tests/unit/test_cohesion_slider.py
git commit -m "feat(gui): add cohesion_mode field to UIStateModel

New field for the cohesion axis (strict/narrow/dynamic/discover, default
dynamic). Legacy 'cohesion' field stays until policy + tests migrate."
```

---

## Task 3: Add `cohesion_mode` derivation in `policy.py`

Add the new derivation and add the key to `POLICY_OWNED_KEYS`. Keep the legacy `COHESION_MAP` fallback block — Task 7 removes it.

**Files:**
- Modify: `src/playlist_gui/policy.py`
- Modify: `tests/unit/test_gui_policy.py` (add new tests; don't yet touch existing ones)

- [ ] **Step 1: Write the failing tests**

Open `tests/unit/test_gui_policy.py` and append at the end:

```python
# ─────────────────────────────────────────────────────────────────────────────
# Cohesion Mode Derivation Tests (sub-project B)
# ─────────────────────────────────────────────────────────────────────────────


class TestCohesionModeDerivation:
    """Tests that policy.derive_runtime_config writes playlists.cohesion_mode."""

    def test_cohesion_mode_in_policy_owned_keys(self):
        from src.playlist_gui.policy import POLICY_OWNED_KEYS

        assert "playlists.cohesion_mode" in POLICY_OWNED_KEYS

    @pytest.mark.parametrize(
        "mode", ["strict", "narrow", "dynamic", "discover"]
    )
    def test_each_valid_mode_writes_through(self, mode):
        from src.playlist_gui.policy import derive_runtime_config
        from src.playlist_gui.ui_state import UIStateModel

        state = UIStateModel(cohesion_mode=mode)
        decisions = derive_runtime_config(state)
        assert (
            decisions.overrides["playlists"]["cohesion_mode"] == mode
        ), f"Expected {mode}, got {decisions.overrides['playlists'].get('cohesion_mode')}"

    def test_invalid_cohesion_mode_falls_back_to_dynamic(self):
        from src.playlist_gui.policy import derive_runtime_config
        from src.playlist_gui.ui_state import UIStateModel

        state = UIStateModel()
        # Bypass dataclass type-checking by setting attr directly
        object.__setattr__(state, "cohesion_mode", "garbage")
        decisions = derive_runtime_config(state)
        assert decisions.overrides["playlists"]["cohesion_mode"] == "dynamic"

    def test_cohesion_mode_default_is_dynamic(self):
        from src.playlist_gui.policy import derive_runtime_config
        from src.playlist_gui.ui_state import UIStateModel

        state = UIStateModel()  # default cohesion_mode = "dynamic"
        decisions = derive_runtime_config(state)
        assert decisions.overrides["playlists"]["cohesion_mode"] == "dynamic"
```

(Note: imports of `pytest` are already present at the top of `test_gui_policy.py` — verify before assuming.)

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/unit/test_gui_policy.py::TestCohesionModeDerivation -v
```

Expected: failures — `POLICY_OWNED_KEYS` does not yet include `playlists.cohesion_mode`, and `derive_runtime_config` does not yet write it.

- [ ] **Step 3: Add the key to `POLICY_OWNED_KEYS`**

In `src/playlist_gui/policy.py` (around line 26), find:

```python
POLICY_OWNED_KEYS: Set[str] = {
    # Mode-derived settings
    "playlists.genre_mode",
    "playlists.sonic_mode",
    "playlists.pace_mode",
```

Insert `"playlists.cohesion_mode"` right after `"playlists.pace_mode"`:

```python
POLICY_OWNED_KEYS: Set[str] = {
    # Mode-derived settings
    "playlists.genre_mode",
    "playlists.sonic_mode",
    "playlists.pace_mode",
    "playlists.cohesion_mode",
```

- [ ] **Step 4: Add the validation set**

Below the existing `VALID_PACE_MODES` line (around line 62), add:

```python
VALID_COHESION_MODES: Set[str] = {"strict", "narrow", "dynamic", "discover"}
```

- [ ] **Step 5: Add the derivation logic**

In `derive_runtime_config()`, find the genre/sonic/pace block (around lines 257-275). After the existing `_set_nested(overrides, "playlists.pace_mode", pace_mode)` line and the three `notes.append(...)` calls, add:

```python
    cohesion_mode = getattr(ui, "cohesion_mode", "dynamic")
    if cohesion_mode not in VALID_COHESION_MODES:
        cohesion_mode = "dynamic"
    _set_nested(overrides, "playlists.cohesion_mode", cohesion_mode)
    notes.append(f"Cohesion mode: {cohesion_mode}")
```

- [ ] **Step 6: Run new tests to verify they pass**

```
pytest tests/unit/test_gui_policy.py::TestCohesionModeDerivation -v
```

Expected: 7 tests pass (1 ownership + 4 parametrized roundtrip + 1 invalid fallback + 1 default).

Run the full policy + panel test suites to confirm no regressions:
```
pytest tests/unit/test_gui_policy.py tests/unit/test_generate_panel.py -q
```

Expected: all existing tests still pass (the legacy `COHESION_MAP` block still exists and is unchanged).

- [ ] **Step 7: Commit**

```
git add src/playlist_gui/policy.py tests/unit/test_gui_policy.py
git commit -m "feat(gui-policy): derive playlists.cohesion_mode from UIStateModel

Adds 'playlists.cohesion_mode' to POLICY_OWNED_KEYS and writes it from
ui_state.cohesion_mode with validation. Legacy COHESION_MAP fallback
stays (removed in Task 7 after test migration)."
```

---

## Task 4: Wire `CohesionSlider` into `GeneratePanel`

Add the new control group card and include `cohesion_mode` in `build_ui_state()`.

**Files:**
- Modify: `src/playlist_gui/widgets/generate_panel.py`

- [ ] **Step 1: Add import**

At the top of `src/playlist_gui/widgets/generate_panel.py`, find the existing imports from `.mode_sliders`:

```python
from .mode_sliders import ModeSliders
```

Add immediately after:

```python
from .cohesion_slider import CohesionSlider
```

- [ ] **Step 2: Update `_header_group_order`**

Find `self._header_group_order` (around line 95) and insert `"cohesion"` between `"mode"` and `"matching"`:

```python
        self._header_group_order = [
            "mode",
            "cohesion",
            "matching",
            "length",
            "freshness",
            "spacing",
            "diversity",
            "actions",
        ]
```

- [ ] **Step 3: Instantiate the widget and create the card**

Find the existing block (around lines 124-128):

```python
        self._create_control_group("mode", "Mode", mode_container)

        # Genre/Sonic/Pace mode sliders (stacked)
        self._mode_sliders = ModeSliders()
        self._create_control_group("matching", "Matching", self._mode_sliders)
```

Insert the cohesion card between them:

```python
        self._create_control_group("mode", "Mode", mode_container)

        # Overall cohesion (pier-bridge beam tuning)
        self._cohesion_slider = CohesionSlider()
        self._create_control_group("cohesion", "OVERALL COHESION", self._cohesion_slider)

        # Genre/Sonic/Pace mode sliders (stacked)
        self._mode_sliders = ModeSliders()
        self._create_control_group("matching", "Matching", self._mode_sliders)
```

- [ ] **Step 4: Extend `build_ui_state()`**

Find `build_ui_state()` (around line 560). Inside the `UIStateModel(...)` call, insert `cohesion_mode=...` right after the `mode=mode` line:

```python
        return UIStateModel(
            mode=mode,
            cohesion_mode=self._cohesion_slider.get_cohesion_mode(),
            genre_mode=self._mode_sliders.get_genre_mode(),
            sonic_mode=self._mode_sliders.get_sonic_mode(),
            pace_mode=self._mode_sliders.get_pace_mode(),
            # ... rest unchanged
```

- [ ] **Step 5: Smoke-import test**

```
python -c "from src.playlist_gui.widgets.generate_panel import GeneratePanel; print('OK')"
```

Expected: `OK` (no ImportError).

- [ ] **Step 6: Run existing panel tests**

```
pytest tests/unit/test_generate_panel.py -v
```

Expected: existing tests still pass (we'll add a new cohesion-specific assertion in Task 6).

- [ ] **Step 7: Commit**

```
git add src/playlist_gui/widgets/generate_panel.py
git commit -m "feat(gui): add OVERALL COHESION card to generate panel

Instantiates CohesionSlider as a new header card between Mode and
Matching, and includes cohesion_mode in build_ui_state(). Narrow-mode
reflow top row becomes [Mode, OVERALL COHESION, Matching, Length]."
```

---

## Task 5: Migrate `test_gui_policy.py` away from `COHESION_MAP`

Rename `cohesion=` fixtures to `cohesion_mode=`, drop the `TestCohesionMapping` class, drop `COHESION_MAP` import. After this task the tests no longer reference the legacy mapping, but `policy.py` still has the `COHESION_MAP` block (deleted in Task 7).

**Files:**
- Modify: `tests/unit/test_gui_policy.py`

- [ ] **Step 1: Audit references**

```
grep -n "cohesion\|COHESION_MAP" tests/unit/test_gui_policy.py
```

You should see references on roughly these lines (verify before editing):
- Line 14: `COHESION_MAP` in the import block
- Line 44: `assert state.cohesion == "balanced"` (default-state test)
- Lines 82-150: entire `TestCohesionMapping` class
- Lines 302-310: "genre pool disabled for tight/balanced/wide" test
- Line 321, 338: `UIStateModel(..., cohesion="discover")` in integration tests
- Lines 579-590: `test_full_seeds_mode_discover_flow` using `cohesion="discover"`
- Lines 626-640: `test_full_artist_mode_tight_flow` using `cohesion="tight"`

- [ ] **Step 2: Remove the `COHESION_MAP` import**

At line 14 (or wherever it appears in the import block), remove `COHESION_MAP` from the import list. Example:

Before:
```python
from src.playlist_gui.policy import (
    COHESION_MAP,
    PolicyDecisions,
    POLICY_OWNED_KEYS,
    derive_runtime_config,
    ...
)
```

After:
```python
from src.playlist_gui.policy import (
    PolicyDecisions,
    POLICY_OWNED_KEYS,
    derive_runtime_config,
    ...
)
```

- [ ] **Step 3: Drop the default-state assertion that depends on the old `cohesion` default**

At line 44 (inside whichever class — likely `TestUIStateDefaults` or similar), remove:

```python
        assert state.cohesion == "balanced"
```

(Leave the surrounding test in place; only that one line is dropped.)

- [ ] **Step 4: Delete the entire `TestCohesionMapping` class**

Find:
```python
# Cohesion Mapping Tests
# ...
class TestCohesionMapping:
    ...
```

Delete the entire class block (header comment + class body). The class is about 60-70 lines (roughly lines 82-150 — verify before deleting).

- [ ] **Step 5: Drop the "genre pool disabled for tight/balanced/wide" test**

Around lines 302-310 there is a test asserting that `genre_pool` is only enabled when `cohesion == "discover"`. The semantics no longer apply (cohesion is independent of genre). Find the test method (it will iterate over `["tight", "balanced", "wide"]` and assert genre pool disabled). Delete the entire method.

If you can't find it cleanly, use:
```
grep -n 'for cohesion in \["tight", "balanced", "wide"\]' tests/unit/test_gui_policy.py
```
to locate, then delete the surrounding test method.

- [ ] **Step 6: Rename remaining `cohesion=` to `cohesion_mode=` in integration tests**

Find every remaining `cohesion="..."` argument in `UIStateModel(...)` calls. Replace with `cohesion_mode="..."` and translate the value:
- `cohesion="tight"` → `cohesion_mode="strict"`
- `cohesion="balanced"` → `cohesion_mode="narrow"`
- `cohesion="wide"` → `cohesion_mode="dynamic"`
- `cohesion="discover"` → `cohesion_mode="discover"`

For each test that previously asserted genre_mode/sonic_mode would derive from cohesion (e.g., `cohesion="tight"` → `genre_mode == "strict"`), update the assertion to assert `cohesion_mode == "<value>"` instead of `genre_mode == "<value>"`. The new architecture: cohesion_mode is a separate axis from genre_mode.

If unsure how to translate a specific assertion, mark it with `pytest.skip("legacy cohesion mapping removed in sub-project B")` and report the test name in your status — do not silently delete an unclear test.

- [ ] **Step 7: Run the policy test suite**

```
pytest tests/unit/test_gui_policy.py -v
```

Expected: all remaining tests pass. The `TestCohesionModeDerivation` tests from Task 3 should still pass.

If any tests fail with `AttributeError: 'UIStateModel' object has no attribute 'cohesion'`, those tests still reference the old field — find them and update.

- [ ] **Step 8: Commit**

```
git add tests/unit/test_gui_policy.py
git commit -m "test(gui-policy): migrate fixtures from cohesion to cohesion_mode

Drops TestCohesionMapping (legacy single-knob → genre+sonic mapping),
removes COHESION_MAP import, renames cohesion=... to cohesion_mode=...
across integration tests. Policy backward-compat block still present
(removed in Task 7)."
```

---

## Task 6: Extend `test_generate_panel.py` with cohesion_mode assertion

**Files:**
- Modify: `tests/unit/test_generate_panel.py`

- [ ] **Step 1: Find the existing `build_ui_state` test**

```
grep -n "build_ui_state\|test_build" tests/unit/test_generate_panel.py
```

If a test exists that calls `panel.build_ui_state()`, add a cohesion assertion there. If no such test exists, add a new one.

- [ ] **Step 2: Add the cohesion assertion**

Append the following test (or extend an existing one):

```python
def test_build_ui_state_includes_cohesion_mode(qtbot):
    """build_ui_state() should capture cohesion_mode from the slider."""
    from src.playlist_gui.widgets.generate_panel import GeneratePanel

    panel = GeneratePanel()
    qtbot.addWidget(panel)

    panel._cohesion_slider.set_cohesion_mode("strict")
    state = panel.build_ui_state()
    assert state.cohesion_mode == "strict"

    panel._cohesion_slider.set_cohesion_mode("discover")
    state = panel.build_ui_state()
    assert state.cohesion_mode == "discover"
```

Note: `qtbot` is a no-op stub in this codebase (`tests/conftest.py:60-70`, called out in CLAUDE.md). The `qtbot.addWidget(panel)` call is for consistency with other tests and does not affect the assertion — we're driving state via `set_cohesion_mode()` directly, which is safe.

If the existing test pattern doesn't use `qtbot` in its signature, mirror the existing pattern in the file. The key assertions are the two `state.cohesion_mode == ...` lines.

- [ ] **Step 3: Run the panel tests**

```
pytest tests/unit/test_generate_panel.py -v
```

Expected: existing tests still pass, plus the new test passes.

- [ ] **Step 4: Commit**

```
git add tests/unit/test_generate_panel.py
git commit -m "test(generate-panel): assert cohesion_mode in build_ui_state output"
```

---

## Task 7: Final cleanup — remove obsolete `cohesion` field, `COHESION_MAP`, and `CohesionDial`

After this task, the codebase has zero references to the legacy cohesion mapping.

**Files:**
- Modify: `src/playlist_gui/ui_state.py` (remove `cohesion` field)
- Modify: `src/playlist_gui/policy.py` (remove `COHESION_MAP` block and the fallback)
- Delete: `src/playlist_gui/widgets/cohesion_dial.py`
- Modify: `docs/ui_ux/plan_gui_just_works.md`, `docs/ui_ux/phase_notes/phase_ui_polish_plan.md`, `docs/ui_ux/phase_notes/phase2_ui_wiring.md` (add superseded footnote)

- [ ] **Step 1: Verify no remaining `cohesion` (old field) references in production code or tests**

```
grep -rn "\.cohesion\b\|COHESION_MAP\|cohesion=" src/ tests/ --include="*.py" | grep -v "cohesion_mode\|cohesion_slider\|CohesionSlider"
```

Expected output: only references inside `src/playlist_gui/widgets/cohesion_dial.py` (about to be deleted), `src/playlist_gui/ui_state.py:37` (the field we're about to remove), and `src/playlist_gui/policy.py:261-265` (the block we're about to remove).

If anything else appears, fix it before continuing.

- [ ] **Step 2: Remove the `cohesion` field from `UIStateModel`**

In `src/playlist_gui/ui_state.py`, find (around line 37):

```python
    cohesion: Literal["tight", "balanced", "wide", "discover"] = "balanced"
    """
    Backward-compatible aggregate control for genre_mode and sonic_mode.
    Newer UI paths may set genre_mode/sonic_mode directly.
    """

    cohesion_mode: Literal["strict", "narrow", "dynamic", "discover"] = "dynamic"
```

Delete the `cohesion` field block (3 lines plus the docstring) so it becomes:

```python
    cohesion_mode: Literal["strict", "narrow", "dynamic", "discover"] = "dynamic"
    """
    Overall cohesion (pier-bridge beam tightness). Independent of
    genre_mode/sonic_mode/pace_mode (which control pool composition).
    Single writer: policy.derive_runtime_config() → playlists.cohesion_mode.
    """
```

- [ ] **Step 3: Remove `COHESION_MAP` and the fallback block from `policy.py`**

In `src/playlist_gui/policy.py`, delete the `COHESION_MAP` constant block (around lines 64-73):

```python
COHESION_MAP: Dict[str, tuple[str, str]] = {
    "tight": ("strict", "strict"),
    "balanced": ("narrow", "narrow"),
    "wide": ("dynamic", "dynamic"),
    "discover": ("discover", "discover"),
}
"""
Backward-compatible mapping from the original cohesion dial to explicit
genre/sonic modes.
"""
```

Then find the fallback usage block (around lines 260-268):

```python
    if (
        getattr(ui, "cohesion", None) in COHESION_MAP
        and ui.genre_mode == "narrow"
        and ui.sonic_mode == "narrow"
    ):
        genre_mode, sonic_mode = COHESION_MAP[ui.cohesion]
    else:
        genre_mode = ui.genre_mode if ui.genre_mode in VALID_MODES else "dynamic"
        sonic_mode = ui.sonic_mode if ui.sonic_mode in VALID_MODES else "dynamic"
```

Replace with the simplified direct-read:

```python
    genre_mode = ui.genre_mode if ui.genre_mode in VALID_MODES else "dynamic"
    sonic_mode = ui.sonic_mode if ui.sonic_mode in VALID_MODES else "dynamic"
```

- [ ] **Step 4: Delete `cohesion_dial.py`**

```
rm src/playlist_gui/widgets/cohesion_dial.py
```

PowerShell equivalent:
```
Remove-Item src/playlist_gui/widgets/cohesion_dial.py
```

- [ ] **Step 5: Add superseded footnotes to obsolete docs**

For each of the three doc files, append at the bottom:

```markdown

---

> **Note (2026-05-24):** The `CohesionDial` widget described in this document was never integrated into the GUI and was superseded by the standalone `CohesionSlider` in sub-project B (`docs/superpowers/specs/2026-05-24-cohesion-slider-design.md`).
```

Files:
- `docs/ui_ux/plan_gui_just_works.md`
- `docs/ui_ux/phase_notes/phase_ui_polish_plan.md`
- `docs/ui_ux/phase_notes/phase2_ui_wiring.md`

- [ ] **Step 6: Run the full test suite**

```
pytest tests/ -q 2>&1 | tail -5
```

Expected: all pass (test count should be `previous total + ~17` for the new cohesion tests). Zero failures.

- [ ] **Step 7: Final repo-wide sanity grep**

```
grep -rn "CohesionDial\|COHESION_MAP\|cohesion_dial" src/ tests/ --include="*.py"
```

Expected: zero matches.

```
grep -rn "ui\.cohesion[^_]\|cohesion: Literal\[\"tight" src/ tests/ --include="*.py"
```

Expected: zero matches.

- [ ] **Step 8: Commit**

```
git add src/playlist_gui/ui_state.py src/playlist_gui/policy.py docs/ui_ux/
git rm src/playlist_gui/widgets/cohesion_dial.py
git commit -m "refactor(gui): remove obsolete CohesionDial / COHESION_MAP machinery

The cohesion-as-single-knob → genre+sonic mapping is obsolete now that
cohesion_mode is an independent axis (sub-project A) with its own slider
(sub-project B Task 1-6). Drops the CohesionDial widget file, the
COHESION_MAP constant + fallback block in policy.py, and the legacy
'cohesion' field from UIStateModel. Adds superseded footnotes to the
three docs that documented the old design."
```

---

## Task 8: Manual verification

The unit tests can't open the GUI (per CLAUDE.md the `qtbot` is a no-op stub). Exercise the feature manually before declaring done.

**Files:** none (verification only)

- [ ] **Step 1: Launch the GUI**

```
python -m playlist_gui.app
```

Expected: window opens without exception.

- [ ] **Step 2: Verify the new card is present**

In the header row, you should see in this order (left to right):
`Mode | OVERALL COHESION | Matching | Length | Freshness | Spacing | Diversity | Actions`

The OVERALL COHESION card contains one horizontal slider with 4 tick marks and a value label showing "Dynamic" at startup.

- [ ] **Step 3: Verify slider moves through all 4 positions**

Drag the slider through each position. The value label should update: Strict → Narrow → Dynamic → Discover.

- [ ] **Step 4: Generate with cohesion=strict**

- Select Artist mode
- Type "The Sundays" in the artist field
- Set Length to 10
- Set OVERALL COHESION to Strict
- Click Generate

Watch the log panel for:
```
INFO Cohesion mode: strict
INFO ... bridge_floor=0.10 ...
```

If you see `bridge_floor=0.02`, the per-mode tuning didn't propagate — check that `playlists.cohesion_mode` actually got the value `strict` in the worker's logging.

- [ ] **Step 5: Generate with cohesion=dynamic**

- Set OVERALL COHESION back to Dynamic
- Click Generate again

Log should show `Cohesion mode: dynamic` and `bridge_floor=0.02`.

- [ ] **Step 6: Window resize / narrow reflow**

Drag the window narrower until the header reflows to 2 rows. Verify:
- Top row: `Mode | OVERALL COHESION | Matching | Length`
- Bottom row: `Freshness | Spacing | Diversity | Actions`

- [ ] **Step 7: Confirm Advanced Panel cohesion dropdown is now redundant**

Open Advanced Settings → Pipeline group. The "Cohesion mode" dropdown is still there. Change it to "Strict" while the main panel slider is set to "Dynamic", then generate. The log should show `Cohesion mode: dynamic` (the main slider wins because policy owns the key). This is the known sub-project C scope: the Advanced Panel control is superseded but not yet removed.

- [ ] **Step 8: Report**

If all 7 steps pass, sub-project B is done. Note the test count and any cosmetic issues for follow-up. If step 4 or 5 shows wrong `bridge_floor`, escalate — the policy/worker wiring has a bug.

---

## Acceptance Criteria

When all tasks complete:

1. `pytest tests/ -q` passes with the new tests included (~19 new tests total: 9 widget + 2 UIStateModel + 7 policy derivation + 1 generate panel)
2. `python -m playlist_gui.app` shows the OVERALL COHESION card between Mode and Matching, with a 4-position slider defaulting to Dynamic
3. Setting the slider to Strict and generating shows `Cohesion mode: strict` and `bridge_floor=0.10` in the log
4. Setting back to Dynamic shows `Cohesion mode: dynamic` and `bridge_floor=0.02`
5. `grep -rn "CohesionDial\|COHESION_MAP\|cohesion_dial" src/ tests/ --include="*.py"` returns zero matches
6. `grep -rn "ui\.cohesion[^_]\|cohesion: Literal\[\"tight" src/ tests/ --include="*.py"` returns zero matches
7. `src/playlist_gui/widgets/cohesion_dial.py` no longer exists
8. The three docs in `docs/ui_ux/` have the "superseded by sub-project B" footnote
