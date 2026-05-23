# Pace Mode Fourth Setting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-tier the pace mode slider into four levels that mirror the genre/sonic slider vocabulary. Currently `strict`/`narrow`/`dynamic` act as Very Strict / Strict / No Rules — there is no "true middle." This plan introduces `off` as the No Rules option and re-purposes `dynamic` to be an actual middle setting that catches double-time mismatches while still permitting natural tempo drift.

**Architecture:** Pure config/preset shuffle. No new modules. The existing BPM gates and rhythm-axis gates are already wired through `candidate_pool` and the beam — we're just changing the threshold values for `dynamic` and adding a fourth preset entry called `off` that matches what `dynamic` used to mean.

**Tech Stack:** Python 3.11, PySide6, pytest. No new dependencies.

**Out of scope:**
- New gate mechanics — both rhythm-axis cosine and BPM log-distance gates already exist
- New module files
- Backward compatibility for the renamed `dynamic` semantics — sole-operator project, intentional behavior shift
- Changes to genre/sonic mode presets

---

## Target preset values

| pace_mode | rhythm-axis admission | rhythm-axis bridge | BPM admission max-log | BPM bridge max-log | Description |
|---|---:|---:|---:|---:|---|
| `strict` | 0.55 | 0.65 | 0.30 | 0.40 | (unchanged) Lock to seed tempo |
| `narrow` | 0.35 | 0.45 | 0.50 | 0.60 | (unchanged) Stay close to seed |
| `dynamic` | **0.20** | **0.25** | **0.75** | **0.85** | **NEW VALUES** — gentle anchoring, catches double-time |
| `off` | 0.00 | 0.00 | ∞ | ∞ | (new entry; matches old `dynamic`) No pace constraint |

Why these numbers for the new `dynamic`:
- BPM 0.75 log-distance ≈ 1.68× tempo ratio. Blocks 2:1 (1.0 octave) but allows 70 → 110 BPM (≈ 0.65 octaves). Catches every double-time case from the original problem report.
- Rhythm-axis cosine 0.20/0.25 is a light floor — most candidates that pass the broad sonic similarity floor will pass this too. Acts as a gentle backstop against the most extreme rhythm-pattern mismatches that BPM alone can't see.

Default remains `dynamic` — but it now means something. Anyone who wants pre-change behavior sets `off` explicitly.

---

## File structure

**Modified files (no new files):**
- `src/playlist/mode_presets.py` — add `off` to `PACE_MODE_PRESETS`; change `dynamic` values
- `src/playlist_gui/widgets/mode_sliders.py` — extend `PACE_MODE_LEVELS` to four items; add label/tooltip for `off`
- `src/playlist_gui/policy.py` — extend `VALID_PACE_MODES` to include `off`
- `src/playlist_gui/ui_state.py` — extend `PaceModeValue` Literal to include `off`
- `main_app.py` — extend `--pace-mode` argparse choices
- `config.example.yaml` — update pace_mode comment
- `docs/PLAYLIST_ORDERING_TUNING.md` — update Knob 5 docs
- `tests/unit/test_pace_mode_presets.py` — update existing tests; add `off` tests

**Goldens that will drift:** Anywhere a golden capture has `pace_bridge_floor: 0.0` or BPM log-distance `Infinity` baked in — these reflected the old `dynamic` semantics and need to be regenerated.

---

## Task 1: Update `PACE_MODE_PRESETS` and add `off`

**Files:**
- Modify: `src/playlist/mode_presets.py`
- Test: extend `tests/unit/test_pace_mode_presets.py`

- [ ] **Step 1: Update the failing tests**

In `tests/unit/test_pace_mode_presets.py`, replace the existing tests that hardcode `dynamic`'s old behavior, and add new tests:

```python
import pytest
from src.playlist.mode_presets import (
    PACE_MODE_PRESETS,
    resolve_pace_mode,
)


def test_pace_mode_presets_has_four_modes():
    assert set(PACE_MODE_PRESETS.keys()) == {"strict", "narrow", "dynamic", "off"}


def test_off_disables_all_gates():
    settings = resolve_pace_mode("off")
    assert settings["admission_floor"] == 0.0
    assert settings["bridge_floor"] == 0.0
    assert settings["bpm_admission_max_log_distance"] == float("inf")
    assert settings["bpm_bridge_max_log_distance"] == float("inf")


def test_dynamic_is_middle_ground_not_disabled():
    """Dynamic now has real (moderate) thresholds — it's no longer a no-op."""
    settings = resolve_pace_mode("dynamic")
    assert settings["admission_floor"] > 0.0
    assert settings["bridge_floor"] > 0.0
    assert settings["bpm_admission_max_log_distance"] < float("inf")
    assert settings["bpm_bridge_max_log_distance"] < float("inf")


def test_dynamic_catches_double_time():
    """The BPM admission floor for dynamic must be < 1.0 (the octave distance)."""
    settings = resolve_pace_mode("dynamic")
    assert settings["bpm_admission_max_log_distance"] < 1.0
    assert settings["bpm_bridge_max_log_distance"] < 1.0


def test_pace_mode_monotonic_strict_to_off():
    """Each successive mode must be at least as permissive as the previous."""
    modes = ["strict", "narrow", "dynamic", "off"]
    settings = [resolve_pace_mode(m) for m in modes]
    for i in range(len(modes) - 1):
        assert settings[i]["admission_floor"] >= settings[i + 1]["admission_floor"], \
            f"{modes[i]} admission_floor must be >= {modes[i+1]} admission_floor"
        assert settings[i]["bridge_floor"] >= settings[i + 1]["bridge_floor"]
        assert settings[i]["bpm_admission_max_log_distance"] <= settings[i + 1]["bpm_admission_max_log_distance"]
        assert settings[i]["bpm_bridge_max_log_distance"] <= settings[i + 1]["bpm_bridge_max_log_distance"]


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="Unknown pace mode"):
        resolve_pace_mode("turbo")


def test_overrides_apply():
    settings = resolve_pace_mode("narrow", {"admission_floor": 0.10})
    assert settings["admission_floor"] == 0.10
    assert settings["bridge_floor"] == PACE_MODE_PRESETS["narrow"]["bridge_floor"]
```

- [ ] **Step 2: Run, expect existing tests to fail**

Run:
```bash
python -m pytest tests/unit/test_pace_mode_presets.py -v
```

Expected: tests for `dynamic = inf` will fail (good — that's what we're changing).

- [ ] **Step 3: Update `PACE_MODE_PRESETS`**

In `src/playlist/mode_presets.py`, change the `dynamic` entry and add an `off` entry:

```python
PACE_MODE_PRESETS: Dict[str, Dict[str, Any]] = {
    "strict": {
        "admission_floor": 0.55,
        "bridge_floor": 0.65,
        "bpm_admission_max_log_distance": 0.30,
        "bpm_bridge_max_log_distance": 0.40,
        "description": "Tight tempo fidelity - stay anchored to seed pace",
        "use_case": "Slow/meditative seeds; mood-locked playlists",
    },
    "narrow": {
        "admission_floor": 0.35,
        "bridge_floor": 0.45,
        "bpm_admission_max_log_distance": 0.50,
        "bpm_bridge_max_log_distance": 0.60,
        "description": "Moderate tempo anchoring",
        "use_case": "Consistent energy with some flex",
    },
    "dynamic": {
        "admission_floor": 0.20,
        "bridge_floor": 0.25,
        "bpm_admission_max_log_distance": 0.75,
        "bpm_bridge_max_log_distance": 0.85,
        "description": "Gentle pace anchoring - catches double-time, allows natural drift",
        "use_case": "General-purpose default; varied playlists with sensible tempo coherence",
    },
    "off": {
        "admission_floor": 0.00,
        "bridge_floor": 0.00,
        "bpm_admission_max_log_distance": float("inf"),
        "bpm_bridge_max_log_distance": float("inf"),
        "description": "No pace constraint - rhythm only contributes via sonic embedding",
        "use_case": "Multi-tempo playlists where pace should not gate candidates",
    },
}
```

- [ ] **Step 4: Run tests, expect pass**

```bash
python -m pytest tests/unit/test_pace_mode_presets.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/playlist/mode_presets.py tests/unit/test_pace_mode_presets.py
git commit -m "feat: re-tier pace mode — dynamic now a middle ground, add off as fourth"
```

---

## Task 2: Add `off` to CLI and GUI

**Files:**
- Modify: `main_app.py` (CLI choices)
- Modify: `src/playlist_gui/policy.py` (`VALID_PACE_MODES`)
- Modify: `src/playlist_gui/ui_state.py` (`PaceModeValue` Literal type)
- Modify: `src/playlist_gui/widgets/mode_sliders.py` (slider levels)

- [ ] **Step 1: Update CLI choices**

In `main_app.py`, find the `--pace-mode` argparse `add_argument` call. Change `choices=["strict", "narrow", "dynamic"]` to `choices=["strict", "narrow", "dynamic", "off"]`. Update the `help=` string to reflect the four levels:

```python
parser.add_argument(
    "--pace-mode",
    choices=["strict", "narrow", "dynamic", "off"],
    help="Pace/rhythm mode: strict / narrow / dynamic (default) / off",
)
```

- [ ] **Step 2: Update GUI policy**

In `src/playlist_gui/policy.py`, find:
```python
VALID_PACE_MODES: Set[str] = {"strict", "narrow", "dynamic"}
```

Change to:
```python
VALID_PACE_MODES: Set[str] = {"strict", "narrow", "dynamic", "off"}
```

- [ ] **Step 3: Update Literal type**

In `src/playlist_gui/ui_state.py`, find:
```python
PaceModeValue = Literal["strict", "narrow", "dynamic"]
```

Change to:
```python
PaceModeValue = Literal["strict", "narrow", "dynamic", "off"]
```

- [ ] **Step 4: Update GUI slider**

In `src/playlist_gui/widgets/mode_sliders.py`:

(a) Update the type alias:
```python
PaceModeLevel = Literal["strict", "narrow", "dynamic", "off"]
```

(b) Update the levels list:
```python
PACE_MODE_LEVELS: list[PaceModeLevel] = ["strict", "narrow", "dynamic", "off"]
```

(c) Update labels:
```python
PACE_MODE_LABELS = {
    "strict": "Strict",
    "narrow": "Narrow",
    "dynamic": "Dynamic",
    "off": "Off",
}
```

(d) Update tooltips:
```python
PACE_MODE_TOOLTIPS = {
    "strict": "Tight rhythm/tempo fidelity",
    "narrow": "Moderate rhythm/tempo anchoring",
    "dynamic": "Gentle anchoring — catches double-time, allows drift",
    "off": "No pace gate — rhythm still influences via sonic embedding",
}
```

Default remains `dynamic` (the slider `setValue(PACE_MODE_LEVELS.index("dynamic"))` line stays — it now points to position 2 out of 4).

- [ ] **Step 5: Verify GUI tests still pass**

```bash
python -m pytest tests/unit/test_generate_panel.py tests/unit/test_gui_policy.py -v
```

If any test enumerates pace mode options and expects exactly 3, update it to expect 4.

- [ ] **Step 6: Commit**

```bash
git add main_app.py src/playlist_gui/policy.py src/playlist_gui/ui_state.py src/playlist_gui/widgets/mode_sliders.py tests/unit/test_generate_panel.py tests/unit/test_gui_policy.py
git commit -m "feat: add 'off' pace mode option to CLI and GUI slider"
```

---

## Task 3: Regenerate or update pipeline goldens

**Files:**
- Modify: any of `tests/unit/goldens/pipeline/*.json` that contain `pace_bridge_floor` or `bpm_*` fields

- [ ] **Step 1: Identify drifted goldens**

```bash
python -m pytest tests/ -m "not slow" -q --tb=short
```

Note any failing tests. They will be golden-comparison tests where the captured effective config snapshot includes the old `dynamic = 0.0 / inf` values and now sees the new `dynamic = 0.25 / 0.85` values.

- [ ] **Step 2: Inspect each failing diff**

For each failing golden test, run it with `-v` and look at the diff between captured and expected JSON. Confirm only the pace/BPM fields drifted (not playlist content). If playlist content also drifted, that is expected — the new dynamic now filters candidates the old one let through.

- [ ] **Step 3: Update the goldens**

For each affected file:
- If only the effective-config snapshot changed (no playlist content drift), open the JSON and update `pace_bridge_floor` and `bpm_bridge_max_log_distance` to the new dynamic values (0.25 and 0.85 respectively).
- If playlist content changed, regenerate the golden using whatever existing mechanism the repo uses (often there's a `--update-goldens` flag or a fixture that auto-writes when the env var is set; look at `tests/unit/goldens/pipeline/discover_with_dj_bridging.json.new` which appears to be an auto-generated update file).

- [ ] **Step 4: Run full suite**

```bash
python -m pytest tests/ -m "not slow" -q
```

All passing.

- [ ] **Step 5: Commit**

```bash
git add tests/unit/goldens/pipeline/
git commit -m "test: regenerate pipeline goldens for new pace dynamic values"
```

---

## Task 4: Update docs

**Files:**
- Modify: `config.example.yaml`
- Modify: `docs/PLAYLIST_ORDERING_TUNING.md`
- Modify: `README.md` (the Pace Mode table)

- [ ] **Step 1: Update `config.example.yaml`**

Find the `pace_mode` block (search for `pace_mode:`) and replace the existing comment block with:

```yaml
  # PACE MODES (rhythm/tempo coherence; orthogonal to sonic_mode):
  #   strict:   Tight tempo fidelity (rhythm floors 0.55/0.65, BPM 0.30/0.40)
  #   narrow:   Moderate tempo anchoring (0.35/0.45 rhythm, 0.50/0.60 BPM)
  #   dynamic:  Gentle anchoring — catches double-time but allows drift (0.20/0.25 rhythm, 0.75/0.85 BPM)
  #   off:      No pace gate — rhythm still influences via the sonic embedding
```

- [ ] **Step 2: Update Knob 5 in `docs/PLAYLIST_ORDERING_TUNING.md`**

Replace the existing pace_mode table with the four-row version (strict / narrow / dynamic / off) using values above. Add a paragraph clarifying: "Default is `dynamic`, which now filters out the worst double-time mismatches while still allowing natural tempo variation. Set `pace_mode: off` to disable both pace gates (rhythm still contributes via the sonic embedding at 20% weight)."

- [ ] **Step 3: Update README.md Pace Mode table**

In `README.md`, find the `### Pace Mode` table and update to four rows:

| Mode | Admission floor | Bridge floor | Use case |
|---|---|---|---|
| `strict` | 0.55 / 0.30 | 0.65 / 0.40 | Lock to seed tempo |
| `narrow` | 0.35 / 0.50 | 0.45 / 0.60 | Moderate anchoring |
| `dynamic` | 0.20 / 0.75 | 0.25 / 0.85 | Gentle — catches double-time |
| `off` | 0 / ∞ | 0 / ∞ | No pace constraint |

(Each cell: rhythm-axis cosine / BPM log-distance.)

Update the "Pace mode is orthogonal..." paragraph to mention the four-level structure.

- [ ] **Step 4: Commit**

```bash
git add config.example.yaml docs/PLAYLIST_ORDERING_TUNING.md README.md
git commit -m "docs: pace mode four-level structure (strict/narrow/dynamic/off)"
```

---

## End-to-end validation

- [ ] **Regenerate the shoegaze playlist with each pace mode**

Same seeds, four runs: `--pace-mode strict|narrow|dynamic|off`. Watch the `BPM admission gate: max_log_distance=X rejected=Y` line in each:

- `strict`: heaviest rejection rate
- `narrow`: moderate rejection
- `dynamic`: light rejection (catches only ~2:1 mismatches)
- `off`: no `BPM admission gate:` log line at all (gates disabled)

- [ ] **Confirm `off` is byte-identical to old `dynamic` behavior**

Generate a playlist with `--pace-mode off` and compare against the pre-Task-1 dynamic baseline. Should match exactly — `off` is the historical no-op behavior.

- [ ] **Confirm full suite passes**

```bash
python -m pytest tests/ -m "not slow"
```

All tests passing.

---

## Risk notes

- **Default behavior change.** Anyone running with implicit `pace_mode=dynamic` (i.e., not specifying it) will now get the new moderate filter instead of no filter. This is intentional but worth noting in commit messages.
- **Golden drift is expected.** Some pipeline goldens captured `pace_bridge_floor: 0.0` and `bpm_*: Infinity` — these reflected the old semantics. Update goldens; do not preserve the old values.
- **Rhythm-axis 0.20 floor is a guess.** May need tuning after a few real playlists. If `dynamic` rejects too many candidates with reasonable rhythm feel, lower to 0.15 or 0.10.
- **BPM 0.75 log-distance for dynamic** is the explicit double-time guardrail. Do not raise above 1.0 — that would let octave mismatches through.
- **Pace-related test fixtures** (e.g., `test_candidate_pool_pace_floor.py`, `test_pier_bridge_bpm_gate.py`) may have used `pace_mode="dynamic"` as shorthand for "no constraint" in setups. They should keep working because we use direct float thresholds in those tests, not the preset name — but verify nothing in the test suite relies on `dynamic` being equivalent to off.
