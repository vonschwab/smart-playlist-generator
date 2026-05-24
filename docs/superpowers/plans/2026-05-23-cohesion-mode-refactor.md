# Cohesion Mode Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the GUI-inaccessible `playlists.ds_pipeline.mode` config key with a new `playlists.cohesion_mode` axis at the same level as `genre_mode`/`sonic_mode`/`pace_mode`, and dedup three duplicate-writer settings to one source each.

**Architecture:** Backend-only refactor. Add `resolve_cohesion_mode()` helper as the sole reader of the new key, propagate the rename through ~25-30 files, and delete the per-`Mode` dicts in `default_ds_config()` for `max_artist_fraction`/`broad_filters`/`min_sonic_similarity` so the GUI-side writers (`apply_mode_presets()`, `policy.py`) become the single source of truth. No GUI changes — sub-project B handles the Cohesion slider.

**Tech Stack:** Python 3.11+, pytest, PySide6 GUI (untouched in this plan).

**Spec:** `docs/superpowers/specs/2026-05-23-cohesion-mode-refactor-design.md`

---

## File Structure

**New files:**
- `tests/unit/test_cohesion_mode_resolution.py` — tests for `resolve_cohesion_mode()` helper
- `tests/unit/test_default_ds_config_dedup.py` — tests that `default_ds_config()` no longer writes the three duplicate settings
- `tests/unit/test_single_writer_settings.py` — integration test verifying the dedup holds end-to-end

**Substantively modified:**
- `src/playlist/config.py` — update `Mode` Literal, add `resolve_cohesion_mode()`, dedup `default_ds_config()` and `get_min_sonic_similarity()`
- `src/playlist_generator.py` — rename `ds_mode_override` → `cohesion_mode_override`, use new helper
- `src/playlist_gui/worker.py` — use new helper to read `cohesion_mode`
- `main_app.py` — CLI flag rename
- `src/playlist_gui/config/settings_schema.py` — update SettingSpec key_path and choices

**Mechanical renames:**
- `src/playlist/reporter.py`, `src/playlist/run_audit.py`, `src/playlist/pipeline/audit_emitter.py`, `src/playlist/pipeline/core.py`, `src/playlist/pipeline/pier_bridge_overrides.py`, `src/playlist/__init__.py`
- `src/playlist_gui/config/config_model.py` — docstring example
- `scripts/diagnose_sonic_floor.py`, `scripts/diagnose_artist_style.py`, `scripts/diagnose_candidate_scores.py`, `scripts/sweep_pier_bridge_dials.py`, `scripts/run_dj_*.py`
- `tests/test_artist_style.py`, `tests/unit/test_mode_threshold_resolution.py`, `tests/unit/test_edge_repair.py`, `tests/unit/test_generate_panel.py`, `tests/unit/test_gui_policy.py`, `tests/unit/test_gui_generation_validation.py`, `tests/unit/test_gui_config.py`

**Config:**
- `config.yaml` (gitignored, local) — drop `playlists.ds_pipeline.mode`, add `playlists.cohesion_mode: dynamic`
- `config.example.yaml` — same edit + update comments

**Docs:**
- `docs/PLAYLIST_ORDERING_TUNING.md` — Knob 6 caveat
- `docs/CONFIG.md` — key reference
- `docs/GOLDEN_COMMANDS.md` — CLI flag note
- `docs/TECHNICAL_PLAYLIST_GENERATION_FLOW.md` — mode resolution section
- `CLAUDE.md` — design principles wording

**Deletions:**
- `~/.claude/projects/C--Users-Dylan-Desktop-PLAYLIST-GENERATOR-V3/memory/project_ds_mode_coupling.md` — describes a problem this refactor fixes

---

## Task 1: Update `Mode` Literal + add `resolve_cohesion_mode()` helper

Foundational change — every later task depends on the new `Mode` set and the helper existing.

**Files:**
- Modify: `src/playlist/config.py` (lines 8, ~345)
- Create: `tests/unit/test_cohesion_mode_resolution.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_cohesion_mode_resolution.py`:

```python
"""Tests for resolve_cohesion_mode() helper and updated Mode Literal."""
from __future__ import annotations

import logging

import pytest

from src.playlist.config import Mode, default_ds_config, resolve_cohesion_mode


class TestResolveCohesionMode:
    def test_valid_strict(self):
        assert resolve_cohesion_mode({"cohesion_mode": "strict"}) == "strict"

    def test_valid_narrow(self):
        assert resolve_cohesion_mode({"cohesion_mode": "narrow"}) == "narrow"

    def test_valid_dynamic(self):
        assert resolve_cohesion_mode({"cohesion_mode": "dynamic"}) == "dynamic"

    def test_valid_discover(self):
        assert resolve_cohesion_mode({"cohesion_mode": "discover"}) == "discover"

    def test_missing_key_defaults_dynamic(self):
        assert resolve_cohesion_mode({}) == "dynamic"

    def test_none_cfg_defaults_dynamic(self):
        assert resolve_cohesion_mode(None) == "dynamic"  # type: ignore[arg-type]

    def test_invalid_value_warns_and_defaults(self, caplog):
        with caplog.at_level(logging.WARNING):
            result = resolve_cohesion_mode({"cohesion_mode": "tight"})
        assert result == "dynamic"
        assert any("Invalid cohesion_mode" in r.message for r in caplog.records)

    def test_value_is_normalized(self):
        assert resolve_cohesion_mode({"cohesion_mode": "  Strict "}) == "strict"

    def test_stale_ds_pipeline_mode_warns(self, caplog):
        cfg = {"ds_pipeline": {"mode": "narrow"}, "cohesion_mode": "dynamic"}
        with caplog.at_level(logging.WARNING):
            result = resolve_cohesion_mode(cfg)
        assert result == "dynamic"
        assert any("ds_pipeline.mode is no longer used" in r.message for r in caplog.records)

    def test_stale_ds_pipeline_mode_alone_ignored(self, caplog):
        cfg = {"ds_pipeline": {"mode": "narrow"}}
        with caplog.at_level(logging.WARNING):
            result = resolve_cohesion_mode(cfg)
        assert result == "dynamic"
        assert any("ds_pipeline.mode is no longer used" in r.message for r in caplog.records)


class TestModeLiteralUpdate:
    def test_strict_is_valid_mode(self):
        # Should not raise
        cfg = default_ds_config("strict", playlist_len=30)
        assert cfg.mode == "strict"

    def test_sonic_only_is_no_longer_valid(self):
        with pytest.raises(ValueError, match="Unsupported mode"):
            default_ds_config("sonic_only", playlist_len=30)
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/unit/test_cohesion_mode_resolution.py -v
```

Expected: ImportError or AttributeError — `resolve_cohesion_mode` does not exist; `default_ds_config("strict", ...)` may pass today but `default_ds_config("sonic_only", ...)` will not raise.

- [ ] **Step 3: Update `Mode` Literal in `src/playlist/config.py:8`**

```python
Mode = Literal["strict", "narrow", "dynamic", "discover"]
```

- [ ] **Step 4: Remove `sonic_only` from validation set in `default_ds_config()` (around line 345)**

Find:
```python
    if mode not in {"strict", "narrow", "dynamic", "discover", "sonic_only"}:
        raise ValueError(f"Unsupported mode {mode}")
```

Replace with:
```python
    if mode not in {"strict", "narrow", "dynamic", "discover"}:
        raise ValueError(f"Unsupported mode {mode}")
```

Note: All per-mode dicts that include `"sonic_only"` keys (lines 362, 366, 371, 398, 410, 423, 488, 492, 496, 500, 505, 510, 514, etc.) will be cleaned up across Tasks 1, 2, and the dedup pass. For this task, just remove the `"sonic_only"` key from each dict you can find with `git grep "sonic_only" src/playlist/config.py`. Leave structure intact, just drop the `, "sonic_only": <value>` entries from each dict literal.

Also handle line 514:
```python
hard_floor=constraints.get(
    "hard_floor",
    True if mode in {"dynamic", "sonic_only"} else False,
),
```
becomes:
```python
hard_floor=constraints.get(
    "hard_floor",
    True if mode == "dynamic" else False,
),
```

And line 411 in `_candidate_per_artist`:
```python
        if mode == "sonic_only":
            return max(3, min(2 * max_per_artist_final, 6))
```
Delete this whole branch — `dynamic` falls through to the same value.

- [ ] **Step 5: Add `resolve_cohesion_mode()` helper to `src/playlist/config.py`**

Add at module level, after the existing imports and before the dataclass definitions (around line 11):

```python
import logging as _logging

_logger = _logging.getLogger(__name__)


def resolve_cohesion_mode(playlists_cfg: Optional[dict]) -> Mode:
    """
    Read playlists.cohesion_mode with validation.

    Sole reader of the cohesion_mode key. Warns (and ignores) if the legacy
    ds_pipeline.mode key is present so stale configs surface immediately.
    """
    if not isinstance(playlists_cfg, dict):
        return "dynamic"

    ds_pipeline = playlists_cfg.get("ds_pipeline")
    if isinstance(ds_pipeline, dict) and "mode" in ds_pipeline:
        _logger.warning(
            "playlists.ds_pipeline.mode is no longer used; remove from config. "
            "Use playlists.cohesion_mode instead."
        )

    raw = str(playlists_cfg.get("cohesion_mode", "dynamic")).strip().lower()
    if raw not in {"strict", "narrow", "dynamic", "discover"}:
        _logger.warning("Invalid cohesion_mode %r; falling back to 'dynamic'", raw)
        return "dynamic"
    return raw  # type: ignore[return-value]
```

- [ ] **Step 6: Run the tests to verify they pass**

```
pytest tests/unit/test_cohesion_mode_resolution.py -v
```

Expected: all 11 tests pass.

- [ ] **Step 7: Run the wider config test suite to confirm no regressions**

```
pytest tests/unit/test_mode_threshold_resolution.py tests/test_artist_style.py -v
```

Expected: all pass. If `test_artist_style.py:test_soft_genre_penalty_per_mode_resolution` fails, do not modify the test — investigate, as it tests `resolve_pier_bridge_tuning()` directly and shouldn't be affected by Task 1.

- [ ] **Step 8: Commit**

```
git add src/playlist/config.py tests/unit/test_cohesion_mode_resolution.py
git commit -m "feat(config): add resolve_cohesion_mode helper + update Mode Literal

Replaces 'sonic_only' with 'strict' in the Mode Literal and adds the
resolve_cohesion_mode() helper that reads playlists.cohesion_mode with
validation. Warns on stale ds_pipeline.mode key so forgotten configs
surface immediately.

Foundation for sub-project A (cohesion mode refactor).
"
```

---

## Task 2: Dedup `default_ds_config()` and `get_min_sonic_similarity()`

Remove per-mode defaults for the three duplicate-writer settings so `apply_mode_presets()` and `policy.py` become single sources of truth.

**Files:**
- Modify: `src/playlist/config.py` (lines ~294, ~360, ~373)
- Create: `tests/unit/test_default_ds_config_dedup.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_default_ds_config_dedup.py`:

```python
"""Tests that default_ds_config() no longer writes the three duplicate-writer settings."""
from __future__ import annotations

import pytest

from src.playlist.config import default_ds_config, get_min_sonic_similarity


class TestMinSonicSimilarityDedup:
    def test_returns_none_when_no_override_set(self):
        # No per-mode default lookup; returns None when caller hasn't set anything
        assert get_min_sonic_similarity({}, "dynamic") is None
        assert get_min_sonic_similarity({}, "strict") is None
        assert get_min_sonic_similarity({}, "narrow") is None
        assert get_min_sonic_similarity({}, "discover") is None

    def test_respects_explicit_override(self):
        assert get_min_sonic_similarity({"min_sonic_similarity": 0.15}, "dynamic") == 0.15

    def test_respects_per_mode_override(self):
        cfg = {"min_sonic_similarity_strict": 0.30}
        assert get_min_sonic_similarity(cfg, "strict") == 0.30

    def test_per_mode_override_wins_over_base(self):
        cfg = {"min_sonic_similarity": 0.10, "min_sonic_similarity_strict": 0.30}
        assert get_min_sonic_similarity(cfg, "strict") == 0.30
        assert get_min_sonic_similarity(cfg, "dynamic") == 0.10

    def test_explicit_none_returns_none(self):
        cfg = {"min_sonic_similarity": None}
        assert get_min_sonic_similarity(cfg, "dynamic") is None


class TestMaxArtistFractionDedup:
    @pytest.mark.parametrize("mode", ["strict", "narrow", "dynamic", "discover"])
    def test_falls_back_to_single_default(self, mode):
        # No per-mode dict; falls back to 0.125 for every mode unless overridden
        cfg = default_ds_config(mode, playlist_len=30)
        assert cfg.candidate.max_artist_fraction_final == 0.125

    def test_respects_explicit_override(self):
        overrides = {"candidate_pool": {"max_artist_fraction": 0.20}}
        cfg = default_ds_config("dynamic", playlist_len=30, overrides=overrides)
        assert cfg.candidate.max_artist_fraction_final == 0.20


class TestBroadFiltersDedup:
    @pytest.mark.parametrize("mode", ["strict", "narrow", "dynamic", "discover"])
    def test_empty_by_default(self, mode):
        # No per-mode dict; empty tuple for every mode unless caller passes one
        cfg = default_ds_config(mode, playlist_len=30)
        assert cfg.candidate.broad_filters == ()

    def test_respects_explicit_override(self):
        overrides = {"candidate_pool": {"broad_filters": ["rock", "indie"]}}
        cfg = default_ds_config("dynamic", playlist_len=30, overrides=overrides)
        assert cfg.candidate.broad_filters == ("rock", "indie")
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/unit/test_default_ds_config_dedup.py -v
```

Expected: most fail. `test_returns_none_when_no_override_set` fails because the current `get_min_sonic_similarity()` returns per-mode defaults (0.05 for dynamic, 0.20 for strict). `test_falls_back_to_single_default` fails for `strict`/`narrow`/`discover` (current values are 0.25/0.20/0.05). `test_empty_by_default` fails for `strict`/`narrow` (currently returns `["rock", "indie", "alternative", "pop"]`).

- [ ] **Step 3: Dedup `get_min_sonic_similarity()` in `src/playlist/config.py` (around line 294)**

Find the current function body. Replace it with this simpler version:

```python
def get_min_sonic_similarity(candidate_pool_cfg: dict, mode: Mode) -> Optional[float]:
    """
    Resolve the sonic similarity floor for the given mode from config.

    Single writer for this setting is apply_mode_presets() (driven by sonic_mode).
    Returns None when nothing is set — apply_mode_presets writes a value in
    the normal config-loading path.

    Priority:
    1) min_sonic_similarity_<mode> (per-mode override)
    2) min_sonic_similarity (base override applied to all modes)
    3) None (no per-mode default; apply_mode_presets is responsible)
    """
    mode = mode.lower()  # type: ignore[assignment]
    mode_key = f"min_sonic_similarity_{mode}"
    if mode_key in candidate_pool_cfg and candidate_pool_cfg.get(mode_key) is None:
        return None
    if "min_sonic_similarity" in candidate_pool_cfg and candidate_pool_cfg.get("min_sonic_similarity") is None:
        return None
    mode_specific = candidate_pool_cfg.get(mode_key)
    base = candidate_pool_cfg.get("min_sonic_similarity")
    resolved = mode_specific if mode_specific is not None else base
    return float(resolved) if resolved is not None else None
```

The only deletion vs. the current version: removed the `default = {...}.get(mode, None)` dict and the `resolved = default if resolved is None else resolved` line.

- [ ] **Step 4: Dedup `max_artist_fraction_final` in `default_ds_config()` (around line 360)**

Find:
```python
    max_artist_fraction_final = candidate_pool.get(
        "max_artist_fraction",
        {"strict": 0.25, "narrow": 0.20, "dynamic": 0.125, "discover": 0.05}[mode],
    )
```

(Note: by this point `sonic_only` should be gone from the dict per Task 1.)

Replace with:
```python
    max_artist_fraction_final = candidate_pool.get("max_artist_fraction", 0.125)
```

- [ ] **Step 5: Dedup `broad_filters` in `default_ds_config()` (around line 373)**

Find:
```python
    broad_filters_cfg_raw = candidate_pool.get("broad_filters", None)
    if broad_filters_cfg_raw is None:
        broad_filters_cfg: list[str] = ["rock", "indie", "alternative", "pop"] if mode in ("strict", "narrow") else []
    elif isinstance(broad_filters_cfg_raw, str):
        broad_filters_cfg = [broad_filters_cfg_raw]
    elif isinstance(broad_filters_cfg_raw, (list, tuple)):
        broad_filters_cfg = [str(b) for b in broad_filters_cfg_raw]
    else:
        try:
            broad_filters_cfg = [str(b) for b in list(broad_filters_cfg_raw)]
        except Exception:
            broad_filters_cfg = []
```

Replace the conditional default with `[]`:
```python
    broad_filters_cfg_raw = candidate_pool.get("broad_filters", None)
    if broad_filters_cfg_raw is None:
        broad_filters_cfg: list[str] = []
    elif isinstance(broad_filters_cfg_raw, str):
        broad_filters_cfg = [broad_filters_cfg_raw]
    elif isinstance(broad_filters_cfg_raw, (list, tuple)):
        broad_filters_cfg = [str(b) for b in broad_filters_cfg_raw]
    else:
        try:
            broad_filters_cfg = [str(b) for b in list(broad_filters_cfg_raw)]
        except Exception:
            broad_filters_cfg = []
```

- [ ] **Step 6: Run dedup tests to verify they pass**

```
pytest tests/unit/test_default_ds_config_dedup.py -v
```

Expected: all 11 tests pass.

- [ ] **Step 7: Run full config + mode preset suites to confirm no regressions**

```
pytest tests/unit/test_cohesion_mode_resolution.py tests/unit/test_mode_threshold_resolution.py tests/test_artist_style.py -v
```

Expected: all pass.

- [ ] **Step 8: Commit**

```
git add src/playlist/config.py tests/unit/test_default_ds_config_dedup.py
git commit -m "refactor(config): dedup three duplicate-writer settings

Removes per-mode defaults from default_ds_config() and
get_min_sonic_similarity() for min_sonic_similarity, max_artist_fraction,
and broad_filters. After this, apply_mode_presets() (driven by sonic_mode/
genre_mode) and policy.py (driven by artist_presence) are the single
writers for these settings.

Part of sub-project A (cohesion mode refactor).
"
```

---

## Task 3: Single-writer integration test

End-to-end sanity check that the full config-resolution chain respects the dedup.

**Files:**
- Create: `tests/unit/test_single_writer_settings.py`

- [ ] **Step 1: Write the integration test**

Create `tests/unit/test_single_writer_settings.py`:

```python
"""Integration test: confirm one writer per setting after the dedup."""
from __future__ import annotations

from src.playlist.config import default_ds_config
from src.playlist.mode_presets import apply_mode_presets


def _build_playlists_cfg(cohesion: str, sonic: str, genre: str) -> dict:
    return {
        "cohesion_mode": cohesion,
        "sonic_mode": sonic,
        "genre_mode": genre,
        "ds_pipeline": {
            "candidate_pool": {},
            "scoring": {},
            "constraints": {},
            "repair": {},
        },
    }


class TestSingleWriterEndToEnd:
    def test_sonic_mode_owns_min_sonic_similarity(self):
        """sonic_mode=narrow should set 0.12 regardless of cohesion_mode."""
        cfg = _build_playlists_cfg(cohesion="strict", sonic="narrow", genre="dynamic")
        apply_mode_presets(cfg)
        ds_cfg = default_ds_config(
            cfg["cohesion_mode"],
            playlist_len=30,
            overrides=cfg["ds_pipeline"],
        )
        # Sonic-narrow preset writes 0.12. Without dedup, cohesion=strict would
        # overwrite to 0.20 inside default_ds_config(). Confirm it does not.
        assert ds_cfg.candidate.min_sonic_similarity == 0.12

    def test_genre_mode_owns_broad_filters(self):
        """genre_mode=dynamic means no broad_filters even if cohesion_mode=strict."""
        cfg = _build_playlists_cfg(cohesion="strict", sonic="dynamic", genre="dynamic")
        apply_mode_presets(cfg)
        ds_cfg = default_ds_config(
            cfg["cohesion_mode"],
            playlist_len=30,
            overrides=cfg["ds_pipeline"],
        )
        # Without dedup, cohesion=strict would set broad_filters to
        # ["rock","indie","alternative","pop"]. Confirm it does not.
        assert ds_cfg.candidate.broad_filters == ()

    def test_genre_strict_writes_broad_filters(self):
        """genre_mode=strict still writes broad_filters via apply_mode_presets."""
        cfg = _build_playlists_cfg(cohesion="dynamic", sonic="dynamic", genre="strict")
        apply_mode_presets(cfg)
        ds_cfg = default_ds_config(
            cfg["cohesion_mode"],
            playlist_len=30,
            overrides=cfg["ds_pipeline"],
        )
        # apply_mode_presets writes broad_filters when genre_mode in {strict, narrow}.
        assert "rock" in ds_cfg.candidate.broad_filters

    def test_max_artist_fraction_uses_caller_override(self):
        """max_artist_fraction comes from caller (policy.py in normal flow), not per-mode dict."""
        cfg = _build_playlists_cfg(cohesion="strict", sonic="dynamic", genre="dynamic")
        cfg["ds_pipeline"]["candidate_pool"]["max_artist_fraction"] = 0.10
        apply_mode_presets(cfg)
        ds_cfg = default_ds_config(
            cfg["cohesion_mode"],
            playlist_len=30,
            overrides=cfg["ds_pipeline"],
        )
        assert ds_cfg.candidate.max_artist_fraction_final == 0.10

    def test_cohesion_still_drives_alpha_schedule(self):
        """Cohesion-owned settings (alpha schedule etc.) still change with cohesion_mode."""
        cfg_strict = _build_playlists_cfg(cohesion="strict", sonic="dynamic", genre="dynamic")
        cfg_dynamic = _build_playlists_cfg(cohesion="dynamic", sonic="dynamic", genre="dynamic")
        apply_mode_presets(cfg_strict)
        apply_mode_presets(cfg_dynamic)
        ds_strict = default_ds_config("strict", playlist_len=30, overrides=cfg_strict["ds_pipeline"])
        ds_dynamic = default_ds_config("dynamic", playlist_len=30, overrides=cfg_dynamic["ds_pipeline"])
        # Cohesion-owned: alpha_schedule changes
        assert ds_strict.construct.alpha_schedule == "constant"
        assert ds_dynamic.construct.alpha_schedule == "arc"
```

- [ ] **Step 2: Run the test**

```
pytest tests/unit/test_single_writer_settings.py -v
```

Expected: all 5 tests pass. If `test_sonic_mode_owns_min_sonic_similarity` fails, Task 2's dedup of `get_min_sonic_similarity()` regressed.

- [ ] **Step 3: Commit**

```
git add tests/unit/test_single_writer_settings.py
git commit -m "test(config): single-writer integration coverage for dedup

Verifies that after apply_mode_presets() + default_ds_config() run in
sequence, each setting has exactly one writer. Locks in the dedup from
Task 2 end-to-end.
"
```

---

## Task 4: Rewire `playlist_generator.py` to use `cohesion_mode`

Replace ~25 references to `ds_mode`/`ds_mode_override`/`ds_mode_effective` and use the new helper as the sole reader.

**Files:**
- Modify: `src/playlist_generator.py` (multiple sites; grep returns ~25 lines)

- [ ] **Step 1: Audit the call sites**

```
grep -n "ds_mode" src/playlist_generator.py
```

Confirm count is ~25-30 references. They cluster in:
- `create_playlist_for_artist()` signature + body (around line 1897)
- `create_playlist_from_seed_tracks()` signature + body (around line 2814)
- `create_playlist_for_genre()` signature + body (around line 3120)
- `create_playlist_batch()` signature + body (around line 1975)
- `_run_ds_pipeline()` or similar helper (around line 2937)
- Several `ds_mode_effective = ds_mode_override or ...` resolution sites

- [ ] **Step 2: Update imports in `src/playlist_generator.py`**

Add to the existing imports from `src.playlist.config`:

```python
from src.playlist.config import (
    # ...existing imports...
    resolve_cohesion_mode,
)
```

- [ ] **Step 3: Rename parameter on `create_playlist_for_artist()`**

Around line 1897, change:

```python
def create_playlist_for_artist(
    self,
    artist: str,
    track_count: int,
    *,
    track_title: Optional[str] = None,
    track_titles: Optional[List[str]] = None,
    dynamic: bool = False,
    ds_mode_override: Optional[str] = None,
    include_collaborations: bool = False,
) -> Optional[dict]:
```

to:

```python
def create_playlist_for_artist(
    self,
    artist: str,
    track_count: int,
    *,
    track_title: Optional[str] = None,
    track_titles: Optional[List[str]] = None,
    dynamic: bool = False,
    cohesion_mode_override: Optional[str] = None,
    include_collaborations: bool = False,
) -> Optional[dict]:
```

Inside the body, find:
```python
ds_mode_effective = ds_mode_override or ("dynamic" if dynamic else ds_cfg.get("mode", "dynamic"))
```

Replace with:
```python
playlists_cfg = self.config.get("playlists", {}) if hasattr(self, "config") else {}
cohesion_mode_effective = (
    cohesion_mode_override
    or ("dynamic" if dynamic else resolve_cohesion_mode(playlists_cfg))
)
```

(Note: confirm the attribute name on `self` that holds the config. Read `src/playlist_generator.py` around the `__init__` to confirm. If it's `self.config` use that; if it's `self.merged_config` or similar, adjust.)

Then rename every later `ds_mode_effective` reference in the same function to `cohesion_mode_effective`. There are around 8-12 references in this function.

- [ ] **Step 4: Repeat the rename for `create_playlist_from_seed_tracks()`**

Around line 2814. Same pattern: parameter rename + body variable rename + replace `ds_cfg.get("mode", "dynamic")` with `resolve_cohesion_mode(playlists_cfg)`.

The signature change:

```python
def create_playlist_from_seed_tracks(
    self,
    seed_tracks: List[str],
    *,
    track_count: int,
    dynamic: bool = False,
    cohesion_mode_override: Optional[str] = None,
    seed_track_ids: Optional[List[str]] = None,
) -> Optional[dict]:
```

Body — find and replace:
```python
ds_mode = ds_mode_override or (self.ds_mode_override if hasattr(self, 'ds_mode_override') and self.ds_mode_override else None)
if not ds_mode:
    ds_mode = "dynamic" if dynamic else ds_cfg.get('mode', 'dynamic')
```

with:
```python
playlists_cfg = self.config.get("playlists", {}) if hasattr(self, "config") else {}
cohesion_mode = cohesion_mode_override or (
    self.cohesion_mode_override
    if hasattr(self, 'cohesion_mode_override') and self.cohesion_mode_override
    else None
)
if not cohesion_mode:
    cohesion_mode = "dynamic" if dynamic else resolve_cohesion_mode(playlists_cfg)
```

Then rename every later `ds_mode` reference in this function to `cohesion_mode`.

- [ ] **Step 5: Repeat for `create_playlist_for_genre()` (around line 3120)**

Same pattern. Signature param rename, body variable renames, helper replacement.

- [ ] **Step 6: Repeat for `create_playlist_batch()` (around line 1975)**

Same pattern.

- [ ] **Step 7: Update `PlaylistGenerator.__init__` if it stores ds_mode_override**

Check around line 35-40:
```python
def __init__(self, config_path: str = "config.yaml", ds_mode_override: Optional[str] = None):
    # ...
    self.ds_mode_override = ds_mode_override
```

(Note: this `__init__` may live in `main_app.py` rather than `playlist_generator.py` — check both. From the file map, it's in `main_app.py:35`, which is handled in Task 6.)

If `PlaylistGenerator` has its own `__init__` accepting `ds_mode_override`, rename to `cohesion_mode_override`.

- [ ] **Step 8: Update log message strings**

In the log message at around line 771:
```python
"Invoking DS pipeline seed=%s mode=%s target_length=%d ..."
```

Leave the format string as-is; the value formatted in is just a string. Same for line 894 (`"DS pipeline success ... mode=%s ..."`). The log lines stay; only the variables passed in change name.

- [ ] **Step 9: Run the test suites that exercise this code**

```
pytest tests/test_artist_style.py tests/unit/test_cohesion_mode_resolution.py tests/unit/test_default_ds_config_dedup.py tests/unit/test_single_writer_settings.py -v
```

Expected: all pass. If tests in `test_artist_style.py` fail with `TypeError: ... got an unexpected keyword argument 'ds_mode_override'`, the test itself needs updating — defer to Task 10 (test renames). For now, if such a failure occurs, add a temporary `cohesion_mode_override` alias in the function signature like:

```python
# TEMPORARY shim, removed in Task 10
ds_mode_override: Optional[str] = None,  # legacy name
cohesion_mode_override: Optional[str] = None,
```

with body logic:
```python
if cohesion_mode_override is None and ds_mode_override is not None:
    cohesion_mode_override = ds_mode_override
```

This shim must be removed in Task 10's commit.

- [ ] **Step 10: Sanity check — make sure no stray `ds_mode` remain**

```
grep -n "ds_mode" src/playlist_generator.py
```

Expected: zero matches, OR only the temporary shim noted in Step 9.

- [ ] **Step 11: Commit**

```
git add src/playlist_generator.py
git commit -m "refactor(playlist_generator): rename ds_mode_override to cohesion_mode_override

Updates the four create_playlist_* signatures plus the internal
resolution helper to use the new cohesion_mode axis via the
resolve_cohesion_mode() helper. Behavior identical with default config
(cohesion_mode defaults to 'dynamic').

Part of sub-project A (cohesion mode refactor).
"
```

---

## Task 5: Rewire `worker.py` to read `cohesion_mode`

The GUI worker reads the mode from config and threads it through. Simple swap.

**Files:**
- Modify: `src/playlist_gui/worker.py` (lines 1135-1213, plus 1141-1148 area)

- [ ] **Step 1: Add import in `src/playlist_gui/worker.py`**

Near the existing imports from `src.playlist.config`, add:

```python
from src.playlist.config import resolve_cohesion_mode
```

- [ ] **Step 2: Replace the ds_mode-reading block (line 1135-1138)**

Find:
```python
        # Get DS pipeline mode from config
        # Note: genre_mode and sonic_mode are SEPARATE settings that control weighting,
        # not the pipeline algorithm. They are already applied to the config via overrides.
        ds_mode = config.get('playlists', {}).get('ds_pipeline', {}).get('mode', 'dynamic')
```

Replace with:
```python
        # Resolve cohesion_mode — drives pier-bridge beam tuning.
        # genre_mode/sonic_mode/pace_mode are independent axes that affect
        # candidate pool composition, not beam scoring.
        cohesion_mode = resolve_cohesion_mode(config.get('playlists', {}))
```

- [ ] **Step 3: Update the log line (around line 1155)**

Find:
```python
        emit_log("INFO", f"DS pipeline mode: {ds_mode}")
```

Replace with:
```python
        emit_log("INFO", f"Cohesion mode: {cohesion_mode}")
```

- [ ] **Step 4: Update the generation call sites (lines 1161, 1170-1171, 1179-1180, 1188-1189, 1197-1198, 1204-1205)**

Six call sites use `ds_mode`. Replace every `ds_mode` reference with `cohesion_mode`:

```python
        emit_log("INFO", f"Running playlist generation with mode={cohesion_mode}")

        if mode == "artist" and artist:
            playlist_data = generator.create_playlist_for_artist(
                artist,
                track_count,
                track_title=track_title,
                track_titles=seed_tracks,
                dynamic=(cohesion_mode == "dynamic"),
                cohesion_mode_override=cohesion_mode,
                include_collaborations=include_collaborations,
            )
        elif mode == "seeds" and seed_tracks:
            playlist_data = generator.create_playlist_from_seed_tracks(
                seed_tracks,
                track_count=track_count,
                dynamic=(cohesion_mode == "dynamic"),
                cohesion_mode_override=cohesion_mode,
                seed_track_ids=seed_track_ids,
            )
        elif mode == "artist" and seed_tracks:
            playlist_data = generator.create_playlist_from_seed_tracks(
                seed_tracks,
                track_count=track_count,
                dynamic=(cohesion_mode == "dynamic"),
                cohesion_mode_override=cohesion_mode,
                seed_track_ids=seed_track_ids,
            )
        elif mode == "genre" and genre:
            playlist_data = generator.create_playlist_for_genre(
                genre,
                track_count,
                dynamic=(cohesion_mode == "dynamic"),
                cohesion_mode_override=cohesion_mode,
            )
        elif mode == "history":
            playlists = generator.create_playlist_batch(
                1,
                dynamic=(cohesion_mode == "dynamic"),
                cohesion_mode_override=cohesion_mode,
            )
            playlist_data = playlists[0] if playlists else None
```

- [ ] **Step 5: Sanity check — no stray ds_mode**

```
grep -n "ds_mode" src/playlist_gui/worker.py
```

Expected: zero matches.

- [ ] **Step 6: Run GUI worker tests**

```
pytest tests/unit/test_gui_generation_validation.py tests/unit/test_generate_panel.py -v
```

Expected: pass. If they fail with `TypeError` about unexpected `cohesion_mode_override` argument, that's because tests still pass `ds_mode_override` — defer that to Task 10.

- [ ] **Step 7: Commit**

```
git add src/playlist_gui/worker.py
git commit -m "refactor(worker): read cohesion_mode via resolve_cohesion_mode helper

Drops the manual ds_pipeline.mode lookup in favor of the new helper.
Worker now logs 'Cohesion mode: <X>' and passes cohesion_mode_override
to all five generation entry points.
"
```

---

## Task 6: Rename CLI flag in `main_app.py`

CLI flag rename from `--ds-mode` to `--cohesion-mode`. No backward-compat alias per spec (single user).

**Files:**
- Modify: `main_app.py` (lines 35, 38, 191, 311, 396, 496, 516, 573, 693, 764)

- [ ] **Step 1: Add import**

Add to existing imports:

```python
from src.playlist.config import resolve_cohesion_mode
```

- [ ] **Step 2: Update `__init__` (line 35)**

Find:
```python
def __init__(self, config_path: str = "config.yaml", ds_mode_override: Optional[str] = None):
    # ...
    self.ds_mode_override = ds_mode_override
```

Replace with:
```python
def __init__(self, config_path: str = "config.yaml", cohesion_mode_override: Optional[str] = None):
    # ...
    self.cohesion_mode_override = cohesion_mode_override
```

- [ ] **Step 3: Update all `self.ds_mode_override` references**

Lines 191, 311, 396, 496, 516 reference `self.ds_mode_override`. Rename all to `self.cohesion_mode_override`. For lines 311 and 396 that set `report.mode = self.ds_mode_override or "dynamic"`, change to `report.mode = self.cohesion_mode_override or "dynamic"`.

- [ ] **Step 4: Update argparse flag definition (around line 573)**

Find:
```python
    parser.add_argument(
        '--ds-mode',
        choices=['narrow', 'dynamic', 'discover', 'sonic_only'],
        help="Select DS pipeline mode (default from config playlists.ds_pipeline.mode)",
    )
```

Replace with:
```python
    parser.add_argument(
        '--cohesion-mode',
        choices=['strict', 'narrow', 'dynamic', 'discover'],
        help="Select cohesion mode (default from config playlists.cohesion_mode)",
    )
```

- [ ] **Step 5: Update the `getattr` call sites (lines 693, 764)**

Line 693:
```python
ds_mode_override=getattr(args, 'ds_mode', None),
```

Replace with:
```python
cohesion_mode_override=getattr(args, 'cohesion_mode', None),
```

Line 764:
```python
dynamic_flag = getattr(args, "ds_mode", None) == "dynamic"
```

Replace with:
```python
dynamic_flag = getattr(args, "cohesion_mode", None) == "dynamic"
```

- [ ] **Step 6: Sanity check**

```
grep -n "ds_mode" main_app.py
```

Expected: zero matches.

- [ ] **Step 7: Smoke test the CLI**

```
python main_app.py --help
```

Expected: `--cohesion-mode` appears in the output with choices `strict, narrow, dynamic, discover`. `--ds-mode` does NOT appear. Exit code 0.

- [ ] **Step 8: Commit**

```
git add main_app.py
git commit -m "refactor(cli): rename --ds-mode flag to --cohesion-mode

Drops the legacy --ds-mode flag (no deprecation alias, single-user
project). Choices updated: strict/narrow/dynamic/discover. The
sonic_only mode is no longer exposed — use genre_mode=off + sonic_mode=
dynamic in the GUI for the same effect.
"
```

---

## Task 7: Update `settings_schema.py` Advanced Panel entry

The Advanced Panel currently exposes `playlists.ds_pipeline.mode`. Rename the key path and update choices.

**Files:**
- Modify: `src/playlist_gui/config/settings_schema.py:400-409`
- Modify: `src/playlist_gui/config/config_model.py:100` (docstring example only)

- [ ] **Step 1: Update the SettingSpec in `settings_schema.py`**

Find (around line 400):
```python
    SettingSpec(
        key_path="playlists.ds_pipeline.mode",
        label="Pipeline mode",
        setting_type=SettingType.CHOICE,
        group="Pipeline",
        default="dynamic",
        choices=["narrow", "dynamic", "discover", "sonic_only"],
        tooltip="Overall pipeline behavior",
        description="Presets that adjust multiple settings: 'narrow' = focused, stay close to seed; 'dynamic' = balanced mix; 'discover' = explore further from seed; 'sonic_only' = ignore genres entirely, pure audio matching."
    ),
```

Replace with:
```python
    SettingSpec(
        key_path="playlists.cohesion_mode",
        label="Cohesion mode",
        setting_type=SettingType.CHOICE,
        group="Pipeline",
        default="dynamic",
        choices=["strict", "narrow", "dynamic", "discover"],
        tooltip="Overall beam tightness (pier-bridge tuning)",
        description="Drives how tightly the beam search bridges between pier tracks: 'strict' = ultra-cohesive transitions; 'narrow' = tight; 'dynamic' = balanced (default); 'discover' = looser transitions for exploration. Independent of Genre, Sonic, and Pace which control candidate pool composition."
    ),
```

- [ ] **Step 2: Update the docstring example in `config_model.py:100`**

Find:
```python
            key_path: Dot-separated path (e.g., "playlists.ds_pipeline.mode")
```

Replace with:
```python
            key_path: Dot-separated path (e.g., "playlists.cohesion_mode")
```

- [ ] **Step 3: Run the GUI config tests**

```
pytest tests/unit/test_gui_config.py -v
```

Expected: pass.

- [ ] **Step 4: Commit**

```
git add src/playlist_gui/config/settings_schema.py src/playlist_gui/config/config_model.py
git commit -m "refactor(gui-config): point Advanced Panel at playlists.cohesion_mode

Updates the SettingSpec key_path and choices to match the new cohesion
axis. Widget remains a dropdown for now; sub-project B will swap to a
slider as part of the four-slider redesign.
"
```

---

## Task 8: Update audit/reporter `ds_mode` field renames

Audit context objects and log helpers carry the mode value through to markdown reports.

**Files:**
- Modify: `src/playlist/reporter.py` (lines ~450-471)
- Modify: `src/playlist/run_audit.py` (lines 38, 145)
- Modify: `src/playlist/pipeline/audit_emitter.py` (if it has matching field — verify with grep)

- [ ] **Step 1: Verify the audit_emitter references**

```
grep -n "ds_mode" src/playlist/pipeline/audit_emitter.py
```

If matches are present, include them in this task. If zero matches, ignore this file.

- [ ] **Step 2: Rename in `src/playlist/run_audit.py`**

Line 38 — context dataclass field:
```python
    ds_mode: str
```
Replace with:
```python
    cohesion_mode: str
```

Line 145 — markdown output:
```python
    lines.append(f"- ds_mode: `{context.ds_mode}`")
```
Replace with:
```python
    lines.append(f"- cohesion_mode: `{context.cohesion_mode}`")
```

- [ ] **Step 3: Find callers that construct the audit context**

```
grep -rn "ds_mode=" src/ scripts/
```

For every callsite that constructs the audit context with `ds_mode=<value>`, rename to `cohesion_mode=<value>`. Same value, just renamed kwarg.

- [ ] **Step 4: Rename in `src/playlist/reporter.py`**

Identify the function containing `last_ds_mode` parameter:

```
grep -n "last_ds_mode" src/playlist/reporter.py
```

Open that function. Rename the parameter `last_ds_mode` → `last_cohesion_mode`, the variable use `last_ds_mode is not None` → `last_cohesion_mode is not None`, and the f-string formatting where it appears (keep the output literal `mode=` — that's the audit context label, not the variable name):

Before:
```python
def <function_name>(
    lines: List[str],
    *,
    last_ds_mode: Optional[str] = None,
    # ...other params...
) -> None:
    pipeline_ctx: List[str] = []
    # ...
    if last_ds_mode is not None:
        pipeline_ctx.append(f"mode={last_ds_mode}")
```

After:
```python
def <function_name>(
    lines: List[str],
    *,
    last_cohesion_mode: Optional[str] = None,
    # ...other params...
) -> None:
    pipeline_ctx: List[str] = []
    # ...
    if last_cohesion_mode is not None:
        pipeline_ctx.append(f"mode={last_cohesion_mode}")
```

Then find callers of this function:

```
grep -rn "last_ds_mode=" src/ scripts/
```

Rename every `last_ds_mode=<value>` keyword arg to `last_cohesion_mode=<value>`. Same value, just renamed.

- [ ] **Step 5: Update audit_emitter if needed**

If Step 1 found matches in `audit_emitter.py`, apply analogous renames.

- [ ] **Step 6: Sanity check**

```
grep -rn "ds_mode" src/playlist/
```

Expected: zero matches across `src/playlist/`. If any remain, address them now.

- [ ] **Step 7: Run audit-touching tests**

```
pytest tests/ -k "audit or report" -v
```

Expected: pass. If no such tests exist, skip.

- [ ] **Step 8: Commit**

```
git add src/playlist/reporter.py src/playlist/run_audit.py src/playlist/pipeline/audit_emitter.py
git commit -m "refactor(audit): rename ds_mode field to cohesion_mode

Audit context objects and the reporter helper now use cohesion_mode
throughout. Markdown audit reports will show 'cohesion_mode: dynamic'
instead of 'ds_mode: dynamic'.
"
```

---

## Task 9: Update pipeline internals + diagnostic scripts

Sweep the remaining `ds_mode` references in pipeline internals and scripts.

**Files:**
- Modify: `src/playlist/pipeline/core.py` (if it has ds_mode references — verify)
- Modify: `src/playlist/pipeline/pier_bridge_overrides.py` (verify)
- Modify: `src/playlist/__init__.py` (verify — likely just re-exports)
- Modify: `src/playlist/pier_bridge/config.py` (verify)
- Modify: `scripts/diagnose_sonic_floor.py`, `diagnose_artist_style.py`, `diagnose_candidate_scores.py`, `sweep_pier_bridge_dials.py`, `run_dj_*.py` (all scripts that grep returned)

- [ ] **Step 1: Audit remaining references**

```
grep -rn "ds_mode\|ds_pipeline.*mode" src/playlist/pipeline/ src/playlist/__init__.py src/playlist/pier_bridge/ scripts/
```

For each match, determine:
- Is it a variable name (rename to `cohesion_mode`)?
- Is it a parameter name (rename to `cohesion_mode`)?
- Is it a string literal `"ds_pipeline.mode"` (replace with `"cohesion_mode"`)?
- Is it a dict lookup like `config['ds_pipeline']['mode']` (replace with `resolve_cohesion_mode(playlists_cfg)`)?

- [ ] **Step 2: Apply renames in pipeline internals**

For each file, do the rename in place. If a file reads the config directly to get the mode, switch to:

```python
from src.playlist.config import resolve_cohesion_mode
# ...
cohesion_mode = resolve_cohesion_mode(playlists_cfg)
```

If a file accepts `ds_mode` as a parameter, rename to `cohesion_mode` and propagate through callers.

- [ ] **Step 3: Apply renames in diagnostic scripts**

For each script in `scripts/`, do find/replace of `ds_mode` → `cohesion_mode`, `ds_mode_override` → `cohesion_mode_override`. Update any `--ds-mode` CLI flags within scripts to `--cohesion-mode`. Update any config dict construction to use `cohesion_mode` at the playlists root.

- [ ] **Step 4: Smoke-run one diagnostic to confirm it imports cleanly**

```
python scripts/diagnose_artist_style.py --help
```

Expected: help message displays without import errors. (Actually running the diagnostic against the artifact is out of scope; just verify imports.)

- [ ] **Step 5: Final sanity check across src/ and scripts/**

```
grep -rn "ds_mode\|ds_pipeline.*mode" src/ scripts/
```

Expected: zero matches.

- [ ] **Step 6: Commit**

```
git add src/playlist/ scripts/
git commit -m "refactor: propagate cohesion_mode rename through pipeline + scripts

Sweeps remaining ds_mode references in pipeline internals, diagnostic
scripts, and sweep tools. Variables, parameters, and config lookups
all use cohesion_mode now.
"
```

---

## Task 10: Update tests (mechanical renames + remove temporary shim)

Sweep all test files for `ds_mode` references and update.

**Files:**
- Modify: `tests/test_artist_style.py`
- Modify: `tests/unit/test_mode_threshold_resolution.py`
- Modify: `tests/unit/test_edge_repair.py`
- Modify: `tests/unit/test_generate_panel.py`
- Modify: `tests/unit/test_gui_policy.py`
- Modify: `tests/unit/test_gui_generation_validation.py`
- Modify: `tests/unit/test_gui_config.py`
- Possibly modify: `src/playlist_generator.py` (remove the temporary shim from Task 4 Step 9 if it was added)

- [ ] **Step 1: Audit references**

```
grep -rn "ds_mode\|ds_pipeline.*mode" tests/
```

For each match:
- `ds_mode_override=` keyword arg → rename to `cohesion_mode_override=`
- Dict construction like `{"ds_pipeline": {"mode": "narrow"}}` → rename to top-level `{"cohesion_mode": "narrow"}` if testing config flow; otherwise keep `{"ds_pipeline": {...}}` only if the test specifically tests the legacy-key warning path
- Variable names containing `ds_mode` → rename to `cohesion_mode`

- [ ] **Step 2: Apply renames file by file**

Work through each test file in the list. For each one, run the file's tests after editing to confirm:

```
pytest <file> -v
```

Some tests may need their fixtures updated to use the new top-level `cohesion_mode` key instead of the nested `ds_pipeline.mode`.

- [ ] **Step 3: If Task 4 added a temporary shim, remove it now**

If `src/playlist_generator.py` has any signature that still accepts `ds_mode_override` as a shim, delete the legacy parameter and the if-block that maps it. The function signature should only have `cohesion_mode_override`.

- [ ] **Step 4: Run the full test suite**

```
pytest tests/ -v
```

Expected: all pass. Note: `tests/conftest.py:60-70`'s `qtbot` stub may cause GUI tests to pass trivially — that's a known limitation flagged in `CLAUDE.md` and out of scope for this plan.

- [ ] **Step 5: Final sanity check across the repo**

```
grep -rn "ds_mode\|ds_pipeline.*mode" src/ tests/ scripts/ main_app.py
```

Expected: zero matches except possibly in `resolve_cohesion_mode()`'s warning string (which intentionally references the legacy key).

- [ ] **Step 6: Commit**

```
git add tests/ src/playlist_generator.py
git commit -m "test: rename ds_mode_override to cohesion_mode_override across tests

Final mechanical sweep. Test fixtures that constructed config dicts with
playlists.ds_pipeline.mode now use playlists.cohesion_mode at the root.
Removes the temporary parameter shim from earlier tasks if present.
"
```

---

## Task 11: Update config files

Drop `playlists.ds_pipeline.mode` from both config files, add `playlists.cohesion_mode: dynamic` at the playlists root.

**Files:**
- Modify: `config.yaml` (gitignored, local only — NOT committed)
- Modify: `config.example.yaml`

- [ ] **Step 1: Update local `config.yaml`**

In `config.yaml` (gitignored — your local config), add at the `playlists` root level (around line 47, near `pace_mode`):

```yaml
  pace_mode: dynamic
  cohesion_mode: dynamic
  pipeline: ds  # legacy | ds
```

Remove the `mode: dynamic` line from inside the `ds_pipeline:` block (around line 51):

```yaml
  ds_pipeline:
    artifact_path: data/artifacts/beat3tower_32k/data_matrices_step1.npz
    # mode: dynamic   # REMOVED — replaced by playlists.cohesion_mode
    random_seed: 0
    enable_logging: true
```

Just delete the `mode: dynamic` line; do not leave a placeholder comment.

- [ ] **Step 2: Update `config.example.yaml` analogously**

Make the same two edits in `config.example.yaml`:
- Add `cohesion_mode: dynamic` at the playlists root, near `pace_mode`
- Delete the `mode: <value>` line from the `ds_pipeline:` block

Also update any comments that reference `playlists.ds_pipeline.mode` to reference `playlists.cohesion_mode` instead. Search for `ds_pipeline.mode` mentions and update them.

- [ ] **Step 3: Smoke test — generate a playlist with the new config**

```
python main_app.py --artist "The Sundays" --tracks 30 --log-level INFO 2>&1 | head -50
```

Expected: log shows `Cohesion mode: dynamic` (not `DS pipeline mode: dynamic`). No `ds_pipeline.mode is no longer used` warning (because we removed the stale key from config.yaml). Generation completes successfully.

- [ ] **Step 4: Commit (config.example.yaml only — config.yaml is gitignored)**

```
git add config.example.yaml
git commit -m "config(example): replace ds_pipeline.mode with cohesion_mode

Drops the GUI-inaccessible ds_pipeline.mode key in favor of the new
cohesion_mode axis at the playlists root level. Existing per-mode
pier_bridge knobs (bridge_floor_strict, weight_bridge_narrow, all
soft_genre_penalty_*_<mode>) are now LIVE — they fire when
cohesion_mode is set to the matching value.
"
```

---

## Task 12: Update documentation

Sweep the docs that reference the old key.

**Files:**
- Modify: `docs/PLAYLIST_ORDERING_TUNING.md`
- Modify: `docs/CONFIG.md`
- Modify: `docs/GOLDEN_COMMANDS.md`
- Modify: `docs/TECHNICAL_PLAYLIST_GENERATION_FLOW.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Audit references**

```
grep -rn "ds_pipeline.mode\|--ds-mode\|ds_mode" docs/ CLAUDE.md
```

- [ ] **Step 2: Update `docs/PLAYLIST_ORDERING_TUNING.md` (Knob 6 section)**

Find the caveat text that says something like "ds_pipeline.mode is fixed at dynamic so per-mode knobs are dead":

The exact phrasing — find by searching for `ds_pipeline.mode` in the file. Replace with text that reflects the new model:

```markdown
> **Note:** Per-mode knobs (`bridge_floor_strict`, `soft_genre_penalty_threshold_narrow`, etc.) are resolved by `playlists.cohesion_mode`. With the default `cohesion_mode: dynamic`, only `*_dynamic` keys apply. Set `cohesion_mode` to `strict`/`narrow`/`discover` to activate those per-mode values.
```

- [ ] **Step 3: Update `docs/CONFIG.md`**

Find every `playlists.ds_pipeline.mode` reference. Replace with `playlists.cohesion_mode`. Update accompanying descriptions if they reference the old name.

- [ ] **Step 4: Update `docs/GOLDEN_COMMANDS.md`**

Find any `--ds-mode` occurrences. Replace with `--cohesion-mode`. Update choice lists (`narrow|dynamic|discover|sonic_only` → `strict|narrow|dynamic|discover`).

- [ ] **Step 5: Update `docs/TECHNICAL_PLAYLIST_GENERATION_FLOW.md`**

Find any mode-resolution explanation. Update to describe the four-axis model: `cohesion_mode` (beam tuning), `genre_mode` (genre pool gating), `sonic_mode` (sonic pool gating), `pace_mode` (rhythm gating).

- [ ] **Step 6: Update `CLAUDE.md`**

In the "Design principles" section, find any wording that conflates `ds_pipeline.mode` with the GUI modes (the project-specific gotchas list mentions this issue). Update to:

```markdown
- **`cohesion_mode` drives the beam, the other three slider axes drive pool composition.** All four axes (`cohesion_mode`, `genre_mode`, `sonic_mode`, `pace_mode`) live at `playlists.<axis>` in config.yaml. The pier-bridge per-mode knobs (`bridge_floor_<mode>`, `weight_bridge_<mode>`, `soft_genre_penalty_*_<mode>`) are keyed by `cohesion_mode`.
```

Delete or update the "DS mode coupling" gotcha if it's still listed.

- [ ] **Step 7: Sanity check**

```
grep -rn "ds_pipeline.mode\|--ds-mode" docs/ CLAUDE.md
```

Expected: zero matches.

- [ ] **Step 8: Commit**

```
git add docs/ CLAUDE.md
git commit -m "docs: update for cohesion_mode refactor

PLAYLIST_ORDERING_TUNING.md Knob 6 caveat, CONFIG.md key references,
GOLDEN_COMMANDS.md CLI flag, TECHNICAL_PLAYLIST_GENERATION_FLOW.md mode
resolution, and CLAUDE.md design principles all updated for the new
four-axis model (cohesion + genre + sonic + pace).
"
```

---

## Task 13: Regression validation + cleanup

Verify nothing broke at runtime, confirm the per-mode pier-bridge knobs actually fire when cohesion_mode is non-dynamic, and clean up obsolete memory.

**Files:**
- Delete: `~/.claude/projects/C--Users-Dylan-Desktop-PLAYLIST-GENERATOR-V3/memory/project_ds_mode_coupling.md`
- Update: `~/.claude/projects/C--Users-Dylan-Desktop-PLAYLIST-GENERATOR-V3/memory/MEMORY.md` (remove the line pointing to the deleted memory)

- [ ] **Step 1: Run the golden test suite**

```
pytest -m golden -v
```

Expected: pass. If a golden test fails, investigate — the dedup may have changed effective config in an unexpected way.

- [ ] **Step 2: Run the full unit test suite**

```
pytest -m "not slow" -v
```

Expected: pass.

- [ ] **Step 3: Manual regression — regenerate the Sundays playlist**

```
python main_app.py --artist "The Sundays" --tracks 30 --log-level INFO 2>&1 > C:\Users\Dylan\Desktop\tmp\sundays_log_post_refactor.txt
```

Compare against `C:\Users\Dylan\Desktop\tmp\sundays_log3.txt` (the pre-refactor calibrated baseline):

```
python -c "import difflib; a=open(r'C:\Users\Dylan\Desktop\tmp\sundays_log3.txt').readlines(); b=open(r'C:\Users\Dylan\Desktop\tmp\sundays_log_post_refactor.txt').readlines(); print(''.join(difflib.unified_diff(a, b, lineterm='')))"
```

Expected differences:
- `DS pipeline mode: dynamic` → `Cohesion mode: dynamic` (log line change)
- `ds_mode=dynamic` → `cohesion_mode=dynamic` (in any audit metadata)

Expected to be IDENTICAL:
- The tracklist (all 30 tracks, same order)
- Transition stats (`min_transition=0.541, mean_transition=0.807, below_floor=0`)
- G/S stats per edge

If the tracklist differs, the dedup accidentally changed behavior — investigate before continuing.

- [ ] **Step 4: Per-mode smoke test — regenerate with cohesion_mode=strict**

Temporarily edit `config.yaml` to set `cohesion_mode: strict`. Then:

```
python main_app.py --artist "The Sundays" --tracks 30 --log-level INFO 2>&1 > C:\Users\Dylan\Desktop\tmp\sundays_log_strict.txt
```

Inspect the log for:

```
INFO: Cohesion mode: strict
INFO: Pier-bridge tuning resolved: mode=strict transition_floor=0.20 bridge_floor=0.10 weight_bridge=0.7 weight_transition=0.3 genre_tiebreak_weight=0.05 genre_penalty_threshold=0.82 genre_penalty_strength=0.40
```

The key indicators:
- `bridge_floor=0.10` (the strict value from `config.yaml`, was 0.02 in dynamic)
- `weight_bridge=0.7` (was 0.6)
- `genre_penalty_threshold=0.82` (was 0.73)
- `genre_penalty_strength=0.40` (was 0.15)

If any of these still show dynamic values, the per-mode resolution did not actually rewire to cohesion_mode — investigate.

Revert `config.yaml` back to `cohesion_mode: dynamic` afterward.

- [ ] **Step 5: Verify the stale-key warning fires**

Temporarily add `mode: dynamic` back to the `ds_pipeline:` block in `config.yaml`. Run:

```
python main_app.py --artist "The Sundays" --tracks 30 --log-level WARNING 2>&1 | grep "ds_pipeline.mode"
```

Expected: one warning line `playlists.ds_pipeline.mode is no longer used; remove from config. Use playlists.cohesion_mode instead.`

Remove the stale key from `config.yaml` again.

- [ ] **Step 6: Delete the obsolete memory file**

```
rm ~/.claude/projects/C--Users-Dylan-Desktop-PLAYLIST-GENERATOR-V3/memory/project_ds_mode_coupling.md
```

(On Windows PowerShell: `Remove-Item "$env:USERPROFILE\.claude\projects\C--Users-Dylan-Desktop-PLAYLIST-GENERATOR-V3\memory\project_ds_mode_coupling.md"`)

- [ ] **Step 7: Remove the pointer from MEMORY.md**

Edit `~/.claude/projects/C--Users-Dylan-Desktop-PLAYLIST-GENERATOR-V3/memory/MEMORY.md` and delete this line:

```
- [DS mode / GUI mode coupling](project_ds_mode_coupling.md) — ds_pipeline.mode (config.yaml) is NOT linked to GUI genre_mode/sonic_mode; per-mode pier-bridge knobs (_narrow, _strict) are dead code until refactored
```

- [ ] **Step 8: Final commit**

The memory file deletion is local-only (outside the repo), so no commit needed for that. Confirm the working tree is clean:

```
git status
```

Expected: clean (all changes already committed in earlier tasks).

If any uncommitted changes remain from earlier tasks (forgotten file), commit them with a descriptive message.

---

## Acceptance Criteria

When all tasks above complete, the following must hold:

1. `grep -rn "ds_mode\|ds_pipeline.*mode" src/ tests/ scripts/ main_app.py config.example.yaml` returns zero matches except inside the warning string of `resolve_cohesion_mode()`.
2. `pytest -m "not slow" -v` passes.
3. `pytest -m golden -v` passes.
4. The Sundays playlist generated post-refactor matches the pre-refactor `sundays_log3.txt` tracklist byte-for-byte.
5. Setting `cohesion_mode: strict` in config.yaml causes the pier-bridge resolved-thresholds log line to show `bridge_floor=0.10 weight_bridge=0.7 genre_penalty_threshold=0.82 genre_penalty_strength=0.40` (the per-mode strict values from config.yaml's `pier_bridge:` block become active).
6. Leaving stale `ds_pipeline.mode` in config.yaml triggers exactly one WARNING log per run, key is ignored.
7. CLI: `python main_app.py --help` shows `--cohesion-mode {strict,narrow,dynamic,discover}` and no `--ds-mode`.
8. `MEMORY.md` no longer references the deleted `project_ds_mode_coupling.md`.
