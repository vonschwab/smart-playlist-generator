# Cohesion Mode Refactor — Design Spec

**Status:** Draft for review
**Date:** 2026-05-23
**Sub-project:** A (of three: A backend, B advanced settings GUI, C presets)
**Scope:** Backend only — GUI work deferred to sub-project B

## Problem

The playlist generator has four mode axes today, but only three are user-reachable:

| Axis | Source | Status |
|---|---|---|
| `playlists.ds_pipeline.mode` | `config.yaml` only | **Frozen at `dynamic`** since GUI inception, no GUI surface, never touched. Per-mode pier-bridge knobs (`bridge_floor_strict`, `weight_bridge_narrow`, all `soft_genre_penalty_*_<mode>`) are dead code in normal use. |
| `playlists.genre_mode` | GUI via `policy.py` | Live |
| `playlists.sonic_mode` | GUI via `policy.py` | Live |
| `playlists.pace_mode` | GUI via `policy.py` | Live |

Three settings have **two writers** that compete:
- `min_sonic_similarity` — written by `sonic_mode` preset *and* defaulted by `get_min_sonic_similarity(ds_mode)`
- `max_artist_fraction` — written by `policy.py` (artist_presence) *and* defaulted by `default_ds_config(ds_mode)`
- `broad_filters` — written by `apply_mode_presets()` (genre_mode) *and* defaulted by `default_ds_config(ds_mode)`

The mode-preset writer wins in practice (it runs first and writes more aggressively), but the duplicate ds_mode defaults survive everywhere the presets don't reach — primarily in pier-bridge tuning.

## Goal

Introduce a new `cohesion_mode` axis at the same level as `genre_mode`/`sonic_mode`/`pace_mode`, replace `ds_pipeline.mode` entirely, and dedup the three settings to one writer each. After this work the per-mode pier-bridge knobs become live and keyed by `cohesion_mode`, ready for sub-project B to add a Cohesion slider.

This is **backend-only**. No GUI changes ship in A; `cohesion_mode` is sourced from `config.yaml` only. Sub-project B adds the slider and policy wiring.

## Non-goals

- Adding a GUI slider for Cohesion (sub-project B)
- Refactoring the Advanced Settings panel (sub-project B)
- Adding a presets / state persistence system (sub-project C)
- Backward-compatible mapping of `ds_pipeline.mode` (you are the only user; rip the band-aid)
- Tuning any per-mode pier-bridge values — those stay as currently calibrated

## Design

### 1. Architecture

```yaml
playlists:
  cohesion_mode: dynamic    # NEW — drives beam tuning (pier-bridge + construction)
  genre_mode: dynamic       # existing — drives genre pool gating + weights
  sonic_mode: dynamic       # existing — drives sonic pool gating + weights
  pace_mode: dynamic        # existing — drives rhythm/BPM gating
  ds_pipeline:
    # mode: dynamic         # REMOVED
    pier_bridge:
      bridge_floor_strict: 0.10      # already exists, was dead; now LIVE via cohesion_mode
      bridge_floor_narrow: 0.05      # ditto
      bridge_floor_dynamic: 0.02     # ditto
      weight_bridge_strict: 0.7      # ditto
      # ...etc all per-mode pier_bridge knobs
```

**Single-writer principle.** Each setting has exactly one writer:

| Setting | Sole writer (after A) | Removed from |
|---|---|---|
| `pier_bridge.bridge_floor`, `weight_bridge/transition`, `transition_floor`, `soft_genre_penalty_*` | Resolved per `cohesion_mode` via `_resolve_mode_number_with_source` | (already this way; just keyed by cohesion_mode now) |
| `candidate_pool.min_sonic_similarity` | `apply_mode_presets()` from `sonic_mode` | per-mode default block removed from `get_min_sonic_similarity()` |
| `candidate_pool.max_artist_fraction` | `policy.py` from artist_presence dropdown (default 0.125 if unset) | per-mode dict removed from `default_ds_config()` |
| `candidate_pool.broad_filters` | `apply_mode_presets()` from `genre_mode` | per-mode default removed from `default_ds_config()` |
| `construct.alpha*`, `alpha_schedule`, `local_top_m`, `target_artists`, `candidates_per_artist`, `max_pool_size`, `similarity_floor` | `default_ds_config()` keyed by `cohesion_mode` | (no change in writer, just key source) |

Pace remains a fully separate axis with its own writer (`resolve_pace_mode()`) — it controls BPM/rhythm gates that `cohesion_mode` doesn't touch. The four axes are genuinely independent after this.

### 2. Components / Files Touched

**Core resolver layer:**
- `src/playlist/config.py`
  - Update `Mode = Literal["strict", "narrow", "dynamic", "discover"]` (add `strict`, drop `sonic_only`)
  - Add `resolve_cohesion_mode(playlists_cfg: dict) -> Mode` helper
  - In `default_ds_config()`: delete per-mode dicts for `max_artist_fraction_final`, `broad_filters_cfg` fallback. Make `min_sonic_similarity` come only from caller's `candidate_pool` overrides.
  - In `get_min_sonic_similarity()`: remove the per-mode default block; return `None` if no override is set (apply_mode_presets is now the only writer).
- `src/playlist/mode_presets.py` — no change. `apply_mode_presets()` already owns `min_sonic_similarity` and `broad_filters` via the genre/sonic presets; we're deleting the competing writer.

**Pipeline orchestration:**
- `src/playlist_generator.py`
  - Every `ds_cfg.get("mode", "dynamic")` becomes `resolve_cohesion_mode(playlists_cfg)`
  - Variable rename: `ds_mode_effective` → `cohesion_mode_effective`
  - Parameter rename: `ds_mode_override` → `cohesion_mode_override` in `create_playlist_for_artist`, `create_playlist_from_seed_tracks`, `create_playlist_for_genre`, `create_playlist_batch`

**Worker / CLI:**
- `src/playlist_gui/worker.py` — line 1138 becomes `cohesion_mode = resolve_cohesion_mode(config.get('playlists', {}))`. Log line at 1155 becomes `Cohesion mode: <X>`. The `ds_mode == "dynamic"` checks become `cohesion_mode == "dynamic"`.
- `main_app.py` — CLI flag `--ds-mode` → `--cohesion-mode`. Parameter plumbing rename to match.

**Settings schema:**
- `src/playlist_gui/config/settings_schema.py:400-409` — `SettingSpec` update: `key_path: "playlists.cohesion_mode"`, `choices: ["strict", "narrow", "dynamic", "discover"]`. Widget shape stays (dropdown) — sub-project B replaces it with a slider.
- `src/playlist_gui/config/config_model.py:100` — docstring example rename.

**Audit / logging:**
- `src/playlist/reporter.py` — `last_ds_mode` parameter rename.
- `src/playlist/run_audit.py` — `ds_mode` field on audit context object renamed.
- `src/playlist/pipeline/audit_emitter.py` — same.

**Pipeline internals:**
- `src/playlist/pipeline/core.py`, `src/playlist/pipeline/pier_bridge_overrides.py` — call sites accept renamed parameter.
- `src/playlist/__init__.py` — update any re-exports if present.
- `src/playlist/pier_bridge_builder.py` — confirmed no direct config read; mode arrives by parameter. No change beyond following the parameter rename through call sites.

**Scripts:**
- `scripts/diagnose_sonic_floor.py`, `diagnose_artist_style.py`, `diagnose_candidate_scores.py`, `sweep_pier_bridge_dials.py`, `run_dj_*.py` — mechanical rename.

**Tests:**
- `tests/test_artist_style.py`, `tests/unit/test_mode_threshold_resolution.py`, `test_edge_repair.py`, `test_generate_panel.py`, `test_gui_policy.py`, `test_gui_generation_validation.py`, `test_gui_config.py` — find/replace `ds_pipeline.mode` → `cohesion_mode`, `ds_mode_override` → `cohesion_mode_override`.

**Config files:**
- `config.yaml` (local, gitignored) — drop `playlists.ds_pipeline.mode`, add `playlists.cohesion_mode: dynamic`.
- `config.example.yaml` — same change, update surrounding comments.

**Docs:**
- `docs/PLAYLIST_ORDERING_TUNING.md` — Knob 6 caveat about "ds_pipeline.mode is fixed at dynamic" updated to reflect new wiring.
- `docs/TECHNICAL_PLAYLIST_GENERATION_FLOW.md` — mode resolution section updated.
- `docs/CONFIG.md` — key references updated.
- `docs/GOLDEN_COMMANDS.md` — CLI flag rename noted.
- `CLAUDE.md` — design principles wording updated where it conflates `ds_pipeline.mode` with GUI modes.

**Memory:**
- `~/.claude/projects/.../memory/project_ds_mode_coupling.md` — deleted (the problem it describes is fixed).

**Estimated touch:** ~25-30 files. Most are mechanical renames; substantive changes concentrated in `config.py` (deletions), `playlist_generator.py` (rename + key-source change), `worker.py` (one-line change).

### 3. Data flow

**Before (today):**

```
config.yaml: playlists.ds_pipeline.mode: "dynamic"
  → worker.py: ds_mode = config['playlists']['ds_pipeline'].get('mode', 'dynamic')
  → generator.create_playlist_*(ds_mode_override=ds_mode)
  → playlist_generator.py: ds_mode_effective = ds_mode_override or ds_cfg.get("mode", "dynamic")
  ├─ default_ds_config(ds_mode_effective, ...)
  │   returns per-mode alpha/target_artists/etc.
  │   AND per-mode min_sonic_similarity/max_artist_fraction/broad_filters
  │   (last three overwritten later by apply_mode_presets/policy)
  └─ resolve_pier_bridge_tuning(mode=ds_mode_effective, ...)
      resolves bridge_floor_<mode>, weight_bridge_<mode>, soft_genre_penalty_*_<mode>
```

**After:**

```
config.yaml: playlists.cohesion_mode: "dynamic"
  → worker.py: cohesion_mode = resolve_cohesion_mode(config['playlists'])
  → generator.create_playlist_*(cohesion_mode_override=cohesion_mode)
  → playlist_generator.py: cohesion_mode_effective = cohesion_mode_override or resolve_cohesion_mode(playlists_cfg)
  ├─ default_ds_config(cohesion_mode_effective, ...)
  │   returns ONLY per-mode alpha/alpha_schedule/local_top_m/target_artists/
  │   candidates_per_artist/max_pool_size/similarity_floor
  │   (no longer writes min_sonic_similarity/max_artist_fraction/broad_filters)
  └─ resolve_pier_bridge_tuning(mode=cohesion_mode_effective, ...)
      resolves bridge_floor_<cohesion_mode>, weight_bridge_<cohesion_mode>, etc.

Pool composition (separate flow, unchanged):
  apply_mode_presets() → min_sonic_similarity, broad_filters, genre weights
  resolve_pace_mode()  → pace_admission_floor, BPM floors
  policy.py            → max_artist_fraction (from artist_presence)
```

**`resolve_cohesion_mode()` signature:**

```python
def resolve_cohesion_mode(playlists_cfg: dict) -> Mode:
    """Read playlists.cohesion_mode with validation. Warn on stale ds_pipeline.mode."""
    if not isinstance(playlists_cfg, dict):
        return "dynamic"

    # Warn if stale key is present (no longer used)
    ds_pipeline = playlists_cfg.get("ds_pipeline", {})
    if isinstance(ds_pipeline, dict) and "mode" in ds_pipeline:
        logger.warning(
            "playlists.ds_pipeline.mode is no longer used; remove from config. "
            "Use playlists.cohesion_mode instead."
        )

    raw = str(playlists_cfg.get("cohesion_mode", "dynamic")).strip().lower()
    if raw not in {"strict", "narrow", "dynamic", "discover"}:
        logger.warning("Invalid cohesion_mode %r; falling back to 'dynamic'", raw)
        return "dynamic"
    return raw  # type: ignore[return-value]
```

This is the only function that reads the key.

### 4. Error handling / Edge cases

**Stale `ds_pipeline.mode` key in config.yaml.** Mitigation: warning in `resolve_cohesion_mode()` ("`ds_pipeline.mode` is no longer used; remove from config. Use `cohesion_mode` instead."). Key is ignored after warning.

**Invalid `cohesion_mode` value.** Covered by `resolve_cohesion_mode()` validator — warn, fall back to `dynamic`.

**`sonic_only` Mode Literal removal.** Today `Mode = Literal["narrow", "dynamic", "discover", "sonic_only"]`. After A: `Mode = Literal["strict", "narrow", "dynamic", "discover"]`. The GUI preset `genre_mode=off + sonic_mode=dynamic` (QUICK_PRESETS["sonic_only"]) covers the same intent and remains. Any `default_ds_config()` branches for `sonic_only` get deleted. 12 source files reference `sonic_only`; most are sonic-pool helpers unrelated to the cohesion axis and stay alone.

**Per-mode pier_bridge keys for removed modes.** Current `config.yaml` has knobs like `bridge_floor_strict: 0.10` — these stay valid. No `*_sonic_only` keys exist today (confirmed via grep), so no cleanup needed there.

**CLI flag breakage.** `--ds-mode` becomes `--cohesion-mode`. No deprecation alias — hard break for the single user (you). Documented in `docs/GOLDEN_COMMANDS.md`.

**Advanced Panel `SettingSpec`.** After renaming `key_path` to `playlists.cohesion_mode`, the `choices` list updates from `["narrow", "dynamic", "discover", "sonic_only"]` to `["strict", "narrow", "dynamic", "discover"]`. Widget continues to function; sub-project B replaces it with a slider.

**Test fixtures.** All tests that construct config dicts with `ds_pipeline.mode` get updated to `cohesion_mode` at the playlists level. All `ds_mode_override=` parameter passes rename to `cohesion_mode_override=`.

**Order-dependence between `apply_mode_presets()` and `default_ds_config()`.** After dedup, `default_ds_config()` no longer writes the three duplicate settings, so order no longer matters for them. Order still matters for the cohesion-owned settings (alpha schedule, target_artists, etc.) but those have a single writer so there's no conflict.

### 5. Testing

**New unit tests:**

1. **`tests/unit/test_cohesion_mode_resolution.py`** — `resolve_cohesion_mode()`:
   - Valid values `strict / narrow / dynamic / discover` → return as-is
   - Missing key → default `"dynamic"`
   - Invalid value → warn, default `"dynamic"`
   - Stale `ds_pipeline.mode` present → warning emitted, value ignored
   - Both `cohesion_mode` and stale `ds_pipeline.mode` → use `cohesion_mode`, warn about stale

2. **`tests/unit/test_default_ds_config_dedup.py`** — `default_ds_config()` no longer writes the three duplicate settings:
   - For each cohesion_mode, assert `candidate.min_sonic_similarity is None` unless caller's `overrides["candidate_pool"]["min_sonic_similarity"]` is set
   - Assert `candidate.max_artist_fraction_final` comes from `overrides["candidate_pool"]["max_artist_fraction"]` or single sensible default (0.125), not per-mode dict
   - Assert `candidate.broad_filters` is empty tuple unless caller passes one

3. **`tests/unit/test_single_writer_settings.py`** — Full resolution chain:
   - Base config with `cohesion_mode=strict`, `sonic_mode=narrow`, `genre_mode=dynamic`
   - Run config_loader → apply_mode_presets → default_ds_config
   - Assert `min_sonic_similarity == 0.12` (from sonic_mode=narrow), NOT overwritten to `0.20` (would-be strict cohesion default)
   - Assert `broad_filters == []` (from genre_mode=dynamic), NOT `["rock","indie","alternative","pop"]` (would-be strict cohesion default)

4. **Mode Literal update** — extend existing tests or add:
   - `"strict"` is now a valid Mode → `default_ds_config(mode="strict", ...)` succeeds
   - `"sonic_only"` is no longer valid → `default_ds_config(mode="sonic_only", ...)` raises `ValueError`

**Existing tests to update:**
- `tests/test_artist_style.py:test_soft_genre_penalty_per_mode_resolution` — already tests `resolve_pier_bridge_tuning(mode=...)` directly; no behavior change needed
- All other tests referencing `ds_pipeline.mode` or `ds_mode_override` — mechanical find/replace

**Integration / regression:**

- **Golden test** (`pytest -m golden`) — known seed against known artifact, compare output. With `cohesion_mode=dynamic` (new default) and unchanged other settings, tracklist must be identical to pre-refactor baseline.
- **Manual regression:** regenerate Sundays playlist with new config. Expected: byte-for-byte identical tracklist + identical T/S/G stats to `sundays_log3.txt`.
- **Per-mode smoke:** generate a playlist with `cohesion_mode=strict`. Confirm pier-bridge logs show `bridge_floor=0.10`, `weight_bridge=0.7`, `soft_genre_penalty_threshold=0.82` (existing per-mode strict values from config.yaml). This is the "did this refactor actually make the dead knobs live?" test.

**Test priority order:**
1. Regression test (golden + Sundays manual) — proves nothing broke
2. `resolve_cohesion_mode()` unit tests — proves the new helper works
3. Single-writer integration test — proves the dedup actually held
4. Per-mode smoke for `cohesion=strict` — proves the refactor delivered its value

## Dependencies for downstream work

After A ships:
- **Sub-project B (GUI)** can add a Cohesion slider widget and wire `policy.py` to write `playlists.cohesion_mode`. The Advanced Panel `SettingSpec` either gets removed or stays as an alternative interface.
- **Sub-project C (Presets)** can save/restore `cohesion_mode` alongside the other slider values without further backend work.

## Open questions

None remaining. All design decisions are settled.
