---
name: playlist-testing
description: How to write and run faithful tests for playlist generation in this repo. Use whenever writing/running a generation test, reproducing a generation bug, or building a calibration/regression harness. Prevents the recurring trap where the test diverges from how the real GUI generates.
---

# Playlist generation testing

Most "playlist bugs" in this repo turn out to be the **test diverging from production**, not a real engine bug. This skill exists to stop that. Read the Core Rule and the Trap Catalog before writing or trusting any generation test.

## Core Rule

**Never hand-build a partial `overrides` dict and pass it to `generate_playlist_ds`.** A direct call does NOT merge `config.yaml` — missing keys silently fall to dataclass defaults that differ from production. Always generate through the harness:

```python
import sys; sys.path.insert(0, "tests")   # if needed
from support.gui_fidelity import generate_like_gui, gui_ui_state, resolve_gui_overrides

res = generate_like_gui(
    seeds=[track_id, ...],            # bundle track_ids used as piers
    cohesion_mode="narrow", genre_mode="narrow", sonic_mode="narrow",
    pace_mode="narrow", artist_spacing="strong",   # the GUI knobs
    length=30, random_seed=0,
)
```

`tests/support/gui_fidelity.py` reuses the **real** production chain:
`policy.derive_runtime_config` → `merge_overrides` → `worker.load_config_with_overrides(config.yaml)` → `playlist_generator.build_ds_overrides` → `ds_pipeline_runner.generate_playlist_ds`.
That single chain (especially `load_config_with_overrides`) is the piece ad-hoc tests skip. **Do not reimplement the slider→config mapping in a test — the drift just moves into the test.** If you need a new GUI knob, wire it through `UIStateModel` + `policy`, not by hand.

## Trap Catalog (every one of these caused wasted debugging)

| Trap | Symptom | Why |
|------|---------|-----|
| Hand-built overrides skip `config.yaml` | `disallow_pier_artists_in_interiors=False`, `artist_identity.enabled=False`, `min_gap`=mode-default | dataclass defaults differ from `config.yaml`. Use the harness. |
| Single-seed topology | "infeasible", 20+ interior segment | pier_a==pier_b==seed; not how artist/seeds mode runs. Use ≥2 piers (seeds) or artist-style. |
| `genre_admission_percentile`/`pier_bridge.*` only set under mode-specific key | knob silently inert | resolve via config, or set `<key>_<mode>` AND base key. |
| Worker `@lru_cache` on `load_artifact_bundle` | rebuilt artifact/sidecar not seen | RESTART the GUI after any artifact/sidecar rebuild. |
| Dense sidecar staleness | stale `X_genre_dense` / vocab drift | `load_artifact_bundle` only guards track_ids, not vocab/content. Rebuild sidecar (Build Artifacts button does both now). |
| Last.fm recency nondeterminism | run-to-run track differences | harness is artifact-level and does not hit Last.fm; full-stack runs do. |
| Mode keys mutated after `Config()` construction | genre/sonic modes silently inert (floors/weights stay at config.yaml values) while pace_mode works | `apply_mode_presets` runs inside `Config.__init__` (config_loader.py); genre/sonic presets bake at load time, pace is read at generation time. Don't drive `PlaylistApp`/`PlaylistGenerator` directly and set `playlists.genre_mode` afterward — use `generate_like_gui` (2026-06-12). |
| Genre gate / hybrid weights silently OFF in the harness | `rejected_genre=0`, no "Candidate pool genre gating" line; any genre-gate experiment is inert (loosening `genre_admission_percentile` changes nothing) | `min_genre_similarity` / `sonic_weight` / `genre_weight` / `genre_method` are **explicit `generate_playlist_ds` params, NOT carried in the overrides dict** — the orchestrator resolves them from `playlists.genre_similarity`. `generate_like_gui` now calls `resolve_gui_genre_params` (shared `src/playlist/genre_ds_params.py::resolve_genre_ds_params`, same fn the orchestrator uses) and splats them in. If you call `generate_playlist_ds` directly, you MUST pass them too or the genre gate is off. Fixed 2026-06-14. |

## What this harness does and does NOT cover

- ✅ Generation **logic** at full config fidelity — seeds mode (and artist mode when piers are supplied). Fast, artifact-level, deterministic.
- ✅ Genre gate + hybrid sonic/genre weights (since 2026-06-14, via `resolve_gui_genre_params`) — the candidate-pool genre admission now fires in the harness exactly as in the GUI.
- ❌ The Qt widget layer (use `pytest-qt`; the old no-op `qtbot` stub is gone).
- ❌ DB-clustering (artist-style pier discovery) and Last.fm recency — those need the full `handle_generate_playlist` worker entry, a separate heavier tier not built here. (Note: per-edge S/G/T *reporting* is also worker-layer — `generate_playlist_ds`'s result carries `track_ids` but empty `metrics`; compute edge stats yourself for harness experiments.)

## Assertion helpers (in `tests/support/gui_fidelity.py`)

- `artist_at_positions(bundle, track_ids)` → list of artist strings in order.
- `find_min_gap_violations(artists, min_gap)` → same-artist pairs closer than `min_gap`.
- `assert_min_gap(bundle, track_ids, min_gap)` → raises on any violation.
- `resolve_gui_overrides(ui, config_path=...)` → the exact ds-overrides the GUI would send (assert config values without generating).

## Test tiers

- **Fast** (`tests/test_gui_fidelity.py`): config-resolution only, no artifact. Guards that the harness still surfaces production defaults. Runs in the default suite.
- **Integration** (`tests/integration/test_gui_fidelity_regressions.py`): end-to-end via the live artifact. Mark `@pytest.mark.integration @pytest.mark.slow` and skip if the artifact is absent.

## Maintenance protocol ("update regularly")

When a generation bug is fixed:
1. Add a case to `tests/integration/test_gui_fidelity_regressions.py` that reproduces it **through `generate_like_gui`** with the GUI knobs that triggered it. Reference the fixing commit.
2. If it exposed a new fidelity trap (a config key not carried, a knob not wired), add a row to the Trap Catalog above and a fast guard in `tests/test_gui_fidelity.py`.
3. Keep this SKILL.md current — it is the index of how we test, not a one-time doc.

## Known follow-ups (not yet pinned)

- **dj_bridging + steering coexistence** (commits `4d5c593`/`8d35a9a`): `dj_bridging_enabled` is a *derived* policy decision, not a simple `UIStateModel` bool, so a clean harness-level regression needs the policy gating traced first. Until then, verified manually.
- **Smog ≟ Bill Callahan**: identity resolution does not collapse same-person/different-project names; a true test needs that capability first.
