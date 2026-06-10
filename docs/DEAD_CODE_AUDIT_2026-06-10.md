# Dead code & unwired config audit — 2026-06-10

Evidence-based kill list for the cleanup program (Phase 0 guardrails → Phase 1
generation-path cleanup + wiring → Phase 2 broad sweep). Nothing in this document
has been deleted yet; every item cites its evidence.

## Method

1. **Static reachability**: AST import graph over `src/`, BFS from all production
   entrypoints (`main_app.py`, `tools/*.py`, `scripts/*.py`, plus the
   subprocess-spawned `src.playlist_gui.worker`). Modules unreachable from every
   entrypoint are static-dead. (76 → 60 candidates after fixing package-`__init__`
   relative-import resolution and adding the worker root.)
2. **Runtime coverage**: two real generations traced under `coverage.py --branch`:
   the exact 10-seed GUI request replayed through the worker NDJSON protocol, and
   an artist-mode CLI run. Files at 0% that are also static-dead are confirmed dead.
3. **Knob audit**: every leaf key in `config.yaml`/`config.example.yaml` checked for
   static reads, dynamic per-mode reads (`cfg.get(f"{key}_{mode}")`), and
   effective-config divergence against the run log.
4. **Zombie tests**: test files importing dead-module candidates.

Scripts + raw reports: `C:\tmp\{dead_modules,knob_audit,cov_summary,zombie_tests}.py`,
`C:\tmp\{dead_modules_report,knob_audit_report,cov_report,zombie_tests_report}.md`.

---

## 1. Wiring bugs found (FIX, not delete) — these change playlist quality

### 1a. Beam search-width knobs are silently unwired (all 6)

`config.yaml` sets them; **no code reads them into `PierBridgeConfig`** — runtime
uses dataclass defaults at half the configured width:

| key | config.yaml | actual runtime |
|---|---|---|
| `pier_bridge.initial_beam_width` | 40 | **20** |
| `pier_bridge.max_beam_width` | 200 | **100** |
| `pier_bridge.initial_neighbors_m` | 200 | **100** |
| `pier_bridge.max_neighbors_m` | 800 | **400** |
| `pier_bridge.initial_bridge_helpers` | 100 | **50** |
| `pier_bridge.max_bridge_helpers` | 400 | **200** |

Evidence: effective-config dump in every run log; the fields are *consumed* at
runtime (`pier_bridge_builder.py:769,791,802,889-894,1242`) but
`resolve_pier_bridge_tuning` never plumbs the YAML keys.
**Fix: wire them** (TDD). Every playlist has been generated with half the
intended search width.

### 1b. The rhythm-shape pace gate is dead code in the live path

`beam.py:1005` gates candidates on rhythm-cosine to an interpolated pier-to-pier
target when `pace_bridge_floor > 0` — but the live caller
(`pier_bridge_builder.py:1144`) **never passes `rhythm_matrix`**, so the gate
silently no-ops. Only `assemble.py` (dead, see §2) wires it. `narrow`/`strict`
pace mode's `bridge_floor` (0.45/…) controls nothing; the only live pace gate in
ordering is the BPM-distance gate at `max_log_distance=0.6` (≈1.8× tempo
tolerance, bypassed for NaN-BPM or low-stability tracks). This is why pace feels
ignored (63→172 BPM swings) and part of why Springsteen→Prince→Loving happened.
**Fix: thread `rhythm_matrix` into `_beam_search_segment`** (TDD; coverage
confirms `pace_gate.py` at 25.5% even in a pace=narrow run).

### 1c. The web GUI silently disables DJ bridging on every run

`policy.py:341-381`: `dj_bridging.enabled` is **policy-owned** — the web layer
overwrites config.yaml on every request. Eligibility requires `seed_artist_keys`,
which the web app **never passes** (`policy.py:352`: "Phase 1 limitation; Seeds
UI must resolve track IDs to artist keys"). Result: in every web GUI run,
`dj_bridging_enabled=false, pooling=baseline` regardless of config — confirmed by
A/B: direct worker replay of the same request shows `true/dj_union`; the GUI run
log shows `false/baseline`. The entire `dj_bridging` config block (waypoint
steering, dj_union pooling, genre coverage) is inert in the GUI you actually use.
**Fix: finish the Phase-1 wiring** (web layer has `seed_track_ids`; resolve to
artist keys) — or explicitly retire dj_bridging from GUI policy and say so in logs.

### 1d. Dead config keys (delete from both YAMLs)

- `playlists.cache_expiry_days` — zero reads anywhere
- `playlists.genre_similarity.use_artist_tags` — zero reads
- `playlists.similar_artists.boost` — zero reads

(The 37 per-mode suffixed keys flagged by naive search are **live** via
`cfg.get(f"{key}_{mode}")` — config.py:144. Don't touch.)

---

## 2. Kill list A — abandoned refactor layer (delete now, ~5,300 LOC)

Static-dead + 0% runtime coverage. The `feature_flags` cluster is an abandoned
architecture migration: a flag system whose flags gate modules nothing imports.
`pipeline/core.py:765` already says "Legacy paths (anchor_builder, …) have been
removed" — the files just weren't deleted.

| module | LOC | note |
|---|---|---|
| `src/playlist/pier_bridge/assemble.py` | 1,970 | fooled us twice (steering + rhythm wiring "existed" here) |
| `src/playlist/anchor_builder.py` | 568 | declared removed in core.py:765 |
| `src/playlist/ds_pipeline_builder.py` | 444 | feature_flags cluster |
| `src/feature_flags.py` | 275 | flag system for unwired modules |
| `src/playlist/config_resolver.py` | 271 | feature_flags cluster |
| `src/similarity/variant_cache.py` | 267 | feature_flags cluster |
| `src/performance_tracker.py` | 274 | |
| `src/playlist/strategies/` (2 files) | 201 | feature_flags cluster |
| `src/genre_normalization.py` | 190 | superseded by `src/genre/normalize_unified.py` |
| `src/artist_cache.py` | 165 | |
| `src/playlist/playlist_factory.py` | 101 | feature_flags cluster |
| `src/eval/` (2 files) | 390 | |
| `src/retry_helper.py` | 93 | |
| `src/logging_config.py` | 80 | superseded by `src/logging_utils.py` |
| `src/similarity/sonic_schema.py` | 30 | |

Plus their zombie tests (see §5).

**Review with user before deleting** (may be staged SP-lane work, not abandoned):
- `src/genre/authority.py` (65) — relates to enriched-genre-authority direction
- `src/genre/vocab_normalization.py` (486) — relates to vocabulary program

## 3. Kill list B — deprecated PySide6 GUI (Phase 2, ~13,000 LOC)

Everything under `src/playlist_gui/` **except**: `worker.py` (spawned by web),
`policy.py` + `ui_state.py` (imported by web app), `request_models.py` shim if
web imports it, `utils/redaction.py` if worker imports it. Includes
`main_window.py` (2,021), the full `widgets/` tree (~6,600), `worker_client.py`
(823), `autocomplete.py` (540), `jobs/` (877), `config/` (1,228),
`diagnostics/` (348), plus the `qtbot` no-op stub in `tests/conftest.py` and
~30 GUI test files. Deliberate removal of a deprecated frontend — not urgent,
but the qtbot stub should go early (it's the test-suite version of silent
no-op wiring).

## 4. Kill list C — legacy pre-DS pipeline (decide)

Imported by `playlist_generator.py` but 5-12% runtime coverage (imports/types
only): `constructor.py` (339 stmts, 5%), `candidate_generator.py` (7.6%),
`ordering.py` (7.5%), `diversity.py` (10.3%), `history_analyzer.py` (12%),
`similarity_calculator.py` (682 stmts, 19.3%), plus `genre_similarity_v2.py`
(11.9%). These are the pre-pier-bridge engine. Deleting them means committing
to `pipeline: ds` as the only engine and untangling `playlist_generator.py`'s
imports — bigger surgery, do after A.

## 5. Zombie tests

**46 of 60 dead modules have passing tests — 47 distinct test files** exercising
code production cannot run (`C:\tmp\zombie_tests_report.md` has the full map).
Delete alongside their modules. Separately: the **13 perma-failing tests** that
the suite has learned to ignore must be fixed or deleted — a suite where "13
failed" means "fine" cannot catch real regressions.

## 6. Phase 0 guardrail (the discipline that prevents regrowth)

*A configured knob that can't act is a startup error, not a silent no-op.*
Concretely, at config resolution: `pace_bridge_floor > 0` without a rhythm
matrix → raise; YAML keys under `pier_bridge` that match no known
`PierBridgeConfig`/`PierBridgeTuning` field and no dynamic per-mode base →
raise (would have caught §1a immediately); policy overrides that flip a
user-set key → log loudly at INFO in the generation log, not in a buried
notes list (§1c).
