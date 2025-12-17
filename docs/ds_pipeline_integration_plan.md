# DS Pipeline Integration Plan

This plan tracks the remaining work to fully integrate the DS playlist pipeline into the main app while keeping the legacy path intact.

## 1) Artifact Lifecycle
- [ ] Decide canonical artifact locations (e.g., `data/ds_artifacts/` or existing `experiments/genre_similarity_lab/artifacts/`).
- [ ] Wire `analyze_library.py` into a scheduled job/CI target; document required env (config.yaml, DB, audio access).
- [ ] Add startup check in the app: verify NPZ + genre-sim files are readable, log timestamps/sizes, and warn if stale or missing.
- [ ] Add a lightweight `verify` stage to the schedule and fail fast if artifacts are invalid (dup track_ids, shape mismatches).

## 2) Pipeline Toggle UX
- [ ] Expose `pipeline` choice in all entrypoints (CLI/UI/API). CLI flag exists; add UI/API surface if applicable.
- [ ] Gate `pipeline=ds` behind artifact existence check; auto-fallback to legacy with a clear warning.
- [ ] Allow DS mode override (narrow/dynamic/discover) from CLI/UI without editing config.

## 3) Metadata Fetch & Performance
- [ ] Add a batch track lookup by track_id in `LocalLibraryClient` (and Plex client if used) to reduce N+1 queries.
- [ ] Preserve seed at position 0 in DS results and keep ordering stable.
- [ ] Profile DS pipeline on typical playlists; tune PCA dims/weights only if needed (keep defaults for now).

## 4) Logging & Telemetry
- [ ] Standardize DS run logs: pipeline=ds, seed_track_id, mode, length, random_seed, min/mean transition, below_floor, distinct_artists, max_artist_share.
- [ ] Ensure logs go to the main log file; include fallback reason when DS is skipped.
- [ ] Optionally add a JSON line snapshot per DS run to aid debugging (size-limited).

## 5) Error Handling & Fallbacks
- [ ] Wrap DS calls so any exception triggers legacy fallback without aborting the workflow.
- [ ] Surface fallback cause to the user (CLI print/UI banner) and to logs.
- [ ] Add guardrails: if artifacts missing/invalid, force legacy and warn once per run.

## 6) Tests
- [ ] Add integration tests driving `PlaylistGenerator` through legacy vs DS with a tiny synthetic artifact; assert ordering and seed inclusion.
- [ ] Add test: `pipeline=ds` + missing artifact -> warning + legacy path used.
- [ ] Add test: DS mode override flows from CLI/UI into runner.
- [ ] Keep existing unit tests for candidate pool/constructor and CLI switches.

## 7) Configs & Docs
- [ ] Update README/DEV docs: how to run `scripts/analyze_library.py`, expected outputs, config keys (`playlists.pipeline`, `ds_pipeline.*`).
- [ ] Make sure `config.example.yaml` and deployment configs have correct artifact paths.
- [ ] Document operational runbook: how to refresh artifacts, how to diagnose DS fallbacks.

## 8) Deployment Readiness
- [ ] Decide where artifacts live in production (shared volume vs local disk) and permissions.
- [ ] Confirm disk usage and retention policy for artifacts and logs.
- [ ] Add a small smoke command to release checklist (`python scripts/analyze_library.py --config ... --dry-run`).

## 9) Nice-to-Have Enhancements
- [ ] Add simple metrics export (e.g., CSV/JSON) for DS runs to inspect diversity and floor metrics over time.
- [ ] Consider embedding caching (reuse PCA fits) if artifact builds become frequent.
- [ ] Add a “verify artifact” command to the CLI for quick checks when debugging.

