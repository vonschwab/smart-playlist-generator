# Changelog

## 3.2
- Windows GUI: accent-insensitive artist autocomplete and export fixes for the track table.
- Artist normalization: shared `normalize_artist_key` across CLI/GUI, DB schema migration (`artist_key`) with backfill.
- MBID workflow: MusicBrainz fetcher (`scripts/fetch_mbids_musicbrainz.py`) with skip markers and force flags; optional `mbid` stage in `analyze_library` to enrich `metadata.db` without touching audio files.
- Matching performance: cached artist/title normalization for Last.FM matching, faster history-mode runs.
- DS modes tightened: higher similarity floors and seed/transition weights (narrow/dynamic/discover) plus higher genre gates (config.example defaults) for more cohesive playlists.
- Pier-bridge tuning: per-mode defaults (dynamic/narrow), centralized resolver with override-source logging, and optional soft genre penalty during ordering (configurable via `playlists.ds_pipeline.pier_bridge.*`).
- Pier-bridge resilience (optional): deterministic `bridge_floor` backoff for infeasible segments plus per-run markdown audits (CLI flags `--pb-backoff`, `--audit-run`, `--audit-run-dir`).
- Pier-bridge segment upgrades: segment-local candidate pools (no neighbor union), A→B progress constraint, robust `artist_key/title_key` dedupe, and default seed-artist piers-only interiors for `--artist` playlists (configurable).
- Recency invariant: Last.fm/local recency exclusions are pre-order only for DS; post-order filtering removed and replaced with strict post-order validation (fails loudly, with audit path when enabled).
