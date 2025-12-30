# Changelog

## 3.1
- Windows GUI: accent-insensitive artist autocomplete and export fixes for the track table.
- Artist normalization: shared `normalize_artist_key` across CLI/GUI, DB schema migration (`artist_key`) with backfill.
- MBID workflow: MusicBrainz fetcher (`scripts/fetch_mbids_musicbrainz.py`) with skip markers and force flags; optional `mbid` stage in `analyze_library` to enrich `metadata.db` without touching audio files.
- Matching performance: cached artist/title normalization for Last.FM matching, faster history-mode runs.
- DS modes tightened: higher similarity floors and seed/transition weights (narrow/dynamic/discover) plus higher genre gates (config.example defaults) for more cohesive playlists.
