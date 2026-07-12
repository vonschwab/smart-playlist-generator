# Changelog

**Latest Release:** Version 6.0.0

For full release notes see [docs/CHANGELOG.md](docs/CHANGELOG.md).

## Summary

- **v6.0.0** — MuQ (contrastive `MuQ-MuLan-large`, 512-d, `center_l2`) became the sole sonic embedding, replacing the hand-built towers and a short-lived MERT era (tower/MERT code paths and the `transition_weights`/`tower_weights` knobs removed; MERT data archived to `data/archive/mert_2026/`; MuQ screened 84% vs MERT's 73% on trusted triplets); weak-edge recovery cascade (variable-bridge add-only → tail-DP → break-glass edge repair → edge delete, least-to-most destructive); collapse prevention (anti-center scoring plus structural mini-piers for bridge-interior sag); artist-mode tag steering (soft pool + pier lean toward the seed artist's own published genres); manual artist links (alias merge + sibling spacing via `data/artist_aliases.yaml`); enriched-genre authority + layered taxonomy graph (grown to v0.30, auto-propagated to publish + artifacts); multisource Claude enrichment with publish/pause safety and a delta migration; Genre Review GUI panel + graph-canonical chips; pace rebuilt on BPM + onset-rate bands plus a soft rhythm penalty; four mode axes (cohesion/genre/sonic/pace); browser GUI as sole front-end; deprecated-code removal
- **v5.0** — Pace mode (rhythm axis independent of timbre/harmony); transition weight alignment fixing beam-reporter mismatch; IDF-weighted genre admission; uncapped seeded candidate pool; opt-in per-edge audit and edge repair; scoped blacklisting (artist/album) in GUI
- **v4.0** — Native GUI overhaul: CLI parity, responsive generation controls, Analyze Library summaries, shared request validation, worker reliability
- **v3.5** — Job cancellation/checkpoints, job-details diagnostics, persistent genre cache, faster library scans, collaboration-aware artist-style clustering
- **v3.4** — DJ Bridge Mode for multi-seed playlists with genre-aware bridging, union pooling, comprehensive diagnostics
- **v3.3** — Seed List mode, Sonic/Genre modes, refactored pipeline, blacklist support
- **v3.2** — Windows GUI improvements, MBID enrichment, artist normalisation, matching performance

See the full documentation at [docs/README.md](docs/README.md).
