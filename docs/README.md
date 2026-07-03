# Documentation Index

Overview of the Playlist Generator docs. `README.md` (repo root) is the listener-facing feature
catalog; [`ARCHITECTURE.md`](ARCHITECTURE.md) is the orientation map, and
[`TECHNICAL_PLAYLIST_GENERATION_FLOW.md`](TECHNICAL_PLAYLIST_GENERATION_FLOW.md) is the most
authoritative implementation walkthrough.

---

## Getting started
- **[GOLDEN_COMMANDS.md](GOLDEN_COMMANDS.md)** — command reference (analyze pipeline, generate, serve)
- **[CONFIG.md](CONFIG.md)** — configuration key reference
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** — common issues

## Architecture & generation
- **[ARCHITECTURE.md](ARCHITECTURE.md)** — system architecture overview (the orientation layer)
- **[TECHNICAL_PLAYLIST_GENERATION_FLOW.md](TECHNICAL_PLAYLIST_GENERATION_FLOW.md)** — end-to-end, code-level generation walkthrough
- **[DJ_BRIDGE_ARCHITECTURE.md](DJ_BRIDGE_ARCHITECTURE.md)** — pier-bridge / beam-search deep-dive
- **[PLAYLIST_ORDERING_TUNING.md](PLAYLIST_ORDERING_TUNING.md)** — knob-by-knob ordering/tuning guide

## Why & current state
- **[DESIGN_RATIONALE.md](DESIGN_RATIONALE.md)** — the "why" canon: what was tried, what won, what was rejected, with evidence
- **[WIRING_STATUS.md](WIRING_STATUS.md)** — living tracker of what's actually live vs shipped-default vs off vs known-broken
- **[CLEANUP_LIST.md](CLEANUP_LIST.md)** — parked features + open tech-debt / bugs

## Sonic
The current sonic space is **MuQ** (contrastive, sole embedding) — see `ARCHITECTURE.md`
§"Sonic feature space" and `DESIGN_RATIONALE.md` §"Sonic embedding". The docs below are
**historical** (the MERT/tower era that MuQ replaced; MERT/tower code is archived):
- **[MERT_WHITEN_NEIGHBORS_20SEEDS.md](MERT_WHITEN_NEIGHBORS_20SEEDS.md)** — *(historical)* MERT neighbour QA
- **[SONIC_PHASE2_HARMONY_FINDINGS.md](SONIC_PHASE2_HARMONY_FINDINGS.md)** — *(historical)* 2DFTM harmony tower investigation

## Genre (authority + taxonomy graph + enrichment)
- **[AI_GENRE_ENRICHMENT.md](AI_GENRE_ENRICHMENT.md)** — enrichment usage
- **[AI_GENRE_ENRICHMENT_DEVELOPMENT_BIBLE.md](AI_GENRE_ENRICHMENT_DEVELOPMENT_BIBLE.md)** — hybrid deterministic/LLM genre model
- **[LAYERED_GENRE_GRAPH_SPEC.md](LAYERED_GENRE_GRAPH_SPEC.md)** — SP3a layered taxonomy graph spec
- The `genre-data-authority` skill is the map of which layer to read/write (the authority is `release_effective_genres` via `src/genre/authority.py`).

## Release & development
- **[../CHANGELOG.md](../CHANGELOG.md)** — release notes (repo root; the authoritative changelog)
- **[LOGGING.md](LOGGING.md)** — logging standards
- **[DEAD_CODE_AUDIT_2026-06-10.md](DEAD_CODE_AUDIT_2026-06-10.md)** — dead-code audit
- Design specs + implementation plans: `docs/superpowers/specs/` and `docs/superpowers/plans/`

---

## Key concepts

**Generation:** seeds become fixed "piers"; a beam search bridges each adjacent pair through the
**MuQ** sonic space, with anti-sag scoring (anti-center, mini-piers) and a post-beam weak-edge
recovery cascade. Four independent axes — `cohesion_mode` (beam tightness) plus `genre_mode` /
`sonic_mode` / `pace_mode` (pool composition).

**Sonic:** a single learned **MuQ** embedding (`MuQ-MuLan-large`, 512-d, contrastive). There is
no runtime variant choice; MERT and the hand-built towers were removed (archived).

**Pace:** BPM + onset-rate log-distance hard bands plus a soft rhythm penalty (DB features, so
embedding-independent) — not a sonic-embedding axis.

**Genre:** the authority is `release_effective_genres` (metadata.db, written only by the
enrichment publish stage, read via `src/genre/authority.py`), resolved against the layered
taxonomy graph and baked into the artifact (`genre_source: graph`). **Tag-steering** lets an
artist-mode user softly lean toward the seed artist's own published genres.

---

**Note:** `docs/archive/` and `docs/run_audits/` are gitignored working areas, not shipped docs.
There is a second, older `docs/CHANGELOG.md` — the repo-root `CHANGELOG.md` above is authoritative.
