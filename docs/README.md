# Documentation Index

Overview of the Playlist Generator docs. `README.md` (repo root) is the listener-facing
feature catalog; `docs/TECHNICAL_PLAYLIST_GENERATION_FLOW.md` is the most authoritative
implementation walkthrough.

---

## Getting started
- **[GOLDEN_COMMANDS.md](GOLDEN_COMMANDS.md)** — command reference (scan, analyze/enrich stages, generate, serve)
- **[CONFIG.md](CONFIG.md)** — configuration key reference
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** — common issues

## Architecture & generation
- **[ARCHITECTURE.md](ARCHITECTURE.md)** — system architecture overview
- **[TECHNICAL_PLAYLIST_GENERATION_FLOW.md](TECHNICAL_PLAYLIST_GENERATION_FLOW.md)** — full pipeline deep-dive
- **[DJ_BRIDGE_ARCHITECTURE.md](DJ_BRIDGE_ARCHITECTURE.md)** — pier-bridge / DJ bridging design
- **[PLAYLIST_ORDERING_TUNING.md](PLAYLIST_ORDERING_TUNING.md)** — knob-by-knob ordering/tuning guide

## Sonic (MERT + towers)
- **[MERT_WHITEN_NEIGHBORS_20SEEDS.md](MERT_WHITEN_NEIGHBORS_20SEEDS.md)** — MERT cross-catalog neighbour QA (the default sonic space)
- **[SONIC_PHASE2_HARMONY_FINDINGS.md](SONIC_PHASE2_HARMONY_FINDINGS.md)** — 2DFTM harmony tower investigation (the rollback space)

## Genre (authority + taxonomy graph + enrichment)
- **[AI_GENRE_ENRICHMENT.md](AI_GENRE_ENRICHMENT.md)** — enrichment usage
- **[AI_GENRE_ENRICHMENT_DEVELOPMENT_BIBLE.md](AI_GENRE_ENRICHMENT_DEVELOPMENT_BIBLE.md)** — hybrid deterministic/LLM genre model (source of truth; overrides older "model prior" specs)
- **[LAYERED_GENRE_GRAPH_SPEC.md](LAYERED_GENRE_GRAPH_SPEC.md)** — SP3a layered taxonomy graph spec
- **[GENRE_DATA_QUALITY_FINDINGS_2026-06-12.md](GENRE_DATA_QUALITY_FINDINGS_2026-06-12.md)** — enrichment fusion + delta-migration findings

## Release & development
- **[CHANGELOG.md](CHANGELOG.md)** — release notes (latest: v6.0)
- **[LOGGING.md](LOGGING.md)** — logging architecture
- **[DEAD_CODE_AUDIT_2026-06-10.md](DEAD_CODE_AUDIT_2026-06-10.md)** — dead-code audit (see also `tools/dead_code_audit.py`)
- Design specs + implementation plans: `docs/superpowers/specs/` and `docs/superpowers/plans/`

---

## Key concepts

**Generation:** seeds become fixed "piers"; beam search builds bridges between adjacent piers
through the **MERT** sonic space (the 162-d towers are a config rollback). Four independent
axes — `cohesion_mode` (beam tightness) plus `genre_mode` / `sonic_mode` / `pace_mode` (pool
composition).

**Pace:** BPM + onset-rate log-distance hard bands plus a soft rhythm penalty (DB features,
MERT-durable) — not a sonic-embedding axis.

**Genre:** the authority is `release_effective_genres` (metadata.db, written only by the
enrichment publish stage, read via `src/genre/authority.py`), resolved against the layered
taxonomy graph and baked into the artifact (`genre_source: graph`). See the
`genre-data-authority` skill for the layer map.

---

**Note:** `docs/archive/` and `docs/run_audits/` are gitignored working areas, not shipped docs.
