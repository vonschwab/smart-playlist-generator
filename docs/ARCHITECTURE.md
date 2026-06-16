# Architecture

High-level map of the system as of **v6.0**. For the code-level walkthrough of a
single generation (file:line references through every phase) see
[`TECHNICAL_PLAYLIST_GENERATION_FLOW.md`](TECHNICAL_PLAYLIST_GENERATION_FLOW.md) —
that is the authoritative implementation doc; this file is the orientation layer.

## System Overview

```
┌──────────────────────────── Offline (analyze pipeline) ─────────────────────────────┐
│  scan ─▶ genres ─▶ discogs ─▶ lastfm ─▶ sonic ─▶ mert ─▶ enrich ─▶ publish           │
│                                          │         │         │         │              │
│                                          ▼         ▼         ▼         ▼              │
│                                    sonic feats  MERT    Claude tag   release_         │
│                                    (towers)    embeds   adjudication  effective_      │
│                                                          + taxonomy   genres          │
│                                                          graph        (authority)     │
│                                          └──────┬──────────────────────┘              │
│                                                 ▼                                      │
│                                  genre-sim ─▶ artifacts ─▶ genre-embedding ─▶ verify  │
│                                                 │                                      │
│                                                 ▼                                      │
│                                      data_matrices_step1.npz  (+ MERT sidecar,        │
│                                      dense genre embedding sidecar)                    │
└───────────────────────────────────────────────┬──────────────────────────────────────┘
                                                 │  (warm path — never re-runs offline work)
┌────────────────────────────── Runtime (generation) ───────┴──────────────────────────┐
│  CLI (main_app.py)        Browser GUI (React) ─NDJSON─▶ worker ─▶ DS pier-bridge       │
│        └──────────────────────────┬──────────────────────────────────┘                │
│                                    ▼                                                    │
│                       artifact bundle ─▶ candidate pool ─▶ beam-searched bridges       │
│                       ─▶ repair pass ─▶ M3U / Plex export                              │
└────────────────────────────────────────────────────────────────────────────────────┘
```

Two halves, by deliberate design (Layer-2 commitment #14, *local-first*):

- **Offline** does all the heavy work once — scan, genre fetch + enrichment, audio
  analysis, artifact build. External APIs (Last.fm, MusicBrainz, Discogs, Claude) only
  run here; they never gate a generation.
- **Runtime** is warm and fast (~25–55s/playlist): load the prebuilt artifact, build a
  candidate pool, beam-search bridges, repair, export. No network, no DB writes to the
  irreplaceable stores.

## Offline: the analyze pipeline

`scripts/analyze_library.py` runs twelve stages (`STAGE_FUNCS`); the canonical order and
per-stage CLI are in [`GOLDEN_COMMANDS.md`](GOLDEN_COMMANDS.md). The stage list is also
the contract the worker + web Tools panel drive (`request_models.AnalyzeLibraryStage`).

| Stage | Role |
|-------|------|
| `scan` | Walk `library.music_directory`, extract metadata (Mutagen), populate `tracks` |
| `genres` | MusicBrainz artist + file-tag genres → normalized genre tables |
| `discogs` | Album genres from Discogs |
| `lastfm` | Last.fm top tags into the enrichment sidecar (no LLM) |
| `sonic` | beat3tower audio features (librosa) → `tracks.sonic_features` |
| `mert` | MERT learned audio embeddings → resumable shards + merged sidecar |
| `enrich` | Adjudicate unknown tags via **Claude** (Agent SDK, no API billing); materialize layered-graph genres into `ai_genre_enrichment.db` |
| `publish` | Resolve graph-where-present-else-legacy into `release_effective_genres` in `metadata.db` (first run backs up the DB, timestamped; idempotent after) |
| `genre-sim` | Genre-similarity matrix (co-occurrence Jaccard or taxonomy-graph) used to smooth genre vectors at build time |
| `artifacts` | Build the `.npz` artifact bundle (sonic variants + genre matrices) |
| `genre-embedding` | Dense PMI-SVD genre embedding sidecar |
| `verify` | Sanity-check the built artifacts |

### Sonic feature space

Two representations coexist in the artifact; one is the live default, the other the
rollback:

- **MERT learned embedding (`X_sonic_variant: mert`) — the v6.0 default.** Folded into
  the artifact by `scripts/fold_mert_into_artifact.py` from the irreplaceable MERT
  shards/sidecar. The artifact loader (`src/features/artifacts.py`) selects the
  `X_sonic_mert` matrix and marks it *pre-scaled*, so the runtime uses it directly with
  no further transform. A configured `artifacts.sonic_variant_override` whose key is
  missing is a **hard startup error**, never a silent fallback.
- **Tower-weighted blend (162-dim) — the rollback.** Rhythm 9 / timbre 57 / harmony 96,
  weighted 0.20 / 0.50 / 0.30, baked at build time. Harmony uses **2DFTM** (key-invariant
  2D Fourier Transform Magnitude of the chromagram, validated 2026-06-03 — replaces the
  legacy absolute-key chroma-median tower). This is the space `sonic_variant.py`'s
  transform variants (`tower_pca`, `tower_weighted`, `robust_whiten`, …) operate on; MERT
  is *not* one of those transforms — it is a pre-scaled artifact matrix.

> The MERT-default flip is committed but the perceptual listen-test gate is still open;
> until it closes, CLAUDE.md Layer-2 principles #8/#17/#18 still describe the tower
> decomposition as the conceptual model.

### Genre authority

The authority for a release's genres is **`release_effective_genres`** in `metadata.db`,
written by the `publish` stage and read through `src/genre/authority.py`. It resolves the
published layered taxonomy graph where present and falls back to legacy raw tags
elsewhere. Any new genre consumer (GUI, export, artifact build) must read through
`authority.py` — **not** the older bandcamp-era sidecar signatures, which are stale.
The artifact's genre source is selected by `ds_pipeline.genre_source`
(`legacy | enriched | graph | hybrid_shadow`). See the `genre-data-authority` skill.

## Runtime: generation

Entry points share one engine:

- **CLI** — `python main_app.py --artist "…" --tracks 30` (full reference in
  `GOLDEN_COMMANDS.md`).
- **Browser GUI** — `python tools/serve_web.py` (default port 8770). The only front-end;
  the PySide6 desktop GUI was removed 2026-06-10.

### Browser GUI wiring

React + TypeScript + Vite + Tailwind + shadcn (`web/`) talk to a **FastAPI** app
(`src/playlist_web/app.py`), which spawns and supervises the **NDJSON IPC worker**
(`src/playlist_gui/worker.py`). The worker runs generation, analyze stages, enrichment,
and the Genre Review backend in a child process; results stream back as newline-delimited
JSON. The policy layer in `src/playlist_gui/` is shared between CLI and web so both
resolve modes identically. Traps (stale `web/dist`, worker-not-restarted, silently
dropped results, end-to-end wiring) are catalogued in the `web-gui` skill — read it
before changing `web/src`, `src/playlist_web`, or worker command handlers.

### The four mode axes

Cohesion-vs-discovery is controlled by four orthogonal axes (all under `playlists.*` in
config; CLI `--cohesion-mode`, `--genre-mode`, `--sonic-mode`, `--pace-mode`):

- **`cohesion_mode`** (`strict | narrow | dynamic | discover`) — drives the **beam**
  tightness. The per-mode pier-bridge knobs (`bridge_floor_<mode>`, `weight_*_<mode>`,
  `soft_genre_penalty_*_<mode>`) are keyed by this.
- **`genre_mode`** (`strict | narrow | dynamic | discover | off`) — genre **pool gating**.
- **`sonic_mode`** (`strict | narrow | dynamic | off`) — sonic **pool gating**.
- **`pace_mode`** (`strict | narrow | dynamic | off`) — rhythm/tempo gating, independent
  of sonic. v6 uses **BPM + onset hard bands plus a soft penalty** (MERT-durable),
  replacing the older rhythm-cosine hard gate. `dynamic` preserves prior behavior;
  `strict`/`narrow` engage the admission bands and bridge gate.

> There is no `sonic_only` mode and no `ds_pipeline.mode` key any more. The closest
> equivalent to the old "sonic only" is `genre_mode: off` + `cohesion_mode: discover`.

### DS pier-bridge engine

The current best playlist topology (Layer-3 method #15). Seeds become **piers**;
**bridges** between consecutive piers are built by **beam search** over a per-segment
candidate pool, with progress monotonic in the sonic space. Genre-arc steering scores each
edge on the dense PMI-SVD embedding (or the taxonomy graph) and adds a renormalized genre
term alongside the bridge harmonic-mean and local transition scores. Diversity (min-gap,
per-artist cap, seed/pier-artist exclusion in interiors) is enforced as a hard constraint.
A post-construction **repair pass** re-works the weakest edges. Full beam mechanics live in
[`dj_bridge_architecture.md`](dj_bridge_architecture.md) and the code-level flow in
`TECHNICAL_PLAYLIST_GENERATION_FLOW.md`.

Quality metrics are first-class output (Layer-4 #21): every generation emits transition
stats (min / mean / p10 / p90), a weakest-edge report, and a distinct-artist count.

## Key data stores

| Path | Contents | Notes |
|------|----------|-------|
| `data/metadata.db` | `tracks`, normalized genre tables, `release_effective_genres`, `track_effective_genres` | **Irreplaceable** — treat like production; back up before any write |
| `data/ai_genre_enrichment.db` | Claude enrichment signatures, suggestions, user overrides, source pages | Authority feeds `publish` |
| `data/artifacts/beat3tower_32k/data_matrices_step1.npz` | Sonic (towers + MERT) + genre matrices | Rebuilt by the `artifacts` stage |
| `data/artifacts/beat3tower_32k/mert_shards/` + `mert_sidecar.npz` | MERT embeddings | **Irreplaceable** — ~55h CPU to regenerate |
| `data/genre_similarity.yaml` | Genre taxonomy overrides / neighbor map | Path is hardcoded at call sites |

Schema column reference (key columns per table) lives in `CLAUDE.md` → "SQLite schema
reference". Always `PRAGMA table_info(<table>)` before querying an unfamiliar table.

## Configuration

All behavior is driven by `config.yaml` (gitignored; copy from `config.example.yaml`).
Tunability over hardcoded behavior is a design principle (Layer-4 #23) — new behavior
ships behind a config flag with legacy defaults. A configured knob that cannot act must
warn loudly or raise; it must never be a silent no-op. Full key reference:
[`CONFIG.md`](CONFIG.md); tuning recipes: [`PLAYLIST_ORDERING_TUNING.md`](PLAYLIST_ORDERING_TUNING.md).

## Extension points

### Add an export format
1. Create the exporter under `src/` (mirror `src/m3u_exporter.py`).
2. Wire it into the export step of `main_app.py`.

### Add a sonic transform variant
1. Add the variant to `_ALLOWED` + `_variant_transform` in `src/similarity/sonic_variant.py`.
   (This is for *runtime transforms* over the tower blend. A whole new pre-scaled space —
   like MERT — is instead folded into the artifact and selected via `X_sonic_variant`.)

### Add / change a mode behavior
1. Per-mode knobs go in `config.yaml` keyed by the relevant axis (`cohesion_mode` for beam
   knobs; `genre_mode`/`sonic_mode`/`pace_mode` for pool gating).
2. Resolve them in the pier-bridge / candidate-pool config layer and document the recipe
   in `PLAYLIST_ORDERING_TUNING.md`. There is no `--ds-mode` flag — use `--cohesion-mode`.
