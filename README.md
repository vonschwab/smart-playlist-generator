# Playlist Generator

**Version 6.0** — Learned MERT sonic embedding, pace bands, enriched-genre authority + layered taxonomy graph, multisource Claude enrichment, and a browser GUI.

## Overview

Generates intelligent playlists from a local music library by fusing learned sonic similarity with genre-aware routing:

- **Learned sonic embedding (MERT)** — the default sonic space is a learned music embedding (MERT-v1-95M, 768-d, anisotropy-corrected) that replaced the hand-built rhythm/timbre/harmony towers as the similarity backbone. The 162-d tower space remains a config-selectable rollback.
- **Pier-Bridge beam search** — seeds become fixed "piers"; beam search builds smooth bridges between each adjacent pair, with monotonic progress through sonic space.
- **Genre routing on a real taxonomy** — published, graph-resolved genres drive a per-segment genre arc with IDF weighting, preserving multi-genre signatures (shoegaze stays shoegaze, not "indie rock").
- **Four independent axes** — `cohesion_mode` tightens the beam; `genre_mode`, `sonic_mode`, and `pace_mode` shape the candidate pool, each tunable from `strict` through `off`.
- **Pace as tempo + rhythmic density** — BPM and onset-rate bands plus a soft rhythm penalty keep slow playlists slow and driving playlists driving — independent of the sonic embedding.
- **Artist identity resolution** — collaboration-aware constraints normalise "X feat. Y", "The X", and ensemble suffixes before diversity enforcement.
- **Browser GUI** — generate, run library analysis/enrichment, and review genres from a local web app.

## What's new in v6.0

### Learned sonic embedding (MERT)
The hand-built towers were perceptually unreliable (the dominant timbre tower rated Metallica ≈ Yeah Yeah Yeahs). v6.0 folds a learned **MERT-v1-95M** embedding into the artifact and makes it the **default** sonic space (`X_sonic_variant: mert`), post-processed with `whiten_l2` (mean-center → per-dim std → L2) fitted on the full library. Cross-catalog neighbour QA shows it beats the towers by ~45–93%. The 162-d `tower_weighted` space (rhythm 9 + timbre 57 + 2DFTM harmony 96, weights 0.20/0.50/0.30) stays in the artifact as the rollback — set `artifacts.sonic_variant_override: tower_weighted` and restart the GUI. (See `docs/MERT_WHITEN_NEIGHBORS_20SEEDS.md`.)

### Pace mode rebuilt on tempo + rhythmic density
The old rhythm-cosine floor (near-noise, unsatisfiable for ambient artists) is gone. Pace now gates on two embedding-independent **hard bands** — BPM log-distance and onset-rate log-distance — plus a **soft** rhythm penalty that demotes (never rejects) off-rhythm bridge edges. Because the bands read DB features, pace survives the MERT migration unchanged, and `narrow` is now usable for a beatless/ambient seed.

### Enriched-genre authority + layered taxonomy graph
Published genres are now the authority: `release_effective_genres` (written only by the enrichment **publish** stage, read via `src/genre/authority.py`), resolved against the **SP3a layered taxonomy graph** (`data/layered_genre_taxonomy.yaml`, ~455 genres). The generation artifact bakes these in (`genre_source: graph`). Genre chips in the GUI are graph-canonical, ordered most-specific → broadest.

### Multisource genre enrichment (Claude backend)
`scripts/analyze_library.py` runs the enrichment pipeline end to end as resumable stages — `scan → genres → discogs → lastfm → sonic → mert → enrich → publish → genre-sim → artifacts → …`. The `enrich` stage adjudicates unknown tags via Claude (no API billing — Agent SDK), with source-quality fusion (label-storefront vs artist-page weighting, never-drop local tags) and incremental skip of unchanged releases. Transient rate-limit failures pause cleanly before publish so partial enrichment never reaches the database.

### Genre Review panel (GUI)
A new **Genre Review** tab queues hybrid-evidence "review terms" per release for human accept/reject; decisions persist as user overrides and rebuild the published genres.

## Quick Start

```bash
# 1. Install (Python 3.11+ required)
pip install -e .[web]        # browser GUI + generation (recommended)
pip install -e .[web,dev]    # contributors: + pytest, ruff, mypy, pre-commit

# 2. Configure
cp config.example.yaml config.yaml
# Edit config.yaml: music_directory, database_path, API keys

# 3. Verify environment
python tools/doctor.py

# 4. Scan + enrich + build, all stages, one command
python scripts/analyze_library.py
#   or a subset:  python scripts/analyze_library.py --stages scan,sonic,mert,artifacts
#   see the stage list:  python scripts/analyze_library.py --help

# 5. Generate playlists
python main_app.py --artist "Slowdive" --tracks 30
python main_app.py --seeds "Alison,Lost and Found,Endless Summer" --tracks 30
python main_app.py --genre "shoegaze" --tracks 30

# 6. Launch the browser GUI (http://127.0.0.1:8770)
python tools/serve_web.py
```

The MERT sonic embedding requires a one-time extraction (`scripts/extract_mert_sidecar.py`,
run via the `mert` analyze stage) and fold (`scripts/fold_mert_into_artifact.py`); extraction
is CPU-heavy but resumable. See [docs/GOLDEN_COMMANDS.md](docs/GOLDEN_COMMANDS.md) for the full
command reference.

## Requirements

- Python 3.11+
- ~8 GB RAM for sonic analysis
- SSD recommended for feature extraction

## Playlist modes

Four independent axes control cohesion vs. discovery. Each can be set in the GUI or via CLI flags. `cohesion_mode` drives the beam search; the other three shape the candidate pool.

### Cohesion mode (`--cohesion-mode`)
Overall beam tightness — how strictly the bridge search holds to a coherent path.

| Mode | Behaviour |
|---|---|
| `strict` | Tightest beam — minimal drift between piers |
| `narrow` | Cohesive (default GUI) |
| `dynamic` | Balanced |
| `discover` | Loosest — allows more exploratory bridges |

### Genre mode (`--genre-mode`)
How closely candidates must match the seed's genre profile: `strict` (single-genre deep dives) · `narrow` (cohesive, default GUI) · `dynamic` (moderate variation) · `discover` (cross-genre) · `off` (ignore genre).

### Sonic mode (`--sonic-mode`)
Sonic-similarity admission floor in the learned MERT space. Floors are calibrated to MERT cosine percentiles (compressed near 0, unlike the old towers):

| Mode | Min sonic similarity |
|---|---|
| `strict` | 0.28 (p75) |
| `narrow` | 0.18 (p50, default GUI) |
| `dynamic` | 0.08 (p25) |
| `discover` | 0.00 |
| `off` | disabled |

### Pace mode (`--pace-mode`)
Tempo + rhythmic-density fidelity, independent of timbre. Two hard log-distance bands (BPM, onset-rate) plus a soft rhythm penalty; bands widen on segment backoff so pace never blows the generation budget.

| Mode | BPM band (adm/bridge, log₂) | Onset band (adm/bridge, log₂) | Rhythm soft penalty (thresh/strength) |
|---|---|---|---|
| `strict`  | 0.30 / 0.40 | 0.30 / 0.40 | 0.35 / 0.20 |
| `narrow`  | 0.50 / 0.60 | 0.50 / 0.60 | 0.25 / 0.15 |
| `dynamic` | 0.75 / 0.85 | 0.75 / 0.85 | 0.15 / 0.10 |
| `off`     | ∞ / ∞ | ∞ / ∞ | 0 / 0 |

Pace mode is orthogonal to sonic mode: `sonic_mode=narrow + pace_mode=strict` means "very similar texture, must also hold tempo and density." Because the bands read database features (`bpm_info`, `onset_rate`), pace works identically whether the sonic space is MERT or the towers.

### Examples

```bash
# Tight shoegaze/slowcore — stays slow, stays cohesive
python main_app.py --seeds "Alison,Felo de Se,Endless Summer" \
    --cohesion-mode narrow --genre-mode narrow --pace-mode strict --tracks 30

# Same genre, explore varied sonic textures
python main_app.py --artist "Bill Evans" \
    --genre-mode narrow --sonic-mode discover --tracks 30

# Pure sonic similarity, no genre constraint
python main_app.py --genre "ambient" \
    --genre-mode off --sonic-mode dynamic --tracks 30
```

## DJ Bridge mode (multi-seed playlists)

Multi-seed playlists use seeds as fixed "piers" and beam-search bridges between each adjacent pair. The genre routing plans a graph-genre arc across each segment using IDF-weighted interpolation.

```bash
python main_app.py --seeds "Slowdive,Beach House,Deerhunter,Helvetia" --tracks 30
```

Produces four segments — Slowdive → Beach House → Deerhunter → Helvetia — with smooth transitions and genre evolution at each bridge. See [docs/DJ_BRIDGE_ARCHITECTURE.md](docs/DJ_BRIDGE_ARCHITECTURE.md).

## Project structure

```
.
├── main_app.py                  # CLI entry point
├── config.example.yaml          # Configuration template (copy to config.yaml)
├── src/
│   ├── playlist/                # Generation pipeline
│   │   ├── pipeline/            # DS pipeline orchestration
│   │   ├── pier_bridge/         # Beam search, beam scoring, pace gate
│   │   ├── repair/              # Post-beam edge repair
│   │   ├── candidate_pool.py    # Admission filtering (sonic, genre, pace, IDF)
│   │   ├── genre_idf.py         # IDF computation for genre weighting
│   │   └── mode_presets.py      # Cohesion/genre/sonic/pace mode presets
│   ├── features/                # Audio feature extraction + artifact resolution
│   ├── similarity/              # Sonic variant computation (MERT / towers)
│   ├── genre/                   # Genre authority, taxonomy graph adapter, granularity
│   ├── ai_genre_enrichment/     # Multisource enrichment (collection + adjudication)
│   ├── playlist_web/            # FastAPI web app (Generate / Tools / Genre Review)
│   └── playlist_gui/            # Generation worker (NDJSON) + shared policy layer
├── web/                         # React + TypeScript + Vite browser front-end
├── scripts/                     # analyze_library orchestrator, scan, sonic/MERT, artifact build
├── tools/
│   ├── doctor.py                # Environment validator
│   ├── serve_web.py             # Browser GUI launcher
│   └── dead_code_audit.py       # Static reachability audit
├── tests/                       # pytest suite (smoke / integration / golden / slow markers)
└── docs/                        # See docs/README.md for the index
```

## Diagnostics

```yaml
# config.yaml — enable per-edge audit
playlists:
  ds_pipeline:
    pier_bridge:
      emit_selected_edge_audit: true   # per-edge T/S/G/bridge + BPM breakdown in logs
      edge_repair:
        enabled: true                  # opt-in post-beam swap for bad edges
```

- **Edge audit** (`emit_selected_edge_audit: true`): logs T, S, G, bridge score, BPM distance, and title flags for every transition.
- **Weakest-edge report**: always on — shows the lowest-T transitions with artist names.
- **Quality metrics**: every generation reports transition stats (min / mean / p10 / p90) and distinct-artist count.

### Track replacement (GUI)
Right-click any non-pier track in the playlist table → **Replace this track…**. The dialog offers Search, Best Match, Different Pace, Different Genre, and Different Sound — the auto modes require the replacement to clear the transition floor against both neighbours.

## Version history

| Version | Highlights |
|---|---|
| **6.0** | Learned MERT sonic embedding (default) + tower rollback; pace rebuilt on BPM + onset bands; enriched-genre authority + layered taxonomy graph; multisource Claude enrichment with publish/pause safety; Genre Review GUI panel + graph-canonical chips; four mode axes (adds cohesion); browser GUI as sole front-end |
| **5.0** | Four-level pace mode; transition weight alignment; IDF admission; uncapped seeded pool; scoped blacklisting |
| **4.0** | Native GUI overhaul; CLI parity; Analyze Library readouts |
| **3.5** | Job cancellation/checkpoints; persistent genre cache; collaboration-aware artist clustering |
| **3.4** | DJ Bridge mode; union pooling; per-run audit reports |
| **3.3** | Seed List mode; sonic/genre modes; blacklist support |

Full release notes: [docs/CHANGELOG.md](docs/CHANGELOG.md)

## License

MIT
