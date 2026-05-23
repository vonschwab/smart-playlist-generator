# Playlist Generator

**Version 5.0** — Pace mode, transition quality improvements, IDF-weighted candidate admission, and scoped blacklisting.

## Overview

Generates intelligent playlists from a local music library by combining deep sonic analysis with genre-aware routing:

- **Beat3Tower Sonic Analysis** — 137-dimensional audio feature vectors split into rhythm, timbre, and harmony towers
- **Pier-Bridge Beam Search** — Seeds become fixed "piers"; beam search builds smooth bridges between each adjacent pair
- **DJ Genre Routing** — Vector-mode interpolation with IDF weighting preserves multi-genre signatures (shoegaze stays shoegaze, not "indie rock")
- **Pace Mode** — Separate rhythm axis for admission and beam search; keeps slow playlists slow and energetic playlists driving
- **Three Independent Axes** — Genre mode, sonic mode, and pace mode each tunable from `strict` through `dynamic`
- **Artist Identity Resolution** — Collaboration-aware constraints normalise "X feat. Y", "The X", and ensemble suffixes before diversity enforcement
- **Scoped Blacklisting** — Block individual tracks, entire artists, or full albums; manual track blocks survive scope removal

## What's New in v5.0

### Pace Mode — rhythm as a separate axis
Genre and sonic modes control *what* music gets admitted. Pace mode controls *how fast* it moves. A `strict` slowcore playlist now stays slow even when fast noisy-guitar bands share the same timbre.

- **Tier 1:** Rhythm-axis admission floor (max-over-seeds cosine on the rhythm PCA sub-vector)
- **Tier 2:** Per-step moving target in beam search — interpolates between pier A and pier B's rhythm vectors, so a slow→fast arc works naturally when the piers themselves differ
- **CLI:** `--pace-mode strict|narrow|dynamic`
- **GUI:** Third mode slider alongside Genre and Sonic

### Transition quality
- `transition_weights` aligned with `tower_weights` (0.20 / 0.50 / 0.30) — fixed a long-standing mismatch where the beam approved edges that the reporter scored poorly
- Per-edge audit table (`emit_selected_edge_audit: true`) showing T, S, G, bridge, and trans_beam per transition
- Opt-in edge repair pass — single-pass post-beam swap for sub-floor edges with do-no-harm guarantees

### Candidate pool improvements
- **IDF-weighted genre admission** — rare tags (slowcore, shoegaze) outweigh common tags (indie rock) during candidate scoring
- **Uncapped seeded pool** — for seeded playlists the global pool is no longer hard-capped at 2400; per-artist cap (6 tracks) still applies, giving ~3500–4000 eligible candidates instead
- Genre conflict gate removed (was rejecting ~50% of legitimate candidates against the 764-dim vocabulary); soft genre compatibility penalty remains

### Scoped blacklisting (GUI)
Right-click any track in the playlist table → **Blacklist this artist** or **Blacklist this album**. Manual per-track blocks are preserved if a scope is later removed.

## Quick Start

```bash
# 1. Install (Python 3.11+ required)
pip install -e .[gui]        # GUI + generation
pip install -e .[gui,dev]    # + pytest, ruff, mypy, pre-commit

# 2. Configure
cp config.example.yaml config.yaml
# Edit config.yaml: music_directory, database_path, API keys

# 3. Verify environment
python tools/doctor.py

# 4. Scan your music library
python scripts/scan_library.py

# 5. (Optional) Fetch MusicBrainz MBIDs
python scripts/fetch_mbids_musicbrainz.py --limit 500

# 6. Extract sonic features
python scripts/update_sonic.py --beat3tower --workers 4

# 7. Fetch genre metadata
python scripts/update_genres_v3_normalized.py --artists

# 8. Build DS pipeline artifacts
python scripts/build_beat3tower_artifacts.py \
    --db-path data/metadata.db \
    --config config.yaml \
    --output data/artifacts/beat3tower_32k/data_matrices_step1.npz

# 9. Generate playlists
python main_app.py --artist "Slowdive" --tracks 30
python main_app.py --seeds "Alison,Lost and Found,Endless Summer" --tracks 30
python main_app.py --genre "shoegaze" --tracks 30

# 10. Launch GUI
python -m playlist_gui.app
```

See [docs/GOLDEN_COMMANDS.md](docs/GOLDEN_COMMANDS.md) for the full command reference.

## Requirements

- Python 3.11+
- ~8 GB RAM for sonic analysis
- SSD recommended for feature extraction

## Playlist Modes

Three independent axes control cohesion vs. discovery. Each can be set in the GUI or via CLI flags.

### Genre Mode (`--genre-mode`)
Controls how closely candidates must match the seed's genre profile.

| Mode | Behaviour |
|---|---|
| `strict` | Ultra-tight — single-genre deep dives |
| `narrow` | Stay close — cohesive exploration (default GUI) |
| `dynamic` | Balanced — moderate genre variation |
| `discover` | Adjacent exploration — cross-genre discovery |
| `off` | Ignore genre tags entirely |

### Sonic Mode (`--sonic-mode`)
Controls overall sonic similarity threshold (timbre + harmony + rhythm combined).

| Mode | Behaviour |
|---|---|
| `strict` | Laser-focused sound |
| `narrow` | Consistent texture (default GUI) |
| `dynamic` | Balanced sonic flow |
| `discover` | Wide sonic palette |
| `off` | Ignore sonic features entirely |

### Pace Mode (`--pace-mode`)
Controls rhythm/tempo fidelity independently from timbre. Use when seeds define an energy level you want to maintain.

| Mode | Admission floor | Bridge floor | Use case |
|---|---|---|---|
| `strict` | 0.55 / BPM 0.30 | 0.65 / BPM 0.40 | Lock to seed tempo |
| `narrow` | 0.35 / BPM 0.50 | 0.45 / BPM 0.60 | Moderate anchoring |
| `dynamic` | 0.20 / BPM 0.75 | 0.25 / BPM 0.85 | Gentle — catches double-time (default) |
| `off` | 0 / ∞ | 0 / ∞ | No pace constraint |

Pace mode is orthogonal to sonic mode. `sonic_mode=narrow + pace_mode=strict` means "very similar timbre, must also stay slow." In `off` mode, no explicit pace gating is applied, but rhythm still contributes to track selection via the sonic embedding at 20% weight.

### Examples

```bash
# Tight shoegaze/slowcore — stays slow, stays noisy
python main_app.py --seeds "Alison,Felo de Se,Endless Summer" \
    --genre-mode narrow --sonic-mode narrow --pace-mode strict --tracks 30

# Same genre, explore varied sonic textures
python main_app.py --artist "Bill Evans" \
    --genre-mode narrow --sonic-mode discover --tracks 30

# Pure sonic similarity, no genre constraint
python main_app.py --genre "ambient" \
    --genre-mode off --sonic-mode dynamic --tracks 30
```

## DJ Bridge Mode (Multi-Seed Playlists)

Multi-seed playlists use seeds as fixed "piers" and beam-search bridges between each adjacent pair. The DJ genre routing system plans a genre arc across each segment using IDF-weighted vector interpolation.

```bash
python main_app.py --seeds "Slowdive,Beach House,Deerhunter,Helvetia" --tracks 30
```

Produces four segments: Slowdive → Beach House → Deerhunter → Helvetia, with smooth transitions and genre evolution at each bridge.

**Key settings** (defaults already tuned; tweak in `config.yaml`):

```yaml
playlists:
  ds_pipeline:
    pier_bridge:
      dj_bridging:
        enabled: true
        dj_ladder_target_mode: vector       # multi-genre interpolation
        dj_genre_use_idf: true              # rare tags weighted higher
        dj_genre_use_coverage: true         # reward seed-signature matching
        dj_waypoint_delta_mode: centered    # prevents waypoint saturation
        dj_waypoint_squash: tanh            # smooth scoring curve
```

See [docs/DJ_BRIDGE_ARCHITECTURE.md](docs/DJ_BRIDGE_ARCHITECTURE.md) for the full architecture.

## Project Structure

```
.
├── main_app.py                  # CLI entry point
├── config.example.yaml          # Configuration template
├── src/
│   ├── playlist/                # Generation pipeline
│   │   ├── pipeline/            # DS pipeline orchestration
│   │   ├── pier_bridge/         # Beam search, beam scoring, pace gate
│   │   ├── repair/              # Post-beam edge repair
│   │   ├── candidate_pool.py    # Admission filtering (sonic, genre, pace, IDF)
│   │   ├── sonic_axes.py        # Per-tower sub-vector slicing
│   │   ├── genre_idf.py         # IDF computation for genre weighting
│   │   ├── transition_metrics.py# Shared T/S/G edge scoring
│   │   └── mode_presets.py      # Genre/sonic/pace mode presets
│   ├── features/                # Audio feature extraction
│   ├── similarity/              # Sonic variant computation (tower_pca)
│   ├── metadata_client.py       # Track DB + scoped blacklisting
│   └── playlist_gui/            # PySide6 GUI
│       └── widgets/
│           ├── mode_sliders.py  # Genre / Sonic / Pace sliders
│           └── track_table.py   # Playlist table + context-menu blacklisting
├── scripts/                     # Library scan, feature extraction, artifact build
├── tools/
│   └── doctor.py                # Environment validator
├── tests/                       # 996-test suite (pytest)
└── docs/
    ├── README.md                # Documentation index
    ├── GOLDEN_COMMANDS.md       # Command reference
    ├── CONFIG.md                # Config key reference
    ├── DJ_BRIDGE_ARCHITECTURE.md# DJ bridge internals
    ├── PLAYLIST_ORDERING_TUNING.md # Knob-by-knob tuning guide
    ├── CANDIDATE_FILTERING_BACKLOG.md # Deferred filtering work
    └── TECHNICAL_PLAYLIST_GENERATION_FLOW.md # Full pipeline walkthrough
```

## Diagnostics

```yaml
# config.yaml — enable per-edge audit
playlists:
  ds_pipeline:
    pier_bridge:
      emit_selected_edge_audit: true   # per-edge T/S/G/bridge breakdown in logs
      edge_repair:
        enabled: true                  # opt-in post-beam swap for bad edges
```

- **Edge audit** (`emit_selected_edge_audit: true`): logs T, T_centered_cos, S, G, bridge score, and title flags for every transition
- **Weakest-edge report**: always on — shows the 3 lowest-T transitions with artist names
- **Pace admission log**: `Pace admission floor applied: floor=0.55 rejected=N` when pace_mode is active
- **GUI debug report**: Help → Copy/Save Debug Report — redacted bundle of env, config, last job, and log tail

## Version History

| Version | Highlights |
|---|---|
| **5.0** | Four-level pace mode (strict/narrow/dynamic/off); transition weight alignment; IDF admission; uncapped seeded pool; scoped blacklisting GUI |
| **4.0** | Native GUI overhaul; CLI parity; responsive generation controls; Analyze Library readouts |
| **3.5** | Job cancellation/checkpoints; job-details dialog; persistent genre cache; collaboration-aware artist clustering |
| **3.4** | DJ Bridge mode; union pooling; per-run audit reports |
| **3.3** | Seed List mode; sonic/genre modes; blacklist support |
| **3.2** | Windows GUI; MBID enrichment; artist normalisation |

Full release notes: [docs/CHANGELOG.md](docs/CHANGELOG.md)

## License

MIT
