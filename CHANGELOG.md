# Changelog

## [3.0.0] - 2025-12-27

### Major Release: Production Refresh + Pier-Bridge Ordering

This release represents a complete production-focused rebuild of the playlist generator with a new track ordering algorithm and cleaned-up codebase.

---

### New Features

#### Pier-Bridge Track Ordering
A new playlist construction strategy that ensures smooth transitions and prevents artist clustering:

- **Seed tracks as "piers"**: Anchor seeds are placed at evenly-spaced fixed positions throughout the playlist
- **Bridge construction**: Beam search finds optimal paths between each pair of piers
- **One-per-artist rule**: Each bridge segment allows only one track per artist, eliminating clustering
- **Transition optimization**: Seeds are reordered to maximize transition quality between consecutive piers

```
30-Track Playlist with 4 Seeds:

[SEED 1] ──bridge── [SEED 2] ──bridge── [SEED 3] ──bridge── [SEED 4]
   ↓         ↓          ↓         ↓          ↓         ↓          ↓
  Pos 1    2-10      Pos 11    12-20      Pos 21    22-29      Pos 30
```

**Benefits:**
- No more 4 songs from one artist in a 5-track span
- Seeds stay locked at calculated positions (no drift during repair)
- Smooth sonic transitions via beam search optimization
- Per-segment diagnostics for debugging

#### Environment Doctor
New `tools/doctor.py` validates your setup before running:

```bash
python tools/doctor.py
```

Checks:
- Python version and dependencies
- Database exists with valid schema
- Artifacts present and correctly sized
- Config file valid
- Actionable fix instructions for each issue

---

### Improvements

#### Codebase Refresh
- **53 production modules** (down from 100+)
- **6 production scripts** (down from 19)
- **6 focused docs** (down from 34+)
- Removed all experiments, archives, and diagnostic folders
- Single clean package structure under `src/`

#### Documentation
- New `docs/GOLDEN_COMMANDS.md` - Canonical production workflows
- New `docs/ARCHITECTURE.md` - System design overview
- New `docs/CONFIG.md` - Complete configuration reference
- New `docs/TROUBLESHOOTING.md` - Common issues and fixes
- Removed stale implementation notes and session logs

#### DS Pipeline Modes
Four playlist generation modes with distinct behaviors:

| Mode | Similarity Floor | Artist Diversity | Use Case |
|------|-----------------|------------------|----------|
| `narrow` | 0.20 | 20% max | Deep focus on specific sound |
| `dynamic` | 0.15 | 12.5% max | Balanced variety (default) |
| `discover` | 0.10 | 5% max | Maximum exploration |
| `sonic_only` | - | - | Pure audio similarity, no genre filter |

```bash
python main_app.py --artist "Radiohead" --ds-mode discover
```

---

### Technical Changes

#### Beat3Tower Integration
- 137-dimensional sonic features (rhythm + timbre + harmony)
- Multi-segment extraction (start, mid, end, full)
- Tower-specific PCA normalization
- Configurable tower weights for similarity

#### Modular Playlist Pipeline
Refactored into discrete, testable components:
- `playlist/anchor_builder.py` - Seed selection
- `playlist/candidate_generator.py` - Candidate pool
- `playlist/filtering.py` - Constraint enforcement
- `playlist/scoring.py` - Track scoring
- `playlist/ordering.py` - Sequence optimization
- `playlist/pier_bridge_builder.py` - New ordering strategy
- `playlist/constructor.py` - Final assembly
- `playlist/reporter.py` - Diagnostics output

#### Configuration
New DS pipeline config options:

```yaml
playlists:
  ds_pipeline:
    mode: dynamic
    tower_weights:
      rhythm: 0.20
      timbre: 0.50
      harmony: 0.30
    transition_weights:
      rhythm: 0.40
      timbre: 0.35
      harmony: 0.25
    constraints:
      min_gap: 6
      hard_floor: true
      transition_floor: 0.20
```

---

### Removed

- `experiments/` - 11 experiment folders
- `archive/` - Legacy code and old diagnostics
- `diagnostics/` - One-off validation reports
- `ui/` - Unused web interface
- `api/` - REST API (can be re-added if needed)
- 13+ diagnostic/fixup scripts
- 30+ stale documentation files

---

### Migration

No migration required. The refreshed repo:
- Uses identical import paths (`from src.X`)
- Works with existing `data/metadata.db`
- Works with existing artifacts
- Uses same `config.yaml` format

Simply use `repo_refreshed/` as your working directory.

---

### Quick Start

```bash
cd repo_refreshed

# Verify environment
python tools/doctor.py

# Generate playlist
python main_app.py --artist "Radiohead" --tracks 30

# Or with specific mode
python main_app.py --artist "Radiohead" --ds-mode discover --tracks 30
```

See `docs/GOLDEN_COMMANDS.md` for complete workflow reference.
