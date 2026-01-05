# Playlist Generator GUI

A native Windows desktop application for Data Science-powered playlist generation, built with PySide6/Qt Widgets.

## Features

### Simple, Clean Interface
- **Artist Mode** (default): Generate playlists from a specific artist and optional seed list (Seed List mode)
- **Genre Mode**: Generate playlists by genre with smart autocomplete showing exact matches and similar genres (similarity ≥ 0.7)
- **History Mode**: Generate playlists from your Last.FM listening history
- **Seed List Mode**: Add multiple explicit seed tracks with per-row autocomplete
- **Predictive Autocomplete**: Artist, seed track, and genre inputs query your music database with accent-insensitive matching
- **Real-time Progress**: Visual progress bar with stage information during generation

### Advanced Settings Panel
- **Schema-Driven Controls**: All settings automatically rendered from a central schema
- **Normalized Weight Groups**: Sliders that automatically adjust to sum to 1.0
- **Expandable Info**: Click the "?" button on any setting to see detailed explanations
- **Configuration File Selection**: Load different config.yaml files for different setups

### Override Highlighting & Reset
- **Modified Indicators**: Settings with values different from the base config are highlighted with:
  - Blue dot (●) indicator
  - Bold label text with blue color
  - Subtle blue background tint
- **Per-Setting Reset**: Click the ↺ button next to any modified setting to reset it to the base config value
- **Group Reset**: Each settings group has a "Reset Group" button to reset all settings in that group
- **Global Reset**: "Reset All Overrides" button at the top confirms before reverting all changes

**Important**: Overrides are stored in memory only - they do NOT modify config.yaml unless explicitly saved as a preset.

### Status Bar Indicator
The status bar shows the current override/preset state:

| Status | Meaning |
|--------|---------|
| `Base config` | No overrides active, using config.yaml values |
| `Overrides active (N)` | N individual settings are overridden |
| `Preset: <name>` | A preset is loaded and matches its saved values |
| `Preset: <name> (modified)` | A preset is loaded but user has changed settings since |

### Presets System
- **Built-in Presets**: Focused, Discovery, Smooth Transitions, Artist Variety, Pure Sonic
- **Save Custom Presets**: Store your favorite settings combinations
- **Import/Export**: Share presets as YAML files
- **Stored in AppData**: `%APPDATA%\PlaylistGenerator\presets\`

### Library Tools
- **Scan Library**: Index your music directory
- **Update Genres**: Fetch genre metadata from MusicBrainz/Discogs
- **Update Sonic Features**: Extract audio fingerprints (rhythm, timbre, harmony)
- **Build Artifacts**: Generate optimized similarity matrices

### Export Playlists
After generating a playlist, export it using the buttons below the track table:

#### Export to Local (M3U8)
- Click "Export to Local (M3U8)" to save the playlist as an M3U8 file
- **Playlist Name**: Editable with default "Auto - <Artist> <Date>"
- **Export Directory**: Configurable with browse button (default: `E:\PLAYLISTS` or `playlists.m3u_export_path` from config)
- **File Preview**: Shows the full path before saving
- Creates directories automatically if they don't exist

#### Export to Plex
- Click "Export to Plex" to create/update a Plex playlist
- **Playlist Name**: Editable with default "Auto - <Artist> <Date>"
- **Replace Existing**: If a playlist with the same name exists, it will be replaced
- **Requirements**: Configure Plex in config.yaml:
  ```yaml
  plex:
    enabled: true
    base_url: "http://your-plex-server:32400"
    music_section: "Music"  # Optional
    verify_ssl: true
    replace_existing: true
  ```
- Set `PLEX_TOKEN` environment variable or `plex.token` in config

### Track Table (Foobar2000-style)
- **Model/View Architecture**: Uses QTableView + QAbstractTableModel for performance with large playlists
- **Sortable Columns**: Click any column header to sort (supports numeric and case-insensitive text sorting)
- **Quick Filter**: Filter tracks by artist/title/album (optional: include file path)
- **Multi-Select**: Extended selection with Shift+Click and Ctrl+Click
- **Double-Click**: Opens the track file with default audio player

#### Context Menu (Right-Click)
- **Copy**: Artist, Title, Album, File Path, Selected Rows as "Artist - Title", File Paths
- **Open**: Open File, Open Containing Folder (with file selection on Windows)
- **Export**: Export selection or whole playlist as M3U8

#### Keyboard Shortcuts
| Shortcut | Action |
|----------|--------|
| `Ctrl+F` | Focus filter box |
| `Esc` | Clear filter (when filter focused) |
| `Ctrl+C` | Copy selected rows as "Artist - Title" |
| Double-click | Open file with default player |

#### Status Display
Shows "Showing N of M tracks" when filter is active.

## Architecture

### Two-Process Model

```
┌─────────────────────────────────────────────────────────────────────┐
│                         GUI Process (Main)                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │ MainWindow   │  │ Advanced     │  │ WorkerClient             │  │
│  │              │  │ Settings     │  │ (QProcess + NDJSON)      │  │
│  │ - Mode       │  │ Panel        │  │                          │  │
│  │ - Artist     │  │              │  │  stdin ──> commands      │  │
│  │ - Track      │  │ - Sliders    │  │  stdout <── events       │  │
│  │ - Generate   │  │ - Spinboxes  │  │                          │  │
│  │ - Progress   │  │ - Checkboxes │  │  Signals:                │  │
│  │ - Tracks     │  │ - Info (?)   │  │  - log_received          │  │
│  │              │  │              │  │  - progress_received     │  │
│  │              │  │              │  │  - result_received       │  │
│  │              │  │              │  │  - error_received        │  │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                                │
                           QProcess
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       Worker Process (Child)                         │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    NDJSON Protocol Handler                    │   │
│  │                                                               │   │
│  │  Commands (stdin):              Events (stdout):              │   │
│  │  - ping                         - log (level, msg)            │   │
│  │  - generate_playlist            - progress (stage, %)         │   │
│  │  - scan_library                 - result (playlist data)      │   │
│  │  - update_genres                - error (message, traceback)  │   │
│  │  - update_sonic                 - done (cmd, ok, detail)      │   │
│  │  - build_artifacts                                            │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                │                                     │
│                                ▼                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                   Playlist Generator Core                     │   │
│  │                                                               │   │
│  │  - DS Pipeline Runner (candidate selection, transition)       │   │
│  │  - Pier-Bridge Ordering (smooth track sequences)              │   │
│  │  - Sonic Feature Analysis (rhythm, timbre, harmony)           │   │
│  │  - Genre Embedding (MusicBrainz, Discogs)                     │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

### Why Two Processes?

1. **Always Responsive UI**: The GUI never freezes during heavy computation
2. **Clean Separation**: UI logic separate from generation logic
3. **Crash Isolation**: Worker crash doesn't kill the GUI
4. **Resource Management**: Worker can be killed/restarted independently

### NDJSON Protocol (Version 1)

Commands and events use newline-delimited JSON (one JSON object per line).

**Protocol Features:**
- All commands include `request_id` (UUID) for correlation
- All events include matching `request_id` to identify the originating command
- `cancel` command for cooperative cancellation of active requests
- Single active job limit (MVP) - new commands rejected while busy

**Commands (GUI → Worker):**
```json
{"cmd": "ping", "request_id": "<uuid>", "protocol_version": 1}
{"cmd": "generate_playlist", "request_id": "<uuid>", "protocol_version": 1, "base_config_path": "config.yaml", "overrides": {...}, "args": {"mode": "artist", "artist": "Lomelda", "track": "Hannah Sun", "tracks": 30}}
{"cmd": "scan_library", "request_id": "<uuid>", "protocol_version": 1, "base_config_path": "config.yaml", "overrides": {}}
{"cmd": "cancel", "request_id": "<uuid-to-cancel>"}
```

**Events (Worker → GUI):**
```json
{"type": "log", "request_id": "<uuid>", "level": "INFO", "msg": "Loading configuration..."}
{"type": "progress", "request_id": "<uuid>", "stage": "generate", "current": 60, "total": 100, "detail": "Building candidate pool"}
{"type": "result", "request_id": "<uuid>", "result_type": "playlist", "playlist": {"name": "...", "tracks": [...]}}
{"type": "error", "request_id": "<uuid>", "message": "...", "traceback": "..."}
{"type": "done", "request_id": "<uuid>", "cmd": "generate_playlist", "ok": true, "detail": "Generated 30 tracks"}
{"type": "done", "request_id": "<uuid>", "cmd": "generate_playlist", "ok": false, "cancelled": true, "detail": "Cancelled by user"}
```

**Cancellation Flow:**
1. GUI sends `cancel` command with the `request_id` to cancel
2. Worker sets internal cancellation flag
3. Worker checks flag at stage boundaries and in loops
4. If cancelled, worker emits `done` event with `cancelled: true`
5. If worker doesn't respond within 5 seconds, GUI force-kills and restarts worker

**Busy State:**
- GUI tracks busy state via `WorkerClient.is_busy()`
- Generate button disabled when busy
- Cancel button enabled when busy
- Tools menu disabled when busy

### Security

- **Secret Redaction**: API keys, tokens, and passwords are automatically redacted from all logs and events
- **Regex Pattern**: Matches common secret patterns (`api_key`, `token`, `secret`, `password`, `credential`, `bearer`)
- **Config Protection**: Secrets loaded from environment variables when possible

## File Structure

```
src/playlist_gui/
├── __init__.py           # Package init
├── __main__.py           # Entry point for `python -m playlist_gui.app`
├── app.py                # QApplication setup and launch
├── main_window.py        # Main window with controls and layout
├── autocomplete.py       # Database-driven artist/track autocomplete
├── worker.py             # Worker process (NDJSON handler)
├── worker_client.py      # QProcess wrapper with Qt signals
├── config/
│   ├── __init__.py
│   ├── config_model.py   # Config loading, merging, dot-path access
│   ├── settings_schema.py # Schema definitions for all settings
│   └── presets.py        # Preset management (save/load/builtin)
└── widgets/
    ├── __init__.py
    ├── advanced_panel.py # Schema-driven settings UI
    ├── log_panel.py      # Filterable log display
    └── track_table.py    # Playlist track table
```

## Installation

### Requirements

```
PySide6>=6.5.0
pyyaml>=6.0
platformdirs>=3.0
```

### Install Dependencies

```bash
pip install -r requirements-gui.txt
```

### Run from Source

```bash
# From project root
python -m playlist_gui.app

# Or using the app entry point
python src/playlist_gui/app.py
```

### Build Windows Executable

```powershell
# Uses PyInstaller
.\scripts\build_windows.ps1
```

## Configuration

### Base Config File

The GUI loads settings from a YAML config file (default: `config.yaml`). You can switch config files via the Advanced Settings panel.

### Override System

Changes made in the GUI are stored as "overrides" that merge with the base config:

```
Final Config = Base Config (YAML) + GUI Overrides
```

This allows you to:
- Keep your base config unchanged
- Experiment with different settings
- Save successful combinations as presets
- Reset to base config at any time

### Normalized Weight Groups

Three weight groups automatically normalize to sum to 1.0:

| Group | Weights | Purpose |
|-------|---------|---------|
| **Embedding Weights** | sonic_weight, genre_weight | Balance audio vs. genre similarity |
| **Tower Weights** | rhythm, timbre, harmony | Weight audio feature towers for candidate selection |
| **Transition Weights** | rhythm, timbre, harmony | Weight features for track-to-track flow |

When you adjust one slider, others in the same group adjust proportionally.

## Settings Reference

### Playlist Settings
| Setting | Default | Description |
|---------|---------|-------------|
| Playlists per batch | 8 | Number of playlists in History mode |
| Tracks per playlist | 30 | Target playlist length |
| Seed count | 5 | Tracks from history to seed from |
| Similar per seed | 20 | Candidates per seed track |

### Hybrid Weights (sum to 1.0)
| Setting | Default | Description |
|---------|---------|-------------|
| Sonic weight | 0.60 | Influence of audio features |
| Genre weight | 0.40 | Influence of genre metadata |

### Tower Weights (sum to 1.0)
| Setting | Default | Description |
|---------|---------|-------------|
| Rhythm | 0.20 | Tempo, beat patterns |
| Timbre | 0.50 | Texture, instrumentation |
| Harmony | 0.30 | Key, chord progressions |

### Scoring
| Setting | Default | Description |
|---------|---------|-------------|
| Alpha | 0.55 | Weight for seed similarity |
| Beta | 0.55 | Weight for transition quality |
| Gamma | 0.04 | Bonus for artist diversity |
| Alpha schedule | arc | constant or arc (narrative) |

### Constraints
| Setting | Default | Description |
|---------|---------|-------------|
| Min artist gap | 6 | Tracks between same artist |
| Transition floor | 0.20 | Minimum transition quality |
| Hard floor | true | Reject vs. penalize bad transitions |

### Candidate Pool
| Setting | Default | Description |
|---------|---------|-------------|
| Similarity floor | 0.20 | Minimum similarity to seed |
| Max pool size | 1200 | Maximum candidates |
| Max artist fraction | 0.125 | Max % from one artist |

### Pipeline Modes
| Mode | Description |
|------|-------------|
| narrow | Focused, stay close to seed |
| dynamic | Balanced mix (default) |
| discover | Explore further from seed |
| sonic_only | Pure audio, ignore genres |

### Genre/Sonic Modes
| Mode | Description |
|------|-------------|
| strict | Ultra-tight matching |
| narrow | Cohesive matching |
| dynamic | Balanced matching (default) |
| discover | Exploratory matching |
| off | Disable the domain (pure sonic or pure genre) |

## Built-in Presets

| Preset | Description | Key Settings |
|--------|-------------|--------------|
| **Focused** | Narrow, cohesive playlists | alpha=0.70, similarity_floor=0.35 |
| **Discovery** | Explore new music | alpha=0.40, similarity_floor=0.10 |
| **Smooth Transitions** | Prioritize flow | beta=0.70, transition_floor=0.30 |
| **Artist Variety** | Maximize diversity | gamma=0.08, min_gap=8 |
| **Pure Sonic** | Audio only, no genres | sonic_weight=1.0, genre_weight=0.0 |

## Testing

```bash
# Run GUI config tests
pytest tests/unit/test_gui_config.py -v
```

Tests cover:
- ConfigModel loading and merging
- Weight normalization logic
- Secret redaction
- Settings schema validation
- Preset management

## Development

### Adding New Settings

1. Add a `SettingSpec` to `SETTINGS_SCHEMA` in `config/settings_schema.py`
2. The Advanced Settings panel automatically renders the control
3. The config model handles merging and validation

Example:
```python
SettingSpec(
    key_path="playlists.ds_pipeline.new_setting",
    label="New Setting",
    setting_type=SettingType.FLOAT,
    group="Scoring",
    default=0.5,
    min_value=0.0,
    max_value=1.0,
    step=0.05,
    tooltip="Short description",
    description="Detailed explanation shown when clicking ?"
)
```

### Adding New Worker Commands

1. Add handler function in `worker.py`
2. Register in `COMMAND_HANDLERS` dict
3. Add corresponding method in `worker_client.py`
4. Connect UI element to trigger the command

## Troubleshooting

### GUI Won't Start
- Check Python version (3.9+ required)
- Verify PySide6 is installed: `pip install PySide6`
- Run with debug: `python -m playlist_gui.app 2>&1`

### Worker Crashes
- Check config.yaml exists and is valid YAML
- Verify database path in config is correct
- Check worker logs in the Log panel

### Autocomplete Not Working
- Ensure database exists at configured path
- Check "Database: X artists, Y tracks" in Advanced Settings
- Run library scan if database is empty

### Playlists Too Short
- Lower similarity_floor (try 0.10-0.15)
- Increase max_pool_size
- Set hard_floor to false
- Lower min_gap constraint

## License

Part of the Playlist Generator project.
