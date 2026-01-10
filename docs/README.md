# Documentation Index

Welcome to the Playlist Generator documentation. This index provides an overview of all available documentation and how to navigate it.

---

## 📖 Core Documentation

### Getting Started
- **[GOLDEN_COMMANDS.md](GOLDEN_COMMANDS.md)** - Essential commands for common workflows
- **[CONFIG.md](CONFIG.md)** - Configuration reference and examples
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Common issues and solutions

### Architecture & Design
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - System architecture overview
- **[DJ_BRIDGE_ARCHITECTURE.md](DJ_BRIDGE_ARCHITECTURE.md)** - Complete DJ bridging design (Phase 1 & 2)
- **[TECHNICAL_PLAYLIST_GENERATION_FLOW.md](TECHNICAL_PLAYLIST_GENERATION_FLOW.md)** - Detailed playlist generation flow

### Release Notes
- **[CHANGELOG_Phase2.md](CHANGELOG_Phase2.md)** - DJ Bridging Phase 2 release notes
- **[TODO.md](TODO.md)** - Current roadmap and pending work

### Development
- **[LOGGING.md](LOGGING.md)** - Logging architecture and usage

---

## 📂 Directory Structure

```
docs/
├── README.md (this file)           # Documentation index
├── ARCHITECTURE.md                 # System architecture
├── CONFIG.md                       # Configuration reference
├── GOLDEN_COMMANDS.md              # Quick command reference
├── TROUBLESHOOTING.md              # Common issues
├── TODO.md                         # Roadmap and pending work
├── CHANGELOG_Phase2.md             # Release notes
├── DJ_BRIDGE_ARCHITECTURE.md       # DJ bridging design
├── TECHNICAL_PLAYLIST_*.md         # Technical deep-dives
├── LOGGING.md                      # Logging architecture
├── diagnostics/                    # Active diagnostic reports
│   └── README.md                   # Diagnostics directory guide
└── archive/                        # Archived documentation (git-ignored)
    ├── README.md                   # Archive index
    ├── dev_cycle_2026-01-02/       # Genre mode development
    └── diagnostics_2026-01/        # DJ bridging diagnostics
```

---

## 🎯 Quick Navigation

### I want to...

**...generate a playlist**
→ Start with [GOLDEN_COMMANDS.md](GOLDEN_COMMANDS.md)

**...configure the system**
→ See [CONFIG.md](CONFIG.md) for all configuration options

**...understand how it works**
→ Read [ARCHITECTURE.md](ARCHITECTURE.md) for high-level overview
→ Read [DJ_BRIDGE_ARCHITECTURE.md](DJ_BRIDGE_ARCHITECTURE.md) for DJ bridging details

**...troubleshoot an issue**
→ Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
→ Review logs in `logs/playlist_generator.log`

**...contribute or extend the codebase**
→ Start with [ARCHITECTURE.md](ARCHITECTURE.md)
→ Review [LOGGING.md](LOGGING.md) for logging conventions
→ Check [TODO.md](TODO.md) for roadmap

**...understand DJ bridging and genre integration**
→ Read [DJ_BRIDGE_ARCHITECTURE.md](DJ_BRIDGE_ARCHITECTURE.md) (comprehensive design doc)
→ Check [CHANGELOG_Phase2.md](CHANGELOG_Phase2.md) for implementation summary

---

## 🔍 Key Concepts

### Playlist Generation Modes
- **Artist Mode**: Generate from a seed artist using sonic similarity
- **Genre Mode**: Generate from a genre with DJ bridging
- **History Mode**: Generate from listening history

### DJ Bridging (Phase 2)
- **Union Pooling**: Combines local + toward + genre candidate pools
- **Waypoint Guidance**: Genre-guided beam search with IDF weighting
- **Coverage Bonus**: Rewards matching anchor top-K genres with schedule decay

### Sonic Analysis
- **beat3tower**: 3-tower feature extraction (rhythm, timbre, harmony)
- **Hybrid Similarity**: Combines multiple sonic dimensions
- **Artifacts**: Pre-computed similarity matrices for fast lookups

---

## 📝 Documentation Standards

### File Naming
- `UPPERCASE.md` - Core documentation (permanent)
- `lowercase_with_underscores.md` - Archived/temporary docs
- `CHANGELOG_*.md` - Release notes and changelogs

### Structure
- Use clear headings and table of contents for long docs
- Include examples for configuration and commands
- Reference source files with line numbers when applicable
- Keep archived docs in `archive/` directory

### Updates
- Update `TODO.md` after completing features
- Add release notes to `CHANGELOG_*.md`
- Archive diagnostics after development cycles complete
- Keep this index updated when adding new core docs

---

## 🗂️ Archive Policy

Diagnostic reports, A/B tests, and design exploration documents are archived after development cycles complete:

- **Location**: `docs/archive/` (git-ignored)
- **Organization**: By date or development cycle
- **Purpose**: Historical reference, not active documentation

See [archive/README.md](archive/README.md) for archive details.

---

**Last Updated:** 2026-01-10
