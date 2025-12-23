# Changelog

## Version 2.0 - Beat3Tower Production Release

**Release Date**: December 2025

### 🎵 Major Features

#### Beat3Tower Sonic Analysis
- **3-Tower Architecture**: Complete rewrite of sonic similarity using rhythm (21-dim), timbre (83-dim), and harmony (33-dim) towers for total 137 dimensions
- **Multi-Segment Analysis**: Analyzes 4 segments per track (full, start, mid, end) for intelligent transition matching
- **Tower PCA Preprocessing**: New default preprocessing with per-tower standardization, PCA dimensionality reduction (8/16/8 components), and weighted combination (0.2/0.5/0.3)
- **Validation Suite**: Comprehensive testing framework with sonic-only and hybrid scoring validation

#### Enhanced Genre System
- **Comprehensive Normalization**: Unified genre normalization across MusicBrainz, Discogs, Last.fm, and file tags
- **Discogs Integration**: Added Discogs API support for album-level genre metadata with rate limiting and caching
- **Multi-Source Ensemble**: Weighted combination of 4 similarity methods (cosine, set overlap, jaccard, asymmetric) with configurable genre gate (default 0.30)

#### Smart Playlist Features
- **Anchor Playlists**: Generate playlists from existing M3U files, maintaining flow while discovering similar tracks
- **Improved Recency Filtering**: Fixed Last.fm integration to properly exclude recently played tracks (14-day lookback)
- **Plex Export**: Native M3U8 export with Plex-compatible track URIs

### 🐛 Bug Fixes

- **Last.fm Recency Filter**: Fixed critical bug where recently played tracks appeared in playlists (match rate improved from 24.7% to 96.0%)
- **Unicode Normalization**: Fixed artist name comparison issues with special characters and accents
- **Bounds Checking**: Added robust validation for PCA components and edge cases

### 📚 Documentation

- **Complete Audit**: Updated all documentation to reflect current codebase
- **Production Ready**: Organized documentation structure for public GitHub release
- **Archived Session Notes**: Moved 29+ internal documents to organized archive structure
- **API Documentation**: Added authentication requirements and configuration examples

### ⚙️ Configuration Changes

- **Default Sonic Variant**: Changed from `robust_whiten` to `tower_pca`
- **Default Pipeline**: Data Science (DS) pipeline is now standard
- **Configurable Weights**: Tower weights and PCA components configurable via environment variables

### 🔧 Technical Improvements

- **Artifact Builder**: Automated beat3tower extraction and matrix building with progress tracking
- **Genre Normalization Pipeline**: Standardized genre processing across all metadata sources
- **Validation Scripts**: Added diagnostic tools for sonic feature validation and quality assurance
- **Database Schema**: Extended metadata schema for Discogs genres and sonic features

### 💾 Data Requirements

- **Beat3Tower Artifacts**: Requires one-time analysis of library (`update_sonic.py --beat3tower`)
- **Genre Metadata**: Fetch from MusicBrainz/Discogs (`update_genres_v3_normalized.py`)
- **Backward Compatible**: Existing databases auto-migrate with new schema

### 📊 Performance

- **Sonic Quality**: Tower PCA passes sonic-only validation tests
- **Hybrid Scoring**: 60% sonic + 40% genre similarity (configurable)
- **Match Rate**: 96% accuracy for Last.fm recency filtering
- **Parallel Processing**: Multi-worker support for library scanning and feature extraction

---

## Version 1.x - Legacy

Previous releases using legacy beat-sync analysis (71 dimensions) and raw sonic preprocessing.

See `docs/archive/` for historical implementation notes.
