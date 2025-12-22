# Sonic Feature Backup & Restore Guide

## Overview

The sonic feature backup system protects valuable audio analysis data, especially beat3tower scans which take significant time and computational resources to generate.

**Key Protection Features:**
- ✅ Compressed backups with metadata
- ✅ Safe mode in scanner (default - won't overwrite existing features)
- ✅ Automatic backup before force mode
- ✅ Explicit confirmation required for overwrites
- ✅ Dry-run restore to preview changes

---

## Quick Reference

### Create a Backup
```bash
# Create timestamped backup
python scripts/backup_sonic_features.py --backup

# Create backup with custom name
python scripts/backup_sonic_features.py --backup --name "before_rescan"
```

### List Backups
```bash
python scripts/backup_sonic_features.py --list
```

### View Backup Info
```bash
python scripts/backup_sonic_features.py --info <backup_name>
```

### Restore from Backup
```bash
# Dry run first (recommended)
python scripts/backup_sonic_features.py --restore <backup_name>

# Actually restore
python scripts/backup_sonic_features.py --restore <backup_name> --no-dry-run
```

### Check Current Statistics
```bash
python scripts/backup_sonic_features.py --stats
```

---

## Understanding the Backup System

### What Gets Backed Up?

The backup system creates a compressed snapshot of:
- `track_id` - Track identifier
- `artist` - Artist name
- `title` - Track title
- `sonic_features` - JSON blob with audio features
- `sonic_features_updated_at` - Timestamp

**Feature Formats:**
- **Beat3tower**: Advanced 3-tower beat-synchronized analysis (137 dimensions)
  - Rhythm tower: 21 beat-sync features
  - Timbre tower: 83 MFCC-based features
  - Harmony tower: 33 chroma features
- **Legacy**: Older single-segment or windowed features

### Backup Location

Backups are stored in: `data/sonic_backups/`

**File structure:**
```
data/sonic_backups/
├── sonic_backup_20231218_143022.db.gz      # Compressed backup
├── sonic_backup_20231218_143022.json       # Metadata
├── sonic_backup_auto_before_force_20231218_150030.db.gz
└── sonic_backup_auto_before_force_20231218_150030.json
```

### Backup Metadata

Each backup includes a JSON metadata file with:
```json
{
  "timestamp": "20231218_143022",
  "name": null,
  "track_count": 30080,
  "backup_size_mb": 42.5,
  "created_at": "2023-12-18T14:30:22",
  "stats": {
    "total_tracks": 30080,
    "beat3tower_tracks": 30080,
    "legacy_tracks": 0,
    "db_size_mb": 156.3
  }
}
```

---

## Sonic Scanner Safety Features

### Safe Mode (Default)

By default, the sonic scanner operates in **safe mode**:
```bash
python scripts/update_sonic.py --beat3tower
```

**Safe mode behavior:**
- ✅ Only analyzes tracks **without existing sonic features**
- ✅ Never overwrites beat3tower or legacy features
- ✅ Safe to run repeatedly

**SQL query used:**
```sql
SELECT track_id, file_path, ...
FROM tracks
WHERE file_path IS NOT NULL
  AND sonic_features IS NULL
```

### Force Mode (Dangerous)

Force mode re-analyzes **ALL** tracks, overwriting existing features:
```bash
python scripts/update_sonic.py --force --beat3tower
```

**Force mode protections:**
1. **Auto-backup**: Automatically creates `auto_before_force` backup if beat3tower features exist
2. **Statistics display**: Shows how many beat3tower tracks will be overwritten
3. **Explicit confirmation**: Requires typing 'YES' to proceed
4. **Abort on backup failure**: If auto-backup fails, force mode is aborted

**Example output:**
```
======================================================================
⚠️  WARNING: FORCE MODE ENABLED ⚠️
======================================================================
This will RE-ANALYZE ALL 34,539 tracks, including:
  • 30,080 tracks with beat3tower features
  • 0 tracks with legacy features

🔒 Auto-backup: Creating safety backup of beat3tower features...
✓ Backup created: sonic_backup_auto_before_force_20231218_150030.db.gz
  Restore with: python scripts/backup_sonic_features.py --restore auto_before_force

Type 'YES' to confirm you want to overwrite existing features: YES

✓ Confirmed. Starting re-analysis...
```

---

## Backup Workflow Examples

### Example 1: Before Major Rescan

You want to upgrade from legacy features to beat3tower:

```bash
# 1. Check current status
python scripts/update_sonic.py --stats

# Output:
# Sonic Analysis Statistics:
# ============================================================
#   Total tracks: 34,539
#   Analyzed: 30,080
#   Pending: 4,459
#
#   Feature formats:
#     Beat3tower (recommended): 30,080
#     Legacy format: 0
#   Librosa source: 30,080
# ============================================================
# ✓ 30,080 tracks protected with beat3tower features

# 2. Create backup before changing anything
python scripts/backup_sonic_features.py --backup --name "before_upgrade"

# 3. Run safe mode to only analyze missing tracks
python scripts/update_sonic.py --beat3tower
```

### Example 2: Recovering from Accidental Overwrite

If you accidentally ran force mode:

```bash
# 1. List available backups
python scripts/backup_sonic_features.py --list

# Output:
# Available backups (2):
# ============================================================
#
# sonic_backup_auto_before_force_20231218_150030.db.gz
#   Created: 2023-12-18T15:00:30
#   Tracks: 30,080
#   Size: 42.50 MB
#
# sonic_backup_before_upgrade_20231218_143022.db.gz
#   Created: 2023-12-18T14:30:22
#   Tracks: 30,080
#   Size: 42.50 MB

# 2. Check what the auto-backup contains
python scripts/backup_sonic_features.py --info auto_before_force

# 3. Dry run restore (preview)
python scripts/backup_sonic_features.py --restore auto_before_force

# Output shows sample records and stats

# 4. Actually restore
python scripts/backup_sonic_features.py --restore auto_before_force --no-dry-run

# Output:
# Restored 30,080 sonic feature records
```

### Example 3: Testing New Analysis Methods

You want to experiment with analysis parameters:

```bash
# 1. Create backup
python scripts/backup_sonic_features.py --backup --name "before_experiment"

# 2. Run experimental analysis
python scripts/update_sonic.py --force --beat3tower --workers 8

# 3. Test playlists with new features
python main_app.py --artist "Built to Spill" --mode dynamic

# 4. If results are worse, restore backup
python scripts/backup_sonic_features.py --restore before_experiment --no-dry-run
```

---

## Backup Statistics

View current sonic feature statistics:

```bash
python scripts/backup_sonic_features.py --stats
```

**Example output:**
```
Current Sonic Feature Statistics:
============================================================
Total tracks with sonic features: 30,080
  Beat3tower format: 30,080
  Legacy format: 0
Database size: 156.30 MB
============================================================
```

View sonic scanner statistics (more detailed):

```bash
python scripts/update_sonic.py --stats
```

**Example output:**
```
Sonic Analysis Statistics:
============================================================
  Total tracks: 34,539
  Analyzed: 30,080
  Pending: 4,459

  Feature formats:
    Beat3tower (recommended): 30,080
    Legacy format: 0
  Librosa source: 30,080
============================================================

✓ 30,080 tracks protected with beat3tower features
  Run without --force to preserve these features (safe mode)
```

---

## Backup Cleanup

Keep only the N most recent backups:

```bash
# Keep 10 most recent backups, delete older ones
python scripts/backup_sonic_features.py --cleanup 10
```

**What happens:**
- Backups are sorted by creation time (newest first)
- The N most recent are kept
- Older backups are deleted (both .db.gz and .json files)

---

## Technical Details

### Compression

Backups use gzip compression (level 6) for space efficiency:
- **Uncompressed**: ~156 MB for 30,000 tracks
- **Compressed**: ~42 MB (73% reduction)

### Restore Behavior

The restore process:
1. Decompresses backup to temporary database
2. Reads all sonic features from backup
3. Updates tracks in main database with `sonic_features` and `sonic_features_updated_at`
4. Only updates tracks that exist in main database (matched by `track_id`)
5. Commits changes in transaction

**Important**: Restore is **additive** - it updates existing tracks but doesn't delete tracks that aren't in the backup.

### Dry Run Mode

Dry run mode (default for restore) shows:
- Number of tracks that would be restored
- Sample of 5 tracks with their extraction methods
- Statistics about feature formats

**No database modifications** are made in dry run mode.

---

## Best Practices

### 1. Regular Backups

Create backups before any risky operations:
- ✅ Before running `--force` mode
- ✅ Before database migrations
- ✅ Before experimenting with new analysis methods
- ✅ After completing a long beat3tower scan

### 2. Name Your Backups

Use descriptive names for manual backups:
```bash
python scripts/backup_sonic_features.py --backup --name "before_genre_migration"
python scripts/backup_sonic_features.py --backup --name "after_full_beat3tower_scan"
python scripts/backup_sonic_features.py --backup --name "before_parameter_tuning"
```

### 3. Verify After Restore

After restoring, check statistics:
```bash
python scripts/update_sonic.py --stats
```

Verify beat3tower count matches what you expected.

### 4. Keep Important Backups

Don't auto-cleanup backups after major milestones:
- Full library beat3tower scan completion
- Before major refactoring
- Known-good configurations

### 5. Use Safe Mode

Always use safe mode unless you have a specific reason to use force mode:
```bash
# Good (safe mode)
python scripts/update_sonic.py --beat3tower

# Dangerous (force mode)
python scripts/update_sonic.py --force --beat3tower
```

---

## Troubleshooting

### "Database is locked" Error

If you get this error during backup or restore:

**Cause**: Another process (like the sonic scanner) has the database open.

**Solution**:
1. Wait for the scanner to finish
2. Or stop the scanner: Find the process and kill it
3. The backup script uses a 60-second timeout, so brief locks are OK

### Backup Failed During Force Mode

If auto-backup fails:

**Behavior**: Force mode is automatically aborted, no tracks are modified.

**What to do**:
1. Check disk space (backups need ~50 MB)
2. Check permissions on `data/sonic_backups/` directory
3. Create backup manually first:
   ```bash
   python scripts/backup_sonic_features.py --backup
   ```
4. Then retry force mode

### Restore Dry Run Shows Wrong Count

If restore dry run shows fewer tracks than expected:

**Possible causes**:
- Backup is from a subset (was created with scanner running)
- Some tracks have been deleted from main database since backup

**What to do**:
- Check backup metadata: `--info <backup_name>`
- Verify `track_count` matches expectations
- Use a different backup if needed

### Beat3tower Count Decreased After Restore

If statistics show fewer beat3tower tracks after restore:

**Possible cause**: Restored backup was from before beat3tower features were added.

**Solution**:
- List all backups: `--list`
- Find a backup with the expected `beat3tower_tracks` count
- Restore from that backup instead

---

## Command Reference

### Backup Script

```bash
# Create backup
python scripts/backup_sonic_features.py --backup [--name NAME]

# List backups
python scripts/backup_sonic_features.py --list

# Show backup info
python scripts/backup_sonic_features.py --info NAME

# Restore (dry run)
python scripts/backup_sonic_features.py --restore NAME

# Restore (actually apply)
python scripts/backup_sonic_features.py --restore NAME --no-dry-run

# Cleanup old backups
python scripts/backup_sonic_features.py --cleanup N

# Show current stats
python scripts/backup_sonic_features.py --stats
```

### Sonic Scanner

```bash
# Safe mode (default - only analyze missing)
python scripts/update_sonic.py [--beat3tower] [--workers N] [--limit N]

# Force mode (re-analyze all - requires confirmation)
python scripts/update_sonic.py --force [--beat3tower] [--workers N]

# Show statistics
python scripts/update_sonic.py --stats
```

---

## Future Improvements

Potential enhancements:
- [ ] Automated scheduled backups (daily/weekly)
- [ ] Backup rotation policy (keep last 10 daily, 4 weekly, 12 monthly)
- [ ] Backup verification (integrity check)
- [ ] Export to portable JSON format
- [ ] Remote backup storage (S3, Google Drive)
- [ ] Incremental backups (only changed tracks)

---

## Summary

The sonic feature backup system provides robust protection for your audio analysis data:

✅ **Safe by default** - Scanner won't overwrite existing features unless explicitly forced
✅ **Automatic backups** - Force mode creates auto-backup before proceeding
✅ **Explicit confirmation** - Requires typing 'YES' to overwrite features
✅ **Dry run mode** - Preview restore changes before applying
✅ **Beat3tower detection** - Shows count of protected features
✅ **Compressed storage** - Efficient disk usage (~73% compression)
✅ **Metadata tracking** - Know what's in each backup

**Remember**: The default behavior is safe. You have to explicitly try to break things, and even then, the system will create a backup and ask for confirmation.
