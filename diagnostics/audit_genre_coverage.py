#!/usr/bin/env python3
"""
Genre Coverage Audit Script

Generates comprehensive reports on genre coverage gaps, especially for collaboration artists.
Produces 4 CSV outputs and 1 markdown summary report.

Usage:
    python diagnostics/audit_genre_coverage.py

Output files:
    - diagnostics/genre_coverage_summary.csv
    - diagnostics/genre_coverage_by_artist_type.csv
    - diagnostics/tracks_without_genres_top200.csv
    - diagnostics/collab_artists_without_genres_top200.csv
    - diagnostics/GENRE_COVERAGE_AUDIT.md
"""

import sqlite3
import csv
import sys
import io
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple

# Fix encoding for Windows console output
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Database path
DB_PATH = Path("data/metadata.db")

# Diagnostics output directory
DIAG_DIR = Path("diagnostics")
DIAG_DIR.mkdir(exist_ok=True)


def get_connection():
    """Get database connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def run_query(conn: sqlite3.Connection, query: str) -> List[Dict]:
    """Execute query and return rows as dictionaries."""
    cursor = conn.cursor()
    cursor.execute(query)
    return [dict(row) for row in cursor.fetchall()]


# ============================================================================
# QUERY 1: Overall Coverage by Source
# ============================================================================

QUERY_OVERALL_COVERAGE = """
WITH track_coverage AS (
  SELECT
    t.track_id,
    t.artist,
    t.album,
    COUNT(DISTINCT CASE
      WHEN ag.genre IS NOT NULL AND ag.genre != '__EMPTY__'
      THEN ag.genre
    END) AS album_genre_count,
    COUNT(DISTINCT CASE
      WHEN artg.genre IS NOT NULL AND artg.genre != '__EMPTY__'
      THEN artg.genre
    END) AS artist_genre_count,
    COUNT(DISTINCT CASE
      WHEN tg.genre IS NOT NULL
      THEN tg.genre
    END) AS file_genre_count
  FROM tracks t
  LEFT JOIN albums alb ON t.album_id = alb.album_id
  LEFT JOIN album_genres ag ON alb.album_id = ag.album_id
  LEFT JOIN artist_genres artg ON t.artist = artg.artist
  LEFT JOIN track_genres tg ON t.track_id = tg.track_id
  WHERE t.sonic_features IS NOT NULL
  GROUP BY t.track_id
)
SELECT
  COUNT(*) AS total_tracks,
  SUM(CASE WHEN album_genre_count > 0 THEN 1 ELSE 0 END) AS tracks_with_album_genres,
  SUM(CASE WHEN artist_genre_count > 0 THEN 1 ELSE 0 END) AS tracks_with_artist_genres,
  SUM(CASE WHEN file_genre_count > 0 THEN 1 ELSE 0 END) AS tracks_with_file_genres,
  SUM(CASE WHEN album_genre_count + artist_genre_count + file_genre_count > 0 THEN 1 ELSE 0 END) AS tracks_with_any_genre,
  ROUND(100.0 * SUM(CASE WHEN album_genre_count > 0 THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct_album_coverage,
  ROUND(100.0 * SUM(CASE WHEN artist_genre_count > 0 THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct_artist_coverage,
  ROUND(100.0 * SUM(CASE WHEN file_genre_count > 0 THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct_file_coverage,
  ROUND(100.0 * SUM(CASE WHEN album_genre_count + artist_genre_count + file_genre_count > 0 THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct_any_coverage
FROM track_coverage;
"""


# ============================================================================
# QUERY 2: Collaboration vs Solo Artist Coverage
# ============================================================================

QUERY_COLLAB_VS_SOLO = """
WITH artist_types AS (
  SELECT
    t.artist,
    CASE
      WHEN t.artist LIKE '%&%'
        OR t.artist LIKE '%feat.%'
        OR t.artist LIKE '%ft.%'
        OR t.artist LIKE '%with %'
        OR t.artist LIKE '%vs.%'
        OR t.artist LIKE '% x %'
      THEN 'collaboration'
      ELSE 'solo'
    END AS artist_type,
    COUNT(DISTINCT t.track_id) AS track_count,
    COUNT(DISTINCT CASE
      WHEN ag.genre IS NOT NULL AND ag.genre != '__EMPTY__'
      THEN ag.genre
    END) + COUNT(DISTINCT CASE
      WHEN artg.genre IS NOT NULL AND artg.genre != '__EMPTY__'
      THEN artg.genre
    END) AS genre_count
  FROM tracks t
  LEFT JOIN albums alb ON t.album_id = alb.album_id
  LEFT JOIN album_genres ag ON alb.album_id = ag.album_id
  LEFT JOIN artist_genres artg ON t.artist = artg.artist
  WHERE t.sonic_features IS NOT NULL
  GROUP BY t.artist
)
SELECT
  artist_type,
  COUNT(*) AS artist_count,
  SUM(track_count) AS total_tracks,
  SUM(CASE WHEN genre_count > 0 THEN 1 ELSE 0 END) AS artists_with_genres,
  SUM(CASE WHEN genre_count = 0 THEN 1 ELSE 0 END) AS artists_without_genres,
  ROUND(100.0 * SUM(CASE WHEN genre_count > 0 THEN 1 ELSE 0 END) / COUNT(*), 2) AS pct_coverage
FROM artist_types
GROUP BY artist_type
ORDER BY artist_type;
"""


# ============================================================================
# QUERY 3: Top 200 Tracks with Empty Genres
# ============================================================================

QUERY_TRACKS_WITHOUT_GENRES = """
WITH track_genre_counts AS (
  SELECT
    t.track_id,
    t.artist,
    t.title,
    t.album,
    COUNT(DISTINCT CASE
      WHEN ag.genre IS NOT NULL AND ag.genre != '__EMPTY__'
      THEN ag.genre
    END) + COUNT(DISTINCT CASE
      WHEN artg.genre IS NOT NULL AND artg.genre != '__EMPTY__'
      THEN artg.genre
    END) + COUNT(DISTINCT CASE
      WHEN tg.genre IS NOT NULL
      THEN tg.genre
    END) AS total_genres
  FROM tracks t
  LEFT JOIN albums alb ON t.album_id = alb.album_id
  LEFT JOIN album_genres ag ON alb.album_id = ag.album_id
  LEFT JOIN artist_genres artg ON t.artist = artg.artist
  LEFT JOIN track_genres tg ON t.track_id = tg.track_id
  WHERE t.sonic_features IS NOT NULL
  GROUP BY t.track_id
)
SELECT
  track_id,
  artist,
  title,
  album,
  total_genres
FROM track_genre_counts
WHERE total_genres = 0
ORDER BY artist, album, title
LIMIT 200;
"""


# ============================================================================
# QUERY 4: Top 200 Collaboration Artists with Empty Genres
# ============================================================================

QUERY_COLLAB_WITHOUT_GENRES = """
WITH collab_artists AS (
  SELECT
    t.artist,
    COUNT(DISTINCT t.track_id) AS track_count,
    COUNT(DISTINCT CASE
      WHEN ag.genre IS NOT NULL AND ag.genre != '__EMPTY__'
      THEN ag.genre
    END) + COUNT(DISTINCT CASE
      WHEN artg.genre IS NOT NULL AND artg.genre != '__EMPTY__'
      THEN artg.genre
    END) AS genre_count,
    GROUP_CONCAT(DISTINCT CASE
      WHEN ag.genre IS NOT NULL AND ag.genre != '__EMPTY__'
      THEN ag.genre
    END) AS genres
  FROM tracks t
  LEFT JOIN albums alb ON t.album_id = alb.album_id
  LEFT JOIN album_genres ag ON alb.album_id = ag.album_id
  LEFT JOIN artist_genres artg ON t.artist = artg.artist
  WHERE t.sonic_features IS NOT NULL
    AND (t.artist LIKE '%&%'
         OR t.artist LIKE '%feat.%'
         OR t.artist LIKE '%ft.%'
         OR t.artist LIKE '%with %'
         OR t.artist LIKE '%vs.%'
         OR t.artist LIKE '% x %')
  GROUP BY t.artist
)
SELECT
  artist,
  track_count,
  genre_count,
  COALESCE(genres, '(none)') AS current_genres
FROM collab_artists
WHERE genre_count = 0
ORDER BY track_count DESC
LIMIT 200;
"""


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_overall_coverage_report(conn: sqlite3.Connection) -> Dict[str, Any]:
    """Generate overall coverage statistics."""
    results = run_query(conn, QUERY_OVERALL_COVERAGE)
    if not results:
        return {}

    report = results[0]

    # Calculate zero genres count
    report['tracks_without_any_genre'] = report['total_tracks'] - report['tracks_with_any_genre']
    report['pct_without_genres'] = 100.0 - report['pct_any_coverage']

    return report


def generate_collab_vs_solo_report(conn: sqlite3.Connection) -> Tuple[Dict, Dict]:
    """Generate collaboration vs solo comparison."""
    results = run_query(conn, QUERY_COLLAB_VS_SOLO)

    solo_data = None
    collab_data = None

    for row in results:
        if row['artist_type'] == 'solo':
            solo_data = dict(row)
        else:
            collab_data = dict(row)

    return solo_data, collab_data


def export_csv(filename: str, data: List[Dict], columns: List[str]):
    """Export data to CSV."""
    output_path = DIAG_DIR / filename

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(data)

    print(f"✅ Exported: {output_path}")


def export_summary_csv(report: Dict[str, Any]):
    """Export summary coverage report."""
    data = [{
        'metric': 'total_tracks',
        'value': report.get('total_tracks', 0)
    }, {
        'metric': 'tracks_with_album_genres',
        'value': report.get('tracks_with_album_genres', 0)
    }, {
        'metric': 'tracks_with_artist_genres',
        'value': report.get('tracks_with_artist_genres', 0)
    }, {
        'metric': 'tracks_with_file_genres',
        'value': report.get('tracks_with_file_genres', 0)
    }, {
        'metric': 'tracks_with_any_genre',
        'value': report.get('tracks_with_any_genre', 0)
    }, {
        'metric': 'tracks_without_any_genre',
        'value': report.get('tracks_without_any_genre', 0)
    }, {
        'metric': 'pct_album_coverage',
        'value': f"{report.get('pct_album_coverage', 0):.2f}%"
    }, {
        'metric': 'pct_artist_coverage',
        'value': f"{report.get('pct_artist_coverage', 0):.2f}%"
    }, {
        'metric': 'pct_file_coverage',
        'value': f"{report.get('pct_file_coverage', 0):.2f}%"
    }, {
        'metric': 'pct_any_coverage',
        'value': f"{report.get('pct_any_coverage', 0):.2f}%"
    }, {
        'metric': 'pct_without_genres',
        'value': f"{report.get('pct_without_genres', 0):.2f}%"
    }]

    export_csv('genre_coverage_summary.csv', data, ['metric', 'value'])


def export_collab_vs_solo_csv(solo: Dict, collab: Dict):
    """Export collaboration vs solo comparison."""
    data = [{
        'artist_type': 'solo',
        'artist_count': solo.get('artist_count', 0),
        'total_tracks': solo.get('total_tracks', 0),
        'artists_with_genres': solo.get('artists_with_genres', 0),
        'artists_without_genres': solo.get('artists_without_genres', 0),
        'pct_coverage': f"{solo.get('pct_coverage', 0):.2f}%"
    }, {
        'artist_type': 'collaboration',
        'artist_count': collab.get('artist_count', 0),
        'total_tracks': collab.get('total_tracks', 0),
        'artists_with_genres': collab.get('artists_with_genres', 0),
        'artists_without_genres': collab.get('artists_without_genres', 0),
        'pct_coverage': f"{collab.get('pct_coverage', 0):.2f}%"
    }]

    export_csv('genre_coverage_by_artist_type.csv', data,
               ['artist_type', 'artist_count', 'total_tracks', 'artists_with_genres',
                'artists_without_genres', 'pct_coverage'])


def generate_markdown_report(overall: Dict, solo: Dict, collab: Dict,
                             empty_tracks_count: int, empty_collab_count: int):
    """Generate markdown report."""

    gap = solo.get('pct_coverage', 0) - collab.get('pct_coverage', 0)

    report_text = f"""# Genre Coverage Audit Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary Metrics

| Metric | Count | Percentage |
|--------|-------|-----------|
| Total tracks (with sonic features) | {overall.get('total_tracks', 0)} | 100% |
| Tracks with album genres | {overall.get('tracks_with_album_genres', 0)} | {overall.get('pct_album_coverage', 0):.2f}% |
| Tracks with artist genres | {overall.get('tracks_with_artist_genres', 0)} | {overall.get('pct_artist_coverage', 0):.2f}% |
| Tracks with file-embedded genres | {overall.get('tracks_with_file_genres', 0)} | {overall.get('pct_file_coverage', 0):.2f}% |
| **Tracks with ANY genre** | **{overall.get('tracks_with_any_genre', 0)}** | **{overall.get('pct_any_coverage', 0):.2f}%** |
| **Tracks with ZERO genres** | **{overall.get('tracks_without_any_genre', 0)}** | **{overall.get('pct_without_genres', 0):.2f}%** ⚠️ |

## Collaboration Coverage Gap

| Artist Type | Artist Count | Total Tracks | Artists w/ Genres | Coverage % |
|-------------|--------------|--------------|-------------------|-----------|
| Solo artists | {solo.get('artist_count', 0)} | {solo.get('total_tracks', 0)} | {solo.get('artists_with_genres', 0)} | {solo.get('pct_coverage', 0):.2f}% |
| Collaboration artists | {collab.get('artist_count', 0)} | {collab.get('total_tracks', 0)} | {collab.get('artists_with_genres', 0)} | {collab.get('pct_coverage', 0):.2f}% |
| **COVERAGE GAP** | | | | **{gap:.2f}% ⚠️** |

**Interpretation**: Collaboration artists have **{gap:.2f}% lower** genre coverage than solo artists.

## Top Offenders

### Tracks with ZERO Genres (Top 200)
- **Count**: {empty_tracks_count}
- **Impact**: These tracks are currently **EXCLUDED from artifact bundles** during `scripts/analyze_library.py`
- **Location**: See `diagnostics/tracks_without_genres_top200.csv`

### Collaboration Artists with ZERO Genres (Top 200)
- **Count**: {empty_collab_count}
- **Impact**: Collaboration tracks missing genres in artist lookups (don't inherit from constituents)
- **Location**: See `diagnostics/collab_artists_without_genres_top200.csv`

## Key Findings

1. **Genre Coverage is Good for Solo Artists**: {solo.get('pct_coverage', 0):.2f}% of solo artists have genres
2. **Genre Coverage is POOR for Collaborations**: {collab.get('pct_coverage', 0):.2f}% of collab artists have genres
3. **Coverage Gap is SIGNIFICANT**: {gap:.2f}% lower for collaborations
4. **Empty-Genre Tracks Exist**: {overall.get('pct_without_genres', 0):.2f}% of all tracks have zero genres

## Recommendations

### Phase 1: Audit (CURRENT)
- ✅ Coverage metrics computed
- ✅ Gap identified and quantified
- ✅ Top offenders identified

### Phase 2: Inheritance (NEXT)
Implement automatic genre inheritance from constituent artists:
- Parse collaboration strings (e.g., "Artist A & Artist B" → ["Artist A", "Artist B"])
- Fetch genres for each constituent from MusicBrainz
- Store inherited genres with source='musicbrainz_artist_inherited'

**Expected Impact**:
- Collaboration coverage: {collab.get('pct_coverage', 0):.2f}% → ~75%
- Total coverage: {overall.get('pct_any_coverage', 0):.2f}% → ~92%

### Phase 3: Materialization (NEXT)
- Create `track_effective_genres` table combining all sources
- Track provenance (file/album/artist/inherited)
- Include tracks with empty vectors (instead of excluding them)

### Phase 4: Verification (NEXT)
- Add regression tests for collaboration parsing
- Verify seed sanity (afrobeat seed shouldn't pull indie/slowcore)
- Manual A/B listening test on collaboration-heavy playlists

## Next Steps

1. Run genre update with inheritance: `python scripts/update_genres_v3_normalized.py --artists --albums`
2. Implement Phase 2 effective genres model
3. Re-run this audit to verify improvement

---

**Report Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    report_path = DIAG_DIR / 'GENRE_COVERAGE_AUDIT.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"✅ Generated: {report_path}")


def main():
    """Run complete audit."""
    print("\n" + "="*80)
    print("GENRE COVERAGE AUDIT")
    print("="*80 + "\n")

    # Connect to database
    try:
        conn = get_connection()
        print(f"✅ Connected to: {DB_PATH}\n")
    except Exception as e:
        print(f"❌ Failed to connect to database: {e}")
        return

    # Query 1: Overall coverage
    print("📊 Query 1: Overall Coverage by Source...")
    overall = generate_overall_coverage_report(conn)
    print(f"   Total tracks: {overall.get('total_tracks', 0)}")
    print(f"   Tracks with any genre: {overall.get('tracks_with_any_genre', 0)} ({overall.get('pct_any_coverage', 0):.2f}%)")
    print(f"   Tracks with ZERO genres: {overall.get('tracks_without_any_genre', 0)} ({overall.get('pct_without_genres', 0):.2f}%)\n")

    # Query 2: Collab vs Solo
    print("📊 Query 2: Collaboration vs Solo Coverage...")
    solo, collab = generate_collab_vs_solo_report(conn)
    print(f"   Solo artists: {solo['pct_coverage']:.2f}% have genres")
    print(f"   Collab artists: {collab['pct_coverage']:.2f}% have genres")
    print(f"   Gap: {solo['pct_coverage'] - collab['pct_coverage']:.2f}% ⚠️\n")

    # Query 3: Tracks without genres
    print("📊 Query 3: Top 200 Tracks with Empty Genres...")
    empty_tracks = run_query(conn, QUERY_TRACKS_WITHOUT_GENRES)
    print(f"   Found: {len(empty_tracks)} tracks\n")

    # Query 4: Collab artists without genres
    print("📊 Query 4: Top 200 Collab Artists without Genres...")
    empty_collabs = run_query(conn, QUERY_COLLAB_WITHOUT_GENRES)
    print(f"   Found: {len(empty_collabs)} collaboration artists\n")

    # Export CSVs
    print("📁 Exporting CSV files...")
    export_summary_csv(overall)
    export_collab_vs_solo_csv(solo, collab)
    export_csv('tracks_without_genres_top200.csv', empty_tracks,
               ['track_id', 'artist', 'title', 'album', 'total_genres'])
    export_csv('collab_artists_without_genres_top200.csv', empty_collabs,
               ['artist', 'track_count', 'genre_count', 'current_genres'])

    # Generate markdown report
    print("\n📝 Generating markdown report...")
    generate_markdown_report(overall, solo, collab, len(empty_tracks), len(empty_collabs))

    print("\n" + "="*80)
    print("AUDIT COMPLETE")
    print("="*80)
    print("\n📋 Summary:")
    print(f"   ✅ Genre coverage: {overall.get('pct_any_coverage', 0):.2f}%")
    print(f"   ⚠️  Collaboration gap: {solo['pct_coverage'] - collab['pct_coverage']:.2f}%")
    print(f"   ⚠️  Tracks excluded from artifacts: {overall.get('tracks_without_any_genre', 0)}")
    print(f"   📁 Reports saved to: {DIAG_DIR}/\n")

    conn.close()


if __name__ == '__main__':
    main()
