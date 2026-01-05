#!/usr/bin/env python3
"""
Fix Compound Genres in track_effective_genres
==============================================
Splits compound genre strings (e.g., "indie rock, alternative") into individual
atomized genre rows in the track_effective_genres table.

This fixes the issue where some genres contain commas and should be split into
separate rows for proper autocomplete functionality.

Usage:
    python fix_compound_genres.py                  # Dry run (preview changes)
    python fix_compound_genres.py --apply          # Apply changes to database
"""
import sys
import sqlite3
import argparse
from pathlib import Path
from typing import List, Tuple, Set

# Add parent directory to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.genre_normalization import normalize_genre_list


def find_compound_genres(conn: sqlite3.Connection) -> List[Tuple]:
    """
    Find all compound genres (containing separators) in track_effective_genres.

    Looks for genres containing:
    - Commas (,) - always a separator
    - Semicolons (;) - always a separator
    - Forward slashes (/) - always a separator
    - Ampersands (&) - MAY be a separator (but preserves R&B, Drum & Bass, etc.)

    Note: The normalization function will intelligently handle ampersands.
    """
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT track_id, genre, source, priority, weight
        FROM track_effective_genres
        WHERE genre LIKE '%,%'
           OR genre LIKE '%;%'
           OR genre LIKE '%/%'
           OR genre LIKE '%&%'
        ORDER BY track_id, priority
    """)
    return cursor.fetchall()


def split_compound_genre(genre: str) -> Set[str]:
    """
    Split compound genre string into individual normalized genres.

    Args:
        genre: Compound genre string (e.g., "indie rock, alternative")

    Returns:
        Set of individual normalized genres
    """
    # Use genre normalization to properly split and normalize
    return normalize_genre_list([genre], filter_broad=False)


def preview_changes(conn: sqlite3.Connection) -> None:
    """Preview what changes would be made."""
    compound_genres = find_compound_genres(conn)

    if not compound_genres:
        print("✓ No compound genres found! Database is clean.")
        return

    print(f"\n📊 Found {len(compound_genres)} compound genre entries\n")
    print("Preview of changes (showing first 20):")
    print("=" * 80)

    for i, (track_id, genre, source, priority, weight) in enumerate(compound_genres[:20], 1):
        atomized = split_compound_genre(genre)
        print(f"\n{i}. Track: {track_id[:8]}... | Source: {source} | Priority: {priority}")
        print(f"   BEFORE: '{genre}'")
        print(f"   AFTER:  {atomized}")
        print(f"   → Will create {len(atomized)} individual genre rows")

    if len(compound_genres) > 20:
        print(f"\n... and {len(compound_genres) - 20} more compound genres")

    print("\n" + "=" * 80)
    print(f"\nTotal: {len(compound_genres)} compound genre entries will be atomized")
    print("\nTo apply these changes, run:")
    print("  python fix_compound_genres.py --apply")


def apply_fixes(conn: sqlite3.Connection, dry_run: bool = True) -> None:
    """Apply fixes to split compound genres into individual rows."""
    compound_genres = find_compound_genres(conn)

    if not compound_genres:
        print("✓ No compound genres found! Database is clean.")
        return

    print(f"\n🔧 Processing {len(compound_genres)} compound genre entries...")

    cursor = conn.cursor()
    total_deleted = 0
    total_inserted = 0

    for track_id, genre, source, priority, weight in compound_genres:
        # Split the compound genre
        atomized = split_compound_genre(genre)

        if len(atomized) <= 1:
            # Not actually compound, skip
            continue

        # Delete the compound genre row
        cursor.execute("""
            DELETE FROM track_effective_genres
            WHERE track_id = ? AND genre = ? AND source = ?
        """, (track_id, genre, source))
        total_deleted += 1

        # Insert individual genre rows
        for individual_genre in atomized:
            cursor.execute("""
                INSERT OR IGNORE INTO track_effective_genres
                (track_id, genre, source, priority, weight, last_updated)
                VALUES (?, ?, ?, ?, ?, datetime('now'))
            """, (track_id, individual_genre, source, priority, weight))
            total_inserted += cursor.rowcount

    if dry_run:
        print("\n⚠️  DRY RUN - No changes committed")
        conn.rollback()
    else:
        conn.commit()
        print(f"\n✅ Changes committed:")
        print(f"   - Deleted: {total_deleted} compound genre rows")
        print(f"   - Inserted: {total_inserted} individual genre rows")
        print(f"   - Net change: {total_inserted - total_deleted:+d} rows")


def verify_results(conn: sqlite3.Connection) -> None:
    """Verify that no compound genres remain."""
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM track_effective_genres WHERE genre LIKE '%,%'")
    remaining = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT genre) FROM track_effective_genres")
    total_genres = cursor.fetchone()[0]

    print(f"\n📊 Final Statistics:")
    print(f"   - Total unique genres: {total_genres}")
    print(f"   - Compound genres remaining: {remaining}")

    if remaining == 0:
        print("\n✅ SUCCESS: All genres are now properly atomized!")
    else:
        print(f"\n⚠️  WARNING: {remaining} compound genres still remain")


def main():
    parser = argparse.ArgumentParser(description="Fix compound genres in track_effective_genres")
    parser.add_argument('--apply', action='store_true',
                       help='Apply changes to database (default is dry run)')
    parser.add_argument('--db', default='data/metadata.db',
                       help='Path to metadata database (default: data/metadata.db)')
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"❌ Error: Database not found: {db_path}")
        sys.exit(1)

    print(f"🔍 Analyzing database: {db_path}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    if args.apply:
        print("\n🚀 APPLYING CHANGES TO DATABASE")
        apply_fixes(conn, dry_run=False)
        verify_results(conn)
    else:
        print("\n👀 DRY RUN MODE (preview only)")
        preview_changes(conn)

    conn.close()


if __name__ == "__main__":
    main()
