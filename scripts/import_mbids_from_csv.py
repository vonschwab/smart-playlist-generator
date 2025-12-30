"""
Import MusicBrainz track IDs (MBIDs) into metadata.db without retagging files.

Usage:
    python scripts/import_mbids_from_csv.py --mapping path/to/mbids.csv --db data/metadata.db

CSV requirements:
    - Must have headers.
    - Acceptable column names for MBID: musicbrainz_id, mbid, track_mbid
    - Acceptable column names for path: file_path, path
    - Extra columns are ignored.

The script matches rows to tracks.file_path and updates tracks.musicbrainz_id.
No audio files are modified.
"""
import argparse
import csv
import sqlite3
from pathlib import Path
from typing import Dict, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import MusicBrainz IDs into metadata.db")
    parser.add_argument(
        "--mapping",
        required=True,
        help="CSV file with file_path and musicbrainz_id columns",
    )
    parser.add_argument(
        "--db",
        default="data/metadata.db",
        help="Path to metadata.db (default: data/metadata.db)",
    )
    return parser.parse_args()


def load_mapping(path: Path) -> Dict[str, str]:
    """Load file_path -> mbid mapping from CSV."""
    mbid_keys = {"musicbrainz_id", "mbid", "track_mbid"}
    path_keys = {"file_path", "path"}

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = {h.strip().lower() for h in reader.fieldnames or []}
        mbid_col = next((h for h in headers if h in mbid_keys), None)
        path_col = next((h for h in headers if h in path_keys), None)
        if not mbid_col or not path_col:
            raise ValueError("CSV must have columns for musicbrainz_id and file_path")

        mapping: Dict[str, str] = {}
        for row in reader:
            mbid = (row.get(mbid_col) or "").strip()
            file_path = (row.get(path_col) or "").strip()
            if not mbid or not file_path:
                continue
            # Normalize path for consistent matching
            normalized_path = str(Path(file_path).resolve())
            mapping[normalized_path] = mbid
        return mapping


def update_db(db_path: Path, mapping: Dict[str, str]) -> Tuple[int, int]:
    """Update metadata.db with MBIDs. Returns (updated, missing)."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    updated = 0
    missing = 0

    for path, mbid in mapping.items():
        cur.execute(
            "SELECT COUNT(1) FROM tracks WHERE file_path = ?",
            (path,),
        )
        exists = cur.fetchone()[0]
        if not exists:
            missing += 1
            continue
        cur.execute(
            "UPDATE tracks SET musicbrainz_id = ? WHERE file_path = ?",
            (mbid, path),
        )
        updated += cur.rowcount

    conn.commit()
    conn.close()
    return updated, missing


def main() -> None:
    args = parse_args()
    mapping_path = Path(args.mapping)
    db_path = Path(args.db)

    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_path}")
    if not db_path.exists():
        raise FileNotFoundError(f"Database file not found: {db_path}")

    mapping = load_mapping(mapping_path)
    print(f"Loaded {len(mapping)} MBID mappings from {mapping_path}")

    updated, missing = update_db(db_path, mapping)
    print(f"Updated {updated} tracks with MBIDs")
    if missing:
        print(f"{missing} paths in mapping not found in tracks.file_path")


if __name__ == "__main__":
    main()
