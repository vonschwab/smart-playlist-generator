#!/usr/bin/env python3
"""Publish authoritative layered genres from the sidecar into metadata.db.

SAFETY: develop/validate against a COPY of metadata.db first. The live run
requires a fresh timestamped backup and explicit confirmation. See
docs/superpowers/specs/2026-06-06-unified-genre-store-design.md.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.genre.genre_publish import publish


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Publish unified genre store into metadata.db")
    parser.add_argument("--metadata-db", default=str(ROOT / "data" / "metadata.db"))
    parser.add_argument("--sidecar-db", default=str(ROOT / "data" / "ai_genre_enrichment.db"))
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute and print stats, then roll back (no writes).")
    args = parser.parse_args(argv)
    stats = publish(args.metadata_db, args.sidecar_db, dry_run=args.dry_run)
    print(json.dumps(stats.as_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
