#!/usr/bin/env python
"""One-time importer: copy backfill adjudications into the sidecar.

The Pass-1/Pass-2 backfill writes to data/adjudication_pass1.db. The analyze pipeline
reads the checkpoint from the sidecar (data/ai_genre_enrichment.db). Run this once after
the backfill so the pipeline inherits all the work instead of re-calling the LLM.

Usage:
  python scripts/research/import_backfill_adjudications.py \
    --src data/adjudication_pass1.db --sidecar data/ai_genre_enrichment.db
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.ai_genre_enrichment.adjudication_store import AdjudicationStore  # noqa: E402

_COLS = (
    "album_id, prompt_version, release_key, input_hash, model, status, response_json, "
    "dropped_file_tags_json, input_tokens, output_tokens, total_tokens, error, updated_at"
)


def import_adjudications(src_db: str, sidecar_db: str) -> int:
    AdjudicationStore(sidecar_db).close()  # ensure the table exists in the sidecar
    src = sqlite3.connect(f"file:{src_db}?mode=ro", uri=True)
    rows = src.execute(f"SELECT {_COLS} FROM adjudications").fetchall()
    src.close()
    dst = sqlite3.connect(sidecar_db)
    placeholders = ",".join(["?"] * len(_COLS.split(",")))
    dst.executemany(
        f"INSERT INTO adjudications ({_COLS}) VALUES ({placeholders}) "
        "ON CONFLICT(album_id, prompt_version) DO UPDATE SET "
        "release_key=excluded.release_key, input_hash=excluded.input_hash, model=excluded.model, "
        "status=excluded.status, response_json=excluded.response_json, "
        "dropped_file_tags_json=excluded.dropped_file_tags_json, input_tokens=excluded.input_tokens, "
        "output_tokens=excluded.output_tokens, total_tokens=excluded.total_tokens, "
        "error=excluded.error, updated_at=excluded.updated_at",
        rows,
    )
    dst.commit()
    n = len(rows)
    dst.close()
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(_ROOT / "data" / "adjudication_pass1.db"))
    ap.add_argument("--sidecar", default=str(_ROOT / "data" / "ai_genre_enrichment.db"))
    args = ap.parse_args()
    n = import_adjudications(args.src, args.sidecar)
    print(f"imported {n} adjudication rows -> {args.sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
