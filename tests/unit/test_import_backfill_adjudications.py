from __future__ import annotations

import sqlite3

from scripts.research.import_backfill_adjudications import import_adjudications
from src.ai_genre_enrichment.adjudication_store import AdjudicationStore


def test_import_copies_rows_into_sidecar(tmp_path):
    src = tmp_path / "adjudication_pass1.db"
    s = AdjudicationStore(src)
    s.save(album_id="a1", prompt_version="pv", input_hash="h", status="complete",
           response={"genres": []}, tokens={"total_tokens": 1})
    s.save(album_id="a2", prompt_version="pv", input_hash="h", status="complete",
           response={"genres": []})
    s.close()

    side = tmp_path / "ai_genre_enrichment.db"
    AdjudicationStore(side).close()  # create the table in the sidecar
    n = import_adjudications(str(src), str(side))
    assert n == 2
    dst = AdjudicationStore(side)
    assert dst.complete_album_ids("pv") == {"a1", "a2"}
    dst.close()
