"""Regression guard: the in-scope generation + GUI files must not reintroduce a
relative metadata.db literal or the config.get('library', {}).get('database_path')
anti-pattern. DB paths in these files go through resolve_database_path()."""

import io
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]

# In-scope per the 2026-07-07 spec (CLI generation + GUI worker). Enrichment
# files (discovery.py, source_extraction.py) are intentionally excluded.
_IN_SCOPE = [
    "main_app.py",
    "src/playlist_generator.py",
    "src/lastfm_client.py",
    "src/playlist_gui/worker.py",
]

_LITERAL = re.compile(r"""["']data/metadata\.db["']""")
_ANTIPATTERN = re.compile(r"""get\(['"]library['"],\s*\{\}\)\.get\(['"]database_path""")


def test_no_relative_db_literal_in_scope_files():
    offenders = []
    for rel in _IN_SCOPE:
        for i, line in enumerate(io.open(_ROOT / rel, encoding="utf-8"), 1):
            if _LITERAL.search(line) or _ANTIPATTERN.search(line):
                offenders.append(f"{rel}:{i}: {line.rstrip()}")
    assert not offenders, "Relative DB literal / anti-pattern found:\n" + "\n".join(offenders)
