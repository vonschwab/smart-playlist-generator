"""Unit tests: the read-only SQLite MCP resolves the DB from config.yaml.

Guards against the satellite stub trap: a clone's tracked data/metadata.db is
a 0-byte placeholder; the MCP must follow config's absolute path instead.
"""

import importlib.util
import pathlib

import pytest

# The read-only SQLite MCP tool imports the `mcp` SDK, which is an optional dev
# dependency (not in pyproject). Skip this test where mcp isn't installed (e.g. CI)
# rather than fail collection.
pytest.importorskip("mcp")

_MOD = pathlib.Path(__file__).resolve().parents[1] / "tools" / "mcp_sqlite_readonly.py"
_spec = importlib.util.spec_from_file_location("mcp_sqlite_readonly", _MOD)
assert _spec is not None and _spec.loader is not None
mcp_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mcp_mod)


def test_config_absolute_db_path_wins(tmp_path):
    canonical_db = tmp_path / "elsewhere" / "metadata.db"
    canonical_db.parent.mkdir()
    canonical_db.write_bytes(b"x")
    (tmp_path / "config.yaml").write_text(
        f"library:\n  database_path: {canonical_db.as_posix()}\n", encoding="utf-8"
    )
    dbs = mcp_mod._load_db_map(root=tmp_path)
    assert dbs["metadata"] == canonical_db.resolve()
    assert dbs["enrichment"] == (canonical_db.parent / "ai_genre_enrichment.db").resolve()


def test_relative_config_path_resolves_against_root(tmp_path):
    (tmp_path / "config.yaml").write_text(
        "library:\n  database_path: data/metadata.db\n", encoding="utf-8"
    )
    dbs = mcp_mod._load_db_map(root=tmp_path)
    assert dbs["metadata"] == (tmp_path / "data" / "metadata.db").resolve()


def test_missing_config_falls_back_to_default(tmp_path):
    dbs = mcp_mod._load_db_map(root=tmp_path)
    assert dbs["metadata"] == (tmp_path / "data" / "metadata.db").resolve()
    assert dbs["enrichment"] == (tmp_path / "data" / "ai_genre_enrichment.db").resolve()


def test_env_override_still_wins(tmp_path, monkeypatch):
    override = tmp_path / "other.db"
    monkeypatch.setenv("SQLITE_RO_DBS", f'{{"metadata": "{override.as_posix()}"}}')
    dbs = mcp_mod._load_db_map(root=tmp_path)
    assert dbs["metadata"] == override.resolve()
