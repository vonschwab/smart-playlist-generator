"""Read-only SQLite MCP server for the Playlist Generator project databases.

Why this exists: we kept hitting SQLite syntax/schema errors from guessing column
names. This server gives structured ``list_tables`` / ``describe_table`` /
``run_query`` tools so the schema is introspected, not guessed.

Safety model (matters because ``data/metadata.db`` is irreplaceable):

* Every connection is opened with SQLite's ``mode=ro`` URI flag, so the driver
  itself rejects any write — even a query that somehow tried to mutate. This is
  the load-bearing guarantee, not the statement guard below.
* Only read tools are surfaced. There is no write/execute/create tool to call.
* ``run_query`` additionally refuses statements that are not SELECT / WITH /
  PRAGMA / EXPLAIN, so misuse fails with a clear message instead of an opaque
  driver error (defense in depth).
* Databases are addressed by a fixed name->path allowlist, so no arbitrary file
  path ever reaches SQLite. The allowlist is configurable via the
  ``SQLITE_RO_DBS`` env var (JSON ``{name: path}``); by default it resolves the
  two project DBs relative to this file, independent of the launch cwd.

Run as a stdio MCP server: ``python tools/mcp_sqlite_readonly.py``.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from pathlib import Path

from mcp.server.fastmcp import FastMCP

# --- Database allowlist -----------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _config_db_path(root: Path) -> Path | None:
    """library.database_path from <root>/config.yaml, resolved; None if unset."""
    cfg = root / "config.yaml"
    try:
        import yaml

        with open(cfg, encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        raw = str(((data.get("library") or {}).get("database_path")) or "").strip()
        if not raw:
            return None
        p = Path(raw)
        return (p if p.is_absolute() else root / p).resolve()
    except Exception:
        return None  # unreadable config -> fall back to default path


def _load_db_map(root: Path = _REPO_ROOT) -> dict[str, Path]:
    """Name -> absolute path for every queryable DB. cwd-independent.

    Honors config.yaml `library.database_path` (satellites point it at the
    canonical checkout by absolute path — the tracked data/metadata.db in a
    clone is a 0-byte stub). `enrichment` lives beside the resolved metadata
    DB so both follow the same data directory.
    """
    raw = os.environ.get("SQLITE_RO_DBS")
    if raw:
        mapping = {name: Path(p) for name, p in json.loads(raw).items()}
    else:
        meta = _config_db_path(root) or (root / "data" / "metadata.db")
        mapping = {
            "metadata": meta,
            "enrichment": meta.parent / "ai_genre_enrichment.db",
        }
    return {name: p.resolve() for name, p in mapping.items()}


_DBS = _load_db_map()

DEFAULT_MAX_ROWS = 200
MAX_ROWS_CAP = 1000
_READ_PREFIXES = ("select", "with", "pragma", "explain")
_LEADING_COMMENT = re.compile(r"^\s*(--[^\n]*\n|/\*.*?\*/)", re.DOTALL)

mcp = FastMCP("sqlite-ro")


def _connect(database: str) -> sqlite3.Connection:
    if database not in _DBS:
        raise ValueError(
            f"Unknown database {database!r}. Configured: {sorted(_DBS)}"
        )
    path = _DBS[database]
    if not path.exists():
        raise FileNotFoundError(f"Database file not found: {path}")
    # mode=ro => driver-level read-only. as_posix() so the URI parses on Windows.
    con = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def _strip_leading(sql: str) -> str:
    """Drop leading whitespace, comments, and an opening paren before prefix check."""
    s = sql
    while True:
        m = _LEADING_COMMENT.match(s)
        if not m:
            break
        s = s[m.end():]
    return s.lstrip().lstrip("(").lstrip()


def _cell(v: object) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, (bytes, bytearray)):
        return f"<{len(v)} bytes>"
    s = str(v).replace("\n", "\\n").replace("\t", " ")
    return s if len(s) <= 300 else s[:297] + "..."


def _format_rows(
    cols: list[str], rows: list[sqlite3.Row], truncated: bool, limit: int
) -> str:
    if not cols:
        return "(query returned no columns)"
    header = " | ".join(cols)
    body = "\n".join(
        " | ".join(_cell(r[i]) for i in range(len(cols))) for r in rows
    )
    footer = f"\n\n({len(rows)} row(s)"
    if truncated:
        footer += f"; truncated at max_rows={limit} — narrow with LIMIT/WHERE"
    footer += ")"
    rule = "-" * min(len(header), 100)
    return f"{header}\n{rule}\n{body}{footer}"


@mcp.tool()
def list_databases() -> str:
    """List the read-only SQLite databases available, with their table counts."""
    lines = []
    for name, path in sorted(_DBS.items()):
        try:
            con = _connect(name)
            n = con.execute(
                "SELECT count(*) FROM sqlite_master WHERE type='table'"
            ).fetchone()[0]
            con.close()
            lines.append(f"- {name}: {path}  ({n} tables)")
        except Exception as exc:  # surface, don't crash the listing
            lines.append(f"- {name}: {path}  (ERROR: {exc})")
    return "Read-only databases:\n" + "\n".join(lines)


@mcp.tool()
def list_tables(database: str) -> str:
    """List tables and views in a database.

    Args:
        database: which DB to inspect ('metadata' or 'enrichment').
    """
    con = _connect(database)
    try:
        rows = con.execute(
            "SELECT type, name FROM sqlite_master "
            "WHERE type IN ('table','view') AND name NOT LIKE 'sqlite_%' "
            "ORDER BY type, name"
        ).fetchall()
    finally:
        con.close()
    if not rows:
        return f"(no tables in {database})"
    return "\n".join(f"{r['type']}\t{r['name']}" for r in rows)


@mcp.tool()
def describe_table(database: str, table: str) -> str:
    """Show columns, primary key, foreign keys, and indexes for a table or view.

    Args:
        database: which DB ('metadata' or 'enrichment').
        table: table or view name.
    """
    con = _connect(database)
    try:
        exists = con.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type IN ('table','view') AND name = ?",
            (table,),
        ).fetchone()
        if not exists:
            raise ValueError(f"No such table/view {table!r} in {database}")
        # table is now a validated identifier; quote-escape defensively anyway.
        q = '"' + table.replace('"', '""') + '"'
        cols = con.execute(f"PRAGMA table_info({q})").fetchall()
        fks = con.execute(f"PRAGMA foreign_key_list({q})").fetchall()
        idx = con.execute(f"PRAGMA index_list({q})").fetchall()
    finally:
        con.close()

    out = [f"# {database}.{table}", "", "Columns:"]
    for c in cols:  # cid, name, type, notnull, dflt_value, pk
        flags = []
        if c["pk"]:
            flags.append(f"PK{c['pk'] if c['pk'] > 1 else ''}")
        if c["notnull"]:
            flags.append("NOT NULL")
        if c["dflt_value"] is not None:
            flags.append(f"DEFAULT {c['dflt_value']}")
        line = f"  {c['name']} {c['type'] or ''}".rstrip()
        if flags:
            line += "  [" + ", ".join(flags) + "]"
        out.append(line)
    if fks:
        out += ["", "Foreign keys:"]
        out += [f"  {f['from']} -> {f['table']}.{f['to']}" for f in fks]
    if idx:
        out += ["", "Indexes:"]
        out += [
            f"  {ix['name']}{' UNIQUE' if ix['unique'] else ''}" for ix in idx
        ]
    return "\n".join(out)


@mcp.tool()
def run_query(database: str, sql: str, max_rows: int = DEFAULT_MAX_ROWS) -> str:
    """Run a read-only SQL query and return the rows.

    Writes are impossible: the connection is opened read-only. Only a single
    SELECT / WITH / PRAGMA / EXPLAIN statement is accepted.

    Args:
        database: which DB ('metadata' or 'enrichment').
        sql: one read-only SQL statement.
        max_rows: row cap (default 200, hard cap 1000).
    """
    if not _strip_leading(sql).lower().startswith(_READ_PREFIXES):
        raise ValueError(
            "Only read-only statements are allowed "
            "(SELECT / WITH / PRAGMA / EXPLAIN)."
        )
    limit = max(1, min(int(max_rows), MAX_ROWS_CAP))
    con = _connect(database)
    try:
        cur = con.execute(sql)  # single-statement; chaining raises naturally
        cols = [d[0] for d in cur.description] if cur.description else []
        rows = cur.fetchmany(limit + 1)
    finally:
        con.close()
    truncated = len(rows) > limit
    return _format_rows(cols, rows[:limit], truncated, limit)


if __name__ == "__main__":
    mcp.run()
