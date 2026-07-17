"""Category A transform golden tables — identity/normalization layer capture.

Corridor Phase 0a, Task 3. Captures byte-identical, deterministic input->output
snapshots of the production identity/alias/genre-authority/dedupe transforms
against the LIVE library (data/metadata.db + the MuQ artifact), so a future
refactor of that layer can be diffed against a known-good baseline.

Tables captured (see docs/corridor_baseline/tables/*.json.gz):
  - artist_identity: every distinct `tracks.artist` string -> its raw/resolved
    identity key + resolve_artist_identity_keys() set.
  - title_keys: every distinct `tracks.title` string -> normalize_title_key().
  - alias_links: data/artist_aliases.yaml groups + per-member resolve_alias()/
    sibling_group_of() results, called the way production actually calls them
    (resolve_alias on the pre-normalized raw key -- identity_keys.py:62;
    sibling_group_of on the raw display-name artist string -- pier_bridge/beam.py:917-924).
  - genre_authority: resolved_genres_by_album() over the full DB.
  - dedup_collapse: dedupe_pool_by_track_key() applied to each artist's full
    bundle-index pool, for the fixed 6-artist CORPUS plus a random.Random(0)
    50-artist sample of all distinct DB artists.

A4 decision (deliberately NOT captured standalone): the solo/collab clustering
partition (`cluster_artist_tracks`, 12+ params) cannot be invoked faithfully
outside the pier-bridge orchestrator without hand-assembling most of its
call-site state, which would make the "golden" capture a reimplementation
rather than a wrap. It is instead covered by Task 4's per-run `solo=N collab=M`
log capture + pier track_ids, taken from real end-to-end generations.

Read-only: the DB is opened via `sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)`.
Zero writes, zero ATTACH, zero PRAGMA journal changes.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import random
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

CODE_ROOT = Path(__file__).resolve().parents[2]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from scripts.corridor_baseline.runner import CORPUS, DETENTS, OUT_DIR, build_cell_config  # noqa: E402

logger = logging.getLogger(__name__)


def write_table(name: str, obj: Any, out_dir: Path | None = None) -> dict:
    out_dir = out_dir or (OUT_DIR / "tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    path = out_dir / f"{name}.json.gz"
    with open(path, "wb") as fh:
        with gzip.GzipFile(fileobj=fh, mode="wb", mtime=0) as gz:  # mtime=0 -> reproducible bytes
            gz.write(raw)
    try:
        rel_path = str(path.relative_to(CODE_ROOT))
    except ValueError:
        # out_dir outside CODE_ROOT (e.g. a test's tmp_path) -- fall back to the raw path.
        rel_path = str(path)
    return {"name": name, "path": rel_path,
            "sha256": hashlib.sha256(raw).hexdigest(),  # hash the JSON, not the gz wrapper
            "rows": len(obj) if isinstance(obj, (dict, list)) else 1}


# ---- production config wiring (no hand-assembled defaults) -----------------

def _artist_identity_config(merged_cfg: dict):
    """Build ArtistIdentityConfig exactly as production does.

    Copy-adapted from src/playlist/pipeline/core.py:310-323 (generate_playlist_ds),
    fed the same `overrides` shape that src/playlist_generator.py:585 builds via
    build_ds_overrides(playlists.ds_pipeline). Reusing build_ds_overrides (rather
    than reading merged_cfg['constraints'] directly) matters: build_ds_overrides
    is what makes `overrides['constraints']` come from `playlists.ds_pipeline.constraints`,
    not the merged config's top-level `constraints` key.
    """
    from src.playlist.artist_identity_resolver import ArtistIdentityConfig
    from src.playlist_generator import build_ds_overrides

    ds_cfg = (merged_cfg.get("playlists", {}) or {}).get("ds_pipeline", {}) or {}
    overrides = build_ds_overrides(ds_cfg)
    constraints_overrides = overrides.get("constraints", {}) if isinstance(overrides, dict) else {}
    artist_identity_overrides = constraints_overrides.get("artist_identity", {})
    if isinstance(artist_identity_overrides, dict):
        return ArtistIdentityConfig(
            enabled=bool(artist_identity_overrides.get("enabled", False)),
            split_delimiters=list(artist_identity_overrides.get("split_delimiters", []))
            if artist_identity_overrides.get("split_delimiters")
            else None or ArtistIdentityConfig().split_delimiters,
            strip_trailing_ensemble_terms=bool(
                artist_identity_overrides.get("strip_trailing_ensemble_terms", True)
            ),
            trailing_ensemble_terms=list(artist_identity_overrides.get("trailing_ensemble_terms", []))
            if artist_identity_overrides.get("trailing_ensemble_terms")
            else None or ArtistIdentityConfig().trailing_ensemble_terms,
        )
    return ArtistIdentityConfig()


# ---- capture functions -------------------------------------------------------

def capture_artist_identity(conn: sqlite3.Connection, cfg) -> dict:
    from src.playlist.artist_identity_resolver import resolve_artist_identity_keys
    from src.playlist.identity_keys import _primary_artist_key_raw, normalize_primary_artist_key

    rows = conn.execute("SELECT DISTINCT artist FROM tracks").fetchall()
    out: dict[str, Any] = {}
    for (artist,) in rows:
        out[artist] = {
            "raw_key": _primary_artist_key_raw(artist),
            "resolved": normalize_primary_artist_key(artist),
            "identity_keys": sorted(resolve_artist_identity_keys(artist, cfg)),
        }
    return out


def capture_title_keys(conn: sqlite3.Connection) -> dict:
    from src.playlist.identity_keys import normalize_title_key

    # Raw title column verified via PRAGMA table_info(tracks): `title`, NOT `norm_title`.
    rows = conn.execute("SELECT DISTINCT title FROM tracks").fetchall()
    return {title: normalize_title_key(title) for (title,) in rows}


def capture_alias_links() -> dict:
    from src.playlist.artist_aliases import read_artist_link_groups, resolve_alias, sibling_group_of
    from src.playlist.identity_keys import _primary_artist_key_raw

    groups = read_artist_link_groups()
    members: dict[str, Any] = {}
    for group in groups:
        for member in group.get("members", []) or []:
            members[member] = {
                # Production only ever calls resolve_alias on the pre-normalized raw
                # key (normalize_primary_artist_key = resolve_alias(_primary_artist_key_raw(v)),
                # identity_keys.py:62) -- feeding it a bare display name would just
                # capture a dict-miss no-op, not real behavior.
                "resolve_alias": resolve_alias(_primary_artist_key_raw(member)),
                # sibling_group_of IS called on the raw display-name artist string in
                # production (pier_bridge/beam.py:917-924) -- it normalizes internally
                # as a fallback.
                "sibling_group": sibling_group_of(member),
            }
    return {"groups": groups, "members": members}


def capture_genre_authority(conn: sqlite3.Connection) -> dict:
    from src.genre.authority import resolved_genres_by_album

    by_album = resolved_genres_by_album(conn)
    return {
        album_id: sorted([r.genre_id, r.assignment_layer, r.confidence, r.source] for r in rows)
        for album_id, rows in by_album.items()
    }


def capture_dedup_collapse(conn: sqlite3.Connection, bundle) -> dict:
    from src.playlist.identity_keys import identity_keys_for_index
    from src.playlist.pipeline.pier_resolver import dedupe_pool_by_track_key

    all_artists = sorted(r[0] for r in conn.execute("SELECT DISTINCT artist FROM tracks").fetchall())
    sample = random.Random(0).sample(all_artists, 50)
    artists = sorted(set(CORPUS) | set(sample))

    # artist (raw display name) -> bundle indices. Matches production's own
    # candidate-pool grouping, which reads bundle.track_artists directly
    # (e.g. pier_bridge/beam.py:917-924) -- never a DB re-query per track.
    by_artist: dict[str, list[int]] = {}
    if bundle.track_artists is not None:
        for idx, artist in enumerate(bundle.track_artists.tolist()):
            by_artist.setdefault(str(artist), []).append(idx)

    out: dict[str, Any] = {}
    for artist in artists:
        pool_indices = by_artist.get(artist, [])
        if not pool_indices:
            out[artist] = {}
            continue
        key_to_indices: dict[tuple, list[int]] = {}
        for idx in pool_indices:
            key = identity_keys_for_index(bundle, idx).track_key
            key_to_indices.setdefault(key, []).append(idx)
        deduped = set(dedupe_pool_by_track_key(bundle, pool_indices))
        mapping: dict[str, list[str]] = {}
        for indices in key_to_indices.values():
            kept = next((i for i in indices if i in deduped), indices[0])
            kept_id = str(bundle.track_ids[kept])
            mapping[kept_id] = sorted(str(bundle.track_ids[i]) for i in indices)
        out[artist] = mapping
    return out


# ---- orchestration -----------------------------------------------------------

def main() -> None:
    # No basicConfig in scripts/ (test_no_basicconfig_in_src_scripts): attach one
    # console handler explicitly instead.
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logging.getLogger().addHandler(_h)
    logging.getLogger().setLevel(logging.INFO)
    t0 = time.time()

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--out-dir", default=None,
        help="Override the output directory for tables/*.json.gz + transforms_summary.json "
             "(default: docs/corridor_baseline/). Use a scratch dir for a paired-capture "
             "comparison run so it never collides with the committed baseline.",
    )
    args = ap.parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    tables_dir = out_dir / "tables"

    merged = build_cell_config(DETENTS["open"])
    db_path = merged["library"]["database_path"]
    artifact_path = merged["playlists"]["ds_pipeline"]["artifact_path"]

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        cfg = _artist_identity_config(merged)

        metas = []
        logger.info("capturing artist_identity ...")
        metas.append(write_table("artist_identity", capture_artist_identity(conn, cfg), out_dir=tables_dir))
        logger.info("capturing title_keys ...")
        metas.append(write_table("title_keys", capture_title_keys(conn), out_dir=tables_dir))
        logger.info("capturing alias_links ...")
        metas.append(write_table("alias_links", capture_alias_links(), out_dir=tables_dir))
        logger.info("capturing genre_authority ...")
        metas.append(write_table("genre_authority", capture_genre_authority(conn), out_dir=tables_dir))

        logger.info("loading artifact bundle (single load) ...")
        from src.features.artifacts import load_artifact_bundle
        bundle = load_artifact_bundle(artifact_path)
        logger.info("capturing dedup_collapse ...")
        metas.append(write_table("dedup_collapse", capture_dedup_collapse(conn, bundle), out_dir=tables_dir))

        db_track_count = conn.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]
    finally:
        conn.close()

    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=str(CODE_ROOT), capture_output=True, text=True, check=True,
    ).stdout.strip()

    summary = {"tables": metas, "generated_on": git_sha, "db_track_count": db_track_count}
    summary_path = out_dir / "transforms_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, sort_keys=True, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info("done in %.1fs; summary -> %s", time.time() - t0, summary_path)


if __name__ == "__main__":
    main()
